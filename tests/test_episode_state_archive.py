import asyncio
import dataclasses
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from types import SimpleNamespace

import pytest
from episode_fixtures import IMAGE
from fake_boxes import FakeBoxes
from fake_episode_env import sandbox_task

from reliquary_sandbox_service.episodes.grader import Grader
from reliquary_sandbox_service.episodes.local_runtime import (
    FileTooLarge,
    LocalRuntime,
    StateUnreadable,
)
from reliquary_sandbox_service.episodes.state_archive import MANIFEST, rebuild_for_restore
from reliquary_sandbox_service.task_runtime import BoxSpec

SPEC = BoxSpec(image=IMAGE, workdir="/work", env={}, user=None, memory_bytes=1, nano_cpus=1,
               pids=16, disk_bytes=1, lifetime_s=60)
BUSYBOX = shutil.which("busybox")


@pytest.fixture
def boxes():
    boxes = FakeBoxes([IMAGE])
    boxes.create("s-1", "s-1", SPEC)
    boxes.create("p", "s-1", SPEC)
    return boxes


def runtime(boxes, box_id="s-1"):
    return LocalRuntime(boxes, box_id, workdir="/work", env={}, user="0:0",
                        deadline=time.monotonic() + 10, output_limit=1024)


def entry(name, kind="file", payload=b"", mode=0o644):
    info = tarfile.TarInfo(name)
    info.mode = mode
    if kind == "file":
        info.size = len(payload)
        return info, payload
    info.type = {"dir": tarfile.DIRTYPE, "sym": tarfile.SYMTYPE, "lnk": tarfile.LNKTYPE,
                 "fifo": tarfile.FIFOTYPE}[kind]
    if kind in ("sym", "lnk"):
        info.linkname = payload
    return info, None


def tar_bytes(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for info, payload in entries:
            tar.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return buffer.getvalue()


def state(*members, roots=("/app",), manifest=True):
    head = []
    if manifest:
        head = [entry(MANIFEST, "file", json.dumps({"version": 1, "roots": list(roots)}).encode())]
    return tar_bytes(head + [entry(*m) for m in members])


def names(data):
    return tarfile.open(fileobj=io.BytesIO(data)).getnames()


def seed_agent_tree(boxes):
    box = boxes.boxes["s-1"]
    box["files"].update({"/app/a.py": b"a", "/app/pkg/b.py": b"b", "/other/secret": b"s"})
    box["links"] = {"/app/link": "pkg/b.py", "/app/venv/bin/python": "/usr/bin/python3"}


# -- archive ----------------------------------------------------------------

def test_archive_names_members_relative_to_their_root_with_a_manifest(boxes):
    seed_agent_tree(boxes)
    data = asyncio.run(runtime(boxes).archive(["/app"], 100_000))
    listed = names(data)
    assert listed[0] == MANIFEST
    assert sorted(listed[1:]) == ["0/a.py", "0/link", "0/pkg/b.py", "0/venv/bin/python"]
    manifest = tarfile.open(fileobj=io.BytesIO(data)).extractfile(MANIFEST).read()
    assert json.loads(manifest) == {"version": 1, "roots": ["/app"]}
    assert boxes.helper_calls[-1][2]["user"] == "0:0"


def test_archive_allows_any_symlink_target(boxes):
    seed_agent_tree(boxes)
    boxes.boxes["s-1"]["links"]["/app/up"] = "../../etc/passwd"
    data = asyncio.run(runtime(boxes).archive(["/app"], 100_000))
    links = {m.name: m.linkname for m in tarfile.open(fileobj=io.BytesIO(data)) if m.issym()}
    assert links["0/up"] == "../../etc/passwd"
    assert links["0/venv/bin/python"] == "/usr/bin/python3"


@pytest.mark.parametrize("paths", [["/"], ["../etc"], ["/app/../etc"], [], ["/app", "/app/x"]])
def test_archive_refuses_root_dotdot_empty_and_nested_paths(boxes, paths):
    with pytest.raises(ValueError):
        asyncio.run(runtime(boxes).archive(paths, 100_000))


def test_archive_over_its_bound_is_too_large(boxes):
    boxes.boxes["s-1"]["files"]["/app/a"] = b"x" * 50_000
    rt = runtime(boxes)
    with pytest.raises(FileTooLarge):
        asyncio.run(rt.archive(["/app"], 20_000))
    assert isinstance(rt.faults[-1], FileTooLarge)


def test_a_tar_failure_on_the_agents_tree_is_state_unreadable(boxes):
    rt = runtime(boxes)
    with pytest.raises(StateUnreadable):
        asyncio.run(rt.archive(["/missing"], 100_000))
    assert isinstance(rt.faults[-1], StateUnreadable)


@pytest.mark.parametrize("raw", [
    [("etc/passwd", "file", b"x")],
    [("app/fifo", "fifo")],
    [("app/h", "lnk", "app/nothing")],
])
def test_archive_refuses_what_busybox_should_not_have_produced(boxes, monkeypatch, raw):
    monkeypatch.setattr(boxes, "tar_create",
                        lambda *a, **k: tar_bytes([entry(*m) for m in raw]))
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime(boxes).archive(["/app"], 100_000))


def test_archive_keeps_hardlinks_as_links_and_strips_privileges(boxes, monkeypatch):
    raw = tar_bytes([entry("app/a", "file", b"data", mode=0o4755), entry("app/h", "lnk", "app/a")])
    monkeypatch.setattr(boxes, "tar_create", lambda *a, **k: raw)
    data = asyncio.run(runtime(boxes).archive(["/app"], 100_000))
    members = {m.name: m for m in tarfile.open(fileobj=io.BytesIO(data))}
    assert members["0/h"].islnk() and members["0/h"].linkname == "0/a"
    assert members["0/a"].mode & 0o6000 == 0 and members["0/a"].uid == 0


def test_an_unarchivable_state_grades_zero_without_counting_an_env_bug(boxes):
    async def extract(rt):
        return await rt.archive(["/missing"], 100_000)

    grader = Grader(SimpleNamespace(episode_grader_output_bytes=1024, episode_grading_attempts=2),
                    boxes)
    task = dataclasses.replace(sandbox_task("train", 0), extract=extract)
    outcome = grader.grade("s-1", "s-1", task, SPEC, env="fake-env")
    assert outcome.facts == {"_grader": {"agent_fault": "state_unreadable"}}
    assert grader.env_failure_counts() == {}


# -- restore ----------------------------------------------------------------

def test_round_trip_keeps_internal_and_absolute_venv_links(boxes):
    seed_agent_tree(boxes)
    data = asyncio.run(runtime(boxes).archive(["/app"], 100_000))
    asyncio.run(runtime(boxes, "p").restore_archive(data, ["/app"]))
    pristine = boxes.boxes["p"]
    assert pristine["files"]["/app/pkg/b.py"] == b"b"
    assert pristine["links"]["/app/link"] == "pkg/b.py"
    assert pristine["links"]["/app/venv/bin/python"] == "/usr/bin/python3"
    assert "/other/secret" not in pristine["files"]
    assert [(d, names(t)) for _, d, t in boxes.extracted] == [
        ("/app", ["a.py", "link", "pkg/b.py", "venv/bin/python"])]


def test_restore_maps_root_i_to_the_given_root(boxes):
    data = state(("0/x", "file", b"1"), ("1/y", "file", b"2"), roots=("/a", "/b"))
    asyncio.run(runtime(boxes, "p").restore_archive(data, ["/srv/one", "/srv/two"]))
    assert boxes.boxes["p"]["files"]["/srv/one/x"] == b"1"
    assert boxes.boxes["p"]["files"]["/srv/two/y"] == b"2"
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime(boxes, "p").restore_archive(data, ["/only-one"]))


def test_writes_never_go_through_a_restored_link_leaving_the_roots(boxes):
    pristine = boxes.boxes["p"]
    pristine["files"]["/usr/bin/python3"] = b"elf"
    data = state(("0/venv/bin/python", "sym", "/usr/bin/python3"), ("0/lib", "sym", "/usr/bin"),
                 ("0/real", "dir"))
    rt = runtime(boxes, "p")
    asyncio.run(rt.restore_archive(data, ["/app"]))
    for path in ("/app/lib/evil", "/app/venv/bin/python/x"):
        with pytest.raises(StateUnreadable):
            asyncio.run(rt.write(path, b"x"))
    assert "/usr/bin/evil" not in pristine["files"]
    asyncio.run(rt.write("/app/real/ok.txt", b"ok"))
    asyncio.run(rt.write("/tmp/tests.diff", b"ok"))  # outside the restored roots: unrestricted
    assert pristine["files"]["/app/real/ok.txt"] == b"ok"


def test_a_root_restored_as_a_symlink_does_not_widen_writes(boxes):
    boxes.boxes["p"]["files"]["/usr/x"] = b"1"
    rt = runtime(boxes, "p")
    asyncio.run(rt.restore_archive(state(("0", "sym", "/usr")), ["/app"]))
    with pytest.raises(StateUnreadable):
        asyncio.run(rt.write("/app/evil", b"x"))


def test_reads_never_follow_a_restored_link_leaving_the_roots(boxes):
    pristine = boxes.boxes["p"]
    pristine["files"]["/tests/expected_output"] = b"reference"
    data = state(("0/output", "sym", "/tests/expected_output"), ("0/dir", "sym", "/tests"),
                 ("0/real/ok.txt", "file", b"ok"), ("0/inside", "sym", "real/ok.txt"))
    rt = runtime(boxes, "p")
    asyncio.run(rt.restore_archive(data, ["/app"]))
    for path in ("/app/output", "/app/dir/expected_output", "/app//dir/./expected_output",
                 "/app/dir/../app/output"):
        with pytest.raises(StateUnreadable):
            asyncio.run(rt.read(path))
    assert asyncio.run(rt.read("/app/real/ok.txt")) == b"ok"
    assert asyncio.run(rt.read("/app/inside")) == b"ok"  # a link that stays inside
    assert asyncio.run(rt.read("/tests/expected_output")) == b"reference"  # outside: the env's
    with pytest.raises(FileNotFoundError):
        asyncio.run(rt.read("/app/missing"))


def test_a_root_restored_as_a_symlink_is_never_read_through(boxes):
    boxes.boxes["p"]["files"]["/tests/output"] = b"reference"
    rt = runtime(boxes, "p")
    asyncio.run(rt.restore_archive(state(("0", "sym", "/tests")), ["/app"]))
    for path in ("/app/output", "/app"):
        with pytest.raises(StateUnreadable):
            asyncio.run(rt.read(path))


@pytest.mark.parametrize("members", [
    [("/abs", "file", b"x")],
    [("../x", "file", b"x")],
    [("0/../../x", "file", b"x")],
    [("5/x", "file", b"x")],
    [("x", "file", b"x")],
    [("01/x", "file", b"x")],
    [("0/l", "sym", "../etc")],
    [("0/l", "sym", "a/../../b")],
    [("0/l", "sym", "")],
    [("0/s", "sym", "."), ("0/t", "sym", "s/../../outside")],
    [("0/a", "sym", "b"), ("0/b", "sym", "a")],
    [("0/l", "sym", "../../..")],
    [("0/l", "sym", "..")],
    [("0/deep", "dir"), ("0/l", "sym", "deep/../../x")],
    [("0/out", "sym", "/tmp"), ("0/t", "sym", "out/../app/x")],
    [("0/out", "sym", "/usr"), ("0/t", "sym", "out/bin")],
    [("0/l", "sym", "/app/../etc")],
    [("0/l", "sym", "/app/x/../../etc/passwd")],
    [("0/s", "sym", "s2"), ("0/s2", "sym", "../x")],
    [("0/d", "sym", "sub"), ("0/d/file", "file", b"x")],
    [("0/d", "sym", "/etc"), ("0/d/passwd", "file", b"x")],
    [("0", "sym", "/etc"), ("0/passwd", "file", b"x")],
    [("0/h", "lnk", "0/missing")],
    [("0/f", "fifo")],
    [("0/a", "file", b"1"), ("0/a", "sym", "/etc")],
])
def test_restore_refuses_members_that_could_escape(boxes, members):
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime(boxes, "p").restore_archive(state(*members), ["/app"]))
    assert boxes.extracted == []  # nothing reached the box


@pytest.mark.parametrize("members", [
    [("0/node_modules/pkg/bin/x", "file", b"#!"),
     ("0/node_modules/.bin/x", "sym", "../pkg/bin/x")],
    [("0/venv/bin/python", "sym", "/usr/bin/python3"), ("0/venv/bin/python3", "sym", "python")],
    [("0/a/b", "dir"), ("0/a/b/l", "sym", "../../c/../a/b")],
    [("0/s", "sym", "."), ("0/t", "sym", "s/x")],
    [("0/abs", "sym", "/app/inside")],
    [("0/x", "sym", "/srv/other/y")],
])
def test_restore_accepts_links_that_resolve_inside_or_are_inert(boxes, members):
    asyncio.run(runtime(boxes, "p").restore_archive(state(*members), ["/app"]))
    assert len(boxes.extracted) == 1


def test_links_between_roots_must_be_absolute(boxes):
    # A relative walk may not step outside the roots, even to come back into another one.
    relative = state(("0/l", "sym", "../srv/x"), ("1/x", "file", b"1"), roots=("/app", "/srv"))
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime(boxes, "p").restore_archive(relative, ["/app", "/srv"]))
    absolute = state(("0/l", "sym", "/srv/x"), ("1/x", "file", b"1"), roots=("/app", "/srv"))
    asyncio.run(runtime(boxes, "p").restore_archive(absolute, ["/app", "/srv"]))


# -- contained links: the bridge's grading restore (image tools follow links) ----------

def restored_paths(boxes, box_id, root="/app"):
    box = boxes.boxes[box_id]
    return sorted(p for p in [*box["files"], *box.get("links", {}), *box["dirs"]]
                  if p == root or p.startswith(root + "/"))


@pytest.mark.parametrize("members", [
    [("0/output", "sym", "/tests/expected_output")],  # an absolute link to the reference
    [("0/sub", "dir"), ("0/sub/x", "sym", "../../tests/x")],  # relative, leaving the roots
    [("0", "sym", "/tests")],  # the root itself is a link
    [("0/a", "sym", "b"), ("0/b", "sym", "/tests/expected_output")],  # out in two hops
    [("0/dangling", "sym", "missing")],  # inside the roots, but nothing there
    [("0/a", "sym", "b"), ("0/b", "sym", "a")],  # a loop
], ids=["absolute", "relative", "root_link", "two_hops", "dangling", "loop"])
def test_a_contained_restore_refuses_any_link_not_leading_inside_the_roots(boxes, members):
    boxes.boxes["p"]["files"]["/tests/expected_output"] = b"reference"
    boxes.boxes["p"]["files"]["/tests/x"] = b"reference"
    rt = runtime(boxes, "p")
    with pytest.raises(StateUnreadable):
        asyncio.run(rt.restore_archive(state(*members), ["/app"], contained_links=True))
    assert restored_paths(boxes, "p") == []  # never left restored
    assert isinstance(rt.faults[-1], StateUnreadable)


def test_a_contained_restore_checks_the_links_in_the_box_itself(boxes):
    # The archive alone cannot tell a dangling link: the box is asked where each link leads.
    rt = runtime(boxes, "p")
    with pytest.raises(StateUnreadable, match="/app/dangling"):
        asyncio.run(rt.restore_archive(state(("0/f", "file", b"1"), ("0/dangling", "sym", "nope")),
                                       ["/app"], contained_links=True))
    assert [name for name, box, _ in boxes.helper_calls if box == "p"].count("resolve_links") == 1
    assert len(boxes.extracted) == 1 and restored_paths(boxes, "p") == []


def test_a_contained_restore_keeps_links_inside_the_roots(boxes):
    data = state(("0/real/ok.txt", "file", b"ok"), ("0/inside", "sym", "real/ok.txt"),
                 ("0/abs", "sym", "/app/real/ok.txt"), ("0/dir", "sym", "real"),
                 ("0/chain", "sym", "inside"), ("1/x", "file", b"x"), ("0/other", "sym", "/srv/x"),
                 roots=("/app", "/srv"))
    rt = runtime(boxes, "p")
    asyncio.run(rt.restore_archive(data, ["/app", "/srv"], contained_links=True))
    for path in ("/app/inside", "/app/abs", "/app/dir/ok.txt", "/app/chain"):
        assert asyncio.run(rt.read(path)) == b"ok"
    assert asyncio.run(rt.read("/app/other")) == b"x"


def test_only_a_contained_restore_refuses_inert_links_before_extraction():
    data = state(("0/python", "sym", "/usr/bin/python3"))
    assert len(rebuild_for_restore(data, ["/app"])) == 1  # the default keeps inert links
    with pytest.raises(StateUnreadable, match="outside the roots"):
        rebuild_for_restore(data, ["/app"], contained_links=True)


@pytest.mark.parametrize("data", [
    state(("0/a", "file", b"1"), manifest=False),
    tar_bytes([entry("0/a", "file", b"1"), entry(MANIFEST, "file", b'{"version":1,"roots":[]}')]),
    tar_bytes([entry(MANIFEST, "file", b"not json")]),
    b"not a tar" * 100,
])
def test_restore_refuses_archives_not_in_the_state_format(boxes, data):
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime(boxes, "p").restore_archive(data, ["/app"]))


def test_restore_keeps_hardlinks_as_links_without_owners_or_setuid(boxes):
    data = state(("0/a", "file", b"data", 0o6755), ("0/h", "lnk", "0/a"))
    asyncio.run(runtime(boxes, "p").restore_archive(data, ["/app"]))
    (_, _, clean), = boxes.extracted
    members = {m.name: m for m in tarfile.open(fileobj=io.BytesIO(clean))}
    assert members["h"].islnk() and members["h"].linkname == "a"
    assert members["a"].mode == 0o755 and members["a"].uid == 0 and members["a"].uname == ""
    assert boxes.boxes["p"]["files"]["/app/h"] == b"data"


def raw_member(name, kind="file", payload=b""):
    info, data = entry(name, kind, payload)
    block = info.tobuf(tarfile.USTAR_FORMAT)
    if data is not None:
        block += data + b"\0" * (-len(data) % 512)
    return block


def crafted_archive():
    """The reviewer's archive: visible members, ONE zero block (Python's tarfile
    stops there, busybox does not), then a hidden hardlink and FIFO."""
    manifest = json.dumps({"version": 1, "roots": ["/app"]}).encode()
    return (raw_member(MANIFEST, "file", manifest) + raw_member("0/ok", "file", b"fine")
            + b"\0" * 512
            + raw_member("0/hidden-link", "lnk", "0/ok") + raw_member("0/hidden-fifo", "fifo")
            + b"\0" * 1024)


def test_members_hidden_after_a_zero_block_are_dropped(boxes):
    asyncio.run(runtime(boxes, "p").restore_archive(crafted_archive(), ["/app"]))
    (_, _, clean), = boxes.extracted
    assert names(clean) == ["ok"]


@pytest.mark.skipif(BUSYBOX is None, reason="busybox is not installed")
def test_real_busybox_never_sees_hidden_members(tmp_path):
    crafted = crafted_archive()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    subprocess.run([BUSYBOX, "tar", "-xof", "-", "-C", str(raw_dir)], input=crafted, check=False,
                   capture_output=True, timeout=30)
    # The attack is real: busybox reads past the zero block.
    assert (raw_dir / "0" / "hidden-fifo").exists() or (raw_dir / "0" / "hidden-link").exists()

    clean_root = tmp_path / "clean" / "app"
    for _, directory, clean in rebuild_for_restore(crafted, [str(clean_root)]):
        os.makedirs(directory, exist_ok=True)  # TAR_EXTRACT's `mkdir -p`
        subprocess.run([BUSYBOX, "tar", "-xof", "-", "-C", directory], input=clean, check=True,
                       capture_output=True, timeout=30)
    assert sorted(p.name for p in clean_root.iterdir()) == ["ok"]
    assert (clean_root / "ok").read_bytes() == b"fine"


# -- round 4: hardlink bombs, long names, cleared roots ----------------------

def hardlink_bomb(prefix, links=300, size=1024 * 1024):
    return [entry(f"{prefix}/big", "file", b"x" * size)] + [
        entry(f"{prefix}/h{i}", "lnk", f"{prefix}/big") for i in range(links)]


def test_a_hardlink_bomb_stays_small_and_memory_stays_bounded(boxes, monkeypatch):
    import tracemalloc

    from reliquary_sandbox_service.episodes.state_archive import normalise_created

    raw = tar_bytes(hardlink_bomb("app"))
    tracemalloc.start()
    try:
        data = normalise_created(raw, ["/app"], 4 * 1024 * 1024)
        _, created_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        rebuilt = rebuild_for_restore(data, ["/app"])
        _, restored_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(data) < 2 * len(raw) and len(rebuilt[0][2]) < 2 * len(raw)
    # Expanding the links would need ~300 MiB; the rebuild holds about one copy.
    assert created_peak < 4 * len(raw) and restored_peak < 4 * len(raw)
    members = tarfile.open(fileobj=io.BytesIO(rebuilt[0][2])).getmembers()
    assert sum(m.islnk() for m in members) == 300


def test_the_archive_bound_is_enforced_while_writing(boxes, monkeypatch):
    raw = tar_bytes([entry(f"app/f{i}", "file", b"y" * 4096) for i in range(20)])
    monkeypatch.setattr(boxes, "tar_create", lambda *a, **k: raw)
    with pytest.raises(FileTooLarge):
        asyncio.run(runtime(boxes).archive(["/app"], 30_000))


@pytest.mark.parametrize("members", [
    [("0/a", "file", b"1"), ("0/h", "lnk", "0/later"), ("0/later", "file", b"2")],
    [("0/d", "dir"), ("0/h", "lnk", "0/d")],
    [("0/s", "sym", "a"), ("0/h", "lnk", "0/s")],
    [("0/f", "file", b"1"), ("0/f/x", "file", b"2")],
])
def test_restore_refuses_bad_hardlinks_and_members_below_files(boxes, members):
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime(boxes, "p").restore_archive(state(*members), ["/app"]))


def test_hardlinks_across_roots_are_refused(boxes):
    data = state(("0/a", "file", b"1"), ("1/h", "lnk", "0/a"), roots=("/a", "/b"))
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime(boxes, "p").restore_archive(data, ["/a", "/b"]))


def test_restore_clears_each_root_first(boxes):
    pristine = boxes.boxes["p"]
    pristine["files"].update({"/app/deleted-by-agent": b"image", "/app/kept": b"old"})
    asyncio.run(runtime(boxes, "p").restore_archive(state(("0/kept", "file", b"new")), ["/app"]))
    assert "/app/deleted-by-agent" not in pristine["files"]
    assert pristine["files"]["/app/kept"] == b"new"
    order = [c[0] for c in boxes.helper_calls if c[0] in ("clear_root", "tar_extract")]
    assert order == ["clear_root", "tar_extract"]


def test_a_root_that_is_a_symlink_in_the_image_aborts_grading(boxes):
    from reliquary_sandbox_service.task_runtime import InfraFault

    boxes.boxes["p"]["links"] = {"/app": "/srv"}
    with pytest.raises(InfraFault):
        asyncio.run(runtime(boxes, "p").restore_archive(state(("0/x", "file", b"1")), ["/app"]))

    async def extract(rt):
        return state(("0/x", "file", b"1"))

    async def grade(rt, data):
        boxes.boxes[rt.box_id].setdefault("links", {})["/app"] = "/srv"  # the image's layout
        await rt.restore_archive(data, ["/app"])

    grader = Grader(SimpleNamespace(episode_grader_output_bytes=1024, episode_grading_attempts=2),
                    boxes)
    task = dataclasses.replace(sandbox_task("train", 0), extract=extract, grade=grade)
    assert grader.grade("s-1", "s-1", task, SPEC, env="fake-env").status == "aborted"


def test_malformed_archive_errors_are_state_unreadable(boxes, monkeypatch):
    import reliquary_sandbox_service.episodes.local_runtime as local_runtime

    def broken(*args, **kwargs):
        raise ValueError("bad header")

    monkeypatch.setattr(local_runtime, "rebuild_for_restore", broken)
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime(boxes, "p").restore_archive(state(("0/x", "file", b"1")), ["/app"]))
    monkeypatch.setattr(local_runtime, "normalise_created", broken)
    seed_agent_tree(boxes)
    with pytest.raises(StateUnreadable):
        asyncio.run(runtime(boxes).archive(["/app"], 100_000))


def host(script):
    from reliquary_sandbox_service.task_runtime import FRAMED_SCRIPTS, HELPER, frame_markers

    def run(*args, stdin=None):
        nonce = ["0123456789abcdef"] if script in FRAMED_SCRIPTS else []
        done = subprocess.run([BUSYBOX, "sh", "-c", script.replace(HELPER, BUSYBOX), "x", *nonce,
                               *args], input=stdin, capture_output=True, timeout=60, cwd="/")
        if nonce:  # the real busybox prints the frame markers
            opened, closed = frame_markers(nonce[0])
            assert done.stdout.startswith(opened) and done.stdout.endswith(closed)
            done.stdout = done.stdout[len(opened):-len(closed)]
        return done
    return run


@pytest.mark.skipif(BUSYBOX is None, reason="busybox is not installed")
def test_real_busybox_round_trip_with_long_names_links_and_a_cleared_root(tmp_path):
    from reliquary_sandbox_service.episodes.state_archive import normalise_created
    from reliquary_sandbox_service.task_runtime import CLEAR_ROOT, TAR_CREATE, TAR_EXTRACT

    agent = tmp_path / "agent" / "app"
    long_dir = agent / ("d" * 100) / ("e" * 100) / ("f" * 100)
    long_dir.mkdir(parents=True)
    (long_dir / "file.txt").write_bytes(b"deep")
    long_rel = long_dir.relative_to(agent) / "file.txt"
    assert len(str(long_rel)) >= 300 and len(str(agent / long_rel)) >= 330
    (agent / "lib").mkdir()  # the agent replaced the image's lib symlink with a directory
    (agent / "lib" / "mod.py").write_bytes(b"mine")
    (agent / "a").write_bytes(b"a")
    (agent / "hard").hardlink_to(agent / "a")
    target = "t" * 150
    (agent / "longlink").symlink_to(target)

    created = host(TAR_CREATE)(str(agent).lstrip("/"))
    assert created.returncode == 0, created.stderr
    data = normalise_created(created.stdout, [str(agent)], 10 * 1024 * 1024)

    image = tmp_path / "image"
    outside = tmp_path / "outside"
    outside.mkdir()
    root = image / "app"
    root.mkdir(parents=True)
    (root / "lib").symlink_to("../../outside")  # an image symlink inside the root
    (root / "deleted-by-agent").write_bytes(b"image")

    for restore_root, directory, clean in rebuild_for_restore(data, [str(root)]):
        cleared = host(CLEAR_ROOT)(restore_root)
        assert cleared.returncode == 0, cleared.stderr
        extracted = host(TAR_EXTRACT)(directory, stdin=clean)
        assert extracted.returncode == 0, extracted.stderr

    assert list(outside.iterdir()) == []  # nothing written through the image's symlink
    assert not (root / "deleted-by-agent").exists()
    assert (root / "lib").is_dir() and not (root / "lib").is_symlink()
    assert (root / "lib" / "mod.py").read_bytes() == b"mine"
    assert (root / long_rel).read_bytes() == b"deep"
    assert os.readlink(root / "longlink") == target
    assert (root / "hard").read_bytes() == b"a"
    assert os.stat(root / "hard").st_ino == os.stat(root / "a").st_ino


@pytest.mark.skipif(BUSYBOX is None, reason="busybox is not installed")
def test_real_busybox_clear_refuses_a_root_that_is_a_symlink(tmp_path):
    from reliquary_sandbox_service.task_runtime import CLEAR_ROOT

    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "keep").write_bytes(b"1")
    (tmp_path / "root").symlink_to(tmp_path / "real")
    assert host(CLEAR_ROOT)(str(tmp_path / "root")).returncode == 3
    assert (tmp_path / "real" / "keep").exists()


# -- round 5: CPU bounds and restoring once -----------------------------------

def pax_state(members, roots=("/app",)):
    entries = [entry(MANIFEST, "file", json.dumps({"version": 1, "roots": list(roots)}).encode())]
    entries += [entry(*m) for m in members]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for info, payload in entries:
            tar.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return buffer.getvalue()


def chain(depth):
    return [("0/" + "/".join(["d"] * level), "dir") for level in range(1, depth + 1)]


def test_a_2040_deep_chain_is_refused_quickly():
    data = pax_state(chain(2040))
    started = time.perf_counter()
    with pytest.raises(StateUnreadable):
        rebuild_for_restore(data, ["/app"])
    assert time.perf_counter() - started < 2.0


def test_a_2040_deep_chain_is_refused_quickly_at_archive_time():
    from reliquary_sandbox_service.episodes.state_archive import normalise_created

    raw = pax_state([(name.replace("0", "app", 1), kind) for name, kind in chain(2040)])
    started = time.perf_counter()
    with pytest.raises(StateUnreadable):
        normalise_created(raw, ["/app"], 64 * 1024 * 1024)
    assert time.perf_counter() - started < 2.0


def test_250_deep_with_3000_links_vets_in_time():
    deepest = "0/" + "/".join(["d"] * 250)
    links = [(f"{deepest}/l{i}", "sym", "../" * 249 + f"t{i}") for i in range(3000)]
    data = pax_state(chain(250) + links)
    started = time.perf_counter()
    rebuilt = rebuild_for_restore(data, ["/app"])
    assert time.perf_counter() - started < 2.0
    assert len(tarfile.open(fileobj=io.BytesIO(rebuilt[0][2])).getmembers()) == 3250


def test_overlong_names_and_targets_are_refused():
    with pytest.raises(StateUnreadable):
        rebuild_for_restore(pax_state(chain(257)), ["/app"])
    rebuild_for_restore(pax_state(chain(256)), ["/app"])
    with pytest.raises(StateUnreadable):
        rebuild_for_restore(pax_state([("0/l", "sym", "/".join(["x"] * 257))]), ["/app"])


def test_member_count_and_resolution_work_are_capped(monkeypatch):
    import reliquary_sandbox_service.episodes.state_archive as state_archive

    monkeypatch.setattr(state_archive, "MAX_MEMBERS", 10)
    with pytest.raises(StateUnreadable):
        rebuild_for_restore(state(*[(f"0/f{i}", "file", b"") for i in range(11)]), ["/app"])
    monkeypatch.setattr(state_archive, "MAX_MEMBERS", 100_000)
    monkeypatch.setattr(state_archive, "MAX_RESOLUTION_STEPS", 50)
    links = [(f"0/l{i}", "sym", "a/b/c/d") for i in range(20)]
    with pytest.raises(StateUnreadable):
        rebuild_for_restore(state(*links), ["/app"])


def test_a_root_is_restored_once_per_runtime(boxes):
    from reliquary_sandbox_service.episodes.local_runtime import RuntimeMisuse

    rt = runtime(boxes, "p")
    asyncio.run(rt.restore_archive(state(("0/x", "file", b"1")), ["/app"]))
    for roots in (["/app"], ["/app/sub"]):
        with pytest.raises(RuntimeMisuse):
            asyncio.run(rt.restore_archive(state(("0/x", "file", b"1")), roots))
    asyncio.run(rt.restore_archive(state(("0/y", "file", b"2")), ["/other"]))


def test_a_repeated_restore_aborts_grading(boxes):
    async def extract(rt):
        return state(("0/x", "file", b"1"))

    async def grade(rt, data):
        await rt.restore_archive(data, ["/app"])
        await rt.restore_archive(data, ["/app"])

    grader = Grader(SimpleNamespace(episode_grader_output_bytes=1024, episode_grading_attempts=2),
                    boxes)
    task = dataclasses.replace(sandbox_task("train", 0), extract=extract, grade=grade)
    outcome = grader.grade("s-1", "s-1", task, SPEC, env="fake-env")
    assert outcome.status == "aborted" and "misuse" in outcome.reason


def test_each_restore_helper_gets_the_time_left_not_the_initial_budget(boxes, monkeypatch):
    data = state(("0/x", "file", b"1"), ("1/y", "file", b"2"), roots=("/a", "/b"))
    clear = boxes.clear_root

    def slow_clear(*args, **kwargs):
        time.sleep(0.3)
        return clear(*args, **kwargs)

    monkeypatch.setattr(boxes, "clear_root", slow_clear)
    boxes.helper_calls.clear()
    asyncio.run(runtime(boxes, "p").restore_archive(data, ["/srv/one", "/srv/two"]))
    budgets = [kwargs["timeout_s"] for _, _, kwargs in boxes.helper_calls]
    # tar_extract after the first slow clear, the resolves after both
    assert budgets == sorted(budgets, reverse=True) and budgets[0] - budgets[-1] >= 0.5
