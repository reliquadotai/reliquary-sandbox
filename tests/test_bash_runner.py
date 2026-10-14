"""The bash tool runner's shell logic, exercised with the host's bash (no Docker)."""

import os
import re
import shutil
import subprocess
import time
import uuid

import pytest

from reliquary_sandbox_service.task_runtime import (
    BASH_RUNNER,
    HELPER,
    KILL_GROUP,
    REFUSED_MARKER,
    TOOLS_DIR,
)

UID = os.getuid()
BUSY = shutil.which("busybox") or ""
ALLOW_SKIP = "RELIQUARY_ALLOW_NO_BUSYBOX"
"""Set to 1 to skip these tests on a machine without busybox; CI installs it."""


@pytest.fixture(autouse=True)
def busybox_required():
    if BUSY:
        return
    if os.environ.get(ALLOW_SKIP) == "1":
        pytest.skip(f"no host busybox ({ALLOW_SKIP}=1)")
    pytest.fail(f"the runner tests need a host busybox (install busybox-static, "
                f"or set {ALLOW_SKIP}=1 to skip them)")


@pytest.fixture
def scripts(tmp_path):
    tools = tmp_path / "tools"

    def local(text):
        return text.replace(TOOLS_DIR, str(tools)).replace(HELPER, BUSY)

    return local(BASH_RUNNER), local(KILL_GROUP), tools


def body(stdout, status):
    """The runner's stdout without its trailer, which must end it with `status`."""
    found = re.fullmatch(rb"(.*)RQ-END [0-9a-f]{32} (\d+)", stdout, re.S)
    assert found and int(found.group(2)) == status, stdout[-80:]
    return found.group(1)


def processes_matching(marker):
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.startswith(marker)]


def test_runner_keeps_stdout_then_stderr_and_exit_code(scripts):
    runner, _, _ = scripts
    result = subprocess.run([BUSY, "sh", "-c", runner, "r", "1000", uuid.uuid4().hex, "echo out; echo err >&2; exit 3", "/"],
                            capture_output=True)
    assert (body(result.stdout, 3), result.stderr, result.returncode) == (b"out\n", b"err\n", 3)


def test_kill_group_stops_the_call_and_its_children_only(scripts):
    runner, kill, tools = scripts
    survivor = f"sleep 31{uuid.uuid4().int % 1000}"
    victim = f"sleep 32{uuid.uuid4().int % 1000}"
    subprocess.run(["bash", "-c", f"nohup {survivor} >/dev/null 2>&1 &"], start_new_session=True)
    call_id = uuid.uuid4().hex
    proc = subprocess.Popen([BUSY, "sh", "-c", runner, "r", "1000", call_id, f"echo go; ({victim} &); {victim}", "/"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    for _ in range(50):
        if (tools / f"{call_id}.{UID}.pgid").exists():
            break
        time.sleep(0.05)
    assert subprocess.run([BUSY, "sh", "-c", kill, "k", call_id]).returncode == 0
    out, _ = proc.communicate(timeout=5)
    assert proc.returncode == 137 and body(out, 137) == b"go\n"
    time.sleep(0.2)
    try:
        assert processes_matching(victim) == []
        assert processes_matching(survivor) != []
    finally:
        subprocess.run(["pkill", "-f", survivor])


def test_kill_after_the_call_ended_is_a_no_op(scripts):
    _, kill, _ = scripts
    assert subprocess.run([BUSY, "sh", "-c", kill, "k", uuid.uuid4().hex]).returncode == 0


def test_runner_refuses_to_run_what_it_could_not_stop(scripts):
    runner, _, tools = scripts
    tools.mkdir()
    marker = f"sleep 33{uuid.uuid4().int % 1000}"
    call_id = uuid.uuid4().hex
    (tools / f"{call_id}.{UID}.pgid").mkdir()  # the pgid file cannot be written
    result = subprocess.run([BUSY, "sh", "-c", runner, "r", "1000", call_id, marker, "/"], capture_output=True, timeout=10)
    assert result.returncode == 125
    assert f"{REFUSED_MARKER} {call_id}".encode() in result.stderr
    assert processes_matching(marker) == []


def run_runner(runner, call_id, command, limit=1000, workdir="/", **kwargs):
    return subprocess.run([BUSY, "sh", "-c", runner, "r", str(limit), call_id, command, workdir],
                          capture_output=True, timeout=20, **kwargs)


def test_a_flood_is_capped_inside_the_box_and_the_command_still_finishes(scripts):
    runner, _, _ = scripts
    result = run_runner(runner, uuid.uuid4().hex,
                        "yes | head -c 3000000; yes err | head -c 3000000 >&2; exit 7", limit=100)
    assert len(body(result.stdout, 7)) == 101 and len(result.stderr) == 101
    assert result.returncode == 7


def test_output_exactly_at_the_limit_is_not_truncated(scripts):
    runner, _, _ = scripts
    result = run_runner(runner, uuid.uuid4().hex, "head -c 100 /dev/zero", limit=100)
    assert len(body(result.stdout, result.returncode)) == 100


def test_a_kill_that_arrives_before_the_group_exists_stops_the_call_from_running(scripts):
    runner, kill, tools = scripts
    call_id = uuid.uuid4().hex
    tools.mkdir()
    (tools / f"{call_id}.{UID}.pgid").write_text("")  # the placeholder: runner started, no group yet
    assert subprocess.run([BUSY, "sh", "-c", kill, "k", call_id]).returncode == 0
    assert (tools / f"{call_id}.killed").exists()
    marker = f"sleep 34{uuid.uuid4().int % 1000}"
    (tools / f"{call_id}.{UID}.pgid").unlink()
    result = run_runner(runner, call_id, f"{marker}; echo ran")
    try:
        assert result.returncode == 137 and b"ran" not in result.stdout
    finally:
        subprocess.run(["pkill", "-f", marker])


@pytest.mark.parametrize("content", ["0", "1", "abc", "-1", "12 34"])
def test_kill_refuses_unsafe_group_ids(scripts, content):
    _, kill, tools = scripts
    call_id = uuid.uuid4().hex
    tools.mkdir()
    (tools / f"{call_id}.{UID}.pgid").write_text(content)
    assert subprocess.run([BUSY, "sh", "-c", kill, "k", call_id]).returncode == 1


def test_kill_never_blocks_on_a_fifo_pgid_file(scripts):
    _, kill, tools = scripts
    call_id = uuid.uuid4().hex
    tools.mkdir()
    subprocess.run(["mkfifo", str(tools / f"{call_id}.{UID}.pgid")], check=True)
    assert subprocess.run([BUSY, "sh", "-c", kill, "k", call_id], timeout=5).returncode == 1


def test_kill_of_an_already_dead_group_succeeds(scripts):
    _, kill, tools = scripts
    call_id = uuid.uuid4().hex
    tools.mkdir()
    (tools / f"{call_id}.{UID}.pgid").write_text("4194000")
    assert subprocess.run([BUSY, "sh", "-c", kill, "k", call_id]).returncode == 0


def timed_run(runner, call_id, command, limit=1000):
    start = time.monotonic()
    result = run_runner(runner, call_id, command, limit)
    return result, time.monotonic() - start


def test_a_background_service_neither_hangs_the_call_nor_is_killed(scripts):
    runner, _, _ = scripts
    marker = f"sleep 35{uuid.uuid4().int % 1000}"
    try:
        result, took = timed_run(runner, uuid.uuid4().hex, f"{marker} & echo started")
        assert result.returncode == 0 and body(result.stdout, 0) == b"started\n" and took < 3
        assert processes_matching(marker) != []
    finally:
        subprocess.run(["pkill", "-f", marker])


def test_a_silenced_background_job_returns_at_once(scripts):
    runner, _, _ = scripts
    marker = f"sleep 36{uuid.uuid4().int % 1000}"
    try:
        result, took = timed_run(runner, uuid.uuid4().hex, f"{marker} >/dev/null 2>&1 & echo up")
        assert result.returncode == 0 and took < 1
    finally:
        subprocess.run(["pkill", "-f", marker])


def test_a_setsid_escapee_cannot_keep_the_runner_alive(scripts):
    runner, _, _ = scripts
    marker = f"sleep 37{uuid.uuid4().int % 1000}"
    try:
        result, took = timed_run(runner, uuid.uuid4().hex, f"setsid {marker} & exit 4")
        assert result.returncode == 4 and took < 3
    finally:
        subprocess.run(["pkill", "-f", marker])


def test_kill_refuses_a_symlinked_pgid_file_and_one_owned_by_someone_else(scripts, tmp_path):
    _, kill, tools = scripts
    call_id = uuid.uuid4().hex
    tools.mkdir()
    real = tmp_path / "real"
    real.write_text("4194000")
    (tools / f"{call_id}.{UID}.pgid").symlink_to(real)
    assert subprocess.run([BUSY, "sh", "-c", kill, "k", call_id]).returncode == 1
    (tools / f"{call_id}.{UID}.pgid").unlink()
    (tools / f"{call_id}.{UID + 1}.pgid").write_text("4194000")  # name says another uid owns it
    assert subprocess.run([BUSY, "sh", "-c", kill, "k", call_id]).returncode == 1


def test_kill_cleans_up_after_a_dead_group_and_the_runner_after_itself(scripts):
    runner, kill, tools = scripts
    call_id = uuid.uuid4().hex
    tools.mkdir()
    for name in (f"{call_id}.{UID}.pgid",):
        (tools / name).write_text("4194000")
    for suffix in ("out", "err"):
        (tools / f"{call_id}.{suffix}").write_text("")
    assert subprocess.run([BUSY, "sh", "-c", kill, "k", call_id]).returncode == 0
    assert [p.name for p in tools.iterdir() if p.name.startswith(call_id)] == []
    done = uuid.uuid4().hex
    run_runner(runner, done, "echo hi")
    assert [p.name for p in tools.iterdir() if p.name.startswith(done)] == []


def test_the_runner_checks_the_tombstone_before_launching_anything(scripts):
    runner, _, tools = scripts
    call_id = uuid.uuid4().hex
    tools.mkdir()
    (tools / f"{call_id}.killed").write_text("")
    marker = f"sleep 38{uuid.uuid4().int % 1000}"
    result = run_runner(runner, call_id, f"{marker}; echo ran")
    assert result.returncode == 137 and body(result.stdout, 137) == b""
    assert processes_matching(marker) == []
