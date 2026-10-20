"""Exercise test-daemon teardown with deterministic process/mount boundaries."""

import shlex
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/test-xfs-docker.sh"
DEFINITIONS, DISPATCH = SCRIPT.read_text().split('\ncase "${1:-}" in\n', 1)


def run_shell(tmp_path, boundaries, command="down", *, dispatch=False):
    root = tmp_path / "owned"
    (root / "mnt").mkdir(parents=True)
    for name in ("disk.img", "daemon.json", "dockerd.log"):
        (root / name).touch()
    # Load the real functions, then replace only external process/mount probes.
    # The initial root guard still runs; no production test mode is introduced.
    setup = f"""
realpath() {{ echo /opt/test-xfs; }}
{DEFINITIONS}
ROOT={shlex.quote(str(root))}
STATE={shlex.quote(str(tmp_path))}
PIDFILE="$STATE/pid"
EXEC_ROOT="$STATE/exec"
BRIDGE_MARKER="$STATE/bridge"
echo 0 > "$STATE/ticks"
ours() {{ echo 4242; }}
daemon_identity() {{ return 1; }}
sleep() {{ local ticks; ticks=$(cat "$STATE/ticks"); echo $((ticks + 1)) > "$STATE/ticks"; }}
kill() {{ echo "kill $*" >> "$STATE/events"; }}
docker() {{ :; }}
findmnt() {{ printf '%s\\n' "$EXEC_ROOT/netns/default" /run/docker/netns/default; }}
mountpoint() {{ [ ! -f "$STATE/unmounted-${{2##*/}}" ]; }}
umount() {{ printf '%s\\n' "$*" >> "$STATE/unmounts"; touch "$STATE/unmounted-${{1##*/}}"; }}
ip() {{ printf '%s\\n' "$*" >> "$STATE/links"; }}
ctr() {{ echo called > "$STATE/ctr"; }}
{boundaries}
"""
    body = '\ncase "${1:-}" in\n' + DISPATCH if dispatch else command
    result = subprocess.run(["bash", "-s", "--", command], input=setup + body,
                            text=True, capture_output=True, timeout=10)
    return result, root


def test_shutdown_waits_for_the_original_process_after_its_pidfile_disappears(tmp_path):
    result, root = run_shell(tmp_path, """
echo 4242 > "$PIDFILE"
ours() { [ -f "$PIDFILE" ] && cat "$PIDFILE"; }
daemon_identity() { [ "$(cat "$STATE/ticks")" -lt 2 ] && echo 100; }
kill() { rm "$PIDFILE"; }
umount() {
  [ "$(cat "$STATE/ticks")" -ge 2 ] || return 1
  printf '%s\\n' "$*" >> "$STATE/unmounts"
  touch "$STATE/unmounted-${1##*/}"
}
""")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "ticks").read_text().strip() == "2"
    assert (tmp_path / "unmounts").read_text().splitlines() == [
        str(tmp_path / "exec/netns/default"), str(root / "mnt")]


@pytest.mark.parametrize("exits_before_kill", [False, True])
def test_pid_reuse_and_a_process_exiting_before_the_signal_are_safe(tmp_path, exits_before_kill):
    boundaries = """
daemon_identity() {
  if [ ! -f "$STATE/observed" ]; then touch "$STATE/observed"; echo 100; else return 1; fi
}
"""
    if exits_before_kill:
        boundaries = """
daemon_identity() { [ ! -f "$STATE/gone" ] && echo 100; }
kill() { touch "$STATE/events" "$STATE/gone"; return 1; }
"""
    result, _ = run_shell(tmp_path, boundaries)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "events").exists() == exits_before_kill


@pytest.mark.parametrize("permanent", [False, True])
def test_busy_mounts_retry_normally_and_failure_prevents_purge(tmp_path, permanent):
    result, root = run_shell(tmp_path, f"""
touch "$BRIDGE_MARKER"
umount() {{
  printf '%s\\n' "$*" >> "$STATE/unmounts"
  if [ {'1' if permanent else '0'} -eq 1 ] || [ "$(cat "$STATE/ticks")" -lt 2 ]; then return 1; fi
  touch "$STATE/unmounted-${{1##*/}}"
}}
""", "purge", dispatch=True)
    attempts = (tmp_path / "unmounts").read_text().splitlines()
    assert all(target.startswith(str(tmp_path / "exec") + "/") or target == str(root / "mnt")
               for target in attempts)
    if permanent:
        assert result.returncode != 0
        assert len(attempts) == 60
        assert (root / "disk.img").exists() and (root / "daemon.json").exists()
        assert (tmp_path / "bridge").exists()
        assert not (tmp_path / "links").exists() and not (tmp_path / "ctr").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert len(attempts) == 4
        assert not root.exists()
        assert (tmp_path / "links").read_text().splitlines() == [
            "-d link show rqxfs0", "link del rqxfs0"]
        assert not (tmp_path / "bridge").exists()
