"""Run the isolated test daemon's teardown with every host operation stubbed."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/test-xfs-docker.sh"


@pytest.mark.parametrize("busy_attempts, removal_fails, expected_code", [
    (2, False, 0), (100, False, 1), (0, True, 1),
])
def test_cleanup_targets_only_its_daemon_and_preserves_failures(
    tmp_path, busy_attempts, removal_fails, expected_code,
):
    # Load definitions without dispatching up/down/purge against a real host.
    definitions, dispatch = SCRIPT.read_text().split('case "${1:-}" in', 1)
    assert dispatch
    trace = tmp_path / "trace"
    stubs = r'''
    stopped=0
    attempts=0
    BRIDGE_MARKER="$TRACE.no-bridge"
    ours() { [ "$stopped" = 0 ] && echo 42; }
    daemon_identity() { [ "$stopped" = 0 ] && echo 100; }
    docker() {
      printf 'docker %s\n' "$*" >> "$TRACE"
      [ "$1" = -H ] && [ "$2" = "unix://$SOCK" ] || return 90
      case "$3" in
        ps) printf 'box-a\nbox-b\n' ;;
        rm) [ "$REMOVAL_FAILS" = 0 ] || return 1 ;;
        *) return 91 ;;
      esac
    }
    kill() { echo "kill $*" >> "$TRACE"; stopped=1; }
    findmnt() { :; }
    mountpoint() { return 0; }
    umount() {
      echo "umount $*" >> "$TRACE"
      attempts=$((attempts + 1))
      [ "$attempts" -gt "$BUSY_ATTEMPTS" ]
    }
    sleep() { echo "sleep $*" >> "$TRACE"; }
    down
    '''
    result = subprocess.run(
        ["bash", "-c", "realpath() { echo /opt/xfs-cleanup-test; }\n" + definitions + stubs],
        text=True, capture_output=True, timeout=5,
        env={**os.environ, "TRACE": str(trace), "XFS_DOCKER_ROOT": "/opt/xfs-cleanup-test",
             "BUSY_ATTEMPTS": str(busy_attempts), "REMOVAL_FAILS": str(int(removal_fails))},
    )
    events = trace.read_text().splitlines()
    assert result.returncode == expected_code, result.stderr
    docker_calls = [event for event in events if event.startswith("docker ")]
    assert docker_calls[0] == "docker -H unix:///run/reliquary-xfs-docker.sock ps -aq"
    removals = [event for event in docker_calls if " rm " in event]
    assert removals
    assert all(event.startswith("docker -H unix:///run/reliquary-xfs-docker.sock rm --force -- ")
               for event in removals)
    if removal_fails:
        assert not any(event.startswith(("kill ", "umount ")) for event in events)
    else:
        assert len(removals) == 2
        assert events.index(removals[-1]) < events.index("kill 42")
        attempts = [event for event in events if event.startswith("umount ")]
        assert len(attempts) == min(busy_attempts + 1, 60)
        assert events.index("kill 42") < events.index(attempts[0])
        if expected_code:
            assert "could not unmount" in result.stderr
