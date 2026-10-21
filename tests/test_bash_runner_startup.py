"""A capper scheduled after relay shutdown still drains its inherited FIFO."""

import os
import shlex
import signal
import subprocess
import time
import uuid

from test_bash_runner import BUSY, body, busybox_required, scripts  # noqa: F401


def test_delayed_cappers_drain_output_and_finish(scripts, tmp_path):  # noqa: F811
    runner, _, tools = scripts
    release = tmp_path / "release-cappers"
    waiting = tmp_path / "waiting-for-cappers"
    delayed = []
    capper_count = 0
    for line in runner.splitlines():
        if line.startswith('{ $B head -c "$n";') and line.endswith(" &"):
            capper_count += 1
            delayed.extend([
                "(",
                f"  while [ ! -e {shlex.quote(str(release))} ]; do $B sleep 0.01; done",
                line[:-2],
                ") 4>&- 5>&- 6>&- 7>&- &",
            ])
        else:
            delayed.append(line)
    assert capper_count == 2
    runner = "\n".join(delayed).replace(
        'wait "$c1" "$c2"',
        f': > {shlex.quote(str(waiting))}\nwait "$c1" "$c2"',
    )
    call_id = uuid.uuid4().hex
    proc = subprocess.Popen(
        [BUSY, "sh", "-c", runner, "r", "1000", call_id,
         "printf 'out\\n'; printf 'err\\n' >&2; exit 7", "/"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 20
    try:
        while not waiting.exists():
            assert proc.poll() is None, "runner exited before the capper rendezvous"
            assert time.monotonic() < deadline, "runner did not reach the capper rendezvous"
            time.sleep(0.01)
        release.touch()
        out, err = proc.communicate(timeout=max(0.01, deadline - time.monotonic()))
        assert (body(out, 7), err, proc.returncode) == (b"out\n", b"err\n", 7)
        assert list(tools.glob(f"{call_id}.*")) == []
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate(timeout=5)
