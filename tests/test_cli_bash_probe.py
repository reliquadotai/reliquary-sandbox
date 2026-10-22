"""Pure Bash CPU fixture checks; no process, Docker, model or gateway is started."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "bash_probe_fixture", Path(__file__).parents[1] / "scripts/cli-probe/bash_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]


def test_bash_fixture_requires_48_generations_and_47_single_signed_commands():
    generator = probe.ScriptedBashGenerator(TOOLS)
    history = [{"role": "user", "content": probe.PROMPT}]
    for turn in range(48):
        ids = generator.render_messages(history, tools=TOOLS, thinking=True,
                                         renderer_id="renderers:qwen38@0.1.11")
        result = generator.generate_turn(ids)
        assert result["token_ids"] == [10 + turn]
        history.append({"role": "assistant", **result["message"]})
        if turn < 47:
            assert result["finish_reason"] == "tool_calls"
            assert len(result["message"]["tool_calls"]) == 1
            assert result["message"]["tool_calls"][0]["name"] == "bash"
            history.append({"role": "tool", "content": "signed output"})
        else:
            assert result["message"] == {"content": "done"}
            assert result["finish_reason"] == "stop"
    with pytest.raises(AssertionError, match="extra generation"):
        generator.generate_turn([1, 2])
    assert generator.calls == generator.renders == 48


@pytest.mark.parametrize("changed", [
    {"tools": []}, {"thinking": False}, {"renderer_id": "unqualified"},
])
def test_bash_fixture_refuses_another_rendering_policy(changed):
    generator = probe.ScriptedBashGenerator(TOOLS)
    with pytest.raises(ValueError, match="history or rendering policy"):
        generator.render_messages([], **{"tools": TOOLS, "thinking": True,
            "renderer_id": "renderers:qwen38@0.1.11", **changed})
    assert generator.calls == generator.renders == 0


async def test_bash_observers_preserve_calls_and_capture_only_owned_runtime_children(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=12345, returncode=0)
    invocations = []

    class Runtime:
        async def start(self):
            self.workdir = tmp_path / "owned-runtime"
            self.workdir.mkdir()

    async def spawn(*argv, **kwargs):
        invocations.append((argv, kwargs))
        return process

    original_start = Runtime.start
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with probe._observe_bash_processes(Runtime) as (workdirs, processes):
        runtime = Runtime()
        await runtime.start()
        assert await asyncio.create_subprocess_exec("fixture-program", cwd=runtime.workdir,
                                                   start_new_session=True) is process
        await asyncio.create_subprocess_exec("unrelated", cwd=tmp_path, start_new_session=True)
    assert processes == [(process, True)] and workdirs == {runtime.workdir.resolve()}
    assert Runtime.start is original_start and asyncio.create_subprocess_exec is spawn
    assert invocations[0] == (("fixture-program",), {"cwd": runtime.workdir, "start_new_session": True})
    runtime.workdir.rmdir()

    def absent(pid, signal):
        assert (pid, signal) == (12345, 0)
        raise ProcessLookupError

    monkeypatch.setattr(probe.os, "killpg", absent)
    probe._assert_owned_cleanup(workdirs, processes)


@pytest.mark.parametrize("fault", ["workdir", "running", "unowned-group", "live-group"])
def test_cleanup_requires_owned_leader_group_and_removed_directory(tmp_path, monkeypatch, fault):
    workdir = tmp_path / "runtime"
    if fault == "workdir":
        workdir.mkdir()
    process = SimpleNamespace(pid=12345, returncode=None if fault == "running" else 0)
    monkeypatch.setattr(probe.os, "killpg", lambda *args: None)
    with pytest.raises(RuntimeError):
        probe._assert_owned_cleanup({workdir}, [(process, fault != "unowned-group")])
