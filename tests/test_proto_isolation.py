"""Our command_session protobuf module never collides with another library's copy (prime
sandboxes, pulled in by verifiers, registers a different command_session.proto in the
default descriptor pool), and the signed-episode modules never load protobuf at all."""

import subprocess
import sys
import textwrap

import pytest


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True,
                          text=True, timeout=120)


def test_the_episode_modules_never_load_protobuf():
    result = run("""
        import sys
        import reliquary_sandbox.attest
        import reliquary_sandbox.episode_client
        import reliquary_sandbox.episode_task
        import reliquary_sandbox.observation
        import reliquary_sandbox.tool_routing
        loaded = sorted(m for m in sys.modules
                        if m.startswith("google.protobuf") or "._proto" in m)
        assert not loaded, loaded
    """)
    assert result.returncode == 0, result.stderr


def test_the_sdk_names_still_import_from_the_package():
    result = run("""
        import reliquary_sandbox
        from reliquary_sandbox import AsyncSandboxClient, SandboxClient, APIError, Sandbox
        assert reliquary_sandbox.TimeoutError is reliquary_sandbox.APITimeoutError
        assert set(reliquary_sandbox.__all__) <= set(dir(reliquary_sandbox))
        for name in reliquary_sandbox.__all__:
            getattr(reliquary_sandbox, name)
    """)
    assert result.returncode == 0, result.stderr


def test_another_command_session_proto_in_the_default_pool_does_not_collide():
    # A different file under the same name and package, as prime-sandboxes 0.3.0 ships.
    result = run("""
        from google.protobuf import descriptor_pb2, descriptor_pool

        other = descriptor_pb2.FileDescriptorProto(
            name="command_session/command_session.proto", package="command_session")
        message = other.message_type.add(name="PTY")
        message.field.add(name="other", number=7, type=1, label=1)
        other.service.add(name="CommandSession")
        descriptor_pool.Default().Add(other)

        from reliquary_sandbox._proto.command_session import command_session_pb2 as ours
        from reliquary_sandbox import rpc_command_session  # noqa: F401
        assert ours.DESCRIPTOR.package == "command_session"
        assert ours.DESCRIPTOR.services_by_name["CommandSession"].full_name == (
            "command_session.CommandSession")
        assert ours.PTY.DESCRIPTOR.fields_by_name["size"].number == 1
        assert ours.DESCRIPTOR.pool is not descriptor_pool.Default()
        start = ours.StartRequest()
        assert type(start).FromString(start.SerializeToString()) == start
    """)
    assert result.returncode == 0, result.stderr


def test_verifiers_and_the_whole_sdk_import_in_one_process():
    pytest.importorskip("verifiers.v1")
    result = run("""
        import verifiers.v1  # noqa: F401
        import prime_sandboxes  # noqa: F401
        import reliquary_sandbox
        from reliquary_sandbox import AsyncSandboxClient  # noqa: F401
        from reliquary_sandbox import rpc_command_session  # noqa: F401
        import reliquary_sandbox_service.episodes.local_gateway  # noqa: F401
        import reliquary_sandbox_verifiers  # noqa: F401
    """)
    assert result.returncode == 0, result.stderr
