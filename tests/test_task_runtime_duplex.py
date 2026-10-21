"""Large exec input and early command exits over a real Unix socket."""

import errno
import socket
import struct
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from test_task_runtime import runtime as runtime_fixture
from test_task_runtime import spec

from reliquary_sandbox_service.episodes.exec_transport import ExecStream, UnixExecTransport
from reliquary_sandbox_service.task_runtime import InfraFault

UPGRADED = b"HTTP/1.1 101 UPGRADED\r\nConnection: Upgrade\r\nUpgrade: tcp\r\n\r\n"
PAYLOAD = b"input" * (1024 * 1024)


def frame(kind, data):
    return struct.pack(">BxxxL", kind, len(data)) + data


def read_input(conn, initial):
    received = bytearray(initial)
    while chunk := conn.recv(65536):
        received.extend(chunk)
    return bytes(received)


@pytest.fixture
def peer(tmp_path, monkeypatch):
    runtime = runtime_fixture.__wrapped__(tmp_path, monkeypatch)
    runtime.create("s-1", "s-1", spec())
    original_create = runtime.client.api.exec_create

    def create(*args, **kwargs):
        made = original_create(*args, **kwargs)
        before = made["Id"]
        after = before.replace("-", "")
        runtime.client.execs[after] = runtime.client.execs.pop(before)
        return {"Id": after}

    monkeypatch.setattr(runtime.client.api, "exec_create", create)
    senders, completed = [], []
    original_send = ExecStream.send_stdin

    def send(stream, data):
        sender = threading.current_thread()
        senders.append(sender)
        try:
            return original_send(stream, data)
        finally:
            completed.append(sender)

    monkeypatch.setattr(ExecStream, "send_stdin", send)

    @contextmanager
    def serve(respond, exit_code=0):
        # A short path also runs on hosts with a small AF_UNIX pathname limit.
        with tempfile.TemporaryDirectory(prefix="exec-duplex-", dir="/tmp") as directory:
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            path = str(Path(directory) / "d.sock")
            listener.bind(path)
            listener.listen(1)
            listener.settimeout(3)
            release, finished = threading.Event(), threading.Event()
            clients, accepted, errors = [], [], []

            def handle():
                try:
                    conn, _ = listener.accept()
                    accepted.append(conn)
                    with conn:
                        conn.settimeout(3)
                        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
                        request = b""
                        while b"\r\n\r\n" not in request:
                            request += conn.recv(65536)
                        head, body = request.split(b"\r\n\r\n", 1)
                        size = int(next(line.split(b":")[1] for line in head.split(b"\r\n")
                                        if line.lower().startswith(b"content-length")))
                        while len(body) < size:
                            body += conn.recv(65536)
                        exec_id = head.split(b"/exec/")[1].split(b"/start")[0].decode()
                        runtime.client.execs[exec_id]["exit"] = exit_code
                        conn.sendall(UPGRADED)
                        respond(conn, body[size:], release)
                except OSError:
                    pass  # A timeout closes the client's side deliberately.
                except Exception as exc:
                    errors.append(exc)
                finally:
                    finished.set()

            transport = UnixExecTransport(path, "1.52")
            original_start = transport.start

            def start(exec_id, on_connect=None):
                def connected(sock):
                    clients.append(sock)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
                    on_connect(sock)
                return original_start(exec_id, on_connect=connected)

            transport.start = start
            runtime.exec_transport = transport
            worker = threading.Thread(target=handle, daemon=True)
            worker.start()
            try:
                yield runtime
                assert runtime._connections == {}
                assert clients and all(sock.fileno() == -1 for sock in clients)
                for sender in senders:
                    sender.join(1)
                    assert sender in completed and not sender.is_alive()
            finally:
                release.set()
                for conn in [*accepted, *clients]:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    conn.close()
                listener.close()
                assert finished.wait(4)
                worker.join(1)
            assert not errors

    return serve


def test_large_input_keeps_an_early_file_error(peer):
    def respond(conn, initial, release):
        conn.sendall(frame(2, b"cat: write error: No space left on device\n"))
        conn.shutdown(socket.SHUT_WR)
        release.wait(3)  # The command has exited without consuming the remaining input.

    with peer(respond, exit_code=1) as runtime:
        with pytest.raises(OSError) as caught:
            runtime.write_file("s-1", "/full", PAYLOAD, timeout_s=1)
        assert caught.value.errno == errno.ENOSPC


def test_output_drains_while_input_is_uploaded_and_remains_capped(peer):
    received = []

    def respond(conn, initial, release):
        conn.sendall(frame(1, b"x" * (1024 * 1024)))
        received.append(read_input(conn, initial))
        conn.sendall(frame(2, b"done\n"))

    with peer(respond) as runtime:
        result = runtime.exec("s-1", ["filter"], stdin=PAYLOAD, timeout_s=2, output_limit=32)
        assert not result.timed_out and result.exit_code == 0
        assert result.stdout == b"x" * 32 and result.stderr == b"done\n"
        assert result.truncated and received == [PAYLOAD]


def test_large_upload_is_complete_and_half_closed(peer):
    received = []

    def respond(conn, initial, release):
        received.append(read_input(conn, initial))
        conn.sendall(frame(1, b"accepted\n"))

    with peer(respond) as runtime:
        result = runtime.exec("s-1", ["upload"], stdin=PAYLOAD, timeout_s=2, output_limit=32)
        assert not result.timed_out and result.exit_code == 0
        assert result.stdout == b"accepted\n" and received == [PAYLOAD]


def test_timeout_closes_the_socket_and_stops_a_blocked_sender(peer):
    with peer(lambda conn, initial, release: release.wait(3)) as runtime:
        result = runtime.exec("s-1", ["blocked"], stdin=PAYLOAD, timeout_s=.2, output_limit=32)
        assert result.timed_out


def test_an_unexpected_sender_error_is_preserved(peer, monkeypatch):
    failure = ValueError("input sender failed")

    def fail(stream, data):
        raise failure

    with peer(lambda conn, initial, release: release.wait(3)) as runtime:
        monkeypatch.setattr(ExecStream, "send_stdin", fail)
        with pytest.raises(InfraFault) as caught:
            runtime.exec("s-1", ["upload"], stdin=PAYLOAD, timeout_s=1, output_limit=32)
        assert caught.value.__cause__ is failure
