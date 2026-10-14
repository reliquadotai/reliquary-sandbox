"""The exec-start transport against a fake Docker daemon on a Unix socket (no Docker)."""

import socket
import struct
import threading
import time

import docker
import pytest
from docker.errors import APIError, NotFound
from docker.utils.socket import frames_iter

from reliquary_sandbox_service.episodes.exec_transport import (
    UnixExecTransport,
    unix_transport_for,
)


def frame(kind, data):
    return struct.pack(">BxxxL", kind, len(data)) + data


UPGRADED = (b"HTTP/1.1 101 UPGRADED\r\nContent-Type: application/vnd.docker.multiplexed-stream\r\n"
            b"Connection: Upgrade\r\nUpgrade: tcp\r\n\r\n")


class FakeDaemon:
    """Answers one exec start per connection. `respond(request, stdin_reader)`
    returns the bytes to send; the first send carries the headers AND the
    first frame in one write, the way the real daemon sometimes does."""

    def __init__(self, path, respond):
        self.path = str(path)
        self.respond = respond
        self.requests = []
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.path)
        self.server.listen(8)
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        with conn:
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(65536)
            head, body = data.split(b"\r\n\r\n", 1)
            length = int(next(line.split(b":")[1] for line in head.split(b"\r\n")
                              if line.lower().startswith(b"content-length")))
            while len(body) < length:
                body += conn.recv(65536)
            stdin = body[length:]
            self.requests.append(head)

            def read_stdin():
                nonlocal stdin
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return stdin
                    stdin += chunk

            for part in self.respond(head, read_stdin):
                conn.sendall(part)


def test_a_first_frame_sent_with_the_headers_is_kept(tmp_path):
    daemon = FakeDaemon(tmp_path / "d.sock", lambda head, _: [
        UPGRADED + frame(1, b"first\n"), frame(2, b"err\n") + frame(1, b"second\n")])
    stream = UnixExecTransport(daemon.path, "1.52").start("abc123")
    assert list(stream.frames()) == [(1, b"first\n"), (2, b"err\n"), (1, b"second\n")]
    assert daemon.requests[0].startswith(b"POST /v1.52/exec/abc123/start HTTP/1.1")
    assert b"Upgrade: tcp" in daemon.requests[0]


def test_docker_py_loses_that_frame(tmp_path):
    """The bug this transport exists for (docker-py 7)."""
    daemon = FakeDaemon(tmp_path / "d.sock", lambda head, _: [
        UPGRADED + frame(1, b"first\n"), frame(1, b"second\n")])
    client = docker.APIClient(base_url=f"unix://{daemon.path}", version="1.52")
    connection = client.exec_start("abc123", socket=True)
    got = [data for _, data in frames_iter(connection, tty=False)]
    assert b"first\n" not in got  # buffered with the headers, never seen


def test_frames_split_across_reads_are_reassembled(tmp_path):
    payload = frame(1, b"x" * 100_000)
    daemon = FakeDaemon(tmp_path / "d.sock", lambda head, _: [
        UPGRADED + payload[:3], payload[3:50], payload[50:]])
    stream = UnixExecTransport(daemon.path, "1.52").start("abc")
    assert list(stream.frames()) == [(1, b"x" * 100_000)]


def test_stdin_is_written_and_half_closed_on_the_same_socket(tmp_path):
    def respond(head, read_stdin):
        yield UPGRADED
        yield frame(1, read_stdin())

    daemon = FakeDaemon(tmp_path / "d.sock", respond)
    stream = UnixExecTransport(daemon.path, "1.52").start("abc")
    stream.send_stdin(b"data" * 1000)
    assert list(stream.frames()) == [(1, b"data" * 1000)]


@pytest.mark.parametrize("status,error", [(404, NotFound), (409, APIError), (500, APIError)])
def test_a_refused_start_raises_the_docker_error_with_its_status(tmp_path, status, error):
    body = b'{"message":"container abc is not running"}'
    daemon = FakeDaemon(tmp_path / "d.sock", lambda head, _: [
        f"HTTP/1.1 {status} X\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body])
    with pytest.raises(error) as caught:
        UnixExecTransport(daemon.path, "1.52").start("abc")
    assert caught.value.status_code == status and "not running" in str(caught.value)


def test_a_closed_connection_is_an_os_error(tmp_path):
    daemon = FakeDaemon(tmp_path / "d.sock", lambda head, _: [b"HTTP/1.1 101"])
    with pytest.raises(OSError):
        UnixExecTransport(daemon.path, "1.52").start("abc")


def test_only_unix_socket_clients_get_a_transport():
    unix = unix_transport_for(docker.DockerClient(base_url="unix:///run/x.sock", version="1.52"))
    assert unix.socket_path == "/run/x.sock" and unix.api_version == "1.52"
    assert unix_transport_for(docker.DockerClient(base_url="tcp://127.0.0.1:2375",
                                                  version="1.52")) is None
    assert unix_transport_for(object()) is None


def chunk(data):
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def hold_open(parts):
    """Send the parts, then keep the connection open (a keep-alive daemon)."""
    def respond(head, read_stdin):
        yield from parts
        read_stdin()  # blocks until the client closes
    return respond


def test_an_error_body_with_a_content_length_returns_without_waiting(tmp_path):
    body = b'{"message":"No such exec instance"}'
    daemon = FakeDaemon(tmp_path / "d.sock", hold_open([
        f"HTTP/1.1 404 Not Found\r\nContent-Length: {len(body)}\r\n\r\n".encode(), body]))
    began = time.monotonic()
    with pytest.raises(NotFound, match="No such exec instance"):
        UnixExecTransport(daemon.path, "1.52").start("abc")
    assert time.monotonic() - began < 0.5


def test_a_chunked_error_body_returns_without_waiting(tmp_path):
    daemon = FakeDaemon(tmp_path / "d.sock", hold_open([
        b"HTTP/1.1 409 Conflict\r\nTransfer-Encoding: chunked\r\n\r\n",
        chunk(b'{"message":"conta'), chunk(b'iner c is not running"}'), b"0\r\n\r\n"]))
    began = time.monotonic()
    with pytest.raises(APIError, match="container c is not running") as caught:
        UnixExecTransport(daemon.path, "1.52").start("abc")
    assert caught.value.status_code == 409 and time.monotonic() - began < 0.5


def test_the_socket_is_handed_over_before_the_handshake(tmp_path):
    """A caller that gives up during the handshake can close the socket and the
    start fails at once instead of waiting for the daemon."""
    daemon = FakeDaemon(tmp_path / "d.sock", hold_open([]))  # never answers
    seen = []

    def on_connect(sock):
        seen.append(sock)
        threading.Timer(0.2, sock.shutdown, args=(socket.SHUT_RDWR,)).start()

    began = time.monotonic()
    with pytest.raises(OSError):
        UnixExecTransport(daemon.path, "1.52", timeout_s=30).start("abc", on_connect=on_connect)
    assert seen and time.monotonic() - began < 5
