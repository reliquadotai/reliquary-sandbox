"""`POST /exec/{id}/start` over the Docker daemon's Unix socket, done by hand.

docker-py's `exec_start(socket=True)` (and `stream=False, demux=True`) hands
back the socket *under* the BufferedReader that parsed the `101 UPGRADED`
response headers. When the daemon sends the first multiplexed frame in the
same read as the headers, those bytes stay in that buffer and are never seen:
a fast exec's output silently vanishes (about 2 % of `head` reads on the test
host). Here the upgrade is read byte-exactly: everything after `\\r\\n\\r\\n` is
kept as the start of the stream, and frames are parsed from our own buffer.

Only Unix sockets are supported (the controller runs on the Docker host);
anything else fails closed.
"""

from __future__ import annotations

import json
import re
import socket
import struct
from collections.abc import Iterator

import requests
from docker.errors import APIError, NotFound

MAX_HEADER_BYTES = 64 * 1024
_STATUS = re.compile(rb"HTTP/1\.[01] (\d{3})")


class ExecStream:
    """A started exec: stdin is written and half-closed on the same socket;
    `frames()` yields (stream number, bytes) from the multiplexed output."""

    def __init__(self, sock, buffered: bytes = b""):
        self.sock = sock
        self._buffer = bytearray(buffered)

    def send_stdin(self, data: bytes) -> None:
        self.sock.settimeout(None)  # a large write must not be cut by a timeout
        self.sock.sendall(data)
        self.sock.shutdown(socket.SHUT_WR)

    def _fill(self, size: int) -> bool:
        while len(self._buffer) < size:
            chunk = self.sock.recv(65536)
            if not chunk:
                return False
            self._buffer.extend(chunk)
        return True

    def frames(self) -> Iterator[tuple[int, bytes]]:
        while self._fill(8):
            kind, length = self._buffer[0], struct.unpack(">L", bytes(self._buffer[4:8]))[0]
            if not self._fill(8 + length):
                return  # a cut frame: the stream ended under us
            data = bytes(self._buffer[8:8 + length])
            del self._buffer[:8 + length]
            yield kind, data

    def close(self) -> None:
        for action in (lambda: self.sock.shutdown(socket.SHUT_RDWR), self.sock.close):
            try:
                action()
            except Exception:
                pass


def _error_body(sock, head: bytes, rest: bytes) -> bytes:
    """The body of a refused start, read by its Content-Length or chunks (the
    daemon keeps the connection alive); else until close, briefly."""
    headers = {}
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        headers[name.strip().lower()] = value.strip()
    data = bytearray(rest)
    sock.settimeout(2)

    def more() -> bool:
        if len(data) > MAX_HEADER_BYTES:
            return False
        try:
            chunk = sock.recv(65536)
        except OSError:
            return False
        data.extend(chunk)
        return bool(chunk)

    if b"content-length" in headers:
        try:
            length = min(int(headers[b"content-length"]), MAX_HEADER_BYTES)
        except ValueError:
            length = 0
        while len(data) < length and more():
            pass
        return bytes(data[:length])
    if b"chunked" in headers.get(b"transfer-encoding", b"").lower():
        body, position = bytearray(), 0
        while True:
            end = data.find(b"\r\n", position)
            if end < 0:
                if not more():
                    return bytes(body)
                continue
            try:
                size = int(bytes(data[position:end]).split(b";")[0], 16)
            except ValueError:
                return bytes(body)
            if size == 0:
                return bytes(body)
            while len(data) < end + 2 + size + 2:
                if not more():
                    return bytes(body + data[end + 2:end + 2 + size])
            body.extend(data[end + 2:end + 2 + size])
            position = end + 2 + size + 2
    while more():
        pass
    return bytes(data)


def _api_error(status: int, body: bytes) -> APIError:
    response = requests.Response()
    response.status_code = status
    text = body.decode("utf-8", "replace").strip()
    try:
        text = json.loads(text).get("message", text)
    except Exception:
        pass
    cls = NotFound if status == 404 else APIError
    return cls(f"{status} exec start failed: {text}", response=response, explanation=text)


class UnixExecTransport:
    def __init__(self, socket_path: str, api_version: str, timeout_s: float = 60):
        self.socket_path = socket_path
        self.api_version = api_version
        self.timeout_s = timeout_s

    def start(self, exec_id: str, on_connect=None) -> ExecStream:
        """`on_connect(sock)` gets the socket before the handshake, so a caller
        that gives up can close it and the start fails at once."""
        if not re.fullmatch(r"[A-Za-z0-9]{1,128}", exec_id):
            raise ValueError("Invalid exec identifier")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if on_connect is not None:
            on_connect(sock)
        try:
            sock.settimeout(self.timeout_s)
            sock.connect(self.socket_path)
            body = b'{"Detach":false,"Tty":false}'
            sock.sendall(
                f"POST /v{self.api_version}/exec/{exec_id}/start HTTP/1.1\r\n"
                "Host: docker\r\nContent-Type: application/json\r\n"
                "Connection: Upgrade\r\nUpgrade: tcp\r\n"
                f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            received = bytearray()
            while b"\r\n\r\n" not in received:
                if len(received) > MAX_HEADER_BYTES:
                    raise OSError("the Docker daemon sent oversized response headers")
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionResetError("the Docker daemon closed the exec stream")
                received.extend(chunk)
            head, rest = bytes(received).split(b"\r\n\r\n", 1)
            match = _STATUS.match(head)
            if not match:
                raise OSError("the Docker daemon sent an unreadable response")
            status = int(match.group(1))
            if status != 101:
                raise _api_error(status, _error_body(sock, head, rest))
            sock.settimeout(None)  # the exec's own timeout is enforced by the caller
            return ExecStream(sock, rest)
        except BaseException:
            sock.close()
            raise


def unix_transport_for(client) -> UnixExecTransport | None:
    """The transport for a docker-py client on a Unix socket, else None."""
    api = getattr(client, "api", None)
    base = getattr(api, "base_url", "")
    if base != "http+docker://localhost":
        return None
    try:
        path = api.get_adapter(base).socket_path
    except Exception:
        return None
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    return UnixExecTransport(path, api.api_version)
