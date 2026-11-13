"""Egress proxy for online task boxes (docs/egress.md). Standard library only.

    python -I -m reliquary_sandbox_service.egress_proxy /etc/reliquary-sandbox/egress.json

One listener per profile, bound to the egress bridge's host address. A box reaches it
through HTTP(S)_PROXY; it is the only thing a box can reach (internal bridge, host
firewall). Per request:

* `CONNECT host:port` (HTTPS, git, anything TLS) and absolute-URI `GET`/`HEAD`
  (plain HTTP, e.g. apt) only; other methods are refused.
* The host must match the profile's allowlist (`example.org` exact, `*.example.org`
  subdomains only) and the port its ports. IP literals never match.
* The proxy resolves the name itself and connects only to a global unicast address
  (no loopback, private, link-local/metadata, CGNAT, multicast), the one it checked.
* Plain HTTP URLs can be rewritten (`rewrites`: URL prefix -> prefix), e.g. an apt
  mirror to a dated snapshot, fetched over HTTPS: what the box installs is pinned.
* Per source address (one box): concurrent connections, connections per minute, bytes
  per window, and a byte rate shared by its connections. A box over a cap is refused
  (429) or slowed; the whole proxy has a connection cap and a total byte rate too.
* One JSON line per connection: source, profile, method, host, port, URL (plain HTTP),
  verdict, bytes each way, seconds. The gateway logs each online box's address, so a
  line maps to its episode.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
import ssl
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

_log = logging.getLogger("reliquary.egress")

HEAD_LIMIT = 16 * 1024
CHUNK = 64 * 1024
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authorization", "proxy-authenticate",
              "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade", "host"}


@dataclass(frozen=True)
class Limits:
    connections_per_source: int = 32
    connects_per_minute: int = 600
    bytes_per_window: int = 8 * 1024**3
    window_s: float = 3600.0
    rate_bytes_per_s: float = 50e6
    connect_timeout_s: float = 10.0
    idle_timeout_s: float = 300.0
    max_tunnel_s: float = 3600.0
    max_connections: int = 2048
    total_rate_bytes_per_s: float = 400e6
    """All boxes together (the host's link is shared with image pulls and the gateway)."""


@dataclass(frozen=True)
class Profile:
    name: str
    hosts: tuple[str, ...]
    ports: tuple[int, ...] = (80, 443)
    rewrites: tuple[tuple[str, str], ...] = ()

    def allows(self, host: str, port: int) -> bool:
        host = host.lower().rstrip(".")
        if port not in self.ports or not host or _is_ip_literal(host):
            return False
        for pattern in self.hosts:
            if pattern.startswith("*."):
                if host.endswith(pattern[1:]) and len(host) > len(pattern) - 1:
                    return True
            elif host == pattern:
                return True
        return False

    def rewrite(self, url: str) -> str:
        for prefix, target in self.rewrites:
            if url.startswith(prefix):
                return target + url[len(prefix):]
        return url


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def address_allowed(text: str) -> bool:
    """A global unicast address (IPv4-mapped IPv6 judged as its IPv4)."""
    try:
        ip = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


@dataclass
class _Source:
    connections: int = 0
    connects: list[float] = field(default_factory=list)
    window_start: float = 0.0
    window_bytes: int = 0
    tokens: float = 0.0
    refilled: float = 0.0


class Refused(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


class EgressProxy:
    def __init__(self, profiles: dict[str, Profile], limits: Limits = Limits(), *,
                 sources: list[str] = (), log_path: str | None = None, clock=time.monotonic):
        self.profiles = profiles
        self.limits = limits
        self.allowed_sources = [ipaddress.ip_network(s) for s in sources]
        self.clock = clock
        self._sources: dict[str, _Source] = {}
        self._active = 0
        now = clock()
        self._total = _Source(window_start=now, tokens=limits.total_rate_bytes_per_s, refilled=now)
        self._log_file = open(log_path, "a", buffering=1) if log_path else None

    # -- policy hooks (overridden in tests) ----------------------------------

    async def resolve(self, host: str, port: int) -> list[str]:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
        return [info[4][0] for info in infos]

    def address_allowed(self, address: str) -> bool:
        return address_allowed(address)

    # -- per-source accounting -----------------------------------------------

    def _source(self, address: str) -> _Source:
        source = self._sources.get(address)
        if source is None:
            now = self.clock()
            source = self._sources[address] = _Source(
                window_start=now, tokens=self.limits.rate_bytes_per_s, refilled=now)
        return source

    def _admit(self, address: str) -> _Source:
        if self.allowed_sources and not any(
                ipaddress.ip_address(address) in net for net in self.allowed_sources):
            raise Refused(403, "source_not_allowed")
        if self._active >= self.limits.max_connections:
            raise Refused(503, "proxy_full")
        source, now = self._source(address), self.clock()
        if source.connections >= self.limits.connections_per_source:
            raise Refused(429, "too_many_connections")
        source.connects = [t for t in source.connects if now - t < 60.0]
        if len(source.connects) >= self.limits.connects_per_minute:
            raise Refused(429, "too_many_connects")
        self._roll_window(source, now)
        if source.window_bytes >= self.limits.bytes_per_window:
            raise Refused(429, "byte_budget_spent")
        source.connects.append(now)
        source.connections += 1
        self._active += 1
        return source

    def _roll_window(self, source: _Source, now: float) -> None:
        if now - source.window_start >= self.limits.window_s:
            source.window_start, source.window_bytes = now, 0

    async def _charge(self, source: _Source, n: int) -> None:
        """Count `n` bytes; sleep while the source's byte rate is spent."""
        now = self.clock()
        self._roll_window(source, now)
        source.window_bytes += n
        rate = self.limits.rate_bytes_per_s
        source.tokens = min(rate, source.tokens + (now - source.refilled) * rate)
        source.refilled = now
        source.tokens -= n
        total, total_rate = self._total, self.limits.total_rate_bytes_per_s
        total.tokens = min(total_rate, total.tokens + (now - total.refilled) * total_rate)
        total.refilled = now
        total.tokens -= n
        wait = max(-source.tokens / rate, -total.tokens / total_rate, 0.0)
        if wait:
            await asyncio.sleep(wait)
        if source.window_bytes > self.limits.bytes_per_window:
            raise Refused(429, "byte_budget_spent")

    def _release(self, address: str, source: _Source) -> None:
        source.connections -= 1
        self._active -= 1
        if source.connections == 0 and self.clock() - source.window_start >= self.limits.window_s:
            self._sources.pop(address, None)

    # -- connections ---------------------------------------------------------

    async def handle(self, profile: Profile, reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername") or ("?", 0)
        entry: dict[str, Any] = {"src": peer[0], "profile": profile.name, "up": 0, "down": 0}
        started, source = self.clock(), None
        try:
            refused = None
            try:
                source = self._admit(peer[0])   # counted from accept: idle heads count too
            except Refused as exc:
                refused = exc
            # Read the head (bounded) even to refuse, so the client never sees a reset.
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"),
                                          self.limits.connect_timeout_s)
            if refused is not None:
                raise refused
            method, target, headers = _parse_head(head)
            entry["method"] = method
            if method == "CONNECT":
                await self._tunnel(profile, source, target, reader, writer, entry)
            elif method in ("GET", "HEAD"):
                await self._forward(profile, source, method, target, headers, writer, entry)
            else:
                raise Refused(405, "method_not_allowed")
            entry.setdefault("verdict", "ok")
        except Refused as exc:
            entry["verdict"] = exc.reason
            await _reply(writer, exc.status, exc.reason)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
            entry["verdict"] = "bad_request"
            await _reply(writer, 400, "bad_request")
        except (asyncio.TimeoutError, TimeoutError):
            entry["verdict"] = "timeout"
        except OSError as exc:
            entry["verdict"] = "upstream_error"
            entry["error"] = type(exc).__name__
            await _reply(writer, 502, "upstream_error")
        finally:
            if source is not None:
                self._release(peer[0], source)
            entry["s"] = round(self.clock() - started, 3)
            entry["at"] = int(time.time())
            self._record(entry)
            writer.close()

    async def _open(self, profile: Profile, host: str, port: int, entry: dict[str, Any], *,
                    tls: bool = False):
        if not profile.allows(host, port):
            raise Refused(403, "host_not_allowed")
        addresses = await asyncio.wait_for(self.resolve(host, port),
                                           self.limits.connect_timeout_s)
        allowed = [a for a in addresses if self.address_allowed(a)]
        if not allowed or len(allowed) != len(addresses):
            raise Refused(403, "address_not_allowed")
        entry["ip"] = allowed[0]
        context = ssl.create_default_context() if tls else None
        return await asyncio.wait_for(asyncio.open_connection(
            allowed[0], port, ssl=context, server_hostname=host if tls else None),
            self.limits.connect_timeout_s)

    async def _tunnel(self, profile, source, target, reader, writer, entry) -> None:
        host, port = _host_port(target)
        entry.update(host=host, port=port)
        up_reader, up_writer = await self._open(profile, host, port, entry)
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        up = asyncio.ensure_future(self._pipe(source, reader, up_writer, entry, "up"))
        try:
            # The client's EOF is passed on (write_eof); the tunnel ends with the upstream.
            await asyncio.wait_for(self._pipe(source, up_reader, writer, entry, "down"),
                                   self.limits.max_tunnel_s)
        finally:
            up.cancel()
            await asyncio.gather(up, return_exceptions=True)
            up_writer.close()

    async def _forward(self, profile, source, method, target, headers, writer, entry) -> None:
        url = target
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname:
            raise Refused(400, "absolute_http_url_required")
        if not profile.allows(parts.hostname, parts.port or 80):
            entry.update(host=parts.hostname, port=parts.port or 80)
            raise Refused(403, "host_not_allowed")
        url = profile.rewrite(url)
        entry["url"] = url[:512]
        parts = urlsplit(url)
        tls = parts.scheme == "https"
        host, port = parts.hostname, parts.port or (443 if tls else 80)
        entry.update(host=host, port=port)
        up_reader, up_writer = await self._open(profile, host, port, entry, tls=tls)
        try:
            path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
            lines = [f"{method} {path} HTTP/1.1", f"Host: {parts.netloc}"]
            lines += [f"{k}: {v}" for k, v in headers if k.lower() not in HOP_BY_HOP]
            lines.append("Connection: close")
            up_writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
            await up_writer.drain()
            head = await asyncio.wait_for(up_reader.readuntil(b"\r\n\r\n"),
                                          self.limits.idle_timeout_s)
            if url != target:
                head = _absolute_location(head, url)
            await self._charge(source, len(head))
            entry["down"] += len(head)
            writer.write(head)
            await self._pipe(source, up_reader, writer, entry, "down")
        finally:
            up_writer.close()

    async def _pipe(self, source, reader, writer, entry, direction) -> None:
        try:
            while True:
                data = await asyncio.wait_for(reader.read(CHUNK), self.limits.idle_timeout_s)
                if not data:
                    break
                await self._charge(source, len(data))
                entry[direction] += len(data)
                writer.write(data)
                await writer.drain()
        finally:
            if direction == "up" and writer.can_write_eof():
                try:
                    writer.write_eof()
                except OSError:
                    pass

    def _record(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        if self._log_file is not None:
            self._log_file.write(line + "\n")
        else:
            _log.info("%s", line)


def _parse_head(head: bytes) -> tuple[str, str, list[tuple[str, str]]]:
    lines = head.decode("latin-1").split("\r\n")
    method, target, version = lines[0].split(" ")
    if not version.startswith("HTTP/1."):
        raise ValueError("not HTTP/1.x")
    headers = []
    for line in lines[1:]:
        if line:
            name, _, value = line.partition(":")
            headers.append((name.strip(), value.strip()))
    return method.upper(), target, headers


def _absolute_location(head: bytes, url: str) -> bytes:
    """A redirect of a rewritten request: its relative Location resolved against the URL
    actually fetched (the client would resolve it against the one it asked for)."""
    from urllib.parse import urljoin

    lines = head.split(b"\r\n")
    for i, line in enumerate(lines[1:], 1):
        name, sep, value = line.partition(b":")
        if sep and name.strip().lower() == b"location":
            location = value.strip().decode("latin-1")
            lines[i] = b"Location: " + urljoin(url, location).encode("latin-1")
    return b"\r\n".join(lines)


def _host_port(target: str) -> tuple[str, int]:
    host, sep, port = target.rpartition(":")
    if not sep or not port.isdigit() or not host:
        raise ValueError("CONNECT target must be host:port")
    return host.strip("[]").lower(), int(port)


async def _reply(writer: asyncio.StreamWriter, status: int, reason: str) -> None:
    try:
        body = (reason + "\n").encode()
        writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Length: {len(body)}\r\n"
                     f"Connection: close\r\n\r\n".encode() + body)
        await writer.drain()
    except OSError:
        pass


def load_config(raw: dict[str, Any]) -> tuple[dict[str, Profile], Limits, list[tuple[str, int, str]],
                                               list[str], str | None]:
    profiles = {}
    for name, spec in raw["profiles"].items():
        hosts = tuple(h.lower().rstrip(".") for h in spec["hosts"])
        if any(_is_ip_literal(h.removeprefix("*.")) or h in ("*", "*.") for h in hosts):
            raise ValueError(f"profile {name}: hosts are names, never addresses or '*'")
        rewrites = tuple((r["from"], r["to"]) for r in spec.get("rewrites", ()))
        if any(not f.startswith("http://") or not t.startswith(("http://", "https://"))
               for f, t in rewrites):
            raise ValueError(f"profile {name}: rewrites map http:// prefixes")
        profiles[name] = Profile(name, hosts, tuple(spec.get("ports", (80, 443))), rewrites)
    listen = [(item["host"], int(item["port"]), item["profile"]) for item in raw["listen"]]
    if any(profile not in profiles for _, _, profile in listen):
        raise ValueError("a listener names an unknown profile")
    limits = Limits(**raw.get("limits", {}))
    return profiles, limits, listen, list(raw.get("sources", ())), raw.get("log")


async def serve(raw: dict[str, Any]) -> None:
    profiles, limits, listen, sources, log_path = load_config(raw)
    proxy = EgressProxy(profiles, limits, sources=sources, log_path=log_path)
    servers = []
    for host, port, name in listen:
        profile = profiles[name]
        servers.append(await asyncio.start_server(
            lambda r, w, p=profile: proxy.handle(p, r, w), host, port, limit=HEAD_LIMIT))
        _log.info("egress profile %s on %s:%d", name, host, port)
    await asyncio.gather(*(server.serve_forever() for server in servers))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(argv[0]) as handle:
        raw = json.load(handle)
    asyncio.run(serve(raw))


if __name__ == "__main__":
    main()
