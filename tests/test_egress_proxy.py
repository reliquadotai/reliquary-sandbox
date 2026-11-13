import asyncio
import json

import pytest

from reliquary_sandbox_service.egress_proxy import (
    EgressProxy,
    Limits,
    Profile,
    address_allowed,
    load_config,
)

PROFILE = Profile("packages", ("pypi.org", "*.pythonhosted.org", "archive.ubuntu.com"),
                  rewrites=(("http://archive.ubuntu.com/ubuntu/",
                             "http://snapshot.test/ubuntu/20261101T000000Z/"),))


class LocalProxy(EgressProxy):
    """Every name resolves to 127.0.0.1, which counts as public here."""

    def __init__(self, *args, upstream_port=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.upstream_port = upstream_port
        self.records = []

    async def resolve(self, host, port):
        return ["127.0.0.1"]

    def address_allowed(self, address):
        return address == "127.0.0.1"

    def _record(self, entry):
        self.records.append(entry)


async def _upstream(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _proxy(proxy, profile=PROFILE):
    server = await asyncio.start_server(lambda r, w: proxy.handle(profile, r, w), "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _request(port, data: bytes, then: bytes = b"") -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(data)
    await writer.drain()
    head = await reader.readuntil(b"\r\n\r\n")
    if then:
        writer.write(then)
        await writer.drain()
        writer.write_eof()
    rest = await asyncio.wait_for(reader.read(), 5)
    writer.close()
    return head + rest


def test_host_allowlist():
    assert PROFILE.allows("pypi.org", 443)
    assert PROFILE.allows("PyPI.org.", 443)
    assert PROFILE.allows("files.pythonhosted.org", 443)
    assert not PROFILE.allows("pythonhosted.org", 443)        # wildcard = subdomains only
    assert not PROFILE.allows("evilpypi.org", 443)
    assert not PROFILE.allows("pypi.org", 22)
    assert not PROFILE.allows("1.1.1.1", 443)


@pytest.mark.parametrize("address,allowed", [
    ("8.8.8.8", True), ("2606:4700:4700::1111", True),
    ("127.0.0.1", False), ("10.1.2.3", False), ("172.17.0.1", False), ("192.168.1.1", False),
    ("169.254.169.254", False), ("100.64.0.1", False), ("0.0.0.0", False), ("224.0.0.1", False),
    ("::1", False), ("fe80::1", False), ("fd00::1", False), ("::ffff:10.0.0.1", False),
    ("not-an-ip", False),
])
def test_address_policy(address, allowed):
    assert address_allowed(address) is allowed


def test_connect_relays_and_records():
    async def run():
        async def echo(reader, writer):
            writer.write(b"pong:" + await reader.read(100))
            await writer.drain()
            writer.close()
        up, up_port = await _upstream(echo)
        profile = Profile("packages", ("pypi.org",), ports=(up_port,))
        proxy = LocalProxy({"packages": profile})
        server, port = await _proxy(proxy, profile)
        out = await _request(port, f"CONNECT pypi.org:{up_port} HTTP/1.1\r\n\r\n".encode(), b"ping")
        server.close()
        up.close()
        return out, proxy.records
    out, records = asyncio.run(run())
    assert out.startswith(b"HTTP/1.1 200") and out.endswith(b"pong:ping")
    assert records[0]["verdict"] == "ok" and records[0]["up"] == 4 and records[0]["down"] == 9


def test_refusals():
    async def run():
        proxy = LocalProxy({"packages": PROFILE})
        server, port = await _proxy(proxy)
        outs = [await _request(port, b"CONNECT evil.com:443 HTTP/1.1\r\n\r\n"),
                await _request(port, b"POST http://pypi.org/x HTTP/1.1\r\n\r\n"),
                await _request(port, b"GET /relative HTTP/1.1\r\n\r\n"),
                await _request(port, b"CONNECT pypi.org:22 HTTP/1.1\r\n\r\n")]
        server.close()
        return outs, proxy.records
    outs, records = asyncio.run(run())
    assert [o.split(b"\r\n")[0] for o in outs] == [
        b"HTTP/1.1 403 host_not_allowed", b"HTTP/1.1 405 method_not_allowed",
        b"HTTP/1.1 400 absolute_http_url_required", b"HTTP/1.1 403 host_not_allowed"]
    assert [r["verdict"] for r in records][0] == "host_not_allowed"


def test_private_resolution_refused():
    class Rebinding(LocalProxy):
        async def resolve(self, host, port):
            return ["93.184.216.34", "169.254.169.254"]

        def address_allowed(self, address):
            return address_allowed(address)

    async def run():
        proxy = Rebinding({"packages": PROFILE})
        server, port = await _proxy(proxy)
        out = await _request(port, b"CONNECT pypi.org:443 HTTP/1.1\r\n\r\n")
        server.close()
        return out
    assert asyncio.run(run()).startswith(b"HTTP/1.1 403 address_not_allowed")


def test_plain_http_rewritten_and_hop_headers_dropped():
    seen = {}

    async def run():
        async def origin(reader, writer):
            seen["head"] = (await reader.readuntil(b"\r\n\r\n")).decode()
            writer.write(b"HTTP/1.1 302 Found\r\nLocation: /file/abc\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()
            writer.close()
        up, up_port = await _upstream(origin)
        profile = Profile("packages", ("archive.ubuntu.com", "snapshot.test"), ports=(80, up_port),
                          rewrites=(("http://archive.ubuntu.com/ubuntu/",
                                     f"http://snapshot.test:{up_port}/ubuntu/20261101T000000Z/"),))
        proxy = LocalProxy({"packages": profile})
        server, port = await _proxy(proxy, profile)
        out = await _request(port, b"GET http://archive.ubuntu.com/ubuntu/dists/noble/InRelease "
                                   b"HTTP/1.1\r\nHost: archive.ubuntu.com\r\nProxy-Authorization: x"
                                   b"\r\nUser-Agent: apt\r\n\r\n")
        server.close()
        up.close()
        return out, proxy.records
    out, records = asyncio.run(run())
    assert out.endswith(b"ok")
    # The relative redirect points at the snapshot, not back at the mirror asked for.
    assert b"Location: http://snapshot.test:" in out and b"/file/abc\r\n" in out
    head = seen["head"]
    assert head.startswith("GET /ubuntu/20261101T000000Z/dists/noble/InRelease HTTP/1.1\r\n")
    assert "Proxy-Authorization" not in head and "User-Agent: apt" in head
    assert "Connection: close" in head and "snapshot.test" in records[0]["url"]


def test_per_source_caps():
    async def run():
        async def sink(reader, writer):
            await reader.read()
            writer.close()
        up, up_port = await _upstream(sink)
        profile = Profile("packages", ("pypi.org",), ports=(up_port,))
        proxy = LocalProxy({"packages": profile}, Limits(connections_per_source=1, bytes_per_window=10))
        server, port = await _proxy(proxy, profile)
        r1, w1 = await asyncio.open_connection("127.0.0.1", port)
        w1.write(f"CONNECT pypi.org:{up_port} HTTP/1.1\r\n\r\n".encode())
        await r1.readuntil(b"\r\n\r\n")
        second = await _request(port, f"CONNECT pypi.org:{up_port} HTTP/1.1\r\n\r\n".encode())
        w1.write(b"x" * 64)                     # over the 10-byte window
        await w1.drain()
        w1.write_eof()
        await asyncio.sleep(0.2)
        w1.close()
        third = await _request(port, f"CONNECT pypi.org:{up_port} HTTP/1.1\r\n\r\n".encode())
        server.close()
        up.close()
        return second, third
    second, third = asyncio.run(run())
    assert second.startswith(b"HTTP/1.1 429 too_many_connections")
    assert third.startswith(b"HTTP/1.1 429 byte_budget_spent")


def test_source_outside_box_subnet_refused():
    async def run():
        proxy = LocalProxy({"packages": PROFILE}, sources=["172.31.250.0/24"])
        server, port = await _proxy(proxy)
        out = await _request(port, b"CONNECT pypi.org:443 HTTP/1.1\r\n\r\n")
        server.close()
        return out
    assert asyncio.run(run()).startswith(b"HTTP/1.1 403 source_not_allowed")


@pytest.mark.parametrize("hosts", [["*"], ["10.0.0.1"], ["*.1.2.3.4"]])
def test_config_refuses_address_and_star_hosts(hosts):
    raw = {"profiles": {"p": {"hosts": hosts}}, "listen": [{"host": "x", "port": 1, "profile": "p"}]}
    with pytest.raises(ValueError):
        load_config(json.loads(json.dumps(raw)))


def test_total_rate_slows_every_source():
    async def run():
        clock = [0.0]
        proxy = LocalProxy({"packages": PROFILE}, Limits(rate_bytes_per_s=1e9, total_rate_bytes_per_s=100),
                           clock=lambda: clock[0])
        slept = []

        async def fake_sleep(s):
            slept.append(s)
        import reliquary_sandbox_service.egress_proxy as mod
        real, mod.asyncio.sleep = mod.asyncio.sleep, fake_sleep
        try:
            a, b = proxy._source("172.31.250.2"), proxy._source("172.31.250.3")
            await proxy._charge(a, 100)
            await proxy._charge(b, 100)     # the shared budget is spent: b waits ~1 s
        finally:
            mod.asyncio.sleep = real
        return slept
    assert asyncio.run(run()) == [1.0]
