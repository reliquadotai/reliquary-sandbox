"""The miner's client against a real gateway (FakeBoxes behind it)."""

import asyncio
import json

import httpx
import pytest
from episode_fixtures import IMAGE, budgets
from gateway_fixtures import fake_gateway

from reliquary_sandbox.attest import canonical_json
from reliquary_sandbox.episode_client import (
    READING,
    Episode,
    EpisodeClient,
    EpisodeClosed,
    encode_body,
)
from reliquary_sandbox.observation import MAX_ARGUMENT_BYTES, NO_OUTPUT, render_observation
from reliquary_sandbox_service.episodes.local_gateway import run_scripted_episode

WRITE = {"command": "write /work/answer.txt 42"}


@pytest.fixture
def gateway(tmp_path):
    with fake_gateway(tmp_path) as both:
        yield both


def token_for(gateway, **overrides):
    return gateway.issue(env="fake-env", split="train", index=7, image=IMAGE,
                         budgets=budgets(**overrides))


def test_bodies_are_compact_utf8_json():
    assert encode_body({"command": "é", "n": 1}) == '{"command":"é","n":1}'.encode()


async def test_a_full_episode_through_the_client(gateway):
    gw, _ = gateway
    async with EpisodeClient(gw.url) as client:
        episode = await client.open(token_for(gw))
        written = await episode.call(0, 0, "bash", WRITE)
        assert written.final is None
        assert render_observation(written.record["body"]) == NO_OUTPUT
        finished = await episode.finish()
        again = await episode.finish()
        transcript = await episode.transcript()
    assert finished.record["body"]["reward"] == 1.0 and finished.state == b"42\n"
    assert again.record == finished.record
    assert gw.verify(transcript["transcript"]).ok


async def test_non_ascii_arguments_at_the_limit_are_recorded_exactly(gateway):
    gw, _ = gateway
    sizes = []
    command = "echo " + "é" * ((120 * 1024 - 19) // 2)   # canonical bytes just under the limit
    hooks = {"request": [lambda request: _note(sizes, request)]}
    async with httpx.AsyncClient(event_hooks=hooks) as http:
        client = EpisodeClient(gw.url, http=http)
        episode = await client.open(token_for(gw))
        result = await episode.call(0, 0, "bash", {"command": command})
    arguments = {"command": command}
    assert result.record["body"]["arguments"] == arguments
    canonical = len(canonical_json(arguments))
    assert MAX_ARGUMENT_BYTES - 8 <= canonical <= MAX_ARGUMENT_BYTES
    sent = encode_body({"turn": 0, "k": 0, "tool": "bash", "arguments": arguments})
    assert sizes[-1] == len(sent)               # the call went out compact UTF-8,
    assert sizes[-1] <= canonical + 64          # not \u-escaped (~6x canonical)


async def _note(sizes, request):
    sizes.append(len(request.content))


class DropFirstCallResponse(httpx.AsyncBaseTransport):
    """The gateway runs the call and answers; the answer is lost on the way back."""

    def __init__(self):
        self.inner = httpx.AsyncHTTPTransport()
        self.dropped = False

    async def handle_async_request(self, request):
        response = await self.inner.handle_async_request(request)
        if request.url.path.endswith("/calls") and not self.dropped:
            await response.aread()
            self.dropped = True
            raise httpx.ReadError("connection lost after the gateway answered", request=request)
        return response


async def test_a_lost_response_is_retried_without_running_the_command_twice(gateway):
    gw, boxes = gateway
    ran = []
    original = boxes.bash

    def counting(*args, **kwargs):
        ran.append(args[2])
        return original(*args, **kwargs)

    boxes.bash = counting
    async with httpx.AsyncClient(transport=DropFirstCallResponse()) as http:
        client = EpisodeClient(gw.url, http=http, sleep=_no_wait)
        episode = await client.open(token_for(gw))
        result = await episode.call(0, 0, "bash", WRITE)
    assert ran == ["write /work/answer.txt 42"]
    assert result.record["body"]["k"] == 0


async def _no_wait(_seconds):
    return None


async def test_a_call_resent_while_it_still_runs_gets_the_same_record(gateway):
    gw, boxes = gateway
    boxes.block.clear()
    statuses = []

    async def note(response):
        statuses.append(response.status_code)

    async with httpx.AsyncClient(event_hooks={"response": [note]}) as http:
        client = EpisodeClient(gw.url, http=http)
        episode = await client.open(token_for(gw))
        first = asyncio.create_task(episode.call(0, 0, "bash", {"command": "hang"}))
        await asyncio.sleep(0.5)
        second = asyncio.create_task(episode.call(0, 0, "bash", {"command": "hang"}))
        await asyncio.sleep(0.5)
        boxes.block.set()
        a, b = await asyncio.gather(first, second)
    assert a.record == b.record
    assert 409 in statuses


async def test_a_call_past_the_budget_raises_with_the_signed_final(gateway):
    gw, _ = gateway
    async with EpisodeClient(gw.url) as client:
        episode = await client.open(token_for(gw, max_calls=1))
        await episode.call(0, 0, "bash", {"command": "echo a"})
        with pytest.raises(EpisodeClosed) as closed:
            await episode.call(1, 0, "bash", {"command": "echo b"})
    assert closed.value.final["body"]["status"] == "budget_exhausted"


async def test_a_finish_after_the_deadline_returns_the_expired_final(gateway):
    gw, _ = gateway
    async with EpisodeClient(gw.url) as client:
        episode = await client.open(token_for(gw, wall_s=1))
        await asyncio.sleep(1.6)
        finished = await episode.finish()
    assert finished.record["body"]["status"] == "expired" and finished.state is None


async def test_run_scripted_episode_drives_and_verifies_an_episode(gateway):
    gw, _ = gateway
    run = await run_scripted_episode(gw, env="fake-env", split="train", index=7,
                                     calls=[("bash", WRITE)])
    assert run.final["status"] == "graded" and run.final["reward"] == 1.0
    assert run.verification.ok and len(run.records) == 1 and run.state == b"42\n"


# -- adaptations to the current gateway: idempotent open, transcript cap, key rotation --


class LoseFirstOpenResponse(httpx.AsyncBaseTransport):
    """The first open reaches the gateway, which goes on opening; its answer never comes
    back. The resend arrives while the episode is still opening."""

    def __init__(self):
        self.inner = httpx.AsyncHTTPTransport()
        self.background = None

    async def handle_async_request(self, request):
        if request.url.path == "/episodes" and self.background is None:
            body = await request.aread()
            copy = httpx.Request(request.method, request.url, headers=request.headers,
                                 content=body)
            self.background = asyncio.create_task(self._forward(copy))
            await asyncio.sleep(0.2)
            raise httpx.ReadError("connection lost while the gateway opens", request=request)
        return await self.inner.handle_async_request(request)

    async def _forward(self, request):
        response = await self.inner.handle_async_request(request)
        await response.aread()
        return response


async def test_a_lost_open_response_resends_the_token_and_gets_the_same_episode(gateway):
    gw, boxes = gateway
    boxes.create_delay = 1.0
    statuses = []

    async def note(response):
        statuses.append(response.status_code)

    transport = LoseFirstOpenResponse()
    async with httpx.AsyncClient(transport=transport, event_hooks={"response": [note]}) as http:
        client = EpisodeClient(gw.url, http=http)
        episode = await client.open(token_for(gw))
        first = (await transport.background).json()
        result = await episode.call(0, 0, "bash", WRITE)
    assert 409 in statuses                     # still opening: waited out, not fatal
    assert episode.key == first["episode_key"] and episode.record0 == first["record"]
    assert len(boxes.boxes) == 1 and result.final is None


async def test_reopening_a_running_episode_returns_its_key_and_record_zero(gateway):
    gw, _ = gateway
    token = token_for(gw)
    async with EpisodeClient(gw.url) as client:
        first = await client.open(token)
        again = await client.open(token)
    assert (again.id, again.key, again.record0) == (first.id, first.key, first.record0)


async def test_reopening_a_closed_episode_raises_with_its_final(gateway):
    gw, _ = gateway
    token = token_for(gw)
    async with EpisodeClient(gw.url) as client:
        episode = await client.open(token)
        await episode.call(0, 0, "bash", WRITE)
        finished = await episode.finish()
        with pytest.raises(EpisodeClosed) as closed:
            await client.open(token)
    assert closed.value.final == finished.record


async def test_the_transcript_cap_ends_the_episode_with_a_signed_final(tmp_path):
    with fake_gateway(tmp_path, episode_max_transcript_bytes=1024**2) as (gw, _):
        async with EpisodeClient(gw.url) as client:
            episode = await client.open(token_for(gw, max_calls=32))
            results = []
            for turn in range(32):
                result = await episode.call(turn, 0, "bash", {"command": "big 70000"})
                results.append(result)
                if result.final is not None:
                    break
            with pytest.raises(EpisodeClosed) as closed:
                await episode.call(turn + 1, 0, "bash", {"command": "echo late"})
            finished = await episode.finish()
    final = results[-1].final["body"]
    assert (final["status"], final["reason"]) == ("budget_exhausted", "transcript_bytes")
    assert closed.value.final == results[-1].final == finished.record
    assert len(results) < 32


def _closed_unsigned(request):
    detail = {"message": "the machine key changed: the episode is void and unsigned",
              "final": None}
    return httpx.Response(409, json={"detail": detail})


async def test_a_key_rotation_raises_closed_without_a_final():
    record0 = {"body": {"budgets": {"per_call_timeout_s": 30}}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(_closed_unsigned)) as http:
        client = EpisodeClient("http://gateway.test", http=http, sleep=_no_wait)
        episode = Episode(client, "s-1", "key", record0)
        with pytest.raises(EpisodeClosed) as on_call:
            await episode.call(0, 0, "bash", WRITE)
        with pytest.raises(EpisodeClosed) as on_finish:
            await episode.finish()
    assert on_call.value.final is None and on_finish.value.final is None
    assert on_call.value.status == 409


# -- fix round 1: retry budget, retryable errors, concurrent transcript reads ------


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class StallFirstCall(httpx.AsyncBaseTransport):
    """The gateway runs the first call, then the connection hangs until the client's
    read timeout (the clock jumps by it) and the answer is lost."""

    def __init__(self, clock):
        self.inner = httpx.AsyncHTTPTransport()
        self.clock = clock
        self.calls = []

    async def handle_async_request(self, request):
        if request.url.path.endswith("/calls"):
            self.calls.append(json.loads(request.content))
            if len(self.calls) == 1:
                response = await self.inner.handle_async_request(request)
                await response.aread()
                self.clock.now += request.extensions["timeout"]["read"]
                raise httpx.ReadTimeout("no answer within the read timeout", request=request)
        return await self.inner.handle_async_request(request)


async def test_a_stalled_call_is_resent_after_its_timeout_and_runs_once(gateway):
    gw, boxes = gateway
    ran = []
    original = boxes.bash

    def counting(*args, **kwargs):
        ran.append(args[2])
        return original(*args, **kwargs)

    boxes.bash = counting
    clock = FakeClock()
    transport = StallFirstCall(clock)
    async with httpx.AsyncClient(transport=transport) as http:
        client = EpisodeClient(gw.url, http=http, retry_s=5, sleep=_no_wait, clock=clock)
        episode = await client.open(token_for(gw))
        result = await episode.call(0, 0, "bash", WRITE)
    assert ran == ["write /work/answer.txt 42"]
    assert [(c["turn"], c["k"]) for c in transport.calls] == [(0, 0), (0, 0)]
    assert result.record["body"]["k"] == 0 and result.final is None


def failing_then_ok(clock, failures, error=httpx.ConnectError, jump=1000.0):
    attempts = []

    def handler(request):
        attempts.append(clock.now)
        if len(attempts) <= failures:
            clock.now += jump
            raise error("down", request=request)
        return httpx.Response(200, json={"record": {"body": {"k": 0}}, "final": None})

    return handler, attempts


RECORD0 = {"body": {"budgets": {"per_call_timeout_s": 30}}}


async def test_three_resends_are_allowed_whatever_the_elapsed_time():
    clock = FakeClock()
    handler, attempts = failing_then_ok(clock, failures=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = EpisodeClient("http://gateway.test", http=http, retry_s=1, sleep=_no_wait,
                               clock=clock)
        result = await Episode(client, "s-1", "key", RECORD0).call(0, 0, "bash", WRITE)
    assert len(attempts) == 4 and result.final is None


async def test_past_three_resends_the_budget_from_the_first_failure_applies():
    clock = FakeClock()
    handler, attempts = failing_then_ok(clock, failures=10)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = EpisodeClient("http://gateway.test", http=http, retry_s=1, sleep=_no_wait,
                               clock=clock)
        with pytest.raises(httpx.ConnectError):
            await Episode(client, "s-1", "key", RECORD0).call(0, 0, "bash", WRITE)
    assert len(attempts) == 4


async def test_no_resend_once_the_caller_deadline_has_passed():
    clock = FakeClock()
    handler, attempts = failing_then_ok(clock, failures=10, jump=10.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = EpisodeClient("http://gateway.test", http=http, retry_s=1000, sleep=_no_wait,
                               clock=clock)
        with pytest.raises(httpx.ConnectError):
            await Episode(client, "s-1", "key", RECORD0).call(
                0, 0, "bash", WRITE, deadline=clock.now + 15)
    assert len(attempts) == 2


async def test_other_transport_errors_are_raised_at_once():
    clock = FakeClock()
    handler, attempts = failing_then_ok(clock, failures=1, error=httpx.UnsupportedProtocol)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = EpisodeClient("http://gateway.test", http=http, sleep=_no_wait, clock=clock)
        with pytest.raises(httpx.UnsupportedProtocol):
            await Episode(client, "s-1", "key", RECORD0).call(0, 0, "bash", WRITE)
    assert len(attempts) == 1


async def test_a_transcript_read_waits_out_a_concurrent_read():
    answers = [httpx.Response(409, json={"detail": READING})] * 2 + [
        httpx.Response(200, json={"status": "RUNNING", "final_status": None,
                                  "transcript": {"token": {}, "records": []}})]
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return answers[len(seen) - 1]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = EpisodeClient("http://gateway.test", http=http, sleep=_no_wait)
        read = await Episode(client, "s-1", "key", RECORD0).transcript()
    assert read["status"] == "RUNNING" and len(seen) == 3
