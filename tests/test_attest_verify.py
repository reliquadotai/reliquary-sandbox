import copy
import time
from types import SimpleNamespace

import pytest
from episode_fixtures import MACHINE, claims, directory_for, make_signer, transcript

from reliquary_sandbox.attest import (
    Ed25519TokenVerifier,
    Expected,
    Reason,
    chain_hash,
    sign_record,
    verify_transcript,
)
from reliquary_sandbox.attest.records import parse_body
from reliquary_sandbox.attest.verify import MachineKey, StaticDirectory


def expect(**overrides):
    values = dict(hotkey="5Hotkey", engagement="corpus:job-1:7", env="fake-env",
                  split="train", index=7, checkpoint="ck-1")
    values.update(overrides)
    return Expected(**values)


@pytest.fixture
def keys(tmp_path):
    validator = make_signer(tmp_path, "validator", "v-1")[0]
    machine = make_signer(tmp_path, "machine", "m-1")[0]
    return validator, machine


def verify(keys, data, expected=None, directory=None):
    validator, machine = keys
    return verify_transcript(
        data,
        directory or directory_for(machine),
        Ed25519TokenVerifier({validator.key_id: validator.public_key_b64}),
        expected or expect(),
    )


def test_honest_transcript_verifies(keys):
    result = verify(keys, transcript(*keys, claims(), outputs=("a\n", "b\n")))
    assert result.ok, result.reasons
    assert [call.output for call in result.calls] == ["a\n", "b\n"]
    assert result.final.reward == 1.0
    assert result.claims.session_id == "s-1"


def test_modified_output_breaks_the_signature(keys):
    data = transcript(*keys, claims())
    data["records"][1]["body"]["output"] = "forged\n"
    assert Reason.BAD_SIGNATURE in verify(keys, data).reasons


def test_removed_record_is_detected(keys):
    data = transcript(*keys, claims(), outputs=("a", "b", "c"))
    del data["records"][2]
    reasons = verify(keys, data).reasons
    assert Reason.INDEX_GAP in reasons and Reason.PREV_MISMATCH in reasons


def test_added_record_needs_the_machine_key(keys, tmp_path):
    data = transcript(*keys, claims(), outputs=("a",))
    forger = make_signer(tmp_path, "forger", "m-1")[0]
    final = data["records"].pop()
    extra = parse_body(copy.deepcopy(data["records"][1]["body"]))
    extra = type(extra)(**{**extra.__dict__, "i": 2, "k": 1, "output": "invented",
                           "prev": chain_hash(data["records"][1]["body"])})
    data["records"].append(sign_record(forger, MACHINE, extra))
    data["records"].append(final)
    reasons = verify(keys, data).reasons
    assert Reason.BAD_SIGNATURE in reasons and Reason.INDEX_GAP in reasons


def test_spliced_chain_from_another_session_is_detected(keys):
    first = transcript(*keys, claims("s-1"), outputs=("a", "b"))
    second = transcript(*keys, claims("s-2"), outputs=("x", "y"))
    first["records"][2] = second["records"][2]
    reasons = verify(keys, first).reasons
    assert Reason.SESSION_MISMATCH in reasons and Reason.PREV_MISMATCH in reasons


def test_reused_session_is_refused(keys):
    expected = expect(seen_session_ids={"s-1"})
    assert verify(keys, transcript(*keys, claims()), expected).reasons == (Reason.SESSION_REUSED,)


def test_key_outside_its_validity_window_is_unknown(keys):
    validator, machine = keys
    data = transcript(*keys, claims())
    late = directory_for(machine, valid_from=claims().issued_at + 3600)
    assert Reason.UNKNOWN_KEY in verify(keys, data, directory=late).reasons


def test_unknown_machine_key_is_refused(keys, tmp_path):
    other = make_signer(tmp_path, "other", "m-1")[0]
    data = transcript(*keys, claims())
    assert Reason.BAD_SIGNATURE in verify(keys, data, directory=directory_for(other)).reasons


def test_token_from_another_validator_is_refused(keys, tmp_path):
    impostor = make_signer(tmp_path, "impostor", "v-1")[0]
    data = transcript(impostor, keys[1], claims())
    assert verify(keys, data).reasons == (Reason.TOKEN_INVALID,)


def test_wrong_token_for_the_records_is_detected(keys):
    data = transcript(*keys, claims("s-1"))
    data["token"] = transcript(*keys, claims("s-1", hotkey="5Other"))["token"]
    reasons = verify(keys, data, expect(hotkey="5Hotkey")).reasons
    assert Reason.TOKEN_HASH_MISMATCH in reasons


def test_opened_after_token_expiry_is_refused(keys):
    expired = claims()
    expired = type(expired)(**{**expired.__dict__, "expires_at": expired.issued_at + 5})
    assert Reason.OPENED_OUTSIDE_VALIDITY in verify(keys, transcript(*keys, expired)).reasons


def test_submission_must_match_the_engagement(keys):
    data = transcript(*keys, claims())
    reasons = verify(keys, data, expect(hotkey="5Else", index=8, checkpoint="ck-2")).reasons
    assert set(reasons) == {Reason.HOTKEY_MISMATCH, Reason.TASK_MISMATCH, Reason.CHECKPOINT_MISMATCH}


def test_abnormal_end_verifies_but_is_not_graded(keys):
    data = transcript(*keys, claims(), status="budget_exhausted")
    assert verify(keys, data).reasons == (Reason.NOT_GRADED,)
    assert verify(keys, data, expect(require_graded=False)).ok


def test_box_failed_verifies_but_is_not_graded(keys):
    data = transcript(*keys, claims(), status="box_failed", reward=None)
    assert verify(keys, data).reasons == (Reason.NOT_GRADED,)
    assert verify(keys, data, expect(require_graded=False)).ok


def test_missing_final_record(keys):
    data = transcript(*keys, claims())
    data["records"].pop()
    assert Reason.NO_FINAL in verify(keys, data).reasons


@pytest.mark.parametrize("data", [None, {}, {"token": {}, "records": []}, {"token": {}, "records": [1]}])
def test_malformed_transcripts(keys, data):
    assert verify(keys, data).reasons in ((Reason.MALFORMED,), (Reason.TOKEN_INVALID,))


def test_call_order_must_be_sequential(keys):
    validator, machine = keys
    data = transcript(*keys, claims(), outputs=("a", "b"))
    second = parse_body(data["records"][2]["body"])
    second = type(second)(**{**second.__dict__, "k": 5})
    data["records"][2] = sign_record(machine, MACHINE, second)
    final = parse_body(data["records"][3]["body"])
    final = type(final)(**{**final.__dict__, "prev": chain_hash(second)})
    data["records"][3] = sign_record(machine, MACHINE, final)
    assert verify(keys, data).reasons == (Reason.CALL_ORDER,)


def resign(signer, body):
    return sign_record(signer, MACHINE, parse_body(body))


def rechain(data, signer, start):
    records = data["records"]
    for j in range(start, len(records)):
        body = dict(records[j]["body"])
        body["i"] = j
        if j:
            body["prev"] = chain_hash(records[j - 1]["body"])
        records[j] = resign(signer, body)
    return data


def test_backdating_with_a_retired_leaked_key_is_refused(tmp_path):
    validator = make_signer(tmp_path, "v", "v-1")[0]
    old = make_signer(tmp_path, "a", "k-old")[0]
    new = make_signer(tmp_path, "b", "k-new")[0]
    now = int(time.time())
    retired_at = now - 100
    directory = StaticDirectory([
        MachineKey(MACHINE, "k-old", old.public_key_b64, 0, retired_at),
        MachineKey(MACHINE, "k-new", new.public_key_b64, retired_at),
    ])
    pair = (validator, new)
    honest = transcript(*pair, claims(now=now), outputs=("real",))
    assert verify(pair, copy.deepcopy(honest), directory=directory).ok
    forged = copy.deepcopy(honest)
    call = dict(forged["records"][1]["body"], output="FORGED", at=retired_at - 1)
    forged["records"][1] = resign(old, call)
    final = dict(forged["records"][2]["body"], at=retired_at - 1, reward=1.0,
                 prev=chain_hash(forged["records"][1]["body"]))
    forged["records"][2] = resign(old, final)
    result = verify(pair, forged, directory=directory)
    assert not result.ok
    assert Reason.KEY_CHANGED in result.reasons and Reason.TIME_OUT_OF_RANGE in result.reasons


def test_mixed_key_ids_are_refused(keys, tmp_path):
    other = make_signer(tmp_path, "m2", "m-2")[0]
    data = transcript(*keys, claims(), outputs=("a", "b"))
    rechain(data, other, 2)
    directory = StaticDirectory([
        MachineKey(MACHINE, "m-1", keys[1].public_key_b64, 0),
        MachineKey(MACHINE, "m-2", other.public_key_b64, 0),
    ])
    assert Reason.KEY_CHANGED in verify(keys, data, directory=directory).reasons


def test_far_future_record_time_is_refused(keys):
    data = transcript(*keys, claims(), outputs=("a", "b"))
    call = dict(data["records"][2]["body"], at=10**12)
    data["records"][2] = resign(keys[1], call)
    rechain(data, keys[1], 3)
    reasons = verify(keys, data).reasons
    assert Reason.TIME_OUT_OF_RANGE in reasons and Reason.TIME_NOT_MONOTONIC in reasons


def test_grading_may_finish_after_the_deadline_within_the_grace(keys):
    data = transcript(*keys, claims(), outputs=("a",))
    final = dict(data["records"][2]["body"], at=claims().expires_at + 600)
    data["records"][2] = resign(keys[1], final)
    assert verify(keys, data).ok
    late = dict(final, at=claims().expires_at + 10_000)
    data["records"][2] = resign(keys[1], late)
    assert Reason.TIME_OUT_OF_RANGE in verify(keys, data).reasons


def test_non_monotonic_time_is_refused(keys):
    data = transcript(*keys, claims(), outputs=("a",))
    call = dict(data["records"][1]["body"], at=data["records"][0]["body"]["at"] - 1)
    data["records"][1] = resign(keys[1], call)
    rechain(data, keys[1], 2)
    assert Reason.TIME_NOT_MONOTONIC in verify(keys, data).reasons


def _with_call_body(keys, **changes):
    data = transcript(*keys, claims(), outputs=("a",))
    data["records"][1]["body"] = {**data["records"][1]["body"], **changes}
    return data


def _deep(levels):
    root = {}
    cursor = root
    for _ in range(levels):
        cursor["x"] = {}
        cursor = cursor["x"]
    return root


@pytest.mark.parametrize("changes", [
    {"arguments": _deep(100)},
    {"arguments": {"x": float("nan")}},
    {"output": "\ud800"},
])
def test_uncanonical_bodies_are_malformed_not_a_crash(keys, changes):
    assert verify(keys, _with_call_body(keys, **changes)).reasons == (Reason.MALFORMED,)


def test_huge_int_in_grading_is_malformed(keys):
    data = transcript(*keys, claims())
    data["records"][-1]["body"] = {**data["records"][-1]["body"], "grading": {"x": 2**70}}
    assert verify(keys, data).reasons == (Reason.MALFORMED,)


def test_expected_bindings_are_required():
    with pytest.raises(TypeError):
        Expected()


def test_seen_session_ids_must_not_be_a_bare_string():
    with pytest.raises(TypeError):
        expect(seen_session_ids="s-12")


def test_more_calls_than_the_budget_is_refused(keys):
    data = transcript(*keys, claims(), outputs=tuple(str(i) for i in range(20)))
    assert verify(keys, data).reasons == (Reason.TOO_MANY_RECORDS,)


def test_a_huge_record_list_is_refused_quickly(keys):
    data = transcript(*keys, claims())
    data["records"] = data["records"][:1] * 100_000
    start = time.perf_counter()
    assert verify(keys, data).reasons == (Reason.TOO_MANY_RECORDS,)
    assert time.perf_counter() - start < 1.0


def test_open_record_must_match_the_token_env_and_budgets(keys):
    data = transcript(*keys, claims())
    opened = dict(data["records"][0]["body"], env="other-env")
    opened["budgets"] = {**opened["budgets"], "max_calls": 10**6}
    data["records"][0] = resign(keys[1], opened)
    rechain(data, keys[1], 1)
    assert Reason.OPEN_MISMATCH in verify(keys, data).reasons


def test_a_non_open_first_record_is_malformed(keys):
    data = transcript(*keys, claims(), outputs=("a",))
    del data["records"][0]
    assert verify(keys, data).reasons == (Reason.MALFORMED,)


@pytest.mark.parametrize("reward", [-0.5, 1.5, -1e-9])
def test_a_graded_reward_outside_zero_one_is_refused(keys, reward):
    result = verify(keys, transcript(*keys, claims(), reward=reward))
    assert result.reasons == (Reason.REWARD_OUT_OF_RANGE,)


@pytest.mark.parametrize("reward", [0.0, 0.25, 1.0])
def test_rewards_at_and_inside_the_bounds_verify(keys, reward):
    assert verify(keys, transcript(*keys, claims(), reward=reward)).ok


@pytest.mark.parametrize("tools", [("bash",), ("bash", "edit")])
def test_the_offered_tools_come_from_record_zero(keys, tools):
    result = verify(keys, transcript(*keys, claims(), tools=tools))
    assert result.ok and result.tools == tools


def test_malformed_offered_tools_are_malformed(keys):
    data = transcript(*keys, claims())
    body = dict(data["records"][0]["body"], tools=["edit", "bash"])
    # Signed as received, like a machine that wrote it: only the parse refuses it.
    signed = SimpleNamespace(to_dict=lambda: body)
    data["records"][0] = sign_record(keys[1], MACHINE, signed)
    data["records"][1] = sign_record(keys[1], MACHINE, SimpleNamespace(
        to_dict=lambda: dict(data["records"][1]["body"], prev=chain_hash(body))))
    result = verify(keys, data)
    assert result.reasons == (Reason.MALFORMED,) and result.tools is None


def test_a_call_to_a_tool_record_zero_does_not_offer_is_refused(keys):
    # The fixture's calls are `bash`; this record 0 offers only `edit`.
    result = verify(keys, transcript(*keys, claims(), tools=("edit",)))
    assert Reason.TOOL_NOT_OFFERED in result.reasons


def test_calls_to_offered_tools_are_not_flagged(keys):
    result = verify(keys, transcript(*keys, claims(), outputs=("a", "b"), tools=("bash",)))
    assert result.ok, result.reasons
