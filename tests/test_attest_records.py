import pytest
from episode_fixtures import IMAGE, budgets, claims, make_signer

from reliquary_sandbox.attest.canonical import digest
from reliquary_sandbox.attest.records import (
    CallBody,
    FinalBody,
    OpenBody,
    RecordError,
    SessionClaims,
    chain_hash,
    issue_session_token,
    parse_body,
    record_signature_valid,
    sign_record,
    token_sha256,
    verify_session_token,
)
from reliquary_sandbox.attest.signing import public_key_from_b64


@pytest.fixture
def validator(tmp_path):
    return make_signer(tmp_path, "validator", "v-1")[0]


def keys(signer):
    return {signer.key_id: public_key_from_b64(signer.public_key_b64)}


def test_token_round_trip_and_hash(validator):
    session = claims(now=1_000_000)
    token = issue_session_token(session, validator)
    assert verify_session_token(token, keys(validator)) == session
    assert token_sha256(token) == digest(token)


@pytest.mark.parametrize("mutate", [
    lambda t: t["claims"].update(hotkey="other"),
    lambda t: t.update(key_id="unknown"),
    lambda t: t.update(signature=t["signature"][:-4] + "AAAA"),
    lambda t: t.update(v=2),
    lambda t: t.update(extra=1),
])
def test_tampered_tokens_are_refused(validator, mutate):
    token = issue_session_token(claims(), validator)
    mutate(token)
    with pytest.raises(RecordError):
        verify_session_token(token, keys(validator))


@pytest.mark.parametrize("override", [
    {"image": "registry.example/task:latest"},
    {"index": -1},
    {"session_id": ""},
    {"budgets": budgets(max_calls=0)},
])
def test_claims_validate_shape(validator, override):
    token = issue_session_token(claims(**override), validator)
    with pytest.raises(RecordError):
        verify_session_token(token, keys(validator))


def test_claims_refuse_booleans_for_integers():
    data = claims().to_dict()
    data["index"] = True
    with pytest.raises(RecordError):
        SessionClaims.from_dict(data)


def test_records_sign_and_parse(tmp_path):
    machine = make_signer(tmp_path, "machine", "m-1")[0]
    body = CallBody(i=1, session_id="s", turn=0, k=0, tool="bash", arguments={"command": "ls"},
                    output="x", exit_code=0, truncated=False, timed_out=False, cpu_ms=1,
                    wall_ms=2, at=3, prev="0" * 64)
    record = sign_record(machine, "machine-1", body)
    assert parse_body(record["body"]) == body
    assert record_signature_valid(public_key_from_b64(machine.public_key_b64), record)
    assert chain_hash(body) == chain_hash(record["body"])
    record["body"]["output"] = "y"
    assert not record_signature_valid(public_key_from_b64(machine.public_key_b64), record)


@pytest.mark.parametrize("status,reward", [("graded", None), ("expired", 1.0), ("graded", True)])
def test_final_reward_present_exactly_when_graded(status, reward):
    data = FinalBody(i=2, session_id="s", status="expired", reward=None, grading=None,
                     state_sha256=None, cpu_total_ms=0, reason="expired", at=1,
                     prev="0" * 64).to_dict()
    data.update(status=status, reward=reward)
    with pytest.raises(RecordError):
        parse_body(data)


def test_image_constant_is_digest_pinned():
    assert "@sha256:" in IMAGE


def _final_dict(**overrides):
    data = FinalBody(i=2, session_id="s", status="graded", reward=1.0, grading={"t": 1},
                     state_sha256="1" * 64, cpu_total_ms=0, reason=None, at=1,
                     prev="0" * 64).to_dict()
    data.update(overrides)
    return data


@pytest.mark.parametrize("reward", [1, 0, True, False])
def test_non_float_reward_is_rejected(reward):
    with pytest.raises(RecordError):
        parse_body(_final_dict(reward=reward))


def test_chain_hash_and_signature_use_the_raw_received_dict(tmp_path):
    # An int reward would hash differently from its float parse; the parser
    # refuses it, and hashes are only ever computed on the raw dict.
    raw = _final_dict(reward=1.0)
    assert chain_hash(raw) == chain_hash(parse_body(raw))
    machine = make_signer(tmp_path, "machine", "m-1")[0]
    record = sign_record(machine, "machine-1", parse_body(raw))
    assert chain_hash(record["body"]) == chain_hash(raw)


@pytest.mark.parametrize("mutate", [
    lambda t: t.update(v=True),
    lambda t: t.update(v=1.0),
])
def test_token_version_must_be_the_integer_one(validator, mutate):
    token = issue_session_token(claims(), validator)
    mutate(token)
    with pytest.raises(RecordError):
        verify_session_token(token, keys(validator))


@pytest.mark.parametrize("override", [
    {"index": 2**63},
    {"issued_at": -1},
    {"expires_at": 2**63},
])
def test_claim_integers_are_bounded_at_parse_time(override):
    data = claims().to_dict()
    data.update(override)
    with pytest.raises(RecordError):
        SessionClaims.from_dict(data)


@pytest.mark.parametrize("field", ["at", "cpu_ms", "wall_ms", "i", "turn", "k"])
def test_call_integers_are_bounded_at_parse_time(field):
    data = CallBody(i=1, session_id="s", turn=0, k=0, tool="bash", arguments={}, output="",
                    exit_code=0, truncated=False, timed_out=False, cpu_ms=1, wall_ms=2, at=3,
                    prev="0" * 64).to_dict()
    for bad in (2**63, -1):
        data[field] = bad
        with pytest.raises(RecordError):
            parse_body(data)


def test_final_integers_are_bounded_at_parse_time():
    with pytest.raises(RecordError):
        parse_body(_final_dict(cpu_total_ms=2**63))


def test_record_signature_check_never_raises_on_malformed_records(tmp_path):
    machine = make_signer(tmp_path, "machine", "m-1")[0]
    public = public_key_from_b64(machine.public_key_b64)
    for bad in ({}, {"body": {}}, {"body": {}, "machine_id": "m", "key_id": "k"}, None, "x"):
        assert record_signature_valid(public, bad) is False


def test_a_signature_from_another_key_is_refused(tmp_path):
    a = make_signer(tmp_path, "a", "k-a")[0]
    b = make_signer(tmp_path, "b", "k-b")[0]
    body = parse_body(_final_dict())
    record = sign_record(a, "machine-1", body)
    assert record_signature_valid(public_key_from_b64(a.public_key_b64), record)
    assert not record_signature_valid(public_key_from_b64(b.public_key_b64), record)


def test_a_record_signature_is_not_valid_under_the_token_domain(tmp_path):
    from reliquary_sandbox.attest.signing import TOKEN_DOMAIN, signature_valid

    machine = make_signer(tmp_path, "machine", "m-1")[0]
    record = sign_record(machine, "machine-1", parse_body(_final_dict()))
    payload = {"body": record["body"], "machine_id": record["machine_id"],
               "key_id": record["key_id"]}
    assert not signature_valid(public_key_from_b64(machine.public_key_b64), TOKEN_DOMAIN,
                               payload, record["signature"])


def test_box_failed_is_a_final_status_without_a_reward():
    data = FinalBody(i=2, session_id="s", status="box_failed", reward=None, grading=None,
                     state_sha256=None, cpu_total_ms=0, reason="box killed", at=1,
                     prev="0" * 64).to_dict()
    assert parse_body(data).status == "box_failed"
    data["reward"] = 1.0
    with pytest.raises(RecordError):
        parse_body(data)


def open_body(**overrides):
    values = dict(i=0, session_id="s", machine_id="machine-1", token_sha256="0" * 64,
                  image=IMAGE, env="e", env_package="e==1", tools_version="reliquary-tools/1",
                  tools=("bash", "edit"), setup_output_sha256="0" * 64, budgets=budgets(), at=1)
    values.update(overrides)
    return OpenBody(**values)


@pytest.mark.parametrize("tools", [("bash",), ("edit",), ("bash", "edit")])
def test_record_zero_carries_the_offered_tools(tools):
    data = open_body(tools=tools).to_dict()
    assert data["tools"] == list(tools)
    digest(data)  # canonical: a JSON list
    assert parse_body(data) == open_body(tools=tools)


@pytest.mark.parametrize("tools", [
    [], ["edit", "bash"], ["bash", "bash"], ["bash", "python"], "bash", ["bash", 1], None,
])
def test_record_zero_tools_must_be_sorted_unique_known_and_non_empty(tools):
    data = open_body().to_dict()
    data["tools"] = tools
    with pytest.raises(RecordError):
        parse_body(data)
    with pytest.raises(RecordError):
        open_body(tools=tools if not isinstance(tools, list) else tuple(tools))


def test_record_zero_without_tools_is_refused():
    data = open_body().to_dict()
    del data["tools"]
    with pytest.raises(RecordError):
        parse_body(data)


def test_the_known_tools_are_the_tasks_tools():
    from reliquary_sandbox.attest.records import TOOLS
    from reliquary_sandbox.episode_task import TOOLS as TASK_TOOLS

    assert TOOLS == TASK_TOOLS
