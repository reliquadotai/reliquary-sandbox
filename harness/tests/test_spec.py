from __future__ import annotations

import dataclasses
import json

import pytest
from reliquary_harness import spec as spec_module
from reliquary_harness.spec import BASH_TOOL, EDIT_TOOL, canonical_json, digest_refusal
from reliquary_harness.spec import BASH_VF_B2E4E81 as SPEC
from reliquary_harness.variant import variant, variant_count, variant_index


def test_tool_sets_are_bash_or_bash_and_edit_in_any_order():
    assert SPEC.tool_specs(("bash",)) == (BASH_TOOL,)
    assert SPEC.tool_specs(["edit", "bash", "edit"]) == (BASH_TOOL, EDIT_TOOL)
    for refused in [(), ("edit",), ("bash", "search")]:
        with pytest.raises(ValueError, match="the bash harness offers bash or bash and edit"):
            SPEC.tool_specs(refused)


def test_the_canonical_spec_is_sorted_compact_utf8():
    data = canonical_json(SPEC.to_dict())
    assert data == json.dumps(json.loads(data), sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False).encode()
    assert len(SPEC.spec_sha256()) == len(SPEC.digest) == 64


def test_the_digest_moves_with_the_spec_and_with_the_code(monkeypatch):
    other = dataclasses.replace(SPEC, version="2")
    assert other.digest != SPEC.digest
    before = SPEC.digest
    monkeypatch.setattr(spec_module, "implementation_sha256", lambda: "0" * 64)
    assert SPEC.digest != before


def test_a_contract_pinning_another_digest_is_refused_not_judged():
    assert digest_refusal(SPEC.id, SPEC.version, SPEC.digest) is None
    assert "the contract pins" in digest_refusal(SPEC.id, SPEC.version, "f" * 64)
    assert "unknown harness" in digest_refusal(SPEC.id, "9", SPEC.digest)


def test_a_spec_without_axes_has_the_one_variant_it_is():
    assert variant_count(SPEC) == 1
    assert variant_index(SPEC, seed=b"seed", engagement="rl-lease:x:0", task_index=3) == 0
    assert variant(SPEC, 0).values == {}
    with pytest.raises(ValueError):
        variant(SPEC, 1)


def test_variants_are_drawn_in_mixed_radix_from_the_seed_alone():
    axes = {"axes": [{"name": "names", "values": ["a", "b"]}, {"name": "prompt", "values": [0, 1, 2]}]}
    spec = dataclasses.replace(SPEC, variants=axes)
    assert variant_count(spec) == 6
    assert [variant(spec, i).values for i in (0, 1, 3, 5)] == [
        {"names": "a", "prompt": 0}, {"names": "a", "prompt": 1}, {"names": "b", "prompt": 0},
        {"names": "b", "prompt": 2}]
    draws = {variant_index(spec, seed=f"s{i}", engagement="e", task_index=1) for i in range(64)}
    assert draws == set(range(6))
    again = variant_index(spec, seed="s1", engagement="e", task_index=1)
    assert again == variant_index(spec, seed=b"s1", engagement="e", task_index=1)
