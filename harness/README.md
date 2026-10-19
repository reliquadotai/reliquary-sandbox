# reliquary-harness

One definition of an agentic harness, imported by every side that renders or checks an
episode: the miner (its own episode loop), the validator (admission, intake, export) and,
later, the sandbox gateway (observations). Nothing here runs a tool or loads a model.

* `spec.py`: `HarnessSpec`, canonical JSON, `digest` (spec + the source of the modules
  below), `digest_refusal`. `BASH_VF_B2E4E81` reproduces verifiers b2e4e81's bash
  harness byte for byte (the compatibility harness: today's tokens).
* `qwen.py`: `QwenTurnRenderer`, render and parse in the pinned Qwen3.8 template.
* `engine.py`: one model turn on a vLLM `/inference/v1/generate` endpoint, as verifiers'
  train client sends it.
* `loop.py`: the episode loop and its stops.
* `variant.py`: the surface variant an episode plays, drawn from the validator's seed.

Tests: `tests/test_upstream_parity.py` holds the spec equal to the pinned verifiers files
(vendored under `tests/upstream/`, MIT) and, when verifiers is installed, the installed
files equal to them. It is never skipped.
