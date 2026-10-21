"""reliquary-harness: the single source of an agentic harness for miners, validators and gateways.

A harness is a `HarnessSpec` (data: prompt, tools, parse filter, observation format, stops,
variant axes) and the pure functions that interpret it: render and parse a turn
(`qwen.QwenTurnRenderer`), run one model turn on a generate endpoint (`engine`), and the
episode loop (`loop.play`). Contracts pin a spec by `{id, version, digest}`; a different
installed digest refuses to serve (`spec.digest_refusal`)."""

from .spec import BASH_VF_B2E4E81, SPECS, HarnessSpec, digest_refusal, get_spec

__version__ = "0.1.0"

__all__ = ["BASH_VF_B2E4E81", "SPECS", "HarnessSpec", "__version__", "digest_refusal", "get_spec"]
