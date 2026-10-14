"""What the model is shown for a tool call, defined once for miners and validators.

A miner's harness answers each tool call with `render_observation(record["body"])` of the
call record the gateway signed. A validator re-derives the same text from the signed
record and requires the observation tokens to be its rendering. The two
sides must import these functions; any copy drifts.

Calls whose arguments the gateway would refuse with nothing recorded (no canonical JSON
object, over the argument limit, a bad tool name) are never sent: `plan_call` answers
them locally with a pinned text, and they take no `(turn, k)` position
(`sent_positions`). So are calls to a tool the episode does not offer, when the caller
passes the offered set (`not_offered`): miner and validator apply the same rule, so a
miner's router never sends a tool its harness did not list. Everything else is sent as parsed, byte for byte; blank commands
and unknown tools included (the gateway records them as errors).

Rendering is forward-only: a record's body determines its observation, and nothing
here maps an observation back to a record (several bodies render alike: an empty and a
whitespace-only output, the exit codes). A validator renders the signed record and
compares; it never parses the observation the model saw.

An observation is never empty. verifiers' tool hook answers "allow" when the rewritten
tool message equals the incoming one, whose content is "", and the bash program would
then run the command on the miner's own host. So an empty output renders as NO_OUTPUT.
Whitespace-only output does too: Qwen's chat template strips tool content, so "\n" would
reach the model as an empty tool response.

Format `reliquary-observation/1`, for tools version `reliquary-tools/1`, a function of
the signed body's `tool`, `output`, `truncated` and `timed_out` only:
* NO_OUTPUT in place of the output when the output is empty or whitespace only
  (`output.strip() == ""`), otherwise the output as recorded (stdout then stderr, UTF-8 with replacement; container-runtime
  lines already replaced by `[runtime output removed]` before signing), never altered;
* then, for `bash` only, TRUNCATED_NOTE and then TIMED_OUT_NOTE when flagged, each on a
  line of its own (a newline is added first unless the output part ends with one).
The exit code is not shown, as in verifiers' bash harness.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .attest.canonical import CanonicalError, canonical_json

OBSERVATION_FORMAT = "reliquary-observation/1"
TOOLS_VERSIONS = frozenset({"reliquary-tools/1"})
MAX_ARGUMENT_BYTES = 120 * 1024
"""Canonical size limit of one call's arguments, inclusive. Fixed, not configurable: the
gateway imports this constant. Below one argv element's kernel limit (128 KiB)."""
MAX_TOOL_NAME_CHARS = 64

NO_OUTPUT = "(no output)"
TRUNCATED_NOTE = "[output truncated]"
TIMED_OUT_NOTE = ("[command timed out; its process group was killed. Run long jobs in the "
                  "background with their output redirected to a file, or narrow the command.]")
INVALID_JSON = "error: the tool arguments are not valid JSON; resend the call with a JSON object"
NOT_AN_OBJECT = "error: the tool arguments must be a JSON object; resend the call as an object"
NOT_CANONICAL = ("error: the tool arguments hold a value that cannot be represented exactly "
                 "(an unpaired surrogate, a non-finite number, an integer outside 64 bits or "
                 "nesting deeper than 64 levels); resend them")
TOO_LARGE = (f"error: the tool arguments exceed {MAX_ARGUMENT_BYTES} bytes; split the work "
             "into smaller calls")
BAD_TOOL_NAME = "error: the tool name must be 1 to 64 characters of valid text"
NOT_OFFERED = "error: that tool is not available in this episode; use one of the tools listed"
NOT_TEXT = "error: the tool arguments must be sent as JSON text; resend the call"


class UnsupportedToolsVersion(ValueError):
    """Record 0 names tools this renderer does not know how to render."""


@dataclass(frozen=True)
class Send:
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Refuse:
    reason: str
    """`not_text`, `invalid_json`, `not_an_object`, `not_canonical`, `too_large`,
    `bad_tool_name` or `not_offered`."""
    observation: str


def require_tools_version(version: str) -> None:
    if version not in TOOLS_VERSIONS:
        raise UnsupportedToolsVersion(f"{version!r} is not one of {sorted(TOOLS_VERSIONS)}")


def plan_call(tool: str, arguments_text: str | None,
              offered: Collection[str] | None = None) -> Send | Refuse:
    """Send the call, or answer it here. `arguments_text` is the call's raw argument
    string as the pinned parser returned it; empty or None means `{}` (verifiers' bash
    program does `json.loads(arguments or "{}")`). Anything but a str or None is refused
    (`not_text`). With `offered`, a well-formed name outside it is refused
    (`not_offered`) before the arguments are looked at."""
    if not isinstance(tool, str) or not 1 <= len(tool) <= MAX_TOOL_NAME_CHARS:
        return Refuse("bad_tool_name", BAD_TOOL_NAME)
    try:
        canonical_json(tool)
    except CanonicalError:
        return Refuse("bad_tool_name", BAD_TOOL_NAME)
    if offered is not None and tool not in offered:
        return Refuse("not_offered", NOT_OFFERED)
    if arguments_text is not None and not isinstance(arguments_text, str):
        return Refuse("not_text", NOT_TEXT)
    try:
        value = json.loads(arguments_text or "{}")
    except (ValueError, RecursionError):
        return Refuse("invalid_json", INVALID_JSON)
    if not isinstance(value, dict):
        return Refuse("not_an_object", NOT_AN_OBJECT)
    try:
        size = len(canonical_json(value))
    except CanonicalError:
        return Refuse("not_canonical", NOT_CANONICAL)
    if size > MAX_ARGUMENT_BYTES:
        return Refuse("too_large", TOO_LARGE)
    return Send(tool, value)


def sent_positions(plans: Sequence[Send | Refuse]) -> list[int | None]:
    """`k` of each call within its turn: sent calls count 0, 1, ...; refused ones None."""
    positions: list[int | None] = []
    k = 0
    for plan in plans:
        if isinstance(plan, Send):
            positions.append(k)
            k += 1
        else:
            positions.append(None)
    return positions


def render_observation(body: Mapping[str, Any]) -> str:
    """The observation for a signed call record's body. Raises ValueError for a body whose
    `tool`/`output` are not text or whose flags are not booleans (no truthiness guessing:
    a miner and a validator must never render one record two ways)."""
    tool, text = body["tool"], body["output"]
    truncated, timed_out = body["truncated"], body["timed_out"]
    if not isinstance(tool, str) or not isinstance(text, str):
        raise ValueError("a call record's tool and output must be strings")
    if type(truncated) is not bool or type(timed_out) is not bool:
        raise ValueError("a call record's truncated and timed_out must be booleans")
    if not text.strip():
        text = NO_OUTPUT
    if tool == "bash":
        notes = [note for flag, note in ((truncated, TRUNCATED_NOTE),
                                         (timed_out, TIMED_OUT_NOTE)) if flag]
        if notes:
            separator = "" if text.endswith("\n") else "\n"
            text = text + separator + "\n".join(notes)
    return text
