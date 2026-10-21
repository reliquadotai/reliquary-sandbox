"""A harness as data: what the model is told, which tools it gets, how a turn becomes calls,
and when an episode stops.

A `HarnessSpec` is everything miner, validator and gateway must agree on to produce the
same tokens. It is serialized as canonical JSON; its `digest` covers that JSON and the
source of the modules that interpret it (render, parse, loop), so a change to either is a
new digest, never a silent drift. A contract pins `{id, version, digest}`; a process
whose installed spec has another digest refuses the job (it never judges an episode with
code the miner did not run).

`BASH_VF_B2E4E81` reproduces, byte for byte, verifiers b2e4e81's bash harness as
Catalyst's signed-episode miners ran it through verifiers' train client: the Qwen3.8
renderer of `renderers` 0.1.11, the system prompt fragments of
`harnesses/bash/harness.py`, the tool specs of `harnesses/bash/program.py`, the call
filter of `clients/train.py:response_from_generate`, the restricted-network notice of
`dialects/base.py`, and the stops of `session.py` and the program's loop. The parity
test (`tests/test_upstream_parity.py`) holds it equal to those files.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMPLEMENTATION_FILES = ("spec.py", "qwen.py", "loop.py", "engine.py", "variant.py", "mini.py", "openhands.py",
                        "claude_code.py", "codex.py", "codex_patch.py")
"""The package modules whose source the digest covers: they decide every token."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _tool(name: str, description: str, properties: Mapping[str, Mapping[str, str]],
          required: Sequence[str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {k: dict(v) for k, v in properties.items()},
                       "required": list(required)}}}


@dataclass(frozen=True)
class HarnessSpec:
    id: str
    version: str
    family: str
    """What the model and the operators call it ("bash")."""
    renderer: str
    """`renderers:<name>@<version>`: the template every turn is rendered with."""
    tools: tuple[dict[str, Any], ...]
    """Tool specs in the order the model is shown them (OpenAI function format)."""
    tool_sets: tuple[tuple[str, ...], ...]
    """The tool sets a job may offer (sorted names); the first tool is always offered."""
    system_prompt: Mapping[str, Any]
    """`fragments`: {tool name: sentence}, joined with `join` in the order of `tools`."""
    network_notice: Mapping[str, Any]
    """Appended to the first user message when the runtime restricts egress (legacy
    corpus replay only): `text`, and `join` between content and notice."""
    parse: Mapping[str, Any]
    observation: Mapping[str, Any]
    stops: Mapping[str, Any]
    think: Mapping[str, Any]
    variants: Mapping[str, Any] = field(default_factory=lambda: {"axes": []})
    upstream: Mapping[str, Any] = field(default_factory=dict)
    actions: Mapping[str, Any] = field(default_factory=lambda: {"mode": "tool-calls"})
    """How a turn names its action: `tool-calls` (the template's function calls), `text`
    (mini-swe-agent: one fenced bash block in the answer; no tool is shown to the model), or
    `translated` (`harness`: the module whose own tools the model calls, written in the answer
    when `calls` is `text`, and which writes each call as the canonical `bash` / `edit` call the
    gateway runs and renders its record back in its own format)."""

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "version": self.version, "family": self.family, "renderer": self.renderer,
                "tools": [dict(t) for t in self.tools], "tool_sets": [list(s) for s in self.tool_sets],
                "system_prompt": dict(self.system_prompt), "network_notice": dict(self.network_notice),
                "parse": dict(self.parse), "observation": dict(self.observation),
                "stops": dict(self.stops), "think": dict(self.think), "variants": dict(self.variants),
                "upstream": dict(self.upstream), "actions": dict(self.actions)}

    def spec_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict())).hexdigest()

    @property
    def digest(self) -> str:
        """sha256 over the canonical spec and the interpreting modules' source."""
        return harness_digest(self)

    # -- what the model is shown -------------------------------------------------------

    def tool_set(self, tools: Iterable[str]) -> tuple[str, ...]:
        names = tuple(sorted(set(tools)))
        if names not in self.tool_sets:
            offered = " or ".join(" and ".join(s) for s in self.tool_sets)
            raise ValueError(f"the {self.family} harness offers {offered}, not {list(names)}")
        return names

    @property
    def text_actions(self) -> bool:
        """The model writes its actions in its answer (no tool is shown through the template)."""
        return self.actions.get("mode") == "text" or (self.translated and self.actions.get("calls") == "text")

    @property
    def translated(self) -> bool:
        return self.actions.get("mode") == "translated"

    def translator(self):
        """The module of a translated harness (`actions.harness`)."""
        return translator_module(self.actions.get("harness"))

    def user_prompt(self, task: str) -> str:
        """The first user message: the task, inside the spec's instance template when it has one."""
        instance = self.system_prompt.get("instance")
        return task if instance is None else instance.replace("{task}", task)

    def surface_names(self, values: Mapping[str, Any] | None = None) -> dict[str, str]:
        """{canonical tool name: the name the model is shown} under a variant's axis values
        (`variant.variant(spec, index).values`; None or no `tools` axis: the canonical names)."""
        surface = (values or {}).get("tools") or {}
        return {tool["function"]["name"]: surface.get(tool["function"]["name"], {}).get(
            "name", tool["function"]["name"]) for tool in self.tools}

    def tool_specs(self, tools: Iterable[str], values: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], ...]:
        """The tool specs the model is shown for `tools` (order and repeats do not matter;
        a set the spec does not offer is refused): the renderer parses with exactly these,
        since parameter coercion follows their schemas. A variant's `tools` axis renames a
        tool and rewords its description; its parameters stay the canonical ones."""
        names = self.tool_set(tools)
        if self.text_actions:
            return ()
        if self.translated:
            return tuple(dict(tool) for tool in self.translator().tool_specs(names))
        surface = (values or {}).get("tools") or {}
        shown = []
        for tool in self.tools:
            name = tool["function"]["name"]
            if name not in names:
                continue
            if name in surface:
                function = dict(tool["function"], name=surface[name]["name"],
                                description=surface[name]["description"])
                tool = dict(tool, function=function)
            shown.append(tool)
        return tuple(shown)

    def system_prompt_for(self, tools: Iterable[str], values: Mapping[str, Any] | None = None) -> str:
        """The enabled fragments joined; a variant's `system_prompt` axis is a fragment per
        tool, with `{bash}`/`{edit}` standing for the tools' surface names."""
        names = self.tool_set(tools)
        if self.translated:
            return self.translator().system_prompt(names)
        if "text" in self.system_prompt:
            return self.system_prompt["text"]
        phrased = (values or {}).get("system_prompt")
        fragments = phrased if phrased is not None else self.system_prompt["fragments"]
        surface = self.surface_names(values)
        render = (lambda text: text.format(**surface)) if phrased is not None else (lambda text: text)
        return self.system_prompt["join"].join(
            render(fragments[tool["function"]["name"]]) for tool in self.tools
            if tool["function"]["name"] in names and tool["function"]["name"] in fragments)

    def observation_format(self, values: Mapping[str, Any] | None = None) -> str:
        """How a signed call record is shown: `plain` (the gateway's rendering as is) or a
        variant's `observation` axis value (`decorate_observation`)."""
        return (values or {}).get("observation", self.observation.get("shown", "plain"))

    def with_network_notice(self, content: str) -> str:
        notice = self.network_notice["text"]
        return f"{content}{self.network_notice['join']}{notice}" if content else notice


TRANSLATORS = ("openhands", "claude_code", "codex", "mini")
"""The translated harnesses this package has: each a module of the package (`openhands.py`, ...)."""


def translator_module(name):
    """The module of translated harness `name`: its tools, prompt, call reading, translation onto the
    canonical tools, and the rendering of their records."""
    if name not in TRANSLATORS:
        raise ValueError(f"no translated harness {name!r} in this package ({list(TRANSLATORS)})")
    import importlib

    return importlib.import_module(f"{__package__}.{name}")


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for name in IMPLEMENTATION_FILES:
        data = (_package_dir() / name).read_bytes()
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def harness_digest(spec: HarnessSpec) -> str:
    return hashlib.sha256(json.dumps({"spec": spec.spec_sha256(),
                                      "implementation": implementation_sha256()},
                                     sort_keys=True).encode()).hexdigest()


VERIFIERS_COMMIT = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"

BASH_SENTENCE = "You are a coding agent. You have access to a bash tool for running shell commands."
EDIT_SENTENCE = "You also have an edit tool for single-occurrence string replacement in a file."
CAPABILITY_NOTICE = (
    "Network protocol blocked fetching a resource. Continue without those capabilities; "
    "use local tools or inline data already present in the conversation, and do not retry "
    "the blocked provider-side operation."
)

BASH_TOOL = _tool("bash", "Run a bash command and return its combined stdout and stderr.",
                  {"command": {"type": "string", "description": "The bash command to run."}},
                  ["command"])
EDIT_TOOL = _tool("edit", "Replace a unique string in a file. old_str must appear exactly once in the file.",
                  {"path": {"type": "string", "description": "File path (relative to cwd or absolute)."},
                   "old_str": {"type": "string", "description": "Exact string to find (must appear exactly once)."},
                   "new_str": {"type": "string", "description": "Replacement string."}},
                  ["path", "old_str", "new_str"])

BASH_VF_B2E4E81 = HarnessSpec(
    id="bash-vf-b2e4e81",
    version="1",
    family="bash",
    renderer="renderers:qwen38@0.1.11",
    tools=(BASH_TOOL, EDIT_TOOL),
    tool_sets=(("bash",), ("bash", "edit")),
    system_prompt={"fragments": {"bash": BASH_SENTENCE, "edit": EDIT_SENTENCE}, "join": " "},
    network_notice={"text": CAPABILITY_NOTICE, "join": "\n\n"},
    parse={
        # response_from_generate: a call needs a non-empty name and a status other than
        # UNKNOWN_TOOL; arguments are the raw string, or json.dumps of the parsed dict.
        "filter": "named-and-not-unknown-tool",
        "arguments": "raw-string-or-json-dumps",
        "call_ids": "parser-id-or-call_<index before filtering>",
        # The renderer's parse, made total: whatever it raises is a refused span.
        "on_error": "refuse-span",
    },
    observation={
        # The tool message is the gateway's rendering of the signed record
        # (reliquary_sandbox.observation), or the refusal text of an unsent call.
        "format": "reliquary-observation/1",
        "tools_versions": ["reliquary-tools/1"],
        "tool_message": {"role": "tool", "tool_call_id": "call_<k>"},
    },
    stops={
        # Checked before each model call, in this order (session.RolloutLimits, the
        # task's @stop); the turn that reaches max_turns still has its calls answered.
        "before_model_call": ["max_turns", "episode_closed"],
        "after_turn": {"no_call": "agent_completed"},
        "overlong_prompt": "context_length",
        "closed_on_a_call": "episode_closed",
        "max_tokens_per_turn": "job",
    },
    think={"prefill": "template", "previous_turns": "bridge", "bridge": "renderer.bridge_to_next_turn"},
    variants={"axes": []},
    upstream={"verifiers": VERIFIERS_COMMIT,
              "files": {
                  "v1/harnesses/bash/harness.py": "6a06f21c33a34756e3c00ddbc4f5eaa9c3fd40cf9a2e54fca81a2fd4405fc33d",
                  "v1/harnesses/bash/program.py": "5ddf3f27512ed9519b29c286756bb74fa3dcbd1d4a558fae36b4111e11d94fdb",
                  "v1/dialects/base.py": "a5c16a5974a5e7456d32ecdc63973286c7385a9282339c13fadb244c5655ec23",
                  "v1/clients/train.py": "3aac433f63e148835cffc094ad9449b547ec4ea5caa93041525d36f5819e0439",
                  "v1/session.py": "c0e466a7f3fd76e0f91d6ba91d19bc11056483df3febda3affdcde75a316ddcf",
              }},
)

OBSERVATION_FORMATS = ("plain", "exit-code", *TRANSLATORS)


def decorate_observation(fmt: str, body: Mapping[str, Any], text: str) -> str:
    """A signed call record's observation in format `fmt`: `text` is the gateway's rendering
    of `body` (`reliquary_sandbox.observation.render_observation`); `exit-code` puts the
    command's signed exit code on a line before it. A refused call (no record) is never
    decorated."""
    if fmt == "plain":
        return text
    if fmt == "exit-code":
        code = body["exit_code"]
        if type(code) is not int:
            raise ValueError("a call record's exit_code must be an integer")
        return f"[exit code: {code}]\n{text}"
    if fmt in TRANSLATORS:
        return translator_module(fmt).observation(body)
    raise ValueError(f"unknown observation format {fmt!r}")


SURFACE_TOOLS = {"name": "tools", "values": [
    {},
    {"bash": {"name": "shell",
              "description": "Execute a shell command and return everything it printed: stdout, then stderr."},
     "edit": {"name": "str_replace",
              "description": "Replace the one occurrence of old_str in the file at path with new_str. "
                             "old_str must occur exactly once."}},
]}
SURFACE_PROMPTS = {"name": "system_prompt", "values": [
    {"bash": "You are a coding agent. You have access to a {bash} tool for running shell commands.",
     "edit": "You also have an {edit} tool for single-occurrence string replacement in a file."},
    {"bash": "You are a software engineering agent working in a repository. Use the {bash} tool to run "
             "shell commands and inspect the results.",
     "edit": "Use the {edit} tool to change a file by replacing one exact occurrence of a string."},
]}

BASH_SURFACE_V1 = dataclasses.replace(
    BASH_VF_B2E4E81,
    id="bash-surface",
    version="1",
    # Two tool namings x two prompt phrasings; variant 0 is bash-vf-b2e4e81's surface.
    variants={"axes": [SURFACE_TOOLS, SURFACE_PROMPTS]},
)


MINI_SWE_AGENT_COMMIT = "04d809ceab9df28f9adaed044884180159172930"


def _mini_swe_v2() -> HarnessSpec:
    from . import mini

    return dataclasses.replace(
        BASH_VF_B2E4E81,
        id="mini-swe",
        version="2",
        family="mini-swe-agent",
        # mini-swe-agent's SWE-bench configuration (the bash-only leaderboard's): its system prompt and instance, its
        # one bash tool through the template's tool calls (edit, when the env offers it, is never shown).
        system_prompt={"harness": "mini", "instance": mini.INSTANCE},
        parse={"filter": "named-calls", "arguments": "raw-string-or-json-dumps",
               "turn": "parse_toolcall_actions: no call, or any call not bash with a command, is the turn's format error",
               "submit": mini.SUBMIT, "errors": {"format": mini.FORMAT_ERROR_CALL, "length": mini.LENGTH_ERROR_CALL},
               "on_error": "refuse-span"},
        observation={"format": "reliquary-observation/1", "tools_versions": ["reliquary-tools/1"], "shown": "mini",
                     "tool_message": {"role": "tool", "tool_call_id": "call_<k>"}, "format_error": {"role": "user"}},
        stops={"before_model_call": ["max_turns", "episode_closed"], "after_turn": {"submit": "agent_completed"},
               "overlong_prompt": "context_length", "closed_on_a_call": "episode_closed", "max_tokens_per_turn": "job"},
        actions={"mode": "translated", "harness": "mini", "calls": "tool-calls"},
        upstream={"mini-swe-agent": MINI_SWE_AGENT_COMMIT,
                  "files": ["src/minisweagent/config/benchmarks/swebench.yaml",
                            "src/minisweagent/models/utils/actions_toolcall.py"]},
    )


MINI_SWE_V2 = _mini_swe_v2()
"""The one mini-swe this package serves: mini-swe-agent's official SWE-bench harness. Its text-based form
(`mini-swe@1`, `config/mini_textbased.yaml`) is not served."""

OPENHANDS_COMMIT = "d39f7ae0e66174c50bdc714304fed4078b5e3b72"


def _openhands_xml_v1() -> HarnessSpec:
    from . import openhands

    return dataclasses.replace(
        BASH_VF_B2E4E81,
        id="openhands-xml",
        version="1",
        family="openhands",
        # The prompt is OpenHands' system prompt plus its function descriptions for the offered tools
        # (`openhands.system_prompt`); the task is wrapped in its SWE-bench instance.
        system_prompt={"harness": "openhands", "instance": openhands.INSTANCE},
        parse={"filter": "one-openhands-function", "regex": openhands.FN_REGEX.pattern,
               "parameters": openhands.PARAM_REGEX.pattern, "submit": openhands.SUBMIT_TOOL,
               "local": [openhands.CONTINUE_CALL, openhands.ERROR_CALL, openhands.THINK_CALL],
               "editor": "python3 str_replace_editor in bash, its first line " + openhands.EDITOR_MARK},
        observation={"format": "reliquary-observation/1", "tools_versions": ["reliquary-tools/1"],
                     "shown": "openhands", "message": {"role": "user"}, "truncate_chars": openhands.MAX_MESSAGE_CHARS},
        stops={"before_model_call": ["max_turns", "episode_closed"], "after_turn": {"submit": "agent_completed"},
               "overlong_prompt": "context_length", "closed_on_a_call": "episode_closed", "max_tokens_per_turn": "job"},
        actions={"mode": "translated", "harness": "openhands", "calls": "text"},
        upstream={"openhands": OPENHANDS_COMMIT,
                  "files": ["openhands/agenthub/codeact_agent/prompts/system_prompt.j2",
                            "openhands/agenthub/codeact_agent/tools/str_replace_editor.py",
                            "openhands/llm/fn_call_converter.py",
                            "evaluation/benchmarks/swe_bench/prompts/swe_default.j2"]},
    )


OPENHANDS_XML_V1 = _openhands_xml_v1()

CLAUDE_CODE_TYPES = "@anthropic-ai/claude-agent-sdk@0.3.306 sdk-tools.d.ts (Claude Code 2.1.306)"


def _claude_code_v1() -> HarnessSpec:
    return dataclasses.replace(
        BASH_VF_B2E4E81,
        id="claude-code",
        version="1",
        family="claude-code",
        # Its own prompt and tool descriptions (`claude_code.SYSTEM`, `claude_code.TOOLS`): Claude Code's tool
        # names and parameters, not its texts. The task is the first user message as is.
        system_prompt={"harness": "claude_code"},
        parse={"filter": "named-calls", "arguments": "raw-string-or-json-dumps",
               "unknown": "?<name>, answered No such tool available", "invalid": "?invalid, answered InputValidationError",
               "files": "python3 Read/Edit/Write program in bash, its first line # claude-code:<tool>",
               "on_error": "refuse-span"},
        observation={"format": "reliquary-observation/1", "tools_versions": ["reliquary-tools/1"],
                     "shown": "claude_code", "tool_message": {"role": "tool", "tool_call_id": "call_<k>"}},
        stops={"before_model_call": ["max_turns", "episode_closed"], "after_turn": {"no_call": "agent_completed"},
               "overlong_prompt": "context_length", "closed_on_a_call": "episode_closed", "max_tokens_per_turn": "job"},
        actions={"mode": "translated", "harness": "claude_code", "calls": "tool-calls"},
        upstream={"claude-code": CLAUDE_CODE_TYPES},
    )


CLAUDE_CODE_V1 = _claude_code_v1()

CODEX_COMMIT = "c7234c1"


def _codex_v1() -> HarnessSpec:
    return dataclasses.replace(
        BASH_VF_B2E4E81,
        id="codex",
        version="1",
        family="codex",
        # Codex CLI's tools and base instructions (`codex.SYSTEM`, `codex.TOOLS`; no model identity); the
        # task is the first user message as is.
        system_prompt={"harness": "codex"},
        parse={"filter": "named-calls", "arguments": "raw-string-or-json-dumps",
               "unknown": "?<name>, answered unsupported call", "invalid": "?invalid, answered its parse error",
               "patch": "python3 apply_patch engine (codex_patch) in bash, its first line # codex:apply_patch",
               "on_error": "refuse-span"},
        observation={"format": "reliquary-observation/1", "tools_versions": ["reliquary-tools/1"],
                     "shown": "codex", "tool_message": {"role": "tool", "tool_call_id": "call_<k>"}},
        stops={"before_model_call": ["max_turns", "episode_closed"], "after_turn": {"no_call": "agent_completed"},
               "overlong_prompt": "context_length", "closed_on_a_call": "episode_closed", "max_tokens_per_turn": "job"},
        actions={"mode": "translated", "harness": "codex", "calls": "tool-calls"},
        upstream={"codex": CODEX_COMMIT,
                  "files": ["codex-rs/core/gpt-5.2-codex_prompt.md", "codex-rs/core/gpt_5_2_prompt.md",
                            "codex-rs/core/src/tools/handlers/shell_spec.rs", "codex-rs/apply-patch/src"]},
    )


CODEX_V1 = _codex_v1()

SPECS: dict[str, HarnessSpec] = {spec.key: spec for spec in (BASH_VF_B2E4E81, BASH_SURFACE_V1, MINI_SWE_V2,
                                                               OPENHANDS_XML_V1, CLAUDE_CODE_V1, CODEX_V1)}


def get_spec(harness_id: str, version: str) -> HarnessSpec:
    try:
        return SPECS[f"{harness_id}@{version}"]
    except KeyError:
        raise ValueError(f"unknown harness {harness_id}@{version}; this install has {sorted(SPECS)}") from None


def digest_refusal(harness_id: str, version: str, digest: str) -> str | None:
    """Why this install cannot serve a contract pinning `{harness_id, version, digest}`, or
    None. A refusal to serve, never a judgment of the other side."""
    try:
        spec = get_spec(harness_id, version)
    except ValueError as exc:
        return str(exc)
    if spec.digest != digest:
        return (f"harness {spec.key} is installed with digest {spec.digest}, "
                f"the contract pins {digest}")
    return None


__all__ = ["BASH_SENTENCE", "BASH_SURFACE_V1", "MINI_SWE_V2", "BASH_TOOL", "BASH_VF_B2E4E81", "OBSERVATION_FORMATS",
           "decorate_observation", "CAPABILITY_NOTICE", "EDIT_SENTENCE", "EDIT_TOOL",
           "HarnessSpec", "SPECS", "VERIFIERS_COMMIT", "canonical_json", "digest_refusal", "get_spec",
           "harness_digest", "implementation_sha256"]
