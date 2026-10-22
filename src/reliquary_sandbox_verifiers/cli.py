"""MCP-only signed tools for actual CLI agents, separate from the legacy surfaces.

ACP transports a CLI process; it does not mediate that process's built-in tools.
Only a reviewed, measured MCP-only capability may enable this route.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from reliquary_sandbox.attest.canonical import canonical_json
from reliquary_sandbox.observation import OBSERVATION_FORMAT, Refuse, plan_call
from reliquary_sandbox.tool_routing import RoutingError, ToolRouter

MCP_SERVER_NAME = "signed_episode"
MCP_PROTOCOL_VERSION = "2024-11-05"
MAX_MCP_BYTES = 128 * 1024
MAX_CLI_TURNS = 48
VERIFIERS_COMMIT = "b2e4e8157783b2c0dffc7821044c87f29f1c3ccf"
CLI_PROMPT_RECIPE = "signed-mcp-fixed-system/1"
CLI_SYSTEM_PROMPT = (
    "You complete the supplied task in the signed workspace. "
    "Use only the supplied signed workspace tools to inspect or change task files. "
    "Treat tool results as observations of that workspace. "
    "After completing the task, return your final answer."
)
CLI_SYSTEM_PROMPT_SHA256 = hashlib.sha256(CLI_SYSTEM_PROMPT.encode()).hexdigest()
CLI_DATE_RECIPE = "claude-current-date/1"
CLI_DATE_MARKER = "<native-episode-date>"
CLI_DATE_REMINDER_TEMPLATE_SHA256 = "bfc995c6dbbbd636bba1a69ea1b97f24e812352799fe15a88e30aa21f1b1f165"
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
CLI_VERSIONS = {
    "signed-codex-cli-v1": ("@openai/codex", "0.147.0",
                             "@agentclientprotocol/codex-acp", "1.2.0"),
    "signed-claude-code-cli-v1": ("@anthropic-ai/claude-code", "2.1.232",
                                   "@agentclientprotocol/claude-agent-acp", "0.67.0"),
}
MCP_TOOLS = (
    {"name": "bash", "description": "Run a command in the signed episode workspace.",
     "inputSchema": {"type": "object", "properties": {"command": {"type": "string"}},
                     "required": ["command"], "additionalProperties": False}},
    {"name": "edit", "description": "Replace one exact string in a signed workspace file.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"}, "old_str": {"type": "string"},
         "new_str": {"type": "string"}}, "required": ["path", "old_str", "new_str"],
         "additionalProperties": False}},
)


class CLIUnavailable(RuntimeError):
    """The actual CLI route has no reviewed qualification or cannot be mediated."""


@dataclass(frozen=True)
class SignedCLICapability:
    protocol_id: str
    runtime_image: str
    # Canonical exact OpenAI function schemas observed at the model boundary.
    model_tools_json: bytes
    model_tool_names: tuple[tuple[str, str], ...]
    # npm package/version/SRI, including transitive and platform artifacts.
    artifacts: tuple[tuple[str, str, str], ...]
    qualification_sha256: str
    # The prepared ACP runner and interpreter, in the immutable runtime image.
    runner_argv: tuple[str, ...]
    runner_sha256: str
    initial_template_sha256: str
    date_recipe: str | None
    prompt_recipe: str = CLI_PROMPT_RECIPE
    system_prompt_sha256: str = CLI_SYSTEM_PROMPT_SHA256
    tool_message_format: str = "text"
    renderer_id: str = "renderers:qwen38@0.1.11"
    tools_version: str = "reliquary-tools/1"
    observation_format: str = OBSERVATION_FORMAT
    max_episode_tokens: int = 65536
    max_tool_calls_per_turn: int = 1

    def __post_init__(self) -> None:
        if (self.protocol_id not in CLI_VERSIONS
                or not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", self.runtime_image)
                or any(not re.fullmatch(r"[0-9a-f]{64}", value)
                       for value in (self.qualification_sha256, self.runner_sha256,
                                     self.initial_template_sha256))
                or self.tool_message_format not in {"text", "text-blocks"}
                or self.date_recipe not in {None, CLI_DATE_RECIPE}
                or self.prompt_recipe != CLI_PROMPT_RECIPE
                or self.system_prompt_sha256 != CLI_SYSTEM_PROMPT_SHA256
                or self.renderer_id != "renderers:qwen38@0.1.11"
                or self.tools_version != "reliquary-tools/1"
                or self.observation_format != OBSERVATION_FORMAT
                or self.max_episode_tokens != 65536
                or self.max_tool_calls_per_turn != 1
                or len(self.runner_argv) != 2
                or any(not value.startswith("/opt/signed-cli/") for value in self.runner_argv)):
            raise ValueError("invalid actual-CLI capability identity")
        if canonical_json(self.model_tools) != self.model_tools_json:
            raise ValueError("the CLI tool schemas must use canonical JSON")
        names = dict(self.model_tool_names)
        if (len(names) != 2 or len(names) != len(self.model_tool_names)
                or set(names.values()) != {"bash", "edit"}
                or {item.get("function", {}).get("name") for item in self.model_tools} != set(names)
                or any(item.get("type") != "function" for item in self.model_tools)):
            raise ValueError("the CLI capability must contain exactly the signed MCP tools")
        required = ((self.versions[0], self.versions[1]), (self.versions[2], self.versions[3]))
        if any(not any(row[:2] == item and row[2].startswith("sha512-")
                       for row in self.artifacts) for item in required):
            raise ValueError("the CLI capability needs exact CLI and ACP artifact integrity")

    @property
    def versions(self) -> tuple[str, str, str, str]:
        return CLI_VERSIONS[self.protocol_id]

    @property
    def model_tools(self) -> list[dict[str, Any]]:
        return json.loads(self.model_tools_json)

    @property
    def model_tools_sha256(self) -> str:
        return hashlib.sha256(self.model_tools_json).hexdigest()

    @property
    def date_template_sha256(self) -> str | None:
        return CLI_DATE_REMINDER_TEMPLATE_SHA256 if self.date_recipe else None

    def require_model_tools(self, tools: Any) -> None:
        if canonical_json(tools) != self.model_tools_json:
            raise RoutingError("the CLI offered tools outside its qualified MCP-only schema")


# Populated only after reviewing a real pinned-CLI CPU probe. A caller-provided
# receipt or a boolean is not authority to add a production capability.
_QUALIFIED: dict[str, SignedCLICapability] = {}


def qualified_capability(protocol_id: str) -> SignedCLICapability:
    try:
        return _QUALIFIED[protocol_id]
    except KeyError:
        raise CLIUnavailable(f"{protocol_id!r} has no approved actual-CLI qualification") from None


def mcp_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def utc_context_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def require_context_date(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("the signed CLI context date must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError("the signed CLI context date must be a canonical ISO date") from None
    if parsed.isoformat() != value:
        raise ValueError("the signed CLI context date must be a canonical ISO date")
    return value


def _canonical_date_block(text: str, context_date: str, template_sha256: str) -> str:
    require_context_date(context_date)
    if not isinstance(text, str):
        raise RoutingError("the native CLI date reminder must be text")
    matches = list(ISO_DATE.finditer(text))
    if len(matches) != 1 or matches[0].group() != context_date:
        raise RoutingError("the native CLI date differs from the signed episode date")
    match = matches[0]
    template = text[:match.start()] + CLI_DATE_MARKER + text[match.end():]
    if not hmac.compare_digest(hashlib.sha256(template.encode()).hexdigest(), template_sha256):
        raise RoutingError("the native CLI date reminder differs from its pinned structure")
    return template


def initial_template_sha256(messages: list[dict[str, Any]], prompt: str, *,
                            context_date: str | None = None,
                            date_recipe: str | None = None) -> str:
    """Hash a measured CLI prefix with exactly one user-task span replaced.

    This permits another trusted task without accepting added startup/local context.
    Dynamic CLI context that changes this hash needs a separately reviewed recipe.
    """
    if not prompt or "<signed-task-prompt>" in prompt:
        raise ValueError("invalid trusted CLI prompt")
    count = 0

    def replace(value):
        nonlocal count
        if isinstance(value, str):
            count += value.count(prompt)
            return value.replace(prompt, "<signed-task-prompt>")
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    template = copy.deepcopy(messages)
    if not template or any(m.get("role") not in {"system", "developer", "user"}
                           for m in template):
        raise RoutingError("the initial CLI prefix contains unsampled messages")
    for message in template:
        if message["role"] == "user":
            message["content"] = replace(message.get("content"))
    if count != 1:
        raise RoutingError("the CLI prefix must contain the trusted task exactly once")
    if date_recipe is not None:
        if date_recipe != CLI_DATE_RECIPE or context_date is None:
            raise RoutingError("the CLI prefix needs its signed native-date recipe")
        users = [message for message in template if message["role"] == "user"]
        content = users[0].get("content") if users else None
        if (not isinstance(content, list) or len(content) != 2
                or any(not isinstance(block, dict) or set(block) != {"type", "text"}
                       or block["type"] != "text" for block in content)):
            raise RoutingError("the native CLI initial user blocks changed")
        content[0]["text"] = _canonical_date_block(
            content[0]["text"], context_date, CLI_DATE_REMINDER_TEMPLATE_SHA256)

        def has_date(value):
            if isinstance(value, str):
                return ISO_DATE.search(value) is not None
            if isinstance(value, list):
                return any(has_date(item) for item in value)
            if isinstance(value, dict):
                return any(has_date(item) for item in value.values())
            return False

        if has_date(template):
            raise RoutingError("the CLI prefix contains another uncontrolled date")
    return hashlib.sha256(canonical_json(template)).hexdigest()


class SignedCLIToolBridge:
    """Join model-generated calls to signed tools and verify the CLI's next request.

    The trusted model proxy registers each response before returning it to the CLI.
    Only one tool call per model turn is permitted: MCP does not carry the model's
    call id, so parallel identical calls cannot be attributed safely.
    """

    def __init__(self, router: ToolRouter, capability: SignedCLICapability, *,
                 prompt: str,
                 max_turns: int = MAX_CLI_TURNS, context_date: str | None = None) -> None:
        if not 1 <= max_turns <= MAX_CLI_TURNS:
            raise ValueError("CLI turns must be between 1 and 48")
        self.router = router
        self.capability = capability
        self.max_turns = max_turns
        self.context_date = (require_context_date(context_date) if capability.date_recipe
                             else context_date)
        self._turns: list[tuple[tuple[str, str, str], ...]] = []
        self._prompt = prompt
        self._history: list[dict[str, Any]] | None = None
        self._served: dict[tuple[int, str], str] = {}
        self._pending: tuple[int, str, str, dict[str, Any]] | None = None
        self._lock = asyncio.Lock()

    @property
    def model_turns(self) -> int:
        return len(self._turns)

    def register_turn(self, turn: int, calls: list[tuple[str, str, str]], *,
                      assistant_message: dict[str, Any]) -> None:
        if self._history is None:
            raise RoutingError("the qualified initial CLI prompt has not been validated")
        if self.router.final is not None or self._pending is not None:
            raise RoutingError("a CLI response arrived before its signed tool completed")
        if turn != len(self._turns) or turn >= self.max_turns:
            raise RoutingError("the CLI model turn is out of sequence or over its limit")
        if len(calls) > 1:
            raise RoutingError("the signed CLI route permits one tool call per model turn")
        translated = []
        pending = None
        names = dict(self.capability.model_tool_names)
        for call_id, name, raw in calls:
            if not isinstance(call_id, str) or not call_id or name not in names:
                raise RoutingError("the CLI response contains an unknown tool or call id")
            tool = names[name]
            plan = plan_call(tool, raw, {"bash", "edit"})
            if isinstance(plan, Refuse):
                raise RoutingError("the CLI response contains invalid tool arguments")
            schema = next(item["inputSchema"] for item in MCP_TOOLS if item["name"] == tool)
            if (set(plan.arguments) != set(schema["required"])
                    or any(not isinstance(v, str) for v in plan.arguments.values())):
                raise RoutingError("the CLI response does not match the signed tool schema")
            translated.append((call_id, tool, canonical_json(plan.arguments).decode()))
            pending = (turn, call_id, tool, plan.arguments)
        message = self._normalise_message(assistant_message)
        if message.get("role") != "assistant":
            raise RoutingError("a model response must be an assistant message")
        wire_calls = [(item.get("id"), item.get("function", {}).get("name"),
                       item.get("function", {}).get("arguments"))
                      for item in message.get("tool_calls") or []]
        expected_calls = [(call_id, name, canonical_json(json.loads(raw)).decode())
                          for call_id, name, raw in calls]
        if wire_calls != expected_calls:
            raise RoutingError("the sampled response and its registered tool calls differ")
        self._turns.append(tuple(translated))
        self._history.append(message)
        self._pending = pending

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if self._pending is None:
                raise RoutingError("an MCP tool was not generated by the checkpoint model")
            turn, call_id, tool, expected = self._pending
            if name != tool or canonical_json(arguments) != canonical_json(expected):
                raise RoutingError("the MCP call differs from the checkpoint model's call")
            records_before = len(self.router.records)
            text = await self.router.answer(turn, self._turns[turn], call_id)
            if len(self.router.records) != records_before + 1:
                raise RoutingError("the MCP call did not produce a signed episode record")
            self._served[turn, call_id] = text
            content = (text if self.capability.tool_message_format == "text"
                       else mcp_result(text)["content"])
            self._history.append({"role": "tool", "tool_call_id": call_id, "content": content})
            self._pending = None
            return mcp_result(text)

    def expected_result(self, turn: int, call_id: str) -> dict[str, Any]:
        try:
            return mcp_result(self._served[turn, call_id])
        except KeyError:
            raise RoutingError("there is no served signed result for this tool message") from None

    def validate_messages(self, messages: list[dict[str, Any]]) -> None:
        """Verify all normalized OpenAI tool messages before another generation.

        A CLI that rewrites, compacts away or inserts the sampled history is refused;
        a measured alternative representation needs a separate qualified capability.
        """
        normal = [self._normalise_message(message) for message in messages]
        if self._history is None:
            if initial_template_sha256(normal, self._prompt, context_date=self.context_date,
                                       date_recipe=self.capability.date_recipe) != self.capability.initial_template_sha256:
                raise RoutingError("the CLI initial prefix differs from its qualified task template")
            self._history = normal
        elif self._pending is not None or canonical_json(normal) != canonical_json(self._history):
            raise RoutingError("the CLI changed sampled history or a signed observation")

    @property
    def has_pending_call(self) -> bool:
        return self._pending is not None

    def require_final_reply(self, reply: str) -> None:
        final = self._history[-1] if self._history else {}
        if (not self._turns or self._pending is not None or final.get("role") != "assistant"
                or final.get("tool_calls") or not isinstance(final.get("content"), str)
                or reply != final["content"]):
            raise RoutingError("the ACP final reply differs from the final checkpoint-model turn")

    @staticmethod
    def _normalise_message(message: dict[str, Any]) -> dict[str, Any]:
        normal = copy.deepcopy(message)
        if not isinstance(normal, dict):
            raise RoutingError("a CLI message must be an object")
        # Both pinned CLI dialect parsers fold absent tool-call text to None.
        if normal.get("role") == "assistant" and normal.get("tool_calls") and normal.get("content") == "":
            normal["content"] = None
        for call in normal.get("tool_calls") or []:
            try:
                function = call["function"]
                function["arguments"] = canonical_json(json.loads(function["arguments"])).decode()
            except (KeyError, TypeError, ValueError):
                raise RoutingError("invalid CLI history tool arguments") from None
        return normal


class SignedMCPApp:
    """Small stateless JSON MCP endpoint, reachable only at a per-run secret path."""

    def __init__(self, bridge: SignedCLIToolBridge, path_secret: str) -> None:
        if len(path_secret) < 32 or not path_secret.isalnum():
            raise ValueError("MCP path secret must contain at least 32 alphanumeric characters")
        self.bridge = bridge
        self._path = "/" + path_secret
        self._loop = asyncio.get_running_loop()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            raise ValueError("the signed MCP endpoint supports HTTP only")
        status, result = 200, None
        if not hmac.compare_digest(scope.get("path", ""), self._path):
            status, result = 404, {"error": "not found"}
        elif scope["method"] != "POST":
            status, result = 405, {"error": "method not allowed"}
        else:
            raw = bytearray()
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                raw.extend(event.get("body", b""))
                if len(raw) > MAX_MCP_BYTES:
                    status, result = 413, {"error": "request too large"}
                    break
                if not event.get("more_body"):
                    break
            if result is None:
                request_id = None
                try:
                    request = json.loads(raw)
                    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                        raise ValueError("invalid MCP request")
                    request_id = request.get("id")
                    method, params = request.get("method"), request.get("params") or {}
                    if method == "notifications/initialized":
                        status, result = 202, None
                    elif method == "initialize":
                        result = {"protocolVersion": MCP_PROTOCOL_VERSION,
                                  "capabilities": {"tools": {}},
                                  "serverInfo": {"name": MCP_SERVER_NAME, "version": "1"}}
                    elif method == "tools/list":
                        result = {"tools": copy.deepcopy(list(MCP_TOOLS))}
                    elif method == "tools/call":
                        call = self.bridge.call(params["name"], params.get("arguments", {}))
                        if asyncio.get_running_loop() is self._loop:
                            result = await call
                        else:
                            # Gateway HTTP connections and model callbacks belong to one
                            # loop even when the ASGI listener runs in a server thread.
                            result = await asyncio.wrap_future(
                                asyncio.run_coroutine_threadsafe(call, self._loop))
                    elif method == "ping":
                        result = {}
                    else:
                        raise ValueError("unsupported MCP method")
                    if status == 200:
                        result = {"jsonrpc": "2.0", "id": request_id, "result": result}
                except (ValueError, TypeError, KeyError, RoutingError):
                    result = {"jsonrpc": "2.0", "id": request_id,
                              "error": {"code": -32602, "message": "signed MCP request refused"}}
        body = b"" if result is None else canonical_json(result)
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
