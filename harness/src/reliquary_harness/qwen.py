"""A harness's turns in the pinned Qwen3.8 template (`renderers` 0.1.11): render and parse.

`QwenTurnRenderer` is the one renderer miners and validators share for a harness spec:
the initial prompt (system prompt and tool specs from the spec), the tool calls read
from an assistant span (verifiers' `response_from_generate` filter), the next prompt
after a turn's observations (`bridge_to_next_turn`), and the checks that keep a span
exportable as a message (its canonical round trip, its closed reasoning).

`renderers` is imported inside functions only: the package itself depends on nothing.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any

from .spec import BASH_VF_B2E4E81, HarnessSpec, decorate_observation
from .variant import variant as variant_of


class UnparsableSpan(ValueError):
    """The pinned parser raised on a span (a parameter nested 10k deep raises
    RecursionError from its JSON coercion): the span is refused, never read."""

    def __init__(self, error: str) -> None:
        super().__init__(f"the pinned parser failed on a span: {error}")
        self.error = error


def _unparsable(exc: BaseException) -> BaseException:
    return UnparsableSpan(type(exc).__name__)


def calls_with_ids(parsed_calls, unknown_status) -> list[tuple[str, str, str]]:
    """`(id, name, arguments)` of the calls the spec's filter keeps (verifiers b2e4e81
    `response_from_generate`, exactly): a non-empty name and a status other than
    UNKNOWN_TOOL (so a block with no name, unclosed or malformed, is no call on either
    side); arguments are the raw string, or the parsed dict through `json.dumps` with its
    defaults; the id is the parser's, else `call_<i>` with i the call's index before
    filtering."""
    return [(getattr(call, "id", None) or f"call_{i}", call.name, call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments or {}))
            for i, call in enumerate(parsed_calls or [])
            if getattr(call, "name", None) and getattr(call, "status", None) != unknown_status]


def pinned_tool_calls(parsed_calls, unknown_status) -> list[tuple[str, str]]:
    """The one parser miners and validators share (spec §5.C amendment 1): `(name,
    arguments)` of `calls_with_ids`."""
    return [(name, arguments) for _, name, arguments in calls_with_ids(parsed_calls, unknown_status)]


def _locked(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


RENDERERS_VERSION = "0.1.11"
"""The `renderers` release every spec's renderer (`renderers:qwen38@0.1.11`) is."""

HIDDEN_PREFIX = "?"
"""A call to a canonical tool name the variant does not show is read as `?<name>`: a tool
nobody offers (`not_offered`), so a hidden name is never executed."""


class QwenTurnRenderer:
    """`trajectory_parse.TurnRenderer` over the pinned `renderers` Qwen3.8 renderer and a
    harness spec's tools and system prompt, in one of its surface variants (`variant`, an
    index; 0 is a spec without axes). The model is shown the variant's tool names; the calls
    read back (`tool_calls`, `calls_of`) carry the canonical names the gateway runs, while the
    assistant message keeps the names the model wrote (it must re-render to the span).
    `refuse(exc)` builds what a span the parser raised on is refused with (default
    `UnparsableSpan`)."""

    def __init__(self, renderer, tools=("bash", "edit"), *, spec: HarnessSpec = BASH_VF_B2E4E81,
                 variant: int = 0, refuse: Callable[[BaseException], BaseException] | None = None) -> None:
        from renderers.base import ToolCallParseStatus

        self.spec = spec
        self.variant = variant_of(spec, variant)
        values = self.variant.values
        offered = spec.tool_set(tools)
        surface = spec.surface_names(values)
        self._canonical = {surface[name]: name for name in offered}
        self._hidden = frozenset(name for name in surface if surface[name] != name)
        self.observation_format = spec.observation_format(values)
        self._text = spec.text_actions
        # A translated harness (OpenHands...): its own module reads, translates and answers the calls.
        self._translator = spec.translator() if spec.translated else None
        self._offered = offered
        self._r = renderer
        # P6: HF tokenizers are not thread-safe; intake, grading and export share this.
        self._lock = threading.RLock()
        self._tokenizer = renderer._tokenizer
        self._unknown = ToolCallParseStatus.UNKNOWN_TOOL
        self._refuse = refuse or _unparsable
        stops = list(renderer.get_stop_token_ids())
        self.terminator_id = int(stops[0])
        self.stop_ids = frozenset(int(t) for t in stops)
        self._open = renderer._token_id("<tool_response>")
        self._close = renderer._token_id("</tool_response>")
        self.turn_markup_ids = frozenset(
            int(renderer._token_id(t)) for t in ("<|im_start|>", "<tool_response>", "</tool_response>"))
        self._tools = [dict(tool) for tool in spec.tool_specs(tools, values)]
        self._system_prompt = spec.system_prompt_for(tools, values)
        self._im_start = int(renderer._token_id("<|im_start|>"))
        self._think = int(renderer._token_id("<think>"))
        self._think_end = int(renderer._token_id("</think>"))
        literals = set(getattr(self._tokenizer, "all_special_tokens", []) or [])
        for added in (getattr(self._tokenizer, "added_tokens_decoder", {}) or {}).values():
            literals.add(str(getattr(added, "content", added)))
        self._literals = tuple(sorted(t for t in literals if t))

    @property
    def renderer(self):
        """The pinned `renderers` renderer underneath (the engine adapter generates with it)."""
        return self._r

    @property
    def tools(self) -> list[dict]:
        return [dict(tool) for tool in self._tools]

    @property
    def _shown_tools(self):
        """The tools the template renders (None when the model is shown none: text actions)."""
        return self._tools or None

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @_locked
    def initial_ids(self, prompt: str) -> list[int]:
        rendered = self._r.render(
            [{"role": "system", "content": self._system_prompt},
             {"role": "user", "content": self.spec.user_prompt(prompt)}],
            tools=self._shown_tools, add_generation_prompt=True)
        return [int(t) for t in rendered.token_ids]

    def canonical_name(self, name: str) -> str:
        """The tool a surface name calls: its canonical name, `?<name>` for a canonical name
        this variant hides, any other name as written (not offered either way)."""
        if name in self._canonical:
            return self._canonical[name]
        return HIDDEN_PREFIX + name if name in self._hidden else name

    def _surface_calls(self, completion_ids) -> list[tuple[str, str]]:
        if self._translator is not None and self._text:
            # The functions written in the answer, and in any call the template parser took out of it.
            parsed = self._parse(completion_ids)
            texts = [parsed.content or "", *(getattr(call, "raw", "") or "" for call in parsed.tool_calls or ())]
            return self._translator.turn_calls(texts, self._offered)
        return self._template_calls(completion_ids)

    def _template_calls(self, completion_ids) -> list[tuple[str, str]]:
        parsed = self._parse(completion_ids).tool_calls
        if self._translator is not None:
            # A translated harness answers a call to a tool it does not have (its module's error text):
            # every named call is read, its arguments as written.
            return [(call.name, call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments or {}))
                    for call in parsed or [] if getattr(call, "name", None)]
        return pinned_tool_calls(parsed, self._unknown)

    @_locked
    def tool_calls(self, completion_ids) -> list[tuple[str, str]]:
        if self._translator is not None:
            surface = self._surface_calls(completion_ids)
            whole_turn = getattr(self._translator, "turn_actions", None)
            if whole_turn is not None:
                # The harness reads the turn as a whole (mini: no call or any bad call is one format error, a
                # submit ends the episode): `ended` is whether the turn stopped on a stop id (not at its cap).
                ended = bool(completion_ids) and int(completion_ids[-1]) in self.stop_ids
                surface = whole_turn(surface, ended=ended)
                if surface is None:
                    return []
            calls = [self._translator.translate(name, arguments, self._offered) for name, arguments in surface]
            # A submit (None) ends the episode: the turn sends nothing.
            return [] if any(call is None for call in calls) else calls
        return [(self.canonical_name(name), arguments) for name, arguments in self._surface_calls(completion_ids)]

    def turn_calls(self, result) -> list[tuple[str, str, str]]:
        """`(id, canonical name, arguments)` of a generate result (`renderers.client.generate`'s dict)."""
        if self._text or self._translator is not None:
            return [(f"call_{i}", name, arguments)
                    for i, (name, arguments) in enumerate(self.tool_calls(result["completion_ids"]))]
        return self.calls_of(result.get("tool_calls"))

    def refusal_observation(self, name: str, arguments: str | None) -> str | None:
        """The harness's own text for a call answered locally (mini's format errors), or None."""
        if self._translator is not None:
            return self._translator.refusal(name, arguments)
        return None

    def calls_of(self, parsed_calls) -> list[tuple[str, str, str]]:
        """`calls_with_ids` of a parse result's calls (the generate endpoint's), canonical names."""
        return [(call_id, self.canonical_name(name), arguments)
                for call_id, name, arguments in calls_with_ids(parsed_calls, self._unknown)]

    def observation(self, body, text: str) -> str:
        """What the model is shown for a signed call record: `text` (the gateway's rendering
        of `body`) in this variant's observation format."""
        return decorate_observation(self.observation_format, body, text)

    def _parse(self, completion_ids):
        """The pinned parser, made total: whatever it raises on a span refuses it
        (`refuse`) instead of escaping the caller."""
        try:
            return self._r.parse_response(list(completion_ids), tools=self._shown_tools)
        except Exception as exc:  # RecursionError is an Exception
            raise self._refuse(exc) from None

    @_locked
    def observations(self, segment_ids) -> list[str]:
        out, start = [], None
        for k, token in enumerate(segment_ids):
            if token == self._open:
                start = k + 1
            elif token == self._close and start is not None:
                text = self._tokenizer.decode(list(segment_ids[start:k]), skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
                # The renderer writes "\n" + content.strip() + "\n".
                text = text[1:] if text.startswith("\n") else text
                text = text[:-1] if text.endswith("\n") else text
                out.append(text)
                start = None
        return out

    def tool_messages(self, observations: Sequence[str]) -> list[dict[str, Any]]:
        """The messages a turn's observations are bridged as: tool messages (spec
        `observation.tool_message`), or, with text actions, user messages."""
        if self._text:
            return [{"role": "user", "content": text} for text in observations]
        as_user = getattr(self._translator, "user_message", None) or (lambda text: False)
        return [{"role": "user", "content": text} if as_user(text)
                else {"role": "tool", "tool_call_id": f"call_{i}", "content": text}
                for i, text in enumerate(observations)]

    @_locked
    def next_prompt(self, prompt_ids, completion_ids, observations) -> list[int] | None:
        rendered = self._r.bridge_to_next_turn(list(prompt_ids), list(completion_ids),
                                               self.tool_messages(observations), tools=self._shown_tools)
        return None if rendered is None else [int(t) for t in rendered.token_ids]

    def _tail(self, ids) -> list[int]:
        """The ids after the last <|im_start|>: role line, and any opened <think>."""
        for k in range(len(ids) - 1, -1, -1):
            if ids[k] == self._im_start:
                return list(ids[k + 1:])
        return list(ids)

    def _opened(self, prompt_ids, completion_ids) -> list[int]:
        """The turn as the parser must read it: when the generation prompt already opened ``<think>`` (a
        thinking template), the span starts inside the reasoning, so the opener is put back in front of it;
        otherwise the parser would take the reasoning and its ``</think>`` for content."""
        ids = [int(t) for t in completion_ids]
        tail = self._tail(prompt_ids)
        if tail and tail[-1] == self._think or (self._think in tail and self._think_end not in tail):
            if self._think not in ids:
                return [self._think, *ids]
        return ids

    @_locked
    def reasoning_unclosed(self, prompt_ids, completion_ids) -> bool:
        ids = list(completion_ids)
        if self._think_end in ids:
            return False
        return self._think in ids or self._think in self._tail(prompt_ids)

    @_locked
    def span_is_canonical(self, prompt_ids, completion_ids) -> bool:
        """Round trip: parse the span to a message, render that message as an
        assistant turn, and require the span's own tokens, so no text can pose
        as turn structure once the message is re-encoded for export."""
        completion = [int(t) for t in completion_ids]
        message = self.assistant_message(self._opened(prompt_ids, completion))
        texts = [message.get("content") or "", message.get("reasoning_content") or ""]
        for call in message.get("tool_calls") or []:
            texts += [call["function"]["name"], call["function"]["arguments"]]
        if any(literal in text for text in texts for literal in self._literals):
            return False
        rendered = self._r.render([{"role": "user", "content": "x"}, message], tools=self._shown_tools,
                                  add_generation_prompt=False).token_ids
        tail = self._tail([int(t) for t in rendered])
        closes = [k for k, t in enumerate(tail) if t == self.terminator_id]
        if not closes:
            return False
        tail = tail[:closes[-1] + 1]                    # drop the template's trailing newline
        if not completion or completion[-1] not in self.stop_ids:
            tail = tail[:-1]                           # capped turn: the bridge adds its own close
        expected = self._tail(prompt_ids) + completion
        if tail == expected:
            return True
        # The pinned parser strips the whitespace around a parameter value (an
        # indented `old_str` loses its indentation: 4 % of honest recorded
        # turns), so an exact comparison would refuse honest edits. Fall back
        # to the same comparison with whitespace erased: what the parser
        # dropped or changed beyond whitespace still fails it, and spelled
        # markup was already refused by the literal check above.
        return self.whitespace_free(tail) == self.whitespace_free(expected)

    @_locked
    def render_messages(self, messages) -> list[int]:
        """A whole conversation (system, user, assistant and tool messages), as
        the pinned template renders it with the harness tools."""
        rendered = self._r.render(list(messages), tools=self._shown_tools, add_generation_prompt=False)
        return [int(t) for t in rendered.token_ids]

    @_locked
    def whitespace_free(self, ids) -> str:
        """The decoded text with every whitespace run erased: what
        ``span_is_canonical`` compares when the parser strips parameter values."""
        return re.sub(r"\s+", "", self._tokenizer.decode(list(ids), skip_special_tokens=False))

    @_locked
    def assistant_message(self, completion_ids) -> dict:
        parsed = self._parse(completion_ids)
        message = {"role": "assistant", "content": parsed.content or ""}
        if parsed.reasoning_content:
            message["reasoning_content"] = parsed.reasoning_content
        calls = [] if self._text else self._template_calls(completion_ids)
        if calls:
            message["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                      "function": {"name": name, "arguments": arguments}}
                                     for i, (name, arguments) in enumerate(calls)]
        return message


def build_renderer(checkpoint_dir: str, spec: HarnessSpec = BASH_VF_B2E4E81):
    """The pinned `renderers` renderer for this checkpoint's tokenizer, as verifiers' train
    client builds it (`create_renderer(load_tokenizer(model), Qwen38RendererConfig())`)."""
    if spec.renderer != "renderers:qwen38@0.1.11":
        raise ValueError(f"{spec.key} renders with {spec.renderer}, not the Qwen3.8 renderer")
    # The digest pins this package's sources, not the renderer's: another `renderers` release
    # renders or bridges other tokens (0.1.12 refuses to bridge a turn closed inside its reasoning).
    from importlib.metadata import version

    installed = version("renderers")
    if installed != RENDERERS_VERSION:
        raise ValueError(f"{spec.key} renders with renderers=={RENDERERS_VERSION}, not the installed {installed}")
    from renderers import create_renderer
    from renderers.base import load_tokenizer
    from renderers.configs import Qwen38RendererConfig

    return create_renderer(load_tokenizer(checkpoint_dir), Qwen38RendererConfig())


def load_turn_renderer(checkpoint_dir: str, tools=("bash", "edit"), *, spec: HarnessSpec = BASH_VF_B2E4E81,
                       variant: int = 0,
                       refuse: Callable[[BaseException], BaseException] | None = None) -> QwenTurnRenderer:
    return QwenTurnRenderer(build_renderer(checkpoint_dir, spec), tools=tools, spec=spec, variant=variant,
                            refuse=refuse)


__all__ = ["HIDDEN_PREFIX", "QwenTurnRenderer", "RENDERERS_VERSION", "UnparsableSpan", "build_renderer", "calls_with_ids",
           "load_turn_renderer", "pinned_tool_calls"]
