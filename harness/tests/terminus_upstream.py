"""Harbor 06d0e86a's Terminus 2 pieces, run from the vendored files (`upstream/terminus/`, Apache-2.0): the JSON
parser module as is, and `terminus_2.py`'s message functions cut out of its source (it imports all of harbor)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

UPSTREAM = Path(__file__).resolve().parent / "upstream" / "terminus"


def source(name: str) -> str:
    return (UPSTREAM / name).read_text(encoding="utf-8")


def parser():
    spec = importlib.util.spec_from_file_location("terminus_json_plain_parser", UPSTREAM / "terminus_json_plain_parser.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TerminusJSONPlainParser()


def method(name: str, relative: str = "terminus_2.py") -> str:
    text = source(relative)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"no {name} in {relative}")


def agent():
    """An object carrying the Terminus2 methods the harness reproduces."""
    import textwrap

    namespace: dict = {}
    for name in ("_limit_output_length", "_get_completion_confirmation_message", "_get_error_response_type"):
        exec(textwrap.dedent(method(name)), namespace)
    fake = SimpleNamespace(options=SimpleNamespace(parser_name="json"))
    for name in ("_limit_output_length", "_get_completion_confirmation_message", "_get_error_response_type"):
        setattr(fake, name, namespace[name].__get__(fake))
    return fake


def rendered() -> dict[str, str]:
    """The texts the parity test compares (`upstream/terminus/rendered.json` holds them)."""
    up = agent()
    return {
        "instance_marker": source("terminus-json-plain.txt").format(instruction="<<TASK>>",
                                                                    terminal_state="Current Terminal Screen:\n"),
        "confirmation": up._get_completion_confirmation_message("New Terminal Output:\nroot@box:/app# "),
        "limited": up._limit_output_length("é" * 6000 + "x" * 3),
    }
