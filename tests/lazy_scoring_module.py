"""Imported by a reward only once its scoring process is ready (after the audit hook):
an import made then still loads its file, `dataclass` and `namedtuple` at module level
included."""

from collections import namedtuple
from dataclasses import dataclass

Pair = namedtuple("Pair", "left right")


@dataclass(frozen=True)
class Verdict:
    score: float


def verdict(text: str) -> Verdict:
    pair = Pair(*text.split(maxsplit=1))
    return Verdict(1.0 if pair.left == "42" else 0.0)


class Annotated:
    """A postponed (string) annotation, resolved at scoring time as pydantic does."""

    score: "float"
