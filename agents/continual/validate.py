"""Strict checks on every value a model hands the harness, from a tool call or from a Refiner edit.

A value that fails raises `Invalid` with a reason that is logged and shown back
to the model; nothing is repaired silently, because a harness that quietly
accepts a malformed call is how the paper's 1,003-turn loop happened.
"""

from __future__ import annotations

import re
from typing import Any

from agents.continual.harness import EFFECTIVENESS

MAX_EDITS = 5  # per Refiner pass, as the memory curator's "max 3-5 changes"
MAX_GUIDANCE = 1200
MAX_TITLE = 80
MAX_CONTENT = 400
MAX_DESCRIPTION = 300
MAX_CODE = 1500
MAX_PATH = 60

NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Invalid(Exception):
    """A value that failed validation; the message is the reason that gets logged."""


def only_fields(value: Any, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Invalid("expected an object")
    unknown = set(value) - allowed
    if unknown:
        raise Invalid(f"unknown fields {sorted(unknown)}; allowed: {sorted(allowed)}")
    missing = required - set(value)
    if missing:
        raise Invalid(f"missing fields {sorted(missing)}")
    return value


def text_field(obj: dict, name: str, limit: int) -> str:
    value = obj[name]
    if not isinstance(value, str) or not value.strip():
        raise Invalid(f'"{name}" must be non-empty text')
    if len(value) > limit:
        raise Invalid(f'"{name}" is {len(value)} characters; the limit is {limit}')
    return value.strip()


def importance_field(obj: dict, default: int = 3) -> int:
    value = obj.get("importance", default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise Invalid(f'"importance" must be a whole number from 1 to 5, not {value!r}')
    return value


def effectiveness_field(obj: dict, default: str = "medium") -> str:
    value = obj.get("effectiveness", default)
    if value not in EFFECTIVENESS:
        raise Invalid(f'"effectiveness" must be one of {list(EFFECTIVENESS)}, not {value!r}')
    return value
