"""Battle specialist: code enforces invariants, the model picks among what is left.

Invariants (from the write-up):
- never pick an option with no uses left
- when the intention is to catch, never pick an attack that could knock the
  opponent out

The model then chooses among the remaining options. If only one remains, no
model call is made.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def allowed_options(battle: Mapping[str, Any], goal: str) -> tuple[list[dict], list[str]]:
    """Options the rules allow, and why each removed option was removed."""
    opponent = battle.get("opponent", {})
    hp = opponent.get("hp")
    allowed, removed = [], []
    for option in battle.get("options", []):
        if option.get("uses_left") is not None and option["uses_left"] <= 0:
            removed.append(f"{option['id']}: no uses left")
        elif goal == "catch" and option.get("kind") == "attack" and hp is not None and option.get("power", 0) >= hp:
            removed.append(f"{option['id']}: could knock out the opponent we are catching")
        elif goal == "catch" and option.get("kind") == "flee":
            removed.append(f"{option['id']}: fleeing gives up the catch")
        else:
            allowed.append(dict(option))
    return allowed, removed


def option_keys(option: Mapping[str, Any]) -> list[str] | None:
    """Keys the adapter says choose this option from the screen, if it gave them."""
    keys = option.get("keys")
    return list(keys) if isinstance(keys, list) and keys else None


def option_index(battle: Mapping[str, Any], option_id: str) -> int:
    ids = [option["id"] for option in battle.get("options", [])]
    return ids.index(option_id)


def choose_prompt(battle: Mapping[str, Any], allowed: Sequence[Mapping[str, Any]], goal: str, removed: Sequence[str]) -> str:
    lines = [
        f"Battle goal: {goal}",
        f"Opponent: {battle.get('opponent', {})}",
        "Options you may choose:",
        *[f"- {option}" for option in allowed],
    ]
    if removed:
        lines += ["Ruled out by the harness:", *[f"- {reason}" for reason in removed]]
    lines.append('Reply with JSON: {"why": "<one sentence>", "option": "<option id>"}')
    return "\n".join(lines)
