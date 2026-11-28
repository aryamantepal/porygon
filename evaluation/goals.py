"""A goal is data: text the player reads, and a progress check the environment decides."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from environments.protocol import Progress


@dataclass(frozen=True)
class Goal:
    task_id: str
    text: str
    success: Mapping[str, Any]  # every key must equal the same key in get_progress()

    def is_met(self, progress: Progress) -> bool:
        return all(key in progress and progress[key] == value for key, value in self.success.items())


@dataclass(frozen=True)
class Level:
    """One rung of a chain: a goal, and optionally how many steps this level may use."""

    goal: Goal
    max_steps: int | None = None  # steps spent in this level; None shares the trial budget
    note: str = ""  # free text kept in the run config (e.g. which ids are unconfirmed)


@dataclass(frozen=True)
class Chain:
    """Goals met in order, in one trial. Each level is judged from `get_progress()` like any goal.

    Later levels are meant to reuse what earlier ones taught, so the player keeps
    its state when a level is met (see `next_goal` in agents/protocol.py).
    """

    task_id: str
    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("a chain needs at least one level")

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> Chain:
        levels = tuple(
            Level(Goal(item["task_id"], item["text"], item["success"]), item.get("max_steps"), item.get("note", ""))
            for item in spec["chain"]
        )
        return cls(spec["task_id"], levels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain": [
                {"task_id": l.goal.task_id, "text": l.goal.text, "success": dict(l.goal.success), "max_steps": l.max_steps, "note": l.note}
                for l in self.levels
            ]
        }


def goal_from_dict(spec: Mapping[str, Any]) -> Goal | Chain:
    """A config's `goal`: a plain goal, or `{"chain": [...], "task_id": ...}`."""
    return Chain.from_dict(spec) if "chain" in spec else Goal(**spec)
