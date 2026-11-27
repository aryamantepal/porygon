"""The stuck ladder: notice going in circles, then escalate one rung at a time.

"Stuck" is measured by what changes in the game, not by time: each planner
decision records a signature of the role snapshot (map, position, context,
and the places on screen). A new signature resets the ladder. Seeing the same
signature again climbs it.

Rungs (from the write-up; thresholds are ours, the write-up gives none):
    0  normal
    1  remember: failed attempts at this spot are shown to the planner
    2  nudge: the planner is told it is going in circles
    3  deep think: the planner call runs with the model's thinking on
    4  ask for help: a help request is logged for the operator; the ladder
       then resets so the run continues
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

RUNGS = ("normal", "remember", "nudge", "deep_think", "ask_for_help")


def signature(state: Mapping[str, Any] | None) -> str:
    if not state:
        return ""
    keys = ("map", "position", "context")
    places = sorted(place.get("id", "") for place in state.get("places", []))
    return json.dumps([state.get(key) for key in keys] + [places])


@dataclass
class Attempt:
    signature: str
    tool: str
    args: Mapping[str, Any]
    result: str


@dataclass
class StuckLadder:
    repeats_per_rung: int = 3
    seen: Counter = field(default_factory=Counter)
    rung: int = 0
    failures: list[Attempt] = field(default_factory=list)

    def observe(self, state: Mapping[str, Any] | None) -> int:
        """Record a planner decision at this state; returns the rung to apply."""
        sig = signature(state)
        self.seen[sig] += 1
        if self.seen[sig] == 1:
            self.rung = 0
        else:
            self.rung = min(len(RUNGS) - 1, (self.seen[sig] - 1) // self.repeats_per_rung + 1)
        return self.rung

    def failed_here(self, state: Mapping[str, Any] | None) -> list[Attempt]:
        sig = signature(state)
        return [attempt for attempt in self.failures if attempt.signature == sig]

    def is_repeat_failure(self, state, tool: str, args: Mapping[str, Any]) -> bool:
        return any(a.tool == tool and a.args == args for a in self.failed_here(state))

    def record_failure(self, state, tool: str, args: Mapping[str, Any], result: str) -> None:
        self.failures.append(Attempt(signature(state), tool, dict(args), result))
        self.failures = self.failures[-50:]

    def help_requested(self, state) -> None:
        """After asking for help, start counting this spot again from zero."""
        self.seen[signature(state)] = 0
        self.rung = 0
