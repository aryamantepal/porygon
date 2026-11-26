"""The interface every player implements.

A player sees a goal, the current observation, and the legal actions, and
returns one decision. It never imports an emulator, reads game memory, or
branches on which game is running.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from agents.models import ModelCall
from environments.protocol import Action, Observation


@dataclass(frozen=True)
class Decision:
    """One decision.

    `action` is None when the model never produced a legal action; the runner
    counts that and presses nothing. `trace` is the architecture's reasoning,
    verbatim (None for a player that has none). `calls` are every model call
    made for this decision.
    """

    action: Action | None
    trace: Mapping[str, Any] | None
    calls: Sequence[ModelCall] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrialOutcome:
    """What the runner tells a player when a trial ends."""

    success: bool
    reason: str  # "goal_met", or which budget ran out
    steps: int


class Player(Protocol):
    name: str

    def mechanisms(self) -> Mapping[str, bool]:
        """Which of this architecture's mechanisms are on, for the run config."""
        ...

    def start_trial(self, goal: str) -> None: ...

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision: ...

    def end_trial(self, outcome: TrialOutcome) -> Sequence[ModelCall]:
        """Called after each trial. Returns any model calls made here (Reflexion)."""
        ...
