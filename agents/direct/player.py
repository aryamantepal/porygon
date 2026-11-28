"""Direct policy: the current snapshot and the legal actions in, one action out.

No earlier turns, reflections, skills, curriculum, or reasoning trace. The
model's thinking is turned off so its output is the decision. This is the
floor the other architectures are measured against.

With `examples`, a few (snapshot, action) pairs from another task come first,
in the same request. They carry no thoughts: this player has no trace.
"""

from __future__ import annotations

from collections.abc import Sequence

from agents.common import describe_actions, parse_action, render_observation
from agents.direct.prompts import EXAMPLES, SYSTEM
from agents.examples import Examples
from agents.models import ChatModel, Message
from agents.protocol import Decision, Review, TrialOutcome
from environments.protocol import Action, Observation


class DirectPolicy:
    name = "direct"

    def __init__(self, model: ChatModel, examples: Examples | None = None) -> None:
        self.model = model
        self._examples = examples.pairs_text() if examples else ""
        self._goal = ""

    def mechanisms(self) -> dict[str, bool]:
        return {"history": False, "thinking": False, "few_shot": bool(self._examples)}

    def start_trial(self, goal: str) -> None:
        self._goal = goal

    def next_goal(self, goal: str) -> None:
        self._goal = goal

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        snapshot, images = render_observation(observation)
        shots = f"{EXAMPLES}\n\n{self._examples}\n\nNow your task.\n" if self._examples else ""
        prompt = (
            f"{shots}Goal: {self._goal}\n{snapshot}\n"
            f"Legal actions: {describe_actions(legal_actions)}\n"
            "Reply with exactly one action name and nothing else."
        )
        call = self.model.chat([Message("system", SYSTEM), Message("user", prompt, images)], think=False)
        action = parse_action(call.output, legal_actions, require_label=False)
        return Decision(action=action, trace=None, calls=(call,))

    def end_trial(self, outcome: TrialOutcome) -> Review:
        return Review()
