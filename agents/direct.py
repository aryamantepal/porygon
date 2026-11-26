"""Direct policy: the current snapshot and the legal actions in, one action out.

No earlier turns, reflections, skills, curriculum, or reasoning trace. The
model's thinking is turned off so its output is the decision. This is the
floor the other architectures are measured against.
"""

from __future__ import annotations

from collections.abc import Sequence

from agents.common import describe_actions, parse_action, render_observation
from agents.models import ChatModel, Message, ModelCall
from agents.protocol import Decision, TrialOutcome
from environments.protocol import Action, Observation

SYSTEM = (
    "You are playing a video game. Each turn you see the current game snapshot "
    "and choose exactly one button to press next."
)


class DirectPolicy:
    name = "direct"

    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self._goal = ""

    def mechanisms(self) -> dict[str, bool]:
        return {"history": False, "thinking": False}

    def start_trial(self, goal: str) -> None:
        self._goal = goal

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        snapshot, images = render_observation(observation)
        prompt = (
            f"Goal: {self._goal}\n{snapshot}\n"
            f"Legal actions: {describe_actions(legal_actions)}\n"
            "Reply with exactly one action name and nothing else."
        )
        call = self.model.chat([Message("system", SYSTEM), Message("user", prompt, images)], think=False)
        action = parse_action(call.output, legal_actions, require_label=False)
        return Decision(action=action, trace=None, calls=(call,))

    def end_trial(self, outcome: TrialOutcome) -> Sequence[ModelCall]:
        return ()
