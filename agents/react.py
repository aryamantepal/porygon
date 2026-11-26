"""ReAct: reason, act, observe, repeat (Yao et al., 2022, arXiv:2210.03629).

Ported from the authors' HotpotQA loop (github.com/ysymyth/ReAct, MIT):
numbered Thought / Action / Observation turns, generation stopped at the next
"Observation", and when the reply carries no action, a second call that asks
for the action alone after the thought.

Differences from the paper, because the setting differs:
- Zero-shot. The paper's few-shot trajectories are for QA and text games; there
  are no in-domain game trajectories to show yet.
- History is the last `history_turns` turns, not the whole trajectory, so long
  games fit in context.
- In visual mode only the current screen is attached; earlier observations in
  the history carry whatever text the mode allows.
- The model's hidden thinking is off: the Thought line is the reasoning, and it
  is logged verbatim.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from agents.common import describe_actions, parse_action, render_observation
from agents.models import ChatModel, Message
from agents.protocol import Decision, Review, TrialOutcome
from environments.protocol import Action, Observation

SYSTEM = (
    "You are playing a video game. Solve the goal by interleaving Thought and Action "
    "steps. A Thought reasons about the current situation and what to do next. An "
    "Action is exactly one of the legal actions, and presses that button once. After "
    "each Action you will see the next Observation."
)

_ACTION = re.compile(r"Action\s*\d*\s*:\s*(.*)", re.IGNORECASE)


@dataclass
class Turn:
    observation: str
    thought: str = ""
    action: str = ""


class ReAct:
    name = "react"

    def __init__(self, model: ChatModel, history_turns: int = 10) -> None:
        self.model = model
        self.history_turns = history_turns
        self._goal = ""
        self._turns: list[Turn] = []

    def mechanisms(self) -> dict[str, bool]:
        return {"reasoning": True, "history": True, "thinking": False}

    def start_trial(self, goal: str) -> None:
        self._goal = goal
        self._turns = []

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        snapshot, images = render_observation(observation)
        self._turns.append(Turn(observation=_history_text(observation)))
        n = len(self._turns)
        prompt = self._prompt(snapshot, legal_actions)

        call = self.model.chat(
            [Message("system", self._system()), Message("user", prompt + f"Thought {n}:", images)],
            think=False,
            stop=[f"\nObservation {n}:", f"\nObservation {n + 1}:"],
        )
        calls = [call]
        thought, action_text = _split(call.output)
        action = parse_action(action_text, legal_actions, require_label=False) if action_text is not None else None

        if action_text is None:
            # The reference loop's recovery: keep the first line as the thought
            # and ask for the action on its own.
            thought = call.output.strip().split("\n")[0]
            retry = self.model.chat(
                [
                    Message("system", self._system()),
                    Message("user", prompt + f"Thought {n}: {thought}\nAction {n}:", images),
                ],
                think=False,
                stop=["\n"],
            )
            calls.append(retry)
            action_text = retry.output.strip()
            action = parse_action(action_text, legal_actions, require_label=False)

        turn = self._turns[-1]
        turn.thought = thought
        turn.action = action.key if action else f"(invalid: {action_text})"
        return Decision(
            action=action,
            trace={"thought": thought, "turn": n, "action_text": action_text, "action_retry": len(calls) > 1},
            calls=tuple(calls),
        )

    def end_trial(self, outcome: TrialOutcome) -> Review:
        return Review()

    def _system(self) -> str:
        return SYSTEM

    def _prompt(self, snapshot: str, legal_actions: Sequence[Action]) -> str:
        n = len(self._turns)
        lines = [
            f"Goal: {self._goal}",
            f"Legal actions: {describe_actions(legal_actions)}",
            f"Reply in exactly this form:\nThought {n}: <your reasoning>\nAction {n}: <one legal action>",
            "",
        ]
        first = max(0, n - 1 - self.history_turns)
        if first > 0:
            lines.append(f"({first} earlier turns omitted)")
        for index in range(first, n - 1):
            turn = self._turns[index]
            lines += [
                f"Observation {index + 1}: {turn.observation}",
                f"Thought {index + 1}: {turn.thought}",
                f"Action {index + 1}: {turn.action}",
            ]
        lines.append(f"Observation {n}: {snapshot}")
        return "\n".join(lines) + "\n"


def _split(output: str) -> tuple[str, str | None]:
    """Split a reply into (thought, action text); action text is None if absent."""
    match = _ACTION.search(output)
    if not match:
        return output.strip(), None
    thought = output[: match.start()].strip()
    thought = re.sub(r"^Thought\s*\d*\s*:\s*", "", thought, flags=re.IGNORECASE)
    return thought, match.group(1).strip().split("\n")[0]


def _history_text(observation: Observation) -> str:
    snapshot, _ = render_observation(observation)
    if observation.state is None:
        return "(screen only; not kept in history)"
    return snapshot.replace("Screen: see the attached image.\n", "")
