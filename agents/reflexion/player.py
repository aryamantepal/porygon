"""Reflexion: ReAct plus verbal reflections on failed trials (Shinn et al., 2023, arXiv:2303.11366).

Ported from the authors' ALFWorld runs (github.com/noahshinn/reflexion, MIT):
the reflection prompt is theirs verbatim, a reflection is written only after
a failed trial, at most the last three reflections are kept in context, and
they enter the prompt under "Your memory for the task below".

Differences from the paper, because the setting differs:
- Zero-shot unless `examples` is given. Theirs are ALFWorld-specific; ours are a
  file of example reflections (and ReAct's trajectories) in this game's format.
- Their early stop on a repeated action is left out: pressing the same
  button twice in a row is normal play here.
- The failure the reflection answers is the environment's verdict (which
  budget ran out), not a model's claim.
- With `action_interface="tools"` (see `agents/react/player.py`) a turn is one
  tool call. The reflection query shows each turn as Observation (which opens
  with how the previous tool ended), Thought, and Action, and adds a Result
  line after the last turn, whose outcome no later Observation shows.

Ablation: `reflections=False` runs the same player with no reflections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agents.examples import Examples
from agents.models import ChatModel, Message
from agents.protocol import Decision, Review, TrialOutcome
from agents.react import ReAct
from agents.reflexion.prompts import EXAMPLES, REFLECTION_PROMPT
from environments.protocol import Action, Observation


@dataclass(frozen=True)
class Reflection:
    id: str
    written_after_trial: int
    failure: str
    text: str


class Reflexion(ReAct):
    name = "reflexion"

    def __init__(
        self,
        model: ChatModel,
        history_turns: int = 10,
        reflections: bool = True,
        memory_size: int = 3,
        reflection_turns: int = 40,
        retry_illegal_action: bool = False,
        examples: Examples | None = None,
        action_interface: str = "buttons",
        specialists: bool = False,
    ) -> None:
        super().__init__(model, history_turns, retry_illegal_action, examples, action_interface, specialists)
        self._example_reflections = examples.reflections_text() if examples else ""
        self.reflections = reflections
        self.memory_size = memory_size
        self.reflection_turns = reflection_turns
        self.memory: list[Reflection] = []
        self._trial = -1
        self._in_context: list[Reflection] = []

    def mechanisms(self) -> dict[str, bool]:
        few_shot = bool(self._examples) or (self.reflections and bool(self._example_reflections))
        return {**super().mechanisms(), "reflections": self.reflections, "few_shot": few_shot}

    def start_trial(self, goal: str) -> None:
        super().start_trial(goal)
        self._trial += 1
        self._in_context = _last(self.memory, self.memory_size) if self.reflections else []

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        decision = super().decide(observation, legal_actions)
        trace = {**(decision.trace or {}), "reflections_in_context": [r.id for r in self._in_context]}
        return Decision(decision.action, trace, decision.calls)

    def end_trial(self, outcome: TrialOutcome) -> Review:
        if outcome.success or not self.reflections:
            return Review(trace={"reflection": None, "reflections_enabled": self.reflections})
        call = self.model.chat([Message("user", self._reflection_query(outcome))], think=False)
        if call.error or not call.output.strip():
            return Review(trace={"reflection": None, "error": call.error or "empty reflection"}, calls=(call,))
        reflection = Reflection(
            id=f"r{len(self.memory)}",
            written_after_trial=self._trial,
            failure=f"{outcome.reason} after {outcome.steps} steps",
            text=call.output.strip(),
        )
        self.memory.append(reflection)
        return Review(trace={"reflection": reflection.__dict__}, calls=(call,))

    def _system(self) -> str:
        system = super()._system()
        if not self._in_context:
            return system
        lines = [system, "", "Your memory for the task below:"]
        for index, reflection in enumerate(self._in_context):
            lines += [f"Trial {index}:", reflection.text]
        return "\n".join(lines)

    def _reflection_query(self, outcome: TrialOutcome) -> str:
        turns = _last(self._turns, self.reflection_turns)
        first = len(self._turns) - len(turns)
        lines = [REFLECTION_PROMPT, ""]
        if self._example_reflections:
            lines += [EXAMPLES, "", self._example_reflections, ""]
        lines += ["Here is the task:", f"Goal: {self._goal}"]
        if first:
            lines.append(f"({first} earlier turns omitted)")
        for index, turn in enumerate(turns, start=first + 1):
            lines += [f"Observation {index}: {turn.observation}", f"Thought {index}: {turn.thought}", f"Action {index}: {turn.action}"]
            lines += self._turn_tail(index, turn, last=index == len(self._turns))
        lines.append(f"STATUS: FAIL ({outcome.reason} after {outcome.steps} steps)")
        past = _last(self.memory, self.memory_size)
        if past:
            lines += ["", "Plans from past attempts:"]
            lines += [f"Trial #{index}: {reflection.text}" for index, reflection in enumerate(past)]
        lines += ["", "New plan:"]
        return "\n".join(lines)


def _last(items: list, n: int) -> list:
    """The last n items; xs[-0:] would be all of them, which breaks a 0 ablation."""
    return items[max(len(items) - n, 0) :]

