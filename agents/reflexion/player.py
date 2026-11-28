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

Two more switches, both off by default (with them off the prompts and the Review trace
are exactly what they were), and both use only the run's own earlier trials:

- `insights` (ExpeL, Zhao et al., AAAI 2024, arXiv:2308.10144; github.com/LeapLabTHU/ExpeL,
  Apache-2.0): after every trial, a success as well as a failure, the model compares
  the trial with a stored trial of the opposite outcome (or, for a success with no failure
  yet, with the successes so far) and replies with ADD / EDIT / AGREE / REMOVE operations
  on a capped list of insights. Votes and removal at zero follow ExpeL
  (`agents/reflexion/insights.py`). Later trials read the list in the system prompt,
  with or without reflections. Success or failure is the runner's `TrialOutcome`.
- `retrieve_trajectories`: at a trial's first decision, the `retrieve_k` most similar
  earlier successful trials go in the system prompt as worked examples, rendered by
  `agents/examples.py` (`agents/reflexion/trajectories.py` says how similarity works).

Both cost context, which a small model has little of: the cap on insights, the turns
shown per trial (`insight_turns`, `retrieved_turns`), and `retrieve_k` bound it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from agents.examples import Examples, render_trajectories
from agents.models import ChatModel, Message, ModelCall
from agents.protocol import Decision, Review, TrialOutcome
from agents.react import ReAct
from agents.react.player import Turn, history_text
from agents.reflexion import prompts
from agents.reflexion.insights import Insight, InsightSet, parse_ops
from agents.reflexion.prompts import EXAMPLES, REFLECTION_PROMPT
from agents.reflexion.trajectories import Trajectory, TrajectoryStore, as_example
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
        insights: bool = False,
        max_insights: int = 10,
        insight_turns: int = 20,
        retrieve_trajectories: bool = False,
        retrieve_k: int = 1,
        retrieved_turns: int = 8,
        trajectories_kept: int = 10,
    ) -> None:
        super().__init__(model, history_turns, retry_illegal_action, examples, action_interface, specialists)
        self._example_reflections = examples.reflections_text() if examples else ""
        self.reflections = reflections
        self.memory_size = memory_size
        self.reflection_turns = reflection_turns
        self.memory: list[Reflection] = []
        self._trial = -1
        self._in_context: list[Reflection] = []
        self.insights = insights
        self.insight_turns = insight_turns
        self.insight_set = InsightSet(limit=max_insights)
        self.retrieve_trajectories = retrieve_trajectories
        self.retrieve_k = retrieve_k
        self.retrieved_turns = retrieved_turns
        self.store = TrajectoryStore(capacity=trajectories_kept)
        self._shown_insights: list[Insight] = []  # the insights this trial's prompts show, fixed at its start
        self._retrieval_done = False
        self._retrieved: list[tuple[Trajectory, float]] = []
        self._retrieved_text = ""

    def mechanisms(self) -> dict[str, bool]:
        few_shot = bool(self._examples) or (self.reflections and bool(self._example_reflections))
        return {
            **super().mechanisms(),
            "reflections": self.reflections,
            "few_shot": few_shot,
            "insights": self.insights,
            "retrieve_trajectories": self.retrieve_trajectories,
        }

    def start_trial(self, goal: str) -> None:
        super().start_trial(goal)
        self._trial += 1
        self._in_context = _last(self.memory, self.memory_size) if self.reflections else []
        self._shown_insights = list(self.insight_set.items) if self.insights else []
        self._retrieval_done = False
        self._retrieved, self._retrieved_text = [], ""

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        if self.retrieve_trajectories and not self._retrieval_done:
            self._retrieve(history_text(observation))
        decision = super().decide(observation, legal_actions)
        trace = {**(decision.trace or {}), "reflections_in_context": [r.id for r in self._in_context]}
        if self.insights:
            trace["insights_in_context"] = [i.id for i in self._shown_insights]
        if self.retrieve_trajectories:
            trace["retrieved_trajectories"] = [t.event(similarity) for t, similarity in self._retrieved]
        return Decision(decision.action, trace, decision.calls)

    def end_trial(self, outcome: TrialOutcome) -> Review:
        review = self._reflect(outcome)
        if not (self.insights or self.retrieve_trajectories):
            return review
        first = self._turns[0].observation if self._turns else ""
        current = Trajectory(self._trial, self._goal, tuple(replace(t) for t in self._turns), outcome.success, outcome.reason, outcome.steps, first)
        trace, calls = {}, ()
        if self.insights:
            trace, calls = self._extract_insights(current)
        if current.turns:
            self.store.add(current)
        if not self.insights:
            return review
        return Review(trace={**(review.trace or {}), "insights": trace}, calls=(*review.calls, *calls))

    def _reflect(self, outcome: TrialOutcome) -> Review:
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
        lines = [super()._system()]
        if self._shown_insights:
            lines += ["", prompts.INSIGHTS_HEADER, *(f"{n}. {i.text}" for n, i in enumerate(self._shown_insights, start=1))]
        if self._in_context:
            lines += ["", "Your memory for the task below:"]
            for index, reflection in enumerate(self._in_context):
                lines += [f"Trial {index}:", reflection.text]
        if self._retrieved_text:
            lines += ["", prompts.RETRIEVED_HEADER, "", self._retrieved_text]
        return "\n".join(lines)

    # Retrieval of earlier successful trials

    def _retrieve(self, first_observation: str) -> None:
        self._retrieval_done = True
        self._retrieved = self.store.rank(self._goal, first_observation, self.retrieve_k)
        if self._retrieved:
            examples = [as_example(t, self.retrieved_turns, self.tools) for t, _ in self._retrieved]
            self._retrieved_text = render_trajectories(examples, self.retrieved_turns)

    # Insights

    def _extract_insights(self, current: Trajectory) -> tuple[dict, tuple[ModelCall, ...]]:
        """ExpeL's insight step after a trial: compare, ask for operations, apply them. Returns the trace and the model calls."""
        mode, pair = self.store.pick_pair(current) if current.turns else ("none", [])
        trace: dict = {"mode": mode, "compared_trials": [t.event() for t in pair], "insights_before": [i.event() for i in self.insight_set.items]}
        if mode == "none":
            trace["skipped"] = "no turns" if not current.turns else "a failed trial with no successful trial to compare it with"
            return trace, ()
        call = self.model.chat(self._insight_messages(mode, pair), think=False)
        trace["raw_reply"] = call.output
        if call.error or not call.output.strip():
            trace["error"] = call.error or "empty reply"
            return trace, (call,)
        ops, rejected = parse_ops(call.output)
        trace["parsed_ops"] = [{"op": o.kind, "number": o.number, "text": o.text} for o in ops]
        trace["rejected_lines"] = [r.__dict__ for r in rejected]
        trace.update(self.insight_set.apply(ops, current.trial))
        return trace, (call,)

    def _insight_messages(self, mode: str, pair: list[Trajectory]) -> list[Message]:
        rules = self.insight_set.numbered() or prompts.INSIGHT_NONE_YET
        if mode == "compare":
            success, failure = pair
            instruction = prompts.INSIGHT_INSTRUCTION_COMPARE.format(reason=failure.reason)
            body = prompts.INSIGHT_COMPARE.format(task=failure.goal, success=self._render_trial(success), failure=self._render_trial(failure), rules=rules)
        else:
            instruction = prompts.INSIGHT_INSTRUCTION_SUCCESS
            body = prompts.INSIGHT_SUCCESS.format(success="\n\n".join(f"{t.goal}\n{self._render_trial(t)}" for t in pair), rules=rules)
        suffix = prompts.INSIGHT_SUFFIX_FULL if self.insight_set.full else prompts.INSIGHT_SUFFIX_NOT_FULL
        system = prompts.INSIGHT_SYSTEM.format(ai_name=prompts.INSIGHT_AI_NAME, instruction=instruction)
        return [Message("system", system), Message("user", body + prompts.INSIGHT_FORMAT.format(max_ops=self.insight_set.max_ops) + suffix)]

    def _render_trial(self, trajectory: Trajectory) -> str:
        return "\n".join([*self._turn_lines(list(trajectory.turns), self.insight_turns), trajectory.outcome_line()])

    def _reflection_query(self, outcome: TrialOutcome) -> str:
        lines = [REFLECTION_PROMPT, ""]
        if self._example_reflections:
            lines += [EXAMPLES, "", self._example_reflections, ""]
        lines += ["Here is the task:", f"Goal: {self._goal}"]
        lines += self._turn_lines(self._turns, self.reflection_turns)
        lines.append(f"STATUS: FAIL ({outcome.reason} after {outcome.steps} steps)")
        past = _last(self.memory, self.memory_size)
        if past:
            lines += ["", "Plans from past attempts:"]
            lines += [f"Trial #{index}: {reflection.text}" for index, reflection in enumerate(past)]
        lines += ["", "New plan:"]
        return "\n".join(lines)

    def _turn_lines(self, all_turns: list[Turn], limit: int) -> list[str]:
        """The last `limit` turns as Observation, Thought, and Action lines, with how the last tool ended under the tools interface."""
        turns = _last(all_turns, limit)
        first = len(all_turns) - len(turns)
        lines = [f"({first} earlier turns omitted)"] if first else []
        for index, turn in enumerate(turns, start=first + 1):
            lines += [f"Observation {index}: {turn.observation}", f"Thought {index}: {turn.thought}", f"Action {index}: {turn.action}"]
            lines += self._turn_tail(index, turn, last=index == len(all_turns))
        return lines


def _last(items: list, n: int) -> list:
    """The last n items; xs[-0:] would be all of them, which breaks a 0 ablation."""
    return items[max(len(items) - n, 0) :]

