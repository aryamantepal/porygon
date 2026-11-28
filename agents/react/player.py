"""ReAct: reason, act, observe, repeat (Yao et al., 2022, arXiv:2210.03629).

Ported from the authors' HotpotQA loop (github.com/ysymyth/ReAct, MIT):
numbered Thought / Action / Observation turns, generation stopped at the next
"Observation", and when the reply carries no action, a second call that asks
for the action alone after the thought.

Differences from the paper, because the setting differs:
- Zero-shot unless `examples` is given. The paper's few-shot trajectories are for
  QA and text games; ours are a file of worked trajectories in this game's
  snapshot format, shown in the system prompt before the task.
- History is the last `history_turns` turns, not the whole trajectory, so long
  games fit in context.
- In visual mode only the current screen is attached; earlier observations in
  the history carry whatever text the mode allows.
- `retry_illegal_action` (off by default, as in the reference loop, which only
  retried a missing action): when on, the second call also happens when the
  action is not a legal one, and it names the legal actions.
- The model's hidden thinking is off: the Thought line is the reasoning, and it
  is logged verbatim.

Two switches change the action interface (both off by default; with them off
the prompts are exactly the buttons-only ones above):
- `action_interface="tools"`: an Action is a tool call (`go_to door_0`,
  `go_to [4, 2]`, `talk_to person`, `press A x3`; grammar in
  `agents/harness/calls.py`), not one button. One ReAct turn is one tool call.
  While a tool runs, `decide` returns its presses with no model call, and when
  it ends the next Observation starts with a line saying how it went. Tools
  need the role snapshot's grid and position; without them only `press` is
  offered and the trace says so. The tools are the Crystal harness's
  navigator (`agents/harness/tools.py`), so every press is still one counted,
  logged environment step. Few-shot examples are shown unchanged (their
  actions are buttons); the system prompt says so.
- `specialists=True` (tools only): code handles busy, dialogue, and battle
  before the model is asked (`agents/harness/specialists.py`). The dialogue
  text and the battle choice are put at the start of the next Observation, so
  the model still reads what the specialists passed through. Off, the model
  handles those contexts itself with `press`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agents.common import describe_actions, parse_action, render_observation
from agents.examples import Examples
from agents.harness import tools
from agents.harness.calls import TOOL_HELP, ToolCall, ToolCallError, check_interface, parse_tool_call
from agents.harness.specialists import Handled, Specialists
from agents.models import ChatModel, Message, ModelCall
from agents.protocol import Decision, Review, TrialOutcome
from agents.react.prompts import EXAMPLES, EXAMPLES_TOOLS, SYSTEM, SYSTEM_TOOLS
from environments.protocol import Action, Observation, Press

_ACTION = re.compile(r"^\s*Action\s*\d*\s*:\s*(.*)", re.IGNORECASE | re.MULTILINE)  # a line of its own, not "the best action: ..." mid-thought

MAX_NOTES = 6  # lines from code (tool results, dialogue text) kept for the next Observation
MAX_DECIDE_LOOPS = 32


@dataclass
class Turn:
    observation: str
    thought: str = ""
    action: str = ""
    result: str = ""  # tools interface: how the tool this turn named ended, once it has


@dataclass
class Offer:
    """What the tools interface offers on one observation."""

    names: list[str]
    place_ids: list[str] | None  # None: place names are not checked (no map to check against)
    places: str  # the places, as shown in the prompt
    keys: str  # the buttons `press` may use
    reason: str = ""  # why only `press` is offered, when it is


class ReAct:
    name = "react"

    def __init__(
        self,
        model: ChatModel,
        history_turns: int = 10,
        retry_illegal_action: bool = False,
        examples: Examples | None = None,
        action_interface: str = "buttons",
        specialists: bool = False,
    ) -> None:
        self.model = model
        self.history_turns = history_turns
        self.retry_illegal_action = retry_illegal_action
        self.tools = check_interface(action_interface, specialists)
        self.specialists = specialists
        self._specialists = Specialists(model)
        self._examples = examples.trajectories_text() if examples else ""
        self._goal = ""
        self._turns: list[Turn] = []
        self._reset_tools()

    def mechanisms(self) -> dict[str, bool]:
        return {
            "reasoning": True,
            "history": True,
            "thinking": False,
            "retry_illegal_action": self.retry_illegal_action,
            "few_shot": bool(self._examples),
            "tools": self.tools,
            "specialists": self.specialists,
        }

    def start_trial(self, goal: str) -> None:
        self._goal = goal
        self._turns = []
        self._reset_tools()

    def next_goal(self, goal: str) -> None:
        """A chain's next level: the history and notes stay; a tool still running for the old goal is dropped."""
        self._goal = goal
        self._run = self._run_call = self._run_turn = None

    def _reset_tools(self) -> None:
        self._run: tools.ToolRun | None = None
        self._run_call: ToolCall | None = None
        self._run_turn: Turn | None = None
        self._notes: list[str] = []
        self._specialists.reset()

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        if self.tools:
            return self._decide_tools(observation, legal_actions)
        snapshot, images = render_observation(observation)
        self._turns.append(Turn(observation=history_text(observation)))
        n = len(self._turns)
        thought, action_text, action, calls = self._ask(
            snapshot,
            images,
            [
                f"Goal: {self._goal}",
                f"Legal actions: {describe_actions(legal_actions)}",
                f"Reply in exactly this form:\nThought {n}: <your reasoning>\nAction {n}: <one legal action>",
                "",
            ],
            lambda text: parse_action(text, legal_actions, require_label=False),
            lambda text: f"({text!r} is not a legal action. Use exactly one of: {describe_actions(legal_actions)}.)",
        )
        turn = self._turns[-1]
        turn.thought = thought
        turn.action = action.key if action else f"(invalid: {action_text})"
        return Decision(
            action=action,
            trace={"thought": thought, "turn": n, "action_text": action_text, "action_retry": len(calls) > 1},
            calls=tuple(calls),
        )

    def _ask(self, snapshot, images, header, parse, illegal_hint):
        """The model's Thought and Action for the turn just added, with the reference loop's retry.

        `parse` turns action text into the interface's action (None if unusable);
        `illegal_hint` is the line naming what is usable, for `retry_illegal_action`.
        """
        n = len(self._turns)
        prompt = self._prompt(snapshot, header)
        call = self.model.chat(
            [Message("system", self._system()), Message("user", prompt + f"Thought {n}:", images)],
            think=False,
            stop=[f"\nObservation {n}:", f"\nObservation {n + 1}:"],
        )
        calls = [call]
        thought, action_text = _split(call.output)
        action = parse(action_text) if action_text is not None else None

        if action is None and (action_text is None or self.retry_illegal_action):
            # The reference loop's recovery: keep the thought and ask for the
            # action on its own. With `retry_illegal_action` it also retries an
            # action that is not legal ("RUN", "INTERACT with the person"),
            # naming the legal ones.
            if action_text is None:
                thought = call.output.strip().split("\n")[0]
                ask = f"Thought {n}: {thought}\nAction {n}:"
            else:
                ask = f"Thought {n}: {thought}\n{illegal_hint(action_text)}\nAction {n}:"
            retry = self.model.chat(
                [Message("system", self._system()), Message("user", prompt + ask, images)],
                think=False,
                stop=["\n"],
            )
            calls.append(retry)
            action_text = retry.output.strip()
            action = parse(action_text)
        return thought, action_text, action, calls

    def end_trial(self, outcome: TrialOutcome) -> Review:
        return Review()

    def _system(self) -> str:
        if self.tools:
            return f"{SYSTEM_TOOLS}\n\n{EXAMPLES_TOOLS}\n\n{self._examples}" if self._examples else SYSTEM_TOOLS
        return f"{SYSTEM}\n\n{EXAMPLES}\n\n{self._examples}" if self._examples else SYSTEM

    def _prompt(self, snapshot: str, header: list[str]) -> str:
        n = len(self._turns)
        lines = list(header)
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

    # Tools interface

    @property
    def _interrupt_on(self) -> tuple[str, ...]:
        return ("battle",) if self.specialists else ()

    def _decide_tools(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        """One decision under the tools interface.

        A running tool or specialist returns its next press with no model call.
        When the run has ended, specialists get the context first (if on), and
        otherwise the model names the next tool call.
        """
        state = observation.state
        keys = tools.controls(state, {action.key for action in legal_actions})
        calls: list[ModelCall] = []
        events: list[dict[str, Any]] = []
        info: dict[str, Any] = {}
        for _ in range(MAX_DECIDE_LOOPS):
            if self._run is not None:
                step = self._run.advance(observation, self._interrupt_on if self._run.chosen else ())
                if isinstance(step, Press):
                    return Decision(step, self._tool_trace(info, events), tuple(calls))
                idle = self._run.chosen and self._run.presses == 0
                self._end_run(step, events)
                if idle:  # nothing was pressed, so there is no step to return; the result is in the next Observation
                    return Decision(None, self._tool_trace(info, events), tuple(calls))
            if self.specialists:
                handled = self._specialists.route(state, keys)
                if handled is not None:
                    self._start_special(handled, calls, events)
                    continue
            decision = self._ask_tool_call(observation, legal_actions, keys, calls, events, info)
            if decision is not None:
                return decision
        events.append({"type": "gave_up", "reason": f"{MAX_DECIDE_LOOPS} loops without a press"})
        return Decision(None, self._tool_trace(info, events), tuple(calls))

    def _start_special(self, handled: Handled, calls: list[ModelCall], events: list[dict[str, Any]]) -> None:
        self._run, self._run_call, self._run_turn = handled.run, None, None
        calls.extend(handled.calls)
        events.extend(handled.events)
        if handled.note and (not self._notes or self._notes[-1] != handled.note):
            self._notes.append(handled.note)
            del self._notes[:-MAX_NOTES]

    def _end_run(self, end: tools.ToolEnd, events: list[dict[str, Any]]) -> None:
        run, call, turn = self._run, self._run_call, self._run_turn
        self._run = self._run_call = self._run_turn = None
        events.append({"type": "tool_end", "handler": run.handler, "tool": run.tool, "args": run.args, "outcome": end.outcome, "result": end.result, "presses": run.presses})
        if run.chosen and call is not None:
            line = _result_line(call, end)
            turn.result = line
            self._notes.append(line)
            del self._notes[:-MAX_NOTES]

    def _offer(self, observation: Observation, legal_actions: Sequence[Action]) -> Offer:
        keys = ", ".join(action.key for action in legal_actions)
        state = observation.state
        if tools.navigable(state):
            places = [place for place in state.get("places", []) if "id" in place]
            described = ", ".join(f"{place['id']} ({place.get('role', 'place')})" for place in places) or "none listed"
            return Offer(["go_to", "talk_to", "press"], [place["id"] for place in places], described, keys)
        return Offer(["press"], None, "", keys, reason="this observation has no grid and position, so there is nothing to walk on")

    def _ask_tool_call(self, observation, legal_actions, keys, calls, events, info) -> Decision | None:
        """Ask the model for a tool call. Returns a Decision if there is nothing to press, else starts the run and returns None."""
        offer = self._offer(observation, legal_actions)
        snapshot, images = render_observation(observation)
        notes, self._notes = self._notes, []
        self._turns.append(Turn(observation="\n".join([*notes, history_text(observation)])))
        turn = self._turns[-1]
        legal_keys = [action.key for action in legal_actions]
        reasons: list[str] = []

        def parse(text: str) -> ToolCall | None:
            try:
                return parse_tool_call(text, legal_keys=legal_keys, offered=offer.names, place_ids=offer.place_ids)
            except ToolCallError as problem:
                reasons.append(str(problem))
                return None

        def hint(text: str) -> str:
            return f"({text!r} is not a valid tool call: {reasons[-1]}. Tools: {', '.join(offer.names)}. Places: {offer.places or 'none'}. Buttons: {offer.keys}.)"

        thought, action_text, call, asked = self._ask(
            "\n".join([*notes, snapshot]), images, self._tool_header(offer), parse, hint
        )
        calls.extend(asked)
        turn.thought = thought
        info.update({"thought": thought, "turn": len(self._turns), "action_text": action_text, "action_retry": len(asked) > 1, "tools": offer.names})
        if offer.reason:
            events.append({"type": "tools_offered", "tools": offer.names, "reason": offer.reason})
        if call is None:
            turn.action = f"(invalid: {action_text})"
            events.append({"type": "invalid_tool_call", "text": action_text, "reason": reasons[-1] if reasons else "no action"})
            return Decision(None, self._tool_trace(info, events), tuple(calls))
        turn.action = call.text()
        try:
            steps = tools.tool_steps(call.tool, call.args, observation, keys, set(legal_keys))
        except tools.ToolError as failure:
            line = _result_line(call, tools.ToolEnd("failed", str(failure)))
            turn.result = line
            self._notes.append(line)
            events.append({"type": "tool_end", "handler": "model", "tool": call.tool, "args": dict(call.args), "outcome": "failed", "result": str(failure), "presses": 0})
            return Decision(None, self._tool_trace(info, events), tuple(calls))
        self._run = tools.ToolRun("model", call.tool, dict(call.args), steps, observation.state, chosen=True)
        self._run_call, self._run_turn = call, turn
        return None

    def _tool_header(self, offer: Offer) -> list[str]:
        n = len(self._turns)
        example = f"go_to {offer.place_ids[0]}" if offer.place_ids else "press A"
        lines = [f"Goal: {self._goal}", "Tools:", *[f"- {TOOL_HELP[name]}" for name in offer.names]]
        if offer.reason:
            lines.append("(go_to and talk_to are not offered: this observation has no grid and position.)")
        else:
            lines.append(f"Places you can name: {offer.places}")
        lines += [
            f"Buttons for press: {offer.keys}",
            f"Reply in exactly this form:\nThought {n}: <your reasoning>\nAction {n}: <one tool call, for example: {example}>",
            "",
        ]
        return lines

    def _tool_trace(self, info: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
        run = self._run
        return {
            "turn": len(self._turns),
            **info,
            "handler": run.handler if run else None,
            "tool": run.tool if run else None,
            "args": run.args if run else None,
            "press": run.presses if run else None,
            "events": events,
        }

    def _turn_tail(self, index: int, turn: Turn, last: bool) -> list[str]:
        """Lines after a turn's Action in a rendered history: under tools, how the last tool ended.

        Earlier turns need none, since the next Observation opens with the result.
        """
        if not self.tools or not last or turn.action.startswith("(invalid"):
            return []
        return [f"Result {index}: {turn.result or '(the trial ended while this tool was still running)'}"]


def _result_line(call: ToolCall, end: tools.ToolEnd) -> str:
    """How a tool ended, as the next Observation opens with it: `go_to door_0: went through to 4.0`."""
    if end.outcome == "done":
        return f"{call.text()}: {end.result}"
    return f"{call.text()} {end.outcome}: {end.result}"


def _split(output: str) -> tuple[str, str | None]:
    """Split a reply into (thought, action text); action text is None if absent."""
    match = _ACTION.search(output)
    if not match:
        return output.strip(), None
    thought = output[: match.start()].strip()
    thought = re.sub(r"^Thought\s*\d*\s*:\s*", "", thought, flags=re.IGNORECASE)
    return thought, match.group(1).strip().split("\n")[0]


def history_text(observation: Observation) -> str:
    snapshot, _ = render_observation(observation)
    if observation.state is None:
        return "(screen only; not kept in history)"
    return snapshot.replace("Screen: see the attached image.\n", "")
