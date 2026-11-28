"""The harness from "Teaching a Local LLM to Play Pokémon Crystal" (playtesting.dev), rebuilt game-agnostically.

One decision at a time. Each `decide` call routes on the snapshot's context:

    battle    -> battle specialist (code rules out illegal options, model picks)
    dialogue  -> read it into the journal as leads, then dismiss it
    menu      -> model picks an option; code moves the cursor
    busy      -> wait (no model call); back out with cancel if it lasts
    otherwise -> planner picks a tool; code runs it press by press

A running tool keeps returning presses until it finishes, fails, or the game
changes context. Then the next decision is routed again. The stuck ladder
watches planner decisions and escalates when the same situation repeats.

What comes from the write-up and what is ours: the component list, the routing
order, the battle invariants, the backbone-plus-notes knowledge graph, the
factual end-of-run summary, and the ladder's rungs are from the write-up. It
publishes no prompts, schemas, or thresholds, so those are ours. Its operator
console and chat inbox are not built yet.

Two switches, off by default, change how the planner's memory is retrieved
(math and deviations in `agents/retrieval.py`):

- utility_retrieval (MemRL, Zhang et al. 2026, arXiv:2601.03192): the notes and
  lessons the agent attached to knowledge nodes are retrieved in two phases. Phase 1
  is the words each shares with the planner's query (goal, intention, situation) and
  its node; phase 2 ranks the top `utility_candidates` by a learned Q, and at most
  `notes_shown` are shown. The reward comes from the navigator, not the model: when a
  planner decision starts a tool, the notes shown in that prompt get r = 1 if the tool
  ends "done" and r = 0 if it ends "failed" (including a call that failed to start).
  A tool the game interrupted (a battle began) is not scored, and neither is a `note`
  call, which is no tool run. A tool that is "done" may still not have moved the game
  toward the goal, so this is a weaker signal than progress read from the game, which
  an agent cannot call. Backbone nodes, edges, and recipes are not scored.
- memory_decay (MemoryBank, Zhong et al. 2023, arXiv:2305.10250): a note's retention
  is `exp(-t / S)`, t the decisions since it was written or last shown, S its strength
  (multiplied by `growth` each showing); below `decay_threshold` it is not shown, and a
  node's full note list drops a forgotten note before the oldest. The journal's leads
  decay the same way within a trial. Both log a `retrieval` event with every candidate's
  similarity, utility, retention, and score, and a `utility_update` event per reward.
  `knowledge_path` saves the Q and strength of each note only while a switch is on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.common import first_json_object, render_observation
from agents.examples import Examples
from agents.harness import prompts, tools
from agents.harness.memory import Journal, KnowledgeGraph
from agents.harness.specialists import Handled, Specialists
from agents.harness.stuck import RUNGS, StuckLadder, signature
from agents.models import ChatModel, Message, ModelCall
from agents.protocol import Decision, Review, TrialOutcome
from agents.retrieval import MemoryParams, Policy, trace_event
from environments.protocol import Action, Observation


@dataclass
class _Turn:
    calls: list[ModelCall] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class CrystalHarness:
    name = "crystal_harness"

    def __init__(
        self,
        model: ChatModel,
        *,
        backbone: dict[str, Any] | None = None,
        knowledge_path: Path | None = None,
        summarize_dialogue: bool = True,
        stuck_ladder: bool = True,
        intention_ticks: int = 60,
        max_calls_per_decision: int = 8,
        busy_wait_limit: int = 6,
        examples: Examples | None = None,
        utility_retrieval: bool = False,
        memory_decay: bool = False,
        memory_params: MemoryParams | None = None,
        notes_shown: int = 6,
    ) -> None:
        self.model = model
        self._example_replies = examples.tool_calls_text() if examples else ""
        self.utility_retrieval, self.memory_decay = utility_retrieval, memory_decay
        self.policy = Policy(utility=utility_retrieval, decay=memory_decay, params=memory_params)
        self.notes_shown = notes_shown
        self.knowledge = KnowledgeGraph(backbone, knowledge_path, self.policy)
        self._clock = self.knowledge.clock  # decisions over the player's life: the time memory_decay counts in
        self.summarize_dialogue = summarize_dialogue
        self.stuck_ladder = stuck_ladder
        self.intention_ticks = intention_ticks
        self.max_calls_per_decision = max_calls_per_decision
        self.busy_wait_limit = busy_wait_limit
        self._specialists = Specialists(model, busy_wait_limit=busy_wait_limit)
        self._trial = -1
        self._reset_trial("")

    def mechanisms(self) -> dict[str, bool]:
        return {
            "navigator": True,
            "battle_specialist": True,
            "dialogue_summaries": self.summarize_dialogue,
            "stuck_ladder": self.stuck_ladder,
            "knowledge_graph": True,
            "few_shot": bool(self._example_replies),
            "thinking": False,
            "utility_retrieval": self.utility_retrieval,
            "memory_decay": self.memory_decay,
        }

    def _reset_trial(self, goal: str) -> None:
        self._goal = goal
        self._journal = Journal(policy=self.policy)
        self._shown_notes: tuple[str, ...] = ()  # the notes in the last planner prompt
        self._chosen_notes: tuple[str, ...] = ()  # the notes in the prompt that chose the running planner tool
        self._ladder = StuckLadder()
        self._running: tools.ToolRun | None = None
        self._intention = ""
        self._intention_age = 0
        self._bad_reply: str | None = None
        self._battle_goal = "defeat"
        self._tick = 0
        self._summarized: set[str] = set()
        self._specialists.reset()
        self._stats = {"tools": 0, "tool_failures": 0, "maps": set(), "dialogues": 0, "help_requests": 0}

    def start_trial(self, goal: str) -> None:
        self._trial += 1
        self._reset_trial(goal)

    def end_trial(self, outcome: TrialOutcome) -> Review:
        stats = self._stats
        summary = (
            f"Trial {self._trial}: {outcome.reason} after {outcome.steps} steps. "
            f"Tools run: {stats['tools']}, failed: {stats['tool_failures']}. "
            f"Maps seen: {', '.join(sorted(map(str, stats['maps']))) or 'none'}. "
            f"Dialogues read: {stats['dialogues']}. Help requests: {stats['help_requests']}."
        )
        self.knowledge.add_run(summary)
        return Review(trace={"summary": summary, "journal": self._journal.entries})

    # Routing

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        self._tick += 1
        self._clock += 1
        self.knowledge.clock = self._clock
        state = observation.state
        keys = tools.controls(state, {action.key for action in legal_actions})
        turn = _Turn()
        if state:
            self._stats["maps"].add(state.get("map"))
        while len(turn.calls) < self.max_calls_per_decision:
            if self._running is not None:
                action = self._advance(observation, turn)
                if action is not None:
                    return Decision(action, self._trace(turn), tuple(turn.calls))
                continue
            context = (state or {}).get("context", "overworld")
            if context != "busy":
                self._specialists.reset_busy()
            if context == "battle" and state.get("battle"):
                self._use(self._specialists.battle(state, keys, self._battle_goal), turn)
            elif context == "dialogue":
                self._read_dialogue(state, keys, turn)
            elif context == "menu" and state.get("menu"):
                self._pick_menu(state, keys, turn)
            elif context == "busy":
                self._use(self._specialists.busy(state, keys), turn)
            else:
                self._plan(observation, keys, legal_actions, turn)
        turn.events.append({"type": "gave_up", "reason": f"{self.max_calls_per_decision} model calls without a press"})
        return Decision(None, self._trace(turn), tuple(turn.calls))

    def _start(self, handler: str, tool: str, args: dict, steps: tools.Step, state) -> None:
        self._running = tools.ToolRun(handler, tool, args, steps, state, chosen=handler == "planner")
        if handler == "planner":
            self._chosen_notes = self._shown_notes

    def _use(self, handled: Handled, turn: _Turn) -> None:
        turn.calls.extend(handled.calls)
        turn.events.extend(handled.events)
        self._running = handled.run

    def _advance(self, observation: Observation, turn: _Turn) -> Action | None:
        running = self._running
        step = running.advance(observation)
        if not isinstance(step, tools.ToolEnd):
            return step
        outcome, result = step.outcome, step.result
        self._running = None
        if running.handler == "planner" and outcome != "interrupted":
            self._credit_notes(self._chosen_notes, float(outcome == "done"), turn)
        turn.events.append({"type": "tool_end", "handler": running.handler, "tool": running.tool, "args": running.args, "outcome": outcome, "result": result, "presses": running.presses})
        if running.handler == "planner":
            self._stats["tools"] += 1
            self._journal.add(self._tick, "tool", f"{running.tool} {json.dumps(running.args)}: {outcome}, {result}")
            if outcome == "failed":
                self._stats["tool_failures"] += 1
                self._ladder.record_failure(running.start_state, running.tool, running.args, result)
        return None

    # Specialists

    def _read_dialogue(self, state, keys, turn: _Turn) -> None:
        text = (state.get("dialogue") or {}).get("text", "")
        if text and text not in self._summarized:
            self._summarized.add(text)
            self._stats["dialogues"] += 1
            self._journal.add(self._tick, "dialogue", text)
            if self.summarize_dialogue:
                call = self.model.chat([Message("user", prompts.DIALOGUE.format(text=text))], think=False)
                turn.calls.append(call)
                leads = [line[2:].strip() for line in call.output.splitlines() if line.startswith("- ")]
                for lead in leads:
                    self._journal.add(self._tick, "lead", lead)
                turn.events.append({"type": "dialogue", "text": text, "leads": leads})
            else:
                turn.events.append({"type": "dialogue", "text": text})
        self._use(self._specialists.dialogue(state, keys), turn)

    def _pick_menu(self, state, keys, turn: _Turn) -> None:
        menu = state["menu"]
        options = menu.get("options", [])
        call = self.model.chat(
            [Message("user", prompts.MENU.format(prompt=menu.get("prompt", ""), options=json.dumps(options), intention=self._intention or self._goal))],
            think=False,
        )
        turn.calls.append(call)
        reply = first_json_object(call.output) or {}
        option = reply.get("option")
        index = option if isinstance(option, int) and not isinstance(option, bool) and 0 <= option < len(options) else 0
        turn.events.append({"type": "menu", "prompt": menu.get("prompt", ""), "choice": index, "why": reply.get("why", "reply was not usable; took the first option")})
        self._start("menu", "choose", {"option": index}, tools.pick_option(menu.get("cursor", 0), index, keys), state)

    # Planner

    def _plan(self, observation: Observation, keys, legal_actions, turn: _Turn) -> None:
        state = observation.state
        rung = self._ladder.observe(state) if self.stuck_ladder else 0
        if RUNGS[rung] == "ask_for_help":
            self._stats["help_requests"] += 1
            self._journal.add(self._tick, "help_request", f"stuck at {json.dumps(_situation(state))}")
            turn.events.append({"type": "help_requested", "situation": _situation(state)})
            self._ladder.help_requested(state)
            rung = 0
        navigable = bool(state and state.get("grid") and state.get("position"))
        names = ["go_to", "talk_to", "press", "note"] if navigable else ["press", "note"]
        tool_text = "\n".join(f"- {prompts.TOOLS[name]}" for name in names).replace("KEYS", ", ".join(a.key for a in legal_actions))
        snapshot, images = render_observation(observation)
        system = prompts.PLANNER.format(tools=tool_text)
        if self._example_replies:
            system += "\n\n" + prompts.EXAMPLE_REPLIES.format(replies=self._example_replies)
        call = self.model.chat(
            [Message("system", system), Message("user", self._planner_input(snapshot, state, rung, turn), images)],
            think=RUNGS[rung] == "deep_think",
        )
        turn.calls.append(call)
        reply = first_json_object(call.output)
        if not isinstance(reply, dict) or reply.get("tool") not in names or not isinstance(reply.get("args") or {}, dict):
            turn.events.append({"type": "plan", "rung": RUNGS[rung], "error": "reply was not a JSON tool call from the list"})
            self._bad_reply = call.output
            return
        self._bad_reply = None
        tool, args, why = reply["tool"], reply.get("args") or {}, reply.get("why", "")
        self._update_intention(reply, why, turn)
        if reply.get("battle_goal") in ("defeat", "catch", "flee"):
            self._battle_goal = reply["battle_goal"]
        event = {"type": "plan", "rung": RUNGS[rung], "tool": tool, "args": args, "why": why, "intention": self._intention, "battle_goal": self._battle_goal}
        if call.thinking:
            event["thinking"] = call.thinking
        turn.events.append(event)
        if self.stuck_ladder and self._ladder.is_repeat_failure(state, tool, args):
            turn.events.append({"type": "skipped_repeat", "tool": tool, "args": args})
            self._ladder.record_failure(state, tool, args, "skipped: this exact call already failed here")
            return
        try:
            steps = self._tool_steps(tool, args, observation, keys, legal_actions)
        except tools.ToolError as failure:
            turn.events.append({"type": "tool_end", "handler": "planner", "tool": tool, "args": args, "outcome": "failed", "result": str(failure), "presses": 0})
            self._credit_notes(self._shown_notes, 0.0, turn)
            self._ladder.record_failure(state, tool, args, str(failure))
            self._stats["tool_failures"] += 1
            return
        if steps is not None:
            self._start("planner", tool, args, steps, state)

    def _tool_steps(self, tool, args, observation, keys, legal_actions):
        if tool == "note":
            ok = self.knowledge.note(str(args.get("node", "")), str(args.get("text", "")))
            if not ok:
                raise tools.ToolError(f"no knowledge node {args.get('node')!r}")
            return None
        return tools.tool_steps(tool, args, observation, keys, {a.key for a in legal_actions})

    def _target(self, args, state) -> tuple[tuple[int, int], str]:
        return tools.resolve_target(args, state)

    def _update_intention(self, reply, why, turn: _Turn) -> None:
        new = str(reply.get("intention") or "").strip()
        self._intention_age += 1
        if new and new != self._intention:
            if self._intention:
                turn.events.append({"type": "intention_changed", "from": self._intention, "to": new, "reason": why})
            self._intention, self._intention_age = new, 0

    def _credit_notes(self, keys: tuple[str, ...], reward: float, turn: _Turn) -> None:
        """MemRL's update for each note the prompt showed: r is 1 when the tool the planner chose finished, 0 when it failed."""
        for key in keys:
            change = self.knowledge.credit(key, reward)
            if change is not None:
                turn.events.append({"type": "utility_update", "store": "knowledge", "id": key, "reward": reward, "old": round(change[0], 4), "new": round(change[1], 4)})

    def _planner_input(self, snapshot: str, state, rung: int, turn: _Turn) -> str:
        lines = [f"Goal: {self._goal}"]
        if self._intention:
            lines.append(f"Intention: {self._intention} ({self._intention_age}/{self.intention_ticks} turns used)")
            if self._intention_age >= self.intention_ticks:
                lines.append("This intention has used its turns. Keep it only if you can say why; otherwise choose a new one.")
        lines.append(snapshot)
        kinds = ("lead",) if self.summarize_dialogue else ("dialogue",)
        leads = self._journal.recent(6, kinds, now=self._tick)
        if self.memory_decay:
            turn.events.append(trace_event("journal", "", self._journal.retrieval(kinds, leads, self._tick)))
            self._journal.recall(leads, self._tick)
        if leads:
            lines += ["Leads:", *[f"- {e['text']}" for e in leads]]
        recent = self._journal.recent(5, ("tool",))
        if recent:
            lines += ["Recent tool results:", *[f"- {e['text']}" for e in recent]]
        query = f"{self._goal} {self._intention} {json.dumps(_situation(state))}"
        knowledge = self.knowledge.relevant(query, notes_shown=self.notes_shown)
        if self.policy.active:
            turn.events.append(trace_event("knowledge", query, self.knowledge.last_rows))
            self._shown_notes = tuple(row.key for row in self.knowledge.last_rows if row.shown)
        if any(knowledge.values()):
            lines.append("What you know: " + json.dumps(knowledge))
        if self._bad_reply is not None:
            lines.append(prompts.BAD_REPLY.format(reply=self._bad_reply.strip()[:300]))
        if rung >= 1:
            failures = self._ladder.failed_here(state)
            if failures:
                lines += ["Already failed in this exact situation:", *[f"- {a.tool} {json.dumps(a.args)}: {a.result}" for a in failures[-5:]]]
        if rung >= 2:
            lines.append(prompts.NUDGE.format(count=self._ladder.seen[signature(state)]))
        return "\n".join(lines)

    def _trace(self, turn: _Turn) -> dict[str, Any]:
        running = self._running
        return {
            "handler": running.handler if running else None,
            "tool": running.tool if running else None,
            "args": running.args if running else None,
            "press": running.presses if running else None,
            "intention": self._intention,
            "rung": RUNGS[self._ladder.rung],
            "events": turn.events,
        }


def _situation(state) -> dict[str, Any]:
    if not state:
        return {}
    return {key: state.get(key) for key in ("context", "map", "position")}
