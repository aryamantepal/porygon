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
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.common import first_json_object, render_observation
from agents.harness import battle as battle_rules
from agents.harness import prompts, tools
from agents.harness.memory import Journal, KnowledgeGraph
from agents.harness.stuck import RUNGS, StuckLadder, signature
from agents.models import ChatModel, Message, ModelCall
from agents.protocol import Decision, Review, TrialOutcome
from environments.protocol import Action, Observation


@dataclass
class Running:
    handler: str
    tool: str
    args: dict[str, Any]
    steps: tools.Step
    start_state: dict[str, Any] | None
    presses: int = 0
    primed: bool = False


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
    ) -> None:
        self.model = model
        self.knowledge = KnowledgeGraph(backbone, knowledge_path)
        self.summarize_dialogue = summarize_dialogue
        self.stuck_ladder = stuck_ladder
        self.intention_ticks = intention_ticks
        self.max_calls_per_decision = max_calls_per_decision
        self.busy_wait_limit = busy_wait_limit
        self._trial = -1
        self._reset_trial("")

    def mechanisms(self) -> dict[str, bool]:
        return {
            "navigator": True,
            "battle_specialist": True,
            "dialogue_summaries": self.summarize_dialogue,
            "stuck_ladder": self.stuck_ladder,
            "knowledge_graph": True,
            "thinking": False,
        }

    def _reset_trial(self, goal: str) -> None:
        self._goal = goal
        self._journal = Journal()
        self._ladder = StuckLadder()
        self._running: Running | None = None
        self._intention = ""
        self._intention_age = 0
        self._bad_reply: str | None = None
        self._battle_goal = "defeat"
        self._tick = 0
        self._summarized: set[str] = set()
        self._busy_streak = 0
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
                self._busy_streak = 0
            if context == "battle" and state.get("battle"):
                self._start_battle(state, keys, turn)
            elif context == "dialogue":
                self._read_dialogue(state, keys, turn)
            elif context == "menu" and state.get("menu"):
                self._pick_menu(state, keys, turn)
            elif context == "busy":
                self._wait_out(state, keys, turn)
            else:
                self._plan(observation, keys, legal_actions, turn)
        turn.events.append({"type": "gave_up", "reason": f"{self.max_calls_per_decision} model calls without a press"})
        return Decision(None, self._trace(turn), tuple(turn.calls))

    def _start(self, handler: str, tool: str, args: dict, steps: tools.Step, state) -> None:
        self._running = Running(handler, tool, args, steps, state)

    def _advance(self, observation: Observation, turn: _Turn) -> Action | None:
        running = self._running
        try:
            action = running.steps.send(observation) if running.primed else next(running.steps)
            running.primed = True
            running.presses += 1
            return action
        except StopIteration as done:
            outcome, result = "done", done.value
            if running.presses == 0 and running.handler == "planner":
                outcome, result = "failed", f"nothing to do: {done.value}"
        except tools.Interrupted as interruption:
            outcome, result = "interrupted", str(interruption)
        except tools.ToolError as failure:
            outcome, result = "failed", str(failure)
        self._running = None
        turn.events.append({"type": "tool_end", "handler": running.handler, "tool": running.tool, "args": running.args, "outcome": outcome, "result": result, "presses": running.presses})
        if running.handler == "planner":
            self._stats["tools"] += 1
            self._journal.add(self._tick, "tool", f"{running.tool} {json.dumps(running.args)}: {outcome}, {result}")
            if outcome == "failed":
                self._stats["tool_failures"] += 1
                self._ladder.record_failure(running.start_state, running.tool, running.args, result)
        return None

    # Specialists

    def _start_battle(self, state, keys, turn: _Turn) -> None:
        battle = state["battle"]
        allowed, removed = battle_rules.allowed_options(battle, self._battle_goal)
        if not battle.get("options"):
            # messages or animations: nothing to choose, so move them along
            self._start("battle", "press", {"key": keys["confirm"]}, tools.press(keys["confirm"], 1), state)
            return
        if not allowed:
            turn.events.append({"type": "battle", "goal": self._battle_goal, "removed": removed, "choice": None})
            self._start("battle", "press", {"key": keys["confirm"]}, tools.press(keys["confirm"], 1), state)
            return
        why = "only legal option"
        choice = allowed[0]["id"]
        if len(allowed) > 1:
            call = self.model.chat(
                [Message("system", prompts.BATTLE), Message("user", battle_rules.choose_prompt(battle, allowed, self._battle_goal, removed))],
                think=False,
            )
            turn.calls.append(call)
            reply = first_json_object(call.output) or {}
            if reply.get("option") in {o["id"] for o in allowed}:
                choice, why = reply["option"], reply.get("why", "")
            else:
                why = f"reply did not name an allowed option; took {choice}"
        turn.events.append({"type": "battle", "goal": self._battle_goal, "allowed": [o["id"] for o in allowed], "removed": removed, "choice": choice, "why": why})
        chosen = next(o for o in allowed if o["id"] == choice)
        given = battle_rules.option_keys(chosen)
        if given:
            steps = tools.press_sequence(given)
        else:
            steps = tools.pick_option(battle.get("cursor", 0), battle_rules.option_index(battle, choice), keys)
        self._start("battle", "choose", {"option": choice}, steps, state)

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
        self._start("dialogue", "press", {"key": keys["confirm"]}, tools.press(keys["confirm"], 1), state)

    def _wait_out(self, state, keys, turn: _Turn) -> None:
        """The game ignores movement (a transition, cutscene, or undescribed menu).

        Wait if the adapter offers a do-nothing action; if the lock outlasts
        `busy_wait_limit` decisions, press cancel to back out of a menu.
        """
        self._busy_streak += 1
        if "wait" in keys and self._busy_streak <= self.busy_wait_limit:
            key, reason = keys["wait"], "waiting for the game to take input again"
        else:
            key, reason = keys["cancel"], f"still busy after {self._busy_streak - 1} waits; backing out"
            self._busy_streak = 0
        turn.events.append({"type": "busy", "press": key, "reason": reason})
        self._start("busy", "press", {"key": key}, tools.press(key, 1), state)

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
        call = self.model.chat(
            [Message("system", prompts.PLANNER.format(tools=tool_text)), Message("user", self._planner_input(snapshot, state, rung), images)],
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
            self._ladder.record_failure(state, tool, args, str(failure))
            self._stats["tool_failures"] += 1
            return
        if steps is not None:
            self._start("planner", tool, args, steps, state)

    def _tool_steps(self, tool, args, observation, keys, legal_actions):
        state = observation.state or {}
        if tool == "note":
            ok = self.knowledge.note(str(args.get("node", "")), str(args.get("text", "")))
            if not ok:
                raise tools.ToolError(f"no knowledge node {args.get('node')!r}")
            return None
        if tool == "press":
            key = str(args.get("key", "")).upper()
            if key not in {a.key for a in legal_actions}:
                raise tools.ToolError(f"{key!r} is not a legal action")
            times = args.get("times", 1)
            if isinstance(times, bool) or not isinstance(times, int):
                raise tools.ToolError(f'"times" must be a whole number, not {times!r}')
            return tools.press(key, max(1, min(times, 20)))
        target, role = self._target(args, state)
        if tool == "go_to":
            return tools.go_to(observation, target, keys, door=role == "door")
        return tools.talk_to(observation, target, keys)

    def _target(self, args, state) -> tuple[tuple[int, int], str]:
        if "place" in args:
            for place in state.get("places", []):
                if place.get("id") == args["place"]:
                    return tuple(place["position"]), place.get("role", "")
            raise tools.ToolError(f"no place {args['place']!r} on this map")
        position = args.get("position")
        if isinstance(position, list) and len(position) == 2 and all(isinstance(v, int) for v in position):
            return tuple(position), ""
        raise tools.ToolError('args need "place" or "position": [x, y]')

    def _update_intention(self, reply, why, turn: _Turn) -> None:
        new = str(reply.get("intention") or "").strip()
        self._intention_age += 1
        if new and new != self._intention:
            if self._intention:
                turn.events.append({"type": "intention_changed", "from": self._intention, "to": new, "reason": why})
            self._intention, self._intention_age = new, 0

    def _planner_input(self, snapshot: str, state, rung: int) -> str:
        lines = [f"Goal: {self._goal}"]
        if self._intention:
            lines.append(f"Intention: {self._intention} ({self._intention_age}/{self.intention_ticks} turns used)")
            if self._intention_age >= self.intention_ticks:
                lines.append("This intention has used its turns. Keep it only if you can say why; otherwise choose a new one.")
        lines.append(snapshot)
        leads = self._journal.recent(6, ("lead",) if self.summarize_dialogue else ("dialogue",))
        if leads:
            lines += ["Leads:", *[f"- {e['text']}" for e in leads]]
        recent = self._journal.recent(5, ("tool",))
        if recent:
            lines += ["Recent tool results:", *[f"- {e['text']}" for e in recent]]
        knowledge = self.knowledge.relevant(f"{self._goal} {self._intention} {json.dumps(_situation(state))}")
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


    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
