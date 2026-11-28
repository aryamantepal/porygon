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

Two more switches, off by default (`agents/harness/plan.py` has the pieces):

- plan (AdaPlanner, Sun et al. 2023, arXiv:2305.16653): the planner may reply with up to 5
  steps, each a tool call and an `expect` over snapshot fields (map, context, position,
  facing). Code runs them in order and checks each `expect` against the adapter's state
  when the step's tool ends, never against the model's claim. A step that fails its
  tool or its check is an in-plan failure. A step the game interrupts (a battle began)
  runs again afterward, up to twice, and then counts as failed.
- explain_replan (DEPS, Wang et al. 2023, arXiv:2302.01560; works without `plan`): on a
  failed tool or a failed check, one model call gets the plan so far, the failed step, the
  tool result, and the state before, after, and the change between. It returns a short
  explanation and a fix: the steps replacing the failed one and all after it (in-plan
  refinement), or a fresh plan, or one tool when `plan` is off (out-of-plan refinement). The
  explanation goes in the journal and the next planner prompt. At `max_replans` calls in
  one stuck-ladder signature the plan is dropped and the ladder takes over.

Deviations from the papers. Both repositories were read as data. DEPS has no LICENSE file,
so nothing of it is copied; AdaPlanner is MIT, but its prompts write Python programs, so
nothing of it is copied either. All prompts here are ours. DEPS's selector is a learned
model that ranks candidate plans by estimated steps to finish; that is out of scope. Here
the selector only drops a leading step identical to a call that already failed in this
exact situation (the stuck ladder's record). DEPS describes with a model and AdaPlanner
asserts in generated code; here code builds the description and the model only explains
and fixes. AdaPlanner's skill discovery (remembering successful plans) is not built.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.common import first_json_object, render_observation
from agents.examples import Examples
from agents.harness import plan as planning
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
        plan: bool = False,
        explain_replan: bool = False,
        max_replans: int = 2,
    ) -> None:
        self.model = model
        self._example_replies = examples.tool_calls_text() if examples else ""
        self.utility_retrieval, self.memory_decay = utility_retrieval, memory_decay
        self.policy = Policy(utility=utility_retrieval, decay=memory_decay, params=memory_params)
        self.notes_shown = notes_shown
        self.plan, self.explain_replan, self.max_replans = plan, explain_replan, max_replans
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
            "plan": self.plan,
            "explain_replan": self.explain_replan,
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
        self._plan_run: planning.PlanRun | None = None
        self._bad_plan: tuple[list[str], str] | None = None  # (problems, reply) of the last rejected plan
        self._replans: Counter[str] = Counter()  # explain-and-replan calls per stuck-ladder signature
        self._legal: Sequence[Action] = ()
        self._summarized: set[str] = set()
        self._specialists.reset()
        self._stats = {"tools": 0, "tool_failures": 0, "maps": set(), "dialogues": 0, "help_requests": 0, "plans": 0, "plan_failures": 0, "replans": 0}

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
        if self.plan or self.explain_replan:
            summary += f" Plans made: {stats['plans']}, steps failed: {stats['plan_failures']}. Explain-and-replan calls: {stats['replans']}."
        self.knowledge.add_run(summary)
        return Review(trace={"summary": summary, "journal": self._journal.entries})

    # Routing

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        self._tick += 1
        self._clock += 1
        self.knowledge.clock = self._clock
        state = observation.state
        self._legal = legal_actions
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
        if self._already_true(running, outcome, result, observation.state):
            outcome = "done"
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
            if self._plan_run is not None and self._plan_run.current is running:
                self._plan_step_ended(self._plan_run, outcome, result, running.start_state, observation.state, turn)
            elif outcome == "failed" and self.explain_replan:
                self._explain_and_replan(None, 0, running.tool, running.args, "the tool failed", result, running.start_state, observation.state, turn)
        return None

    def _already_true(self, running: tools.ToolRun, outcome: str, result: str, state) -> bool:
        """A plan step whose tool had nothing to do (a step the game interrupted, run again once the fight was over,
        is already where it was going) is done when its `expect` holds in the game."""
        run = self._plan_run
        if run is None or run.current is not running or outcome != "failed" or not result.startswith("nothing to do"):
            return False
        expect = run.steps[run.index].expect
        return bool(expect) and not planning.check_expect(expect, state)

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
        if self._plan_run is not None:
            self._continue_plan(observation, keys, legal_actions, turn)
            return
        state = observation.state
        rung = self._ladder.observe(state) if self.stuck_ladder else 0
        if RUNGS[rung] == "ask_for_help":
            self._stats["help_requests"] += 1
            self._journal.add(self._tick, "help_request", f"stuck at {json.dumps(_situation(state))}")
            turn.events.append({"type": "help_requested", "situation": _situation(state)})
            self._ladder.help_requested(state)
            self._replans[signature(state)] = 0
            rung = 0
        names = _tool_names(state)
        tool_text = _tool_text(names, legal_actions)
        snapshot, images = render_observation(observation)
        system = prompts.PLANNER.format(tools=tool_text)
        if self._example_replies:
            system += "\n\n" + prompts.EXAMPLE_REPLIES.format(replies=self._example_replies)
        if self.plan:
            system += "\n\n" + prompts.PLAN_ADDENDUM.format(max_steps=planning.MAX_PLAN_STEPS, fields=", ".join(planning.EXPECT_FIELDS))
        call = self.model.chat(
            [Message("system", system), Message("user", self._planner_input(snapshot, state, rung, turn), images)],
            think=RUNGS[rung] == "deep_think",
        )
        turn.calls.append(call)
        reply = first_json_object(call.output)
        if self.plan and isinstance(reply, dict) and "plan" in reply:
            self._accept_plan(reply, call, names, rung, turn)
            return
        if not isinstance(reply, dict) or reply.get("tool") not in names or not isinstance(reply.get("args") or {}, dict):
            turn.events.append({"type": "plan", "rung": RUNGS[rung], "error": "reply was not a JSON tool call from the list"})
            self._bad_reply = call.output
            self._bad_plan = None
            return
        self._bad_reply = None
        self._bad_plan = None
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
            if self.explain_replan:
                self._explain_and_replan(None, 0, tool, args, "the tool failed", str(failure), state, state, turn)
            return
        if steps is not None:
            self._start("planner", tool, args, steps, state)

    # Plans with checked steps, and explain-and-replan (agents/harness/plan.py)

    def _accept_plan(self, reply: dict[str, Any], call: ModelCall, names: list[str], rung: int, turn: _Turn) -> None:
        steps, problems = planning.parse_steps(reply["plan"], names)
        if problems:
            turn.events.append({"type": "plan", "rung": RUNGS[rung], "error": "plan rejected: " + "; ".join(problems), "problems": problems})
            self._bad_plan, self._bad_reply = (problems, call.output.strip()), None
            return
        self._bad_plan = self._bad_reply = None
        why = reply.get("why", "")
        self._update_intention(reply, why, turn)
        if reply.get("battle_goal") in ("defeat", "catch", "flee"):
            self._battle_goal = reply["battle_goal"]
        event = {
            "type": "plan",
            "rung": RUNGS[rung],
            "tool": steps[0].tool,
            "args": dict(steps[0].args),
            "steps": [step.to_json() for step in steps],
            "why": why,
            "intention": self._intention,
            "battle_goal": self._battle_goal,
        }
        if call.thinking:
            event["thinking"] = call.thinking
        turn.events.append(event)
        self._stats["plans"] += 1
        self._plan_run = planning.PlanRun(steps, "planner")

    def _continue_plan(self, observation: Observation, keys, legal_actions, turn: _Turn) -> None:
        """Start the plan's next step, or end the plan."""
        run, state = self._plan_run, observation.state
        kept, dropped = planning.select(run.steps[run.index :], self._ladder, state)
        if dropped:
            turn.events.append({"type": "plan_selector", "dropped": [step.to_json() for step in dropped], "reason": "each already failed in this exact situation"})
            run.steps = run.steps[: run.index] + kept
        if run.index >= len(run.steps):
            turn.events.append({"type": "plan_done", "steps": len(run.steps), "source": run.source})
            self._plan_run = None
            return
        step = run.steps[run.index]
        turn.events.append({"type": "plan_step", "index": run.index, "of": len(run.steps), **step.to_json()})
        try:
            generator = self._tool_steps(step.tool, step.args, observation, keys, legal_actions)
        except tools.ToolError as failure:
            turn.events.append({"type": "tool_end", "handler": "planner", "tool": step.tool, "args": dict(step.args), "outcome": "failed", "result": str(failure), "presses": 0})
            self._credit_notes(self._shown_notes, 0.0, turn)
            self._ladder.record_failure(state, step.tool, step.args, str(failure))
            self._stats["tool_failures"] += 1
            self._plan_step_ended(run, "failed", str(failure), state, state, turn)
            return
        self._start("planner", step.tool, dict(step.args), generator, state)
        run.current = self._running

    def _plan_step_ended(self, run: planning.PlanRun, outcome: str, result: str, before, after, turn: _Turn) -> None:
        """Check the step that just ended against the adapter's state, then go on, retry, or explain and re-plan."""
        index, step = run.index, run.steps[run.index]
        run.current = None
        if outcome == "interrupted":
            run.interrupts += 1
            turn.events.append({"type": "plan_check", "index": index, "tool": step.tool, "outcome": "interrupted", "result": result, "retries": run.interrupts, "passed": None})
            if run.interrupts <= planning.MAX_INTERRUPTS:
                return  # the step runs again once the game is back on the overworld
            outcome, result = "failed", f"interrupted {run.interrupts} times: {result}"
            self._ladder.record_failure(before, step.tool, step.args, result)
        mismatches = planning.check_expect(step.expect, after) if outcome == "done" else []
        run.results.append(planning.StepResult(index, step, outcome, result, mismatches))
        passed = outcome == "done" and not mismatches
        turn.events.append(
            {
                "type": "plan_check",
                "index": index,
                "tool": step.tool,
                "outcome": outcome,
                "result": result,
                "expect": dict(step.expect),
                "actual": {key: (after or {}).get(key) for key in step.expect},
                "passed": passed,
            }
        )
        if passed:
            run.index, run.interrupts = index + 1, 0
            if run.index >= len(run.steps):
                turn.events.append({"type": "plan_done", "steps": len(run.steps), "source": run.source})
                self._plan_run = None
            return
        self._stats["plan_failures"] += 1
        if mismatches:
            what = "; ".join(f"{m['field']} is {json.dumps(m['actual'])}, expected {json.dumps(m['expected'])}" for m in mismatches)
            result = f"{result}. The game does not match the step's expect: {what}"
            self._ladder.record_failure(before, step.tool, step.args, result)
            kind = "the tool finished, but the game does not match the step's expect"
        else:
            kind = "the tool failed"
        self._explain_and_replan(run, index, step.tool, step.args, kind, result, before, after, turn)

    def _explain_and_replan(self, run, index: int, tool: str, args, kind: str, result: str, before, after, turn: _Turn) -> None:
        """DEPS's describe, explain, re-plan, in one model call. `run` is the plan the failed step belonged to, or None
        for a single tool. With a plan the reply replaces the failed step and the rest (in-plan refinement); without one
        it is a fresh plan, or one tool when `plan` is off (out-of-plan refinement). The explanation goes in the journal
        and so into the next planner prompt. At `max_replans` calls in one situation the plan is dropped and the stuck
        ladder (already told of the failure) takes over."""
        signature_ = signature(before)
        if not self.explain_replan or self._replans[signature_] >= self.max_replans:
            if self.explain_replan:
                turn.events.append({"type": "replan_cap", "tool": tool, "args": dict(args), "cap": self.max_replans, "fallback": "stuck_ladder"})
            self._drop_plan(run, "no re-plan left in this situation" if self.explain_replan else "explain_replan is off", turn)
            return
        self._replans[signature_] += 1
        self._stats["replans"] += 1
        in_plan = run is not None
        names = _tool_names(before or after)
        fields = ", ".join(planning.EXPECT_FIELDS)
        if in_plan:
            revision, shape = prompts.REVISE_IN_PLAN.format(fields=fields), prompts.SHAPE_PLAN
        elif self.plan:
            revision, shape = prompts.REVISE_NEW_PLAN.format(max_steps=planning.MAX_PLAN_STEPS, fields=fields), prompts.SHAPE_PLAN
        else:
            revision, shape = prompts.REVISE_ONE_TOOL, prompts.SHAPE_TOOL
        progress = planning.render_progress(run, index) if in_plan else f"The step that failed: {tool} {json.dumps(dict(args))}"
        earlier = self._ladder.failed_here(before)
        failed = "\n" + "\n".join(["Already failed in this exact situation:", *[f"- {a.tool} {json.dumps(a.args)}: {a.result}" for a in earlier[-5:]]]) if earlier else ""
        call = self.model.chat(
            [
                Message("system", prompts.EXPLAIN_REPLAN.format(tools=_tool_text([n for n in names if n != "note"], self._legal), revision=revision, shape=shape)),
                Message(
                    "user",
                    prompts.FAILURE.format(
                        goal=self._goal,
                        intention=self._intention or "none yet",
                        progress=progress,
                        kind=kind,
                        result=result,
                        before=json.dumps(planning.brief(before)),
                        after=json.dumps(planning.brief(after)),
                        change=planning.describe_change(before, after),
                        failed=failed,
                    ),
                ),
            ],
            think=False,
        )
        turn.calls.append(call)
        reply = first_json_object(call.output)
        event: dict[str, Any] = {"type": "explanation", "scope": "in_plan" if in_plan else "out_of_plan", "tool": tool, "args": dict(args), "failure": kind, "step": index if in_plan else None, "replans_in_situation": self._replans[signature_]}
        if not isinstance(reply, dict):
            turn.events.append({**event, "error": "reply was not a JSON object"})
            self._drop_plan(run, "the explain reply was not usable", turn)
            return
        explanation = str(reply.get("explanation") or "").strip()[:400]
        event["explanation"] = explanation
        if explanation:
            self._journal.add(self._tick, "explanation", f"{tool} {json.dumps(dict(args))} ({kind}): {explanation}")
        steps, problems = planning.parse_reply(reply, names, plan_allowed=self.plan)
        if problems:
            turn.events.append({**event, "error": "revision rejected: " + "; ".join(problems), "problems": problems})
            self._drop_plan(run, "the revision was not usable", turn)
            return
        turn.events.append({**event, "revision": [step.to_json() for step in steps]})
        if in_plan:
            run.steps = run.steps[:index] + steps
            run.results = run.results[:index]
            run.index, run.interrupts, run.current = index, 0, None
        else:
            self._plan_run = planning.PlanRun(steps, "revision")
        turn.events.append({"type": "plan_revised", "scope": "in_plan" if in_plan else "out_of_plan", "from_step": index if in_plan else 0, "steps": [step.to_json() for step in steps]})

    def _drop_plan(self, run, reason: str, turn: _Turn) -> None:
        if run is not None:
            turn.events.append({"type": "plan_dropped", "at_step": run.index, "reason": reason})
            self._plan_run = None

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
        if self.explain_replan:
            explanations = self._journal.recent(2, ("explanation",))
            if explanations:
                lines += [prompts.EXPLANATIONS, *[f"- {e['text']}" for e in explanations]]
        if self._bad_reply is not None:
            lines.append(prompts.BAD_REPLY.format(reply=self._bad_reply.strip()[:300]))
        if self._bad_plan is not None:
            problems, reply = self._bad_plan
            lines.append(prompts.PLAN_REJECTED.format(problems="\n".join(f"- {p}" for p in problems), reply=reply[:300]))
        if rung >= 1:
            failures = self._ladder.failed_here(state)
            if failures:
                lines += ["Already failed in this exact situation:", *[f"- {a.tool} {json.dumps(a.args)}: {a.result}" for a in failures[-5:]]]
        if rung >= 2:
            lines.append(prompts.NUDGE.format(count=self._ladder.seen[signature(state)]))
        return "\n".join(lines)

    def _trace(self, turn: _Turn) -> dict[str, Any]:
        running = self._running
        trace = {
            "handler": running.handler if running else None,
            "tool": running.tool if running else None,
            "args": running.args if running else None,
            "press": running.presses if running else None,
            "intention": self._intention,
            "rung": RUNGS[self._ladder.rung],
            "events": turn.events,
        }
        if self._plan_run is not None:
            trace["plan"] = {"step": self._plan_run.index, "of": len(self._plan_run.steps), "source": self._plan_run.source}
        return trace


def _tool_names(state) -> list[str]:
    navigable = bool(state and state.get("grid") and state.get("position"))
    return ["go_to", "talk_to", "press", "note"] if navigable else ["press", "note"]


def _tool_text(names: Sequence[str], legal_actions: Sequence[Action]) -> str:
    return "\n".join(f"- {prompts.TOOLS[name]}" for name in names).replace("KEYS", ", ".join(a.key for a in legal_actions))


def _situation(state) -> dict[str, Any]:
    if not state:
        return {}
    return {key: state.get(key) for key in ("context", "map", "position")}
