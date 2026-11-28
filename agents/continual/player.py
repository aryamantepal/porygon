"""Continual Harness: a minimal agent whose harness is rewritten mid-episode (Karten et al., 2026, arXiv:2605.09998).

Ported from the authors' code (github.com/sethkarten/continual-harness, MIT,
Copyright (c) 2026 Seth Karten, Tersoo Upaa Jr, Jake Grigsby, Stephanie Milani,
Kiran Vodrahalli, Amy Zhang, Fei Fang, Yuke Zhu, Chi Jin): the schedule and the
passes of `HarnessEvolver` (agents/utils/harness_evolver.py), the prompt pass of
`PromptOptimizer` (agents/utils/prompt_optimizer.py), the seed prompts under
agents/prompts/, and the evolve hooks and prompt layout of the main agent loop.

One model plays through a minimal interface: the snapshot and the legal buttons.
The harness H = (guidance p, skills K, memory M) starts empty. On a code-driven
schedule the same model, acting as a text-only Refiner, reads the recent
trajectory window and edits H in place, one pass per component. The agent can
also write memory and ask for a refinement itself. Nothing resets between
refinements.

    decide: [scheduled refine] -> model picks a tool call
      press {key, times}             one or more presses
      run_skill {name, args?}        a skill from the library, one press per step
      write_memory {title, ...}      no game time; the model picks again
      refine {reason, num_steps}     no game time; rate-limited; the model picks again

Schedule (`should_refine`) is the code's, not the paper's prose: nothing before
step 25, then every 25 steps through step 200, then every 100. The code also
holds a second, unused 50-step warmup inside `PromptOptimizer.should_optimize`;
`HarnessEvolver` never calls it, so the prompt pass runs from step 25 here too.
The trajectory window is 50 steps.

Differences from the authors' code, because the setting differs:
- Skills are programs in the skill language of `agents/voyager/skills.py`, not
  Python run with exec. A Refiner-written skill must parse against the legal
  actions or the edit is rejected and logged. A skill runs one press per
  `decide`, so the runner counts every press. This is the main deviation.
- The prompt pass rewrites a bounded "learned guidance" section, not the whole
  base prompt, so the fixed instructions cannot be edited away.
- Sub-agents are not built. The Refiner has no sub-agent pass and the agent has
  no sub-agent tools. The specialists in `agents/harness` are the nearest analogue.
- The agent's tools are `press`, `run_skill`, `write_memory`, and `refine`. It
  cannot write skills directly (the original's `process_skill`) or run code, so
  skills come only from the Refiner. Memory is shown inline, capped, with no
  read tool. One tool call per turn, not several.
- A step is one press the runner counts. The original counts agent turns, in
  which a skill call is one turn.
- The agent-triggered refine is rate-limited (`refine_cooldown`, `max_agent_refines`);
  the original's is not. Skill and memory counts are capped, and a full store
  evicts its least important entry.
- A reply that is not a valid tool call is rejected, shown back to the model with
  the reason, and asked again, up to `max_calls_per_decision` calls. Repeating one
  call with no change in the state adds a note to the next prompt.
- Trials are independent by default: each starts from an empty harness, or from
  `harness_path` if that file exists when the player is built. `carry_across_trials`
  keeps the harness, as the paper's bootstrapping does with a prior run's artifacts.
  `harness_path` is rewritten with the final harness at the end of every trial.
- `refiner_model` may be a different model. The paper found the smallest model
  fell below the baseline as its own Refiner.
- The original's skills and memory stores live on a server and are inspected
  through tools; here they are in-process and every edit is in the trace.
- skill_parameters (off by default; SkillWeaver, Zheng et al. 2025, arXiv:2504.07079):
  a skill may declare parameters (`skill walk_to target`), and `run_skill` takes
  `args`, one value per parameter, bound by position (`{"name": "walk_to", "args": [9]}`).
  A wrong count is rejected with the reason before any press. This player has no
  tools interface, so a parameter stands where a JSON value goes (`until x == $target
  (max 20):`) or in a `call`, not in `go_to`. A Refiner-written skill may declare
  parameters. Off, parameter syntax is rejected and `run_skill` takes only `name`.
  Skills are not tested before they enter the library (that is Voyager's `test_skills`).
- utility_retrieval (MemRL, Zhang et al. 2026, arXiv:2601.03192) and memory_decay
  (MemoryBank, Zhong et al. 2023, arXiv:2305.10250), off by default, the math in
  `agents/retrieval.py`. This store has no query, so phase 1 keeps the order it always
  had (importance, then recency) and phase 2 ranks the top `utility_candidates` by a
  learned Q per entry; the prompt shows the top `memory_shown` memories and the top
  `skills_shown` skills (it shows every skill when the switch is off). The Refiner
  still sees everything. The reward is read from the state after one tool run (a press
  or a `run_skill`): r = 1 when the state the player sees (the structured state, or the
  frame in visual mode) differs from the state the run started in and the run raised
  no error, else r = 0. Every memory entry in the prompt that led to the run is
  updated with that r, and so is the skill the run executed. It is the player's own
  view of the state, not `get_progress`, and a press into a wall is a 0 for every
  memory that was in the prompt. Under decay a memory's retention is `exp(-t / S)` with
  t in presses over the player's life, and a memory below `decay_threshold` is not
  shown, and is the first evicted when the store is full. Skills do not decay.
  Rewriting an entry's content starts its Q and strength over. `harness_path` saves
  both only while a switch is on. Events: `retrieval` (every candidate with its
  similarity, utility, retention, score) and `utility_update`.
- Few-shot: `examples` tool_calls whose tool is one of ours are shown to the
  agent, and `examples` programs are shown to the Refiner's skills pass as
  language examples. They are not stored in the library.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from agents.common import first_json_object, render_observation
from agents.continual import prompts
from agents.continual.harness import HarnessState, dumps
from agents.continual.refiner import Refiner
from agents.continual.validate import (
    MAX_CONTENT,
    MAX_TITLE,
    Invalid,
    importance_field,
    only_fields,
    text_field,
)
from agents.examples import Examples
from agents.models import ChatModel, Message, ModelCall
from agents.protocol import Decision, Review, TrialOutcome
from agents.retrieval import MemoryParams, Policy, trace_event
from agents.voyager.skills import Program, Run, SkillError, Tap, execute
from environments.protocol import Action, Observation

TOOLS = ("press", "run_skill", "write_memory", "refine")
MAX_WINDOW = 100  # the most trajectory steps one refinement may read
VALUE_CHARS = 80


def should_refine(step: int, *, warmup: int = 25, early_every: int = 25, early_until: int = 200, late_every: int = 100) -> bool:
    """HarnessEvolver.should_evolve: after the warmup, every `early_every` steps through
    `early_until`, then every `late_every`. Steps after `early_until` that are not
    multiples of `late_every` do not fire, so the first late refinement is at 300."""
    if step <= 0 or step < warmup:
        return False
    return step % (early_every if step <= early_until else late_every) == 0


@dataclass
class _Running:
    tool: str
    label: str
    program: Program
    steps: Generator[Action, Observation, None]
    run: Run
    start: Observation
    why: str
    skill: str | None = None
    primed: bool = False
    shown_memory: tuple[str, ...] = ()  # the memory entries in the prompt that chose this run (utility_retrieval)


@dataclass
class _Turn:
    calls: list[ModelCall] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class ContinualHarness:
    name = "continual"

    def __init__(
        self,
        model: ChatModel,
        *,
        refiner_model: ChatModel | None = None,
        refine_prompt: bool = True,
        refine_skills: bool = True,
        refine_memory: bool = True,
        agent_meta_tools: bool = True,
        carry_across_trials: bool = False,
        refine_warmup: int = 25,
        refine_early_every: int = 25,
        refine_early_until: int = 200,
        refine_late_every: int = 100,
        refine_window: int = 50,
        refine_cooldown: int = 10,
        max_agent_refines: int = 3,
        harness_path: Path | None = None,
        history_turns: int = 10,
        max_press_times: int = 10,
        repeat_limit: int = 3,
        max_memory: int = 30,
        max_skills: int = 12,
        memory_shown: int = 8,
        max_calls_per_decision: int = 10,
        examples: Examples | None = None,
        skill_parameters: bool = False,
        utility_retrieval: bool = False,
        memory_decay: bool = False,
        memory_params: MemoryParams | None = None,
        skills_shown: int = 6,
    ) -> None:
        self.model = model
        self.skill_parameters = skill_parameters
        self.utility_retrieval, self.memory_decay = utility_retrieval, memory_decay
        self.policy = Policy(utility=utility_retrieval, decay=memory_decay, params=memory_params)
        self.skills_shown = skills_shown
        self.refiner_model = refiner_model
        self.refine_prompt, self.refine_skills, self.refine_memory = refine_prompt, refine_skills, refine_memory
        self.agent_meta_tools = agent_meta_tools
        self.carry_across_trials = carry_across_trials
        self.schedule = {"warmup": refine_warmup, "early_every": refine_early_every, "early_until": refine_early_until, "late_every": refine_late_every}
        self.refine_window = min(refine_window, MAX_WINDOW)
        self.refine_cooldown = refine_cooldown
        self.max_agent_refines = max_agent_refines
        self.harness_path = harness_path
        self.history_turns = history_turns
        self.max_press_times = max_press_times
        self.repeat_limit = repeat_limit
        self.max_memory, self.max_skills = max_memory, max_skills
        self.memory_shown = memory_shown
        self.max_calls_per_decision = max_calls_per_decision
        self._example_programs = examples.programs_text() if examples else ""
        ours = [call for call in (examples.tool_calls if examples else ()) if call.reply.get("tool") in TOOLS]
        self._example_replies = dataclasses.replace(examples, tool_calls=tuple(ours)).tool_calls_text() if ours else ""
        self.refiner = Refiner(
            refiner_model or model, prompt=refine_prompt, skills=refine_skills, memory=refine_memory,
            example_programs=self._example_programs, skill_parameters=skill_parameters,
        )
        self._seed = json.loads(harness_path.read_text()) if harness_path is not None and harness_path.exists() else None
        self.harness = self._new_harness()
        self._trial = -1
        self._reset_trial("")

    def mechanisms(self) -> dict[str, bool]:
        return {
            "refine_prompt": self.refine_prompt,
            "refine_skills": self.refine_skills,
            "refine_memory": self.refine_memory,
            "agent_meta_tools": self.agent_meta_tools,
            "carry_across_trials": self.carry_across_trials,
            "separate_refiner_model": self.refiner_model is not None,
            "few_shot": bool(self._example_replies or self._example_programs),
            "thinking": False,
            "skill_parameters": self.skill_parameters,
            "utility_retrieval": self.utility_retrieval,
            "memory_decay": self.memory_decay,
        }

    def _new_harness(self) -> HarnessState:
        if self._seed is not None:
            return HarnessState.from_dict(self._seed, max_memory=self.max_memory, max_skills=self.max_skills, policy=self.policy)
        return HarnessState(max_memory=self.max_memory, max_skills=self.max_skills, policy=self.policy)

    def _reset_trial(self, goal: str) -> None:
        self._goal = goal
        self._steps = 0  # presses so far this trial
        self._trajectory: list[dict[str, Any]] = []
        self._last_press: tuple[dict[str, Any], Observation] | None = None
        self._shown_memory: tuple[str, ...] = ()
        self._running: _Running | None = None
        self._results: list[str] = []
        self._bad_reply: tuple[str, str] | None = None
        self._keys: list[str] = []
        self._last_call: tuple[str, str] | None = None
        self._last_signature = ""
        self._streak = 0
        self._last_refine: int | None = None
        self._agent_refines = 0
        self._refinements: list[dict[str, Any]] = []
        self.refiner.reset()

    def start_trial(self, goal: str) -> None:
        self._trial += 1
        if not self.carry_across_trials:
            self.harness = self._new_harness()
        self._reset_trial(goal)

    def end_trial(self, outcome: TrialOutcome) -> Review:
        if self.harness_path is not None:
            self.harness_path.parent.mkdir(parents=True, exist_ok=True)
            self.harness_path.write_text(dumps(self.harness))
        return Review(
            trace={"trial": self._trial, "steps": self._steps, "refinements": self._refinements, "harness": self.harness.to_dict()}
        )

    # One decision

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        self._keys = [action.key for action in legal_actions]
        self.harness.bind(self._keys, parameters=self.skill_parameters)
        self._close_last_press(observation)
        turn = _Turn()
        if self._due():
            self._refine("schedule", "", None, turn)
        while len(turn.calls) < self.max_calls_per_decision:
            if self._running is not None:
                action = self._advance(observation, turn)
                if action is not None:
                    return Decision(action, self._trace(turn), tuple(turn.calls))
                continue
            self._plan(observation, turn)
        turn.events.append({"type": "gave_up", "reason": f"{self.max_calls_per_decision} model calls without a press"})
        return Decision(None, self._trace(turn), tuple(turn.calls))

    def _tool_names(self) -> list[str]:
        names = ["press"]
        if self.refine_skills or self.harness.library.skills:
            names.append("run_skill")
        if self.agent_meta_tools:
            names.append("write_memory")
            if self.refine_prompt or self.refine_skills or self.refine_memory:
                names.append("refine")
        return names

    # Planning

    def _plan(self, observation: Observation, turn: _Turn) -> None:
        snapshot, images = render_observation(observation)
        call = self.model.chat(
            [Message("system", self._system()), Message("user", self._user(snapshot, observation, turn), images)], think=False
        )
        turn.calls.append(call)
        try:
            tool, args, why = self._parse(call.output)
        except Invalid as problem:
            self._bad_reply = (str(problem), call.output.strip()[:300])
            self._note_failure(f"unusable reply: {problem}")
            turn.events.append({"type": "plan", "error": str(problem), "reply": call.output[:300]})
            return
        self._bad_reply = None
        turn.events.append({"type": "plan", "tool": tool, "args": args, "why": why})
        if tool == "write_memory":
            self._write_memory(args, turn)
        elif tool == "refine":
            self._agent_refine(args, turn)
        else:
            self._start(tool, args, why, observation)

    def _parse(self, output: str) -> tuple[str, dict[str, Any], str]:
        reply = first_json_object(output)
        if reply is None:
            raise Invalid("the reply has no JSON object")
        names = self._tool_names()
        tool = reply.get("tool")
        if tool not in names:
            raise Invalid(f"unknown tool {tool!r}; choose one of {names}")
        args = reply.get("args")
        args = {} if args is None else args
        if not isinstance(args, dict):
            raise Invalid('"args" must be an object')
        why = reply.get("why", "")
        try:
            return tool, getattr(self, f"_check_{tool}")(args), why[:300] if isinstance(why, str) else ""
        except Invalid as problem:
            raise Invalid(f"{tool}: {problem}") from None

    def _check_press(self, args: dict) -> dict[str, Any]:
        only_fields(args, {"key", "times"}, {"key"})
        by_upper = {key.upper(): key for key in self._keys}
        key = args["key"]
        if not isinstance(key, str) or key.strip().upper() not in by_upper:
            raise Invalid(f'"key" must be one of {self._keys}, not {key!r}')
        times = args.get("times", 1)
        if isinstance(times, bool) or not isinstance(times, int) or not 1 <= times <= self.max_press_times:
            raise Invalid(f'"times" must be a whole number from 1 to {self.max_press_times}, not {times!r}')
        return {"key": by_upper[key.strip().upper()], "times": times}

    def _check_run_skill(self, args: dict) -> dict[str, Any]:
        only_fields(args, {"name", "args"} if self.skill_parameters else {"name"}, {"name"})
        name, skills = args["name"], sorted(self.harness.library.skills)
        if not isinstance(name, str) or name not in skills:
            raise Invalid(f"no skill named {name!r}; the library has: {skills or 'no skills'}")
        if not self.skill_parameters:
            return {"name": name}
        given = args.get("args", [])
        if not isinstance(given, list):
            raise Invalid('"args" must be a list of values, one per parameter')
        params = self.harness.library.skills[name].program.params
        if len(given) != len(params):
            wanted = f"{len(params)} argument{'s' if len(params) != 1 else ''} ({', '.join(params)})" if params else "no arguments"
            raise Invalid(f"skill {name} takes {wanted} but got {len(given)}")
        return {"name": name, "args": given}

    def _check_write_memory(self, args: dict) -> dict[str, Any]:
        only_fields(args, {"title", "content", "importance"}, {"title", "content"})
        return {
            "title": text_field(args, "title", MAX_TITLE),
            "content": text_field(args, "content", MAX_CONTENT),
            "importance": importance_field(args),
        }

    def _check_refine(self, args: dict) -> dict[str, Any]:
        only_fields(args, {"reason", "num_steps"}, set())
        reason = args.get("reason", "")
        if not isinstance(reason, str):
            raise Invalid('"reason" must be text')
        num_steps = args.get("num_steps")
        if num_steps is not None and (isinstance(num_steps, bool) or not isinstance(num_steps, int) or not 1 <= num_steps <= MAX_WINDOW):
            raise Invalid(f'"num_steps" must be a whole number from 1 to {MAX_WINDOW}, not {num_steps!r}')
        return {"reason": reason[:300], "num_steps": num_steps}

    # Tools that take no game time

    def _write_memory(self, args: dict, turn: _Turn) -> None:
        entry = self.harness.find_memory(args["title"])
        evicted: list[str] = []
        if entry is not None:
            self.harness.revise_memory(entry, args["content"], args["importance"], step=self._steps)
            op = "updated"
        else:
            entry, evicted = self.harness.add_memory(args["title"], args["content"], args["importance"], step=self._steps, source="agent")
            op = "added"
        if entry is None:
            result = "write_memory: not saved; memory is full of entries that are more important than this one."
            turn.events.append({"type": "memory", "op": "refused", "title": args["title"]})
        else:
            result = f"write_memory: {op} {entry.id} ({entry.title!r}, importance {entry.importance})."
            turn.events.append({"type": "memory", "op": op, "id": entry.id, "evicted": evicted})
        self._results.append(result)

    def _agent_refine(self, args: dict, turn: _Turn) -> None:
        reason = None
        if self._last_refine is not None and self._steps - self._last_refine < self.refine_cooldown:
            reason = f"the last refinement was at step {self._last_refine}; wait until step {self._last_refine + self.refine_cooldown}"
        elif self._agent_refines >= self.max_agent_refines:
            reason = f"you have used all {self.max_agent_refines} refinements you may ask for this trial"
        if reason:
            turn.events.append({"type": "refine_refused", "reason": reason})
            self._results.append(f"refine: refused; {reason}.")
            return
        if self._refine("agent", args["reason"], args["num_steps"], turn):
            self._agent_refines += 1

    # Refinement

    def _due(self) -> bool:
        passes = self.refine_prompt or self.refine_skills or self.refine_memory
        return passes and self._steps != self._last_refine and should_refine(self._steps, **self.schedule)

    def _refine(self, trigger: str, reason: str, num_steps: int | None, turn: _Turn) -> bool:
        """Run the passes on the recent window; False when there was nothing to review."""
        count = min(num_steps or self.refine_window, MAX_WINDOW, self._steps)
        window, failures = self._window(count)
        if not window:
            turn.events.append({"type": "refine_skipped", "trigger": trigger, "reason": "no trajectory yet"})
            self._results.append("refine: nothing to review yet.")
            return False
        self._last_refine = self._steps
        refinement = self.refiner.refine(
            self.harness, goal=self._goal, window=window, failures=failures, count=count,
            fixed=self._system() + "\n\n" + self._base(), keys=self._keys, step=self._steps,
        )
        turn.calls.extend(refinement.calls)
        self._refinements.append({"step": self._steps, "trigger": trigger})
        turn.events.append(
            {
                "type": "refine", "trigger": trigger, "reason": reason, "step": self._steps, "window": count,
                "passes": {name: result.trace() for name, result in refinement.passes.items()},
            }
        )
        self._results.append(_summary(refinement))
        return True

    def _window(self, count: int) -> tuple[str, str]:
        """The last `count` steps as text for the Refiner, and the failures among them."""
        entries = [e for e in self._trajectory if e["step"] > self._steps - count]
        presses = [e for e in entries if "key" in e]
        lines, failures = [], []
        for entry in entries:
            if "key" not in entry:
                failures.append(f"- Step {entry['step']}: {entry['text']}")
                continue
            line = f"Step {entry['step']}: {entry['key']} via {entry['via']}"
            if entry["why"]:
                line += f' | reasoning: "{entry["why"]}"'
            line += f" | result: {entry.get('result', 'not seen yet')}"
            lines.append(line)
            index = presses.index(entry)
            if entry.get("same"):
                lines.append("  Issue: the state did not change after this press (possibly blocked)")
            if index >= 2 and presses[index - 2]["signature"] == entry["signature"]:
                lines.append("  Pattern: same state as 2 presses ago (possible loop)")
        failure_text = "## Tool Failures Detected\n" + "\n".join(failures) if failures else ""
        return "\n".join(lines), failure_text

    # Running a press or a skill

    def _start(self, tool: str, args: dict, why: str, observation: Observation) -> None:
        self._note_repeat(tool, args, observation)
        if tool == "press":
            label = f"press x{args['times']}"
            program = Program("press", (Tap(args["key"], args["times"]),), f"press {args['key']} x{args['times']}")
            skill = None
        else:
            skill, label = args["name"], f"skill {args['name']}"
            program = self.harness.library.skills[skill].program
            self.harness.meta[skill].uses += 1
            if args.get("args"):
                label += f" {json.dumps(args['args'])}"
        run = Run()
        steps = execute(program, self.harness.library.programs(), observation, run, None, args.get("args", ()))
        self._running = _Running(tool, label, program, steps, run, observation, why, skill, shown_memory=self._shown_memory)

    def _advance(self, observation: Observation, turn: _Turn) -> Action | None:
        """The next press of the running tool, or None once it has finished."""
        running = self._running
        try:
            action = running.steps.send(observation) if running.primed else next(running.steps)
            running.primed = True
        except StopIteration:
            error = ""
        except SkillError as failure:
            error = str(failure)
        else:
            self._steps += 1
            self.harness.clock += 1
            entry = {
                "step": self._steps, "key": action.key, "via": running.label, "why": running.why if running.run.presses == 1 else "",
                "signature": _signature(observation),
            }
            self._trajectory = (self._trajectory + [entry])[-400:]
            self._last_press = (entry, observation)
            self._results = []  # shown to the call that produced this press
            return action
        self._finish(running, observation, error, turn)
        return None

    def _finish(self, running: _Running, observation: Observation, error: str, turn: _Turn) -> None:
        self._running = None
        if self.utility_retrieval:
            self._credit(running, observation, error, turn)
        if running.skill is None and not error:
            return
        change = _state_change(running.start, observation)
        turn.events.append(
            {"type": "tool_end", "tool": running.tool, "skill": running.skill, "presses": running.run.presses, "error": error, "result": change}
        )
        if running.skill is None:
            return
        if error:
            self.harness.meta[running.skill].failures += 1
            self._note_failure(f"skill {running.skill} failed after {running.run.presses} presses: {error}")
            self._results.append(f"run_skill {running.label.removeprefix('skill ')}: FAILED after {running.run.presses} presses: {error}")
        else:
            self._results.append(f"run_skill {running.label.removeprefix('skill ')}: finished, {running.run.presses} presses. {change}")

    def _credit(self, running: _Running, observation: Observation, error: str, turn: _Turn) -> None:
        """MemRL's update after a tool run: r = 1 when the state moved and nothing failed, else 0, for each
        memory in the prompt that chose the run and for the skill it ran."""
        reward = float(not error and _signature(running.start) != _signature(observation))
        keys = [(f"memory:{entry_id}", "memory", entry_id) for entry_id in running.shown_memory]
        if running.skill is not None:
            keys.append((f"skill:{running.skill}", "skills", running.skill))
        for key, store, name in keys:
            change = self.harness.credit(key, reward)
            if change is not None:
                turn.events.append({"type": "utility_update", "store": store, "id": name, "reward": reward, "old": round(change[0], 4), "new": round(change[1], 4)})

    def _close_last_press(self, observation: Observation) -> None:
        if self._last_press is None:
            return
        entry, before = self._last_press
        self._last_press = None
        entry["result"] = _state_change(before, observation)
        entry["same"] = _signature(before) == _signature(observation)

    def _note_failure(self, text: str) -> None:
        self._trajectory.append({"step": self._steps, "text": text[:200]})

    def _note_repeat(self, tool: str, args: dict, observation: Observation) -> None:
        call, signature = (tool, json.dumps(args, sort_keys=True)), _signature(observation)
        same = call == self._last_call and signature == self._last_signature
        self._streak = self._streak + 1 if same else 1
        self._last_call, self._last_signature = call, signature

    # Prompts

    def _system(self) -> str:
        names = self._tool_names()
        text = prompts.system(
            "\n\n".join(self._tool_text(name) for name in names), names
        )
        if self._example_replies:
            text += "\n\n" + prompts.EXAMPLE_REPLIES.format(replies=self._example_replies)
        return text

    def _tool_text(self, name: str) -> str:
        text = prompts.RUN_SKILL_PARAMETERS if name == "run_skill" and self.skill_parameters else prompts.TOOLS[name]
        return text.format(keys="|".join(self._keys), max_times=self.max_press_times)

    def _shows_guidance(self) -> bool:
        return bool(self.harness.guidance) or self.refine_prompt

    def _base(self) -> str:
        return prompts.base(self._tool_names(), learned=self._shows_guidance())

    def _user(self, snapshot: str, observation: Observation, turn: _Turn) -> str:
        pressed = [e for e in self._trajectory if "key" in e][-self.history_turns :]
        recent = "\n".join(f"[{e['step']}] {e['key']} ({e['via']}) -> {e.get('result', 'not seen yet')}" for e in pressed) or "No previous actions recorded."
        parts = [f"# Step: {self._steps + 1}", self._base()]
        if self._shows_guidance():
            parts.append("## Learned guidance\n" + (self.harness.guidance or "(none yet)"))
        parts += ["## CONTEXT FOR THIS STEP", f"### GOAL\n{self._goal}", f"### RECENT ACTIONS (last {len(pressed)}):\n{recent}"]
        if self._results:
            parts.append("### RESULTS FROM PREVIOUS STEP\n" + "\n".join(self._results))
        parts.append("### CURRENT STATE\n" + snapshot)
        self.harness.retrieved = {}
        parts.append("### LONG-TERM MEMORY\n" + self.harness.memory_text(self.memory_shown, rank=True))
        if "run_skill" in self._tool_names():
            parts.append("### SKILL LIBRARY\n" + self.harness.skill_text(rank=self.skills_shown))
        self._shown_memory = tuple(row.key for row in self.harness.retrieved.get("memory", ()) if row.shown)
        for store, rows in self.harness.retrieved.items():
            turn.events.append(trace_event(store, "", rows))
        if self._streak >= self.repeat_limit and self._last_call is not None:
            tool, args = self._last_call
            turn.events.append({"type": "repeat_note", "call": f"{tool} {args}", "count": self._streak})
            parts.append(prompts.REPEAT_NOTE.format(call=f"{tool} {args}", count=self._streak))
        if self._bad_reply is not None:
            parts.append(prompts.BAD_REPLY.format(error=self._bad_reply[0], reply=self._bad_reply[1]))
        return "\n\n".join(parts)

    def _trace(self, turn: _Turn) -> dict[str, Any]:
        running = self._running
        return {
            "tool": running.tool if running else None,
            "skill": running.skill if running else None,
            "press": running.run.presses if running else None,
            "step": self._steps,
            "events": turn.events,
        }


def _summary(refinement) -> str:
    """What changed, for the next prompt (the original's evolution summary)."""
    lines = [prompts.REFINED]
    for name, label in (("skills", "Skills"), ("memory", "Memory")):
        result = refinement.passes.get(name)
        if result is None:
            continue
        added = [a["id"] for a in result.applied if a["op"] == "add"]
        updated = [a["id"] for a in result.applied if a["op"] == "update"]
        if added or updated:
            lines.append(f"  {label}: " + "; ".join(part for part in (f"new: {added}" if added else "", f"updated: {updated}" if updated else "") if part))
            if result.analysis:
                lines.append(f"    Reason: {result.analysis[:200]}")
    prompt = refinement.passes.get("prompt")
    if prompt is not None and prompt.applied:
        lines.append("  Learned guidance: rewritten from your recent steps")
    rejected = sum(len(r.rejected) for r in refinement.passes.values())
    if rejected:
        lines.append(f"  {rejected} proposed edits were rejected as invalid.")
    if len(lines) == 1:
        lines.append("  No changes.")
    else:
        lines.append("Review the updated learned guidance, SKILL LIBRARY, and LONG-TERM MEMORY below.")
    return "\n".join(lines)


def _signature(observation: Observation) -> str:
    """A fingerprint of what the player can see, to tell whether anything changed."""
    if observation.state is not None:
        return hashlib.sha1(json.dumps(observation.state, sort_keys=True, default=str).encode()).hexdigest()[:12]
    if observation.frame is not None:
        return hashlib.sha1(observation.frame.rgba).hexdigest()[:12]
    return ""


def _short(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= VALUE_CHARS else text[: VALUE_CHARS - 3] + "..."


def _state_change(before: Observation, after: Observation) -> str:
    if before.state is None or after.state is None:
        return "state not available in this observation mode"
    changes = [
        f"{key}: {_short(before.state.get(key))} -> {_short(after.state.get(key))}"
        for key in sorted(set(before.state) | set(after.state))
        if before.state.get(key) != after.state.get(key)
    ]
    return "; ".join(changes) or "no state change"
