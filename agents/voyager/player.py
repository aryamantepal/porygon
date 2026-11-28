"""Voyager: automatic curriculum, skill library, iterative prompting (Wang et al., 2023).

The loop follows voyager/voyager.py in github.com/MineDojo/Voyager (MIT):

    curriculum proposes a task
      -> action agent writes a program (retrieved skills in its prompt)
      -> the program runs
      -> critic checks the task (self-verification)
      -> success: describe the program and add it to the skill library
         failure: rewrite with the last code, execution error, environment
                  feedback, and critique, up to `max_attempts` times

Our runner asks for one action per step, so a program runs one press per
`decide` call; model calls happen only between programs. Every press is an
environment step the runner counts and logs, and the trace says which task,
program, and attempt it belongs to.

Mechanisms, each switchable for ablations:
- curriculum: off means the task is always the trial's goal.
- skill_library: off means no retrieval, no `call`, nothing stored.
- iterative_prompting: off means one attempt per task, with no feedback.
- self_verification: off means a program that runs without error counts as success.
- few_shot: `examples` programs from another task are shown in the action-agent
  prompt after the retrieved skills, as Voyager's original prompt shows example
  code. They are not stored in the library and cannot be called.
- tools (`action_interface="tools"`): the skill language gains `go_to <place_id>`,
  `go_to [x, y]`, and `talk_to <place_id>`, which expand through the Crystal
  harness's navigator one press per `decide`, each press counted against
  MAX_PRESSES and logged by the runner (`agents/voyager/skills.py`). A failed tool
  is an execution error fed back to the next attempt, and what each tool
  statement returned joins the environment feedback. Without a grid and position
  in the observation the prompt lists no tool statements and the trace says so.
- specialists (tools only): code handles busy, dialogue, and battle before any
  model call (`agents/harness/specialists.py`), between programs and during
  them. A battle that starts while a program runs ends it with
  "interrupted: battle started" (the critic is skipped and the attempt is spent);
  the battle specialist then plays the battle and the task resumes. Dialogue and
  busy do not interrupt a program, so `talk_to` can finish in a conversation;
  the specialist presses through it once the program is over.
- skill_parameters (SkillWeaver, Zheng et al. 2025, arXiv:2504.07079): a skill
  may declare parameters (`skill walk_through door`) and use `$door` where a
  place id or a JSON value goes (`go_to $door`, `until map != $start_map (max 10):`,
  `call other_skill $door`), so one skill serves any target. `call` passes
  arguments positionally; a wrong count is an execution error. The code block
  may hold several skills: the last one runs with no arguments and calls the
  earlier ones with arguments, so a new parameterized skill is written and used
  in one reply. There is no separate `run` form: the last skill's body is the
  invocation. Earlier skills are stored (when the task succeeds) beside the last.
- test_skills (needs skill_parameters and the library): SkillWeaver tests a new
  skill in a sandbox before keeping it. An agent here cannot restore the
  environment's state, so a skill that takes parameters instead enters the
  library as a "candidate", with the argument lists of the run that got it
  accepted, and is promoted to "verified" the first time a later program that
  calls it with different arguments succeeds (critic accepts, or no error when
  self-verification is off, and the run had no error). Uses and successes are
  counted per skill. Retrieval scales a candidate's score down, so a verified
  skill wins a near tie, and the prompt marks a candidate with a comment line.
  A parameterized skill the program never called, or whose run ended in an
  error, is not stored. Writing a stored skill again with the same source is a
  use of it, not a new version; changed source is a new, untested candidate.
  The test is weaker than SkillWeaver's: the critic judges the whole program's
  final state, not the skill alone, and it runs on whatever state the world is in.
- utility_retrieval (MemRL, Zhang et al. 2026, arXiv:2601.03192; needs the library):
  retrieval keeps the bag-of-words cosine as phase 1 (the top `candidates`), then
  ranks those by a learned utility Q per skill, `Q <- Q + alpha * (r - Q)`
  (`agents/retrieval.py`). The reward is per program run, for each library skill the
  program called: r = 1 when the program succeeded by the test this player already
  uses, and r = 0 when it did not. The test is the critic's acceptance with
  self_verification on (a model reading the before and after snapshots, not the
  program's own claim), and "ran without an error" with it off; a run with an
  execution error is always 0. A program a battle interrupted is not scored, and a
  skill the program did not call is not scored, even if it was in the prompt. This
  is the agent's signal. Agents cannot call `get_progress`, so it is a stand-in for
  progress read from the game, and a critic that accepts too easily rewards skills
  that did nothing. A new version of a skill starts at `q_init` again.
- memory_decay (MemoryBank, Zhong et al. 2023, arXiv:2305.10250; needs the library):
  a skill's retention is `exp(-t / S)`, t the decisions since it was added or last
  shown in a prompt, S its strength, multiplied by `growth` on each showing. A skill
  below `threshold` is not retrieved, and retention multiplies the cosine. Both
  switches log a `retrieval` event with each candidate's similarity, utility,
  retention, and score, and a `utility_update` event per reward. With a
  `library_path` the Q, strength, and last access of each skill are saved beside
  it, and only while a switch is on. Deviations from both papers are listed in
  `agents/retrieval.py`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from agents.common import first_json_object, render_observation
from agents.examples import Examples
from agents.harness import tools
from agents.harness.calls import check_interface
from agents.harness.specialists import Specialists
from agents.models import ChatModel, Message, ModelCall
from agents.protocol import Decision, Review, TrialOutcome
from agents.retrieval import MemoryParams, Policy, trace_event
from agents.voyager import prompts
from agents.voyager.skills import Program, Run, SkillError, SkillInterrupted, SkillLibrary, execute, parse_all
from environments.protocol import Action, Observation, Press

MAX_TOOL_NOTES = 10  # tool results kept in a program's environment feedback
_CODE = re.compile(r"```(?:skill)?\s*\n(.*?)```", re.DOTALL)
_SECTION = re.compile(r"^(Explain|Plan|Code)\s*:", re.IGNORECASE | re.MULTILINE)


@dataclass
class Task:
    text: str
    context: str = ""
    attempt: int = 0
    last_code: str = ""
    last_error: str = ""
    last_feedback: str = ""
    last_critique: str = ""


@dataclass
class Execution:
    program: Program
    steps: Generator[Action, Observation, None]
    run: Run
    start: Observation
    primed: bool = False
    definitions: tuple[Program, ...] = ()  # skills defined earlier in the same code block (`skill_parameters`)

    @property
    def source(self) -> str:
        """The whole code block, so a rewrite sees the definitions too."""
        return "\n\n".join(p.source for p in (*self.definitions, self.program))


@dataclass
class _Turn:
    """Model calls and events produced inside one `decide` call."""

    calls: list[ModelCall] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class Voyager:
    name = "voyager"

    def __init__(
        self,
        model: ChatModel,
        *,
        curriculum: bool = True,
        skill_library: bool = True,
        iterative_prompting: bool = True,
        self_verification: bool = True,
        max_attempts: int = 4,
        retrieval_k: int = 5,
        library_path: Path | None = None,
        max_calls_per_decision: int = 12,
        examples: Examples | None = None,
        action_interface: str = "buttons",
        specialists: bool = False,
        skill_parameters: bool = False,
        test_skills: bool = False,
        utility_retrieval: bool = False,
        memory_decay: bool = False,
        memory_params: MemoryParams | None = None,
    ) -> None:
        self.model = model
        self.tools = check_interface(action_interface, specialists)
        self.specialists = specialists
        if test_skills and not (skill_parameters and skill_library):
            raise ValueError("test_skills only works with skill_parameters and the skill library: only a skill with parameters is tested")
        if (utility_retrieval or memory_decay) and not skill_library:
            raise ValueError("utility_retrieval and memory_decay rank the skill library, so they need skill_library")
        self.skill_parameters = skill_parameters
        self.test_skills = test_skills
        self.utility_retrieval, self.memory_decay = utility_retrieval, memory_decay
        self.policy = Policy(utility=utility_retrieval, decay=memory_decay, params=memory_params)
        self._specialists = Specialists(model)
        self._example_programs = examples.programs_text() if examples else ""
        self.curriculum = curriculum
        self.skill_library = skill_library
        self.iterative_prompting = iterative_prompting
        self.self_verification = self_verification
        self.max_attempts = max_attempts if iterative_prompting else 1
        self.retrieval_k = retrieval_k
        self.max_calls_per_decision = max_calls_per_decision
        self.library = SkillLibrary(library_path if skill_library else None, self.policy)
        self._clock = self.library.clock  # decisions so far, over the player's life: the time memory_decay counts in
        # The curriculum's memory is lifelong, as in Voyager: it carries across trials.
        self.completed_tasks: list[str] = []
        self.failed_tasks: list[str] = []
        self._goal = ""
        self._task: Task | None = None
        self._execution: Execution | None = None
        self._special: tools.ToolRun | None = None

    def mechanisms(self) -> dict[str, bool]:
        return {
            "curriculum": self.curriculum,
            "skill_library": self.skill_library,
            "iterative_prompting": self.iterative_prompting,
            "self_verification": self.self_verification,
            "few_shot": bool(self._example_programs),
            "thinking": False,
            "tools": self.tools,
            "specialists": self.specialists,
            "skill_parameters": self.skill_parameters,
            "test_skills": self.test_skills,
            "utility_retrieval": self.utility_retrieval,
            "memory_decay": self.memory_decay,
        }

    def start_trial(self, goal: str) -> None:
        self._goal = goal
        self._task = None
        self._execution = None
        self._special = None
        self._specialists.reset()

    def end_trial(self, outcome: TrialOutcome) -> Review:
        trace = {
            "completed_tasks": list(self.completed_tasks),
            "failed_tasks": list(self.failed_tasks),
            "skills": sorted(self.library.skills),
        }
        if self.test_skills:
            trace["skill_status"] = {
                name: {"status": s.status, "uses": s.uses, "successes": s.successes, "proven_with": list(s.proven_with)}
                for name, s in sorted(self.library.skills.items())
            }
        if self.policy.active:
            trace["memory"] = {
                name: stats.to_dict(utility=self.utility_retrieval, decay=self.memory_decay) for name, stats in sorted(self.library.stats.items())
            }
        return Review(trace=trace)

    # One step

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        keys = [action.key for action in legal_actions]
        self.library.bind(keys, tools=self.tools, parameters=self.skill_parameters)
        if self.policy.active:
            self._clock += 1
            self.library.clock = self._clock
        turn = _Turn()
        while len(turn.calls) < self.max_calls_per_decision:
            if self._special is not None:
                step = self._special.advance(observation)
                if isinstance(step, Press):
                    return Decision(step, self._trace(turn), tuple(turn.calls))
                self._special = None
            if self._execution is not None:
                action = self._advance(observation, turn)
                if action is not None:
                    return Decision(action, self._trace(turn), tuple(turn.calls))
                continue
            if self.specialists:
                handled = self._specialists.route(observation.state, tools.controls(observation.state, set(keys)))
                if handled is not None:
                    self._special = handled.run
                    turn.calls.extend(handled.calls)
                    turn.events.extend(handled.events)
                    continue
            if self._task is None:
                self._task = self._propose_task(observation, turn)
            self._write_program(observation, keys, turn)
        turn.events.append({"type": "gave_up", "reason": f"{self.max_calls_per_decision} model calls without a press"})
        return Decision(None, self._trace(turn), tuple(turn.calls))

    def _advance(self, observation: Observation, turn: _Turn) -> Action | None:
        """The next press of the running program, or None once it has finished."""
        execution = self._execution
        interrupted = False
        if self.specialists and execution.primed and (observation.state or {}).get("context") == "battle":
            execution.steps.close()
            error, interrupted = "interrupted: battle started", True
        else:
            try:
                if execution.primed:
                    return execution.steps.send(observation)
                execution.primed = True
                return next(execution.steps)
            except StopIteration:
                error = ""
            except SkillInterrupted as failure:
                error, interrupted = str(failure), True
            except SkillError as failure:
                error = str(failure)
        self._finish_program(observation, error, turn, interrupted)
        return None

    # Curriculum

    def _propose_task(self, observation: Observation, turn: _Turn) -> Task:
        if not self.curriculum:
            turn.events.append({"type": "task", "task": self._goal, "source": "goal (curriculum off)"})
            return Task(self._goal)
        snapshot, images = render_observation(observation)
        user = (
            f"{snapshot}\n"
            f"Completed tasks so far: {', '.join(self.completed_tasks) or 'None'}\n"
            f"Failed tasks that are too hard: {', '.join(self.failed_tasks) or 'None'}"
        )
        call = self.model.chat(
            [Message("system", prompts.CURRICULUM.format(goal=self._goal)), Message("user", user, images)], think=False
        )
        turn.calls.append(call)
        task = _field(call.output, "Task")
        reasoning = _field(call.output, "Reasoning")
        if not task:
            turn.events.append({"type": "task", "task": self._goal, "source": "goal (curriculum reply had no Task)", "reasoning": reasoning})
            return Task(self._goal)
        turn.events.append({"type": "task", "task": task, "source": "curriculum", "reasoning": reasoning})
        return Task(task)

    # Action agent

    def _write_program(self, observation: Observation, keys: Sequence[str], turn: _Turn) -> None:
        task = self._task
        query = f"{task.text} {task.context}"
        rows = []
        if not self.skill_library:
            retrieved = []
        elif self.policy.active:
            retrieved, rows = self.library.retrieve_ranked(query, self.retrieval_k)
        else:
            retrieved = self.library.retrieve(query, self.retrieval_k)
        turn.events.append({"type": "retrieved_skills", "skills": [s.program.name for s in retrieved]})
        if self.policy.active:
            turn.events.append(trace_event("skills", query, rows))
        programs = "\n\n".join(s.shown for s in retrieved) or "(none yet)"
        if self._example_programs:
            programs += "\n\n" + prompts.EXAMPLE_PROGRAMS.format(programs=self._example_programs)
        offered = self.tools and tools.navigable(observation.state)
        if self.tools and not offered:
            turn.events.append({"type": "tools_offered", "tools": [], "reason": "this observation has no grid and position, so the language lists no tool statements"})
        system = prompts.ACTION_TEMPLATE.format(
            language=prompts.skill_language(tools=bool(offered), parameters=self.skill_parameters, testing=self.test_skills).replace("<ACTION>", "|".join(keys)),
            programs=programs,
            response_format=prompts.ACTION_RESPONSE_FORMAT,
        )
        snapshot, images = render_observation(observation)
        feedback = self.iterative_prompting
        user = (
            f"Code from the last round: {task.last_code if feedback and task.last_code else 'No code in the first round'}\n"
            f"Execution error: {task.last_error if feedback and task.last_error else 'No error'}\n"
            f"Environment feedback: {task.last_feedback if feedback and task.last_feedback else 'None'}\n"
            f"{snapshot}\n"
            f"Task: {task.text}\n"
            f"Context: {task.context or 'None'}\n"
            f"Critique: {task.last_critique if feedback and task.last_critique else 'None'}"
        )
        call = self.model.chat([Message("system", system), Message("user", user, images)], think=False)
        turn.calls.append(call)
        task.attempt += 1
        explain, plan, code = _sections(call.output)
        try:
            if code is None:
                raise SkillError("the reply has no ```skill code block")
            *definitions, program = parse_all(code, keys, tools=self.tools, parameters=self.skill_parameters)
        except SkillError as failure:
            turn.events.append({"type": "program", "attempt": task.attempt, "explain": explain, "plan": plan, "source": code, "error": str(failure)})
            self._attempt_failed(task, code or "", str(failure), "", "", turn)
            return
        event = {"type": "program", "attempt": task.attempt, "name": program.name, "explain": explain, "plan": plan, "source": program.source}
        if definitions:
            event["definitions"] = {d.name: d.source for d in definitions}
        turn.events.append(event)
        run = Run()
        # skills defined in this block are callable now, and stored only if the task succeeds
        library = {**self.library.programs(), **{d.name: d for d in definitions}} if self.skill_library else None
        steps = execute(program, library, observation, run, keys if self.tools else None)
        self._execution = Execution(program, steps, run, observation, definitions=tuple(definitions))

    # Self-verification

    def _finish_program(self, observation: Observation, error: str, turn: _Turn, interrupted: bool = False) -> None:
        execution, task = self._execution, self._task
        self._execution = None
        feedback = _state_change(execution.start, observation) + f"; {execution.run.presses} presses"
        if execution.run.notes:
            feedback += "; tools: " + "; ".join(execution.run.notes[-MAX_TOOL_NOTES:])
        end = {"type": "program_end", "name": execution.program.name, "presses": execution.run.presses, "error": error, "feedback": feedback}
        turn.events.append({**end, "interrupted": True} if interrupted else end)
        if interrupted:
            # the game changed context under the program; nothing to verify, and a model call mid-battle would only guess
            success, critique = False, ""
        elif self.self_verification:
            success, critique = self._critic(execution.start, observation, task, turn)
        else:
            success, critique = not error, ""
        if self.test_skills and not interrupted:
            self._record_uses(execution, success and not error, turn)
        if self.utility_retrieval and not interrupted:
            self._credit_skills(execution, success and not error, turn)
        if success:
            self.completed_tasks.append(task.text)
            turn.events.append({"type": "task_completed", "task": task.text})
            if self.skill_library:
                if self.skill_parameters:
                    self._add_skills(execution, task, error, turn)
                else:
                    self._add_skill(execution.program, task, turn)
            self._task = None
        else:
            self._attempt_failed(task, execution.source, error, feedback, critique, turn)

    def _critic(self, before: Observation, after: Observation, task: Task, turn: _Turn) -> tuple[bool, str]:
        before_text, _ = render_observation(before)
        after_text, images = render_observation(after)
        user = (
            f"State before: {_state_only(before_text)}\n"
            f"State after: {_state_only(after_text)}\n"
            f"Screen: {'attached' if images else 'not available'}\n"
            f"Task: {task.text}\n"
            f"Context: {task.context or 'None'}"
        )
        call = self.model.chat([Message("system", prompts.CRITIC), Message("user", user, images)], think=False)
        turn.calls.append(call)
        verdict = first_json_object(call.output)
        if not isinstance(verdict, dict) or not isinstance(verdict.get("success"), bool):
            turn.events.append({"type": "critic", "success": False, "error": "reply was not the requested JSON"})
            return False, "The critic's reply could not be parsed."
        turn.events.append({"type": "critic", "success": verdict["success"], "reasoning": verdict.get("reasoning", ""), "critique": verdict.get("critique", "")})
        return verdict["success"], str(verdict.get("critique", ""))

    def _attempt_failed(self, task: Task, code: str, error: str, feedback: str, critique: str, turn: _Turn) -> None:
        task.last_code, task.last_error, task.last_feedback, task.last_critique = code, error, feedback, critique
        if task.attempt >= self.max_attempts:
            self.failed_tasks.append(task.text)
            turn.events.append({"type": "task_failed", "task": task.text, "attempts": task.attempt})
            self._task = None

    # Skill library

    def _add_skill(self, program: Program, task: Task, turn: _Turn, *, status: str = "verified", proven_with: Sequence[Sequence[Any]] = ()) -> None:
        call = self.model.chat(
            [Message("system", prompts.SKILL_DESCRIPTION), Message("user", f"{program.source}\n\nThe skill is `{program.name}`.")],
            think=False,
        )
        turn.calls.append(call)
        description = call.output.strip().splitlines()[0] if call.output.strip() else task.text
        skill = self.library.add(program, description, task.text, status=status, proven_with=proven_with)
        event = {"type": "skill_added", "name": program.name, "version": skill.version, "description": description}
        if self.skill_parameters:
            event |= {"params": list(program.params), "status": skill.status}
        turn.events.append(event)

    def _fresh(self, execution: Execution) -> set[str]:
        """The block's definitions that are new or changed. Repeating a stored skill's exact source is a use of it, not a new version."""
        stored = self.library.skills
        return {d.name for d in execution.definitions if not (self.test_skills and d.name in stored and stored[d.name].program.source == d.source)}

    def _add_skills(self, execution: Execution, task: Task, error: str, turn: _Turn) -> None:
        """Store the block's definitions, then its program. With `test_skills` a skill that takes
        parameters enters as a candidate with the arguments this run called it with, and is
        not stored when the run never called it or ended in an error."""
        fresh = self._fresh(execution)
        for skill in (*execution.definitions, execution.program):
            called = [args for name, args in execution.run.calls if name == skill.name]
            if skill is not execution.program and skill.name not in fresh:
                continue  # unchanged, and `_record_uses` counted this run as a use of it
            if not (self.test_skills and skill.params):
                self._add_skill(skill, task, turn)
            elif called and not error:
                self._add_skill(skill, task, turn, status="candidate", proven_with=called)
            else:
                reason = "the program ended in an error" if called else "the program never called it"
                turn.events.append({"type": "skill_untested", "name": skill.name, "reason": f"{reason}, so it was not stored"})

    def _record_uses(self, execution: Execution, success: bool, turn: _Turn) -> None:
        """Count each library skill the run called, once per distinct arguments, and promote candidates."""
        fresh = self._fresh(execution)
        for name, args in dict.fromkeys((n, json.dumps(a)) for n, a in execution.run.calls if n not in fresh and n in self.library.skills):
            skill, promoted = self.library.record_use(name, json.loads(args), success)
            turn.events.append(
                {"type": "skill_used", "name": name, "args": json.loads(args), "success": success, "status": skill.status, "uses": skill.uses, "successes": skill.successes}
            )
            if promoted:
                turn.events.append({"type": "skill_promoted", "name": name, "args": json.loads(args), "successes": skill.successes})

    def _credit_skills(self, execution: Execution, success: bool, turn: _Turn) -> None:
        """MemRL's update for each library skill the run called, once per run: r is 1 when the program succeeded, else 0."""
        defined = {d.name for d in execution.definitions}
        for name in dict.fromkeys(n for n, _ in execution.run.calls):
            change = None if name in defined else self.library.credit(name, float(success))
            if change is not None:
                turn.events.append({"type": "utility_update", "store": "skills", "id": name, "reward": float(success), "old": round(change[0], 4), "new": round(change[1], 4)})

    def _trace(self, turn: _Turn) -> dict[str, Any]:
        execution = self._execution
        trace = {
            "task": self._task.text if self._task else None,
            "attempt": self._task.attempt if self._task else None,
            "skill": execution.program.name if execution else None,
            "press": execution.run.presses if execution else None,
            "events": turn.events,
        }
        if self.tools:
            trace["statement"] = (execution.run.current or None) if execution else None
            special = self._special
            trace["special"] = {"handler": special.handler, "tool": special.tool, "args": special.args, "press": special.presses} if special else None
        return trace


def _field(text: str, name: str) -> str:
    match = re.search(rf"^\s*{name}\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _sections(text: str) -> tuple[str, str, str | None]:
    code = _CODE.search(text)
    parts: dict[str, str] = {}
    marks = list(_SECTION.finditer(text))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        parts[mark.group(1).lower()] = text[mark.end() : end].strip()
    return parts.get("explain", ""), parts.get("plan", ""), code.group(1) if code else None


def _state_only(snapshot: str) -> str:
    lines = [line for line in snapshot.splitlines() if line.startswith("State: ")]
    return lines[0][len("State: ") :] if lines else "(no structured state in this observation mode)"


def _state_change(before: Observation, after: Observation) -> str:
    if before.state is None or after.state is None:
        return "state not available in this observation mode"
    changes = [
        f"{key}: {json.dumps(before.state.get(key))} -> {json.dumps(after.state.get(key))}"
        for key in sorted(set(before.state) | set(after.state))
        if before.state.get(key) != after.state.get(key)
    ]
    return "; ".join(changes) or "no state change"
