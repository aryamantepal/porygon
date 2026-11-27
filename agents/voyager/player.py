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
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from agents.common import render_observation
from agents.models import ChatModel, Message, ModelCall
from agents.protocol import Decision, Review, TrialOutcome
from agents.voyager import prompts
from agents.voyager.skills import Program, Run, SkillError, SkillLibrary, execute, parse
from environments.protocol import Action, Observation

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
    ) -> None:
        self.model = model
        self.curriculum = curriculum
        self.skill_library = skill_library
        self.iterative_prompting = iterative_prompting
        self.self_verification = self_verification
        self.max_attempts = max_attempts if iterative_prompting else 1
        self.retrieval_k = retrieval_k
        self.max_calls_per_decision = max_calls_per_decision
        self.library = SkillLibrary(library_path if skill_library else None)
        # The curriculum's memory is lifelong, as in Voyager: it carries across trials.
        self.completed_tasks: list[str] = []
        self.failed_tasks: list[str] = []
        self._goal = ""
        self._task: Task | None = None
        self._execution: Execution | None = None

    def mechanisms(self) -> dict[str, bool]:
        return {
            "curriculum": self.curriculum,
            "skill_library": self.skill_library,
            "iterative_prompting": self.iterative_prompting,
            "self_verification": self.self_verification,
            "thinking": False,
        }

    def start_trial(self, goal: str) -> None:
        self._goal = goal
        self._task = None
        self._execution = None

    def end_trial(self, outcome: TrialOutcome) -> Review:
        return Review(
            trace={
                "completed_tasks": list(self.completed_tasks),
                "failed_tasks": list(self.failed_tasks),
                "skills": sorted(self.library.skills),
            }
        )

    # One step

    def decide(self, observation: Observation, legal_actions: Sequence[Action]) -> Decision:
        keys = [action.key for action in legal_actions]
        self.library.bind(keys)
        turn = _Turn()
        while len(turn.calls) < self.max_calls_per_decision:
            if self._execution is not None:
                action = self._advance(observation, turn)
                if action is not None:
                    return Decision(action, self._trace(turn), tuple(turn.calls))
                continue
            if self._task is None:
                self._task = self._propose_task(observation, turn)
            self._write_program(observation, keys, turn)
        turn.events.append({"type": "gave_up", "reason": f"{self.max_calls_per_decision} model calls without a press"})
        return Decision(None, self._trace(turn), tuple(turn.calls))

    def _advance(self, observation: Observation, turn: _Turn) -> Action | None:
        """The next press of the running program, or None once it has finished."""
        execution = self._execution
        try:
            if execution.primed:
                return execution.steps.send(observation)
            execution.primed = True
            return next(execution.steps)
        except StopIteration:
            error = ""
        except SkillError as failure:
            error = str(failure)
        self._finish_program(observation, error, turn)
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
        retrieved = self.library.retrieve(f"{task.text} {task.context}", self.retrieval_k) if self.skill_library else []
        turn.events.append({"type": "retrieved_skills", "skills": [s.program.name for s in retrieved]})
        programs = "\n\n".join(s.program.source for s in retrieved) or "(none yet)"
        system = prompts.ACTION_TEMPLATE.format(
            language=prompts.SKILL_LANGUAGE.replace("<ACTION>", "|".join(keys)),
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
            program = parse(code, keys)
        except SkillError as failure:
            turn.events.append({"type": "program", "attempt": task.attempt, "explain": explain, "plan": plan, "source": code, "error": str(failure)})
            self._attempt_failed(task, code or "", str(failure), "", "", turn)
            return
        turn.events.append({"type": "program", "attempt": task.attempt, "name": program.name, "explain": explain, "plan": plan, "source": program.source})
        run = Run()
        library = self.library.programs() if self.skill_library else None
        self._execution = Execution(program, execute(program, library, observation, run), run, observation)

    # Self-verification

    def _finish_program(self, observation: Observation, error: str, turn: _Turn) -> None:
        execution, task = self._execution, self._task
        self._execution = None
        feedback = _state_change(execution.start, observation) + f"; {execution.run.presses} presses"
        turn.events.append({"type": "program_end", "name": execution.program.name, "presses": execution.run.presses, "error": error, "feedback": feedback})
        if self.self_verification:
            success, critique = self._critic(execution.start, observation, task, turn)
        else:
            success, critique = not error, ""
        if success:
            self.completed_tasks.append(task.text)
            turn.events.append({"type": "task_completed", "task": task.text})
            if self.skill_library:
                self._add_skill(execution.program, task, turn)
            self._task = None
        else:
            self._attempt_failed(task, execution.program.source, error, feedback, critique, turn)

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
        verdict = _json_object(call.output)
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

    def _add_skill(self, program: Program, task: Task, turn: _Turn) -> None:
        call = self.model.chat(
            [Message("system", prompts.SKILL_DESCRIPTION), Message("user", f"{program.source}\n\nThe skill is `{program.name}`.")],
            think=False,
        )
        turn.calls.append(call)
        description = call.output.strip().splitlines()[0] if call.output.strip() else task.text
        skill = self.library.add(program, description, task.text)
        turn.events.append({"type": "skill_added", "name": program.name, "version": skill.version, "description": description})

    def _trace(self, turn: _Turn) -> dict[str, Any]:
        execution = self._execution
        return {
            "task": self._task.text if self._task else None,
            "attempt": self._task.attempt if self._task else None,
            "skill": execution.program.name if execution else None,
            "press": execution.run.presses if execution else None,
            "events": turn.events,
        }


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


def _json_object(text: str) -> Any:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


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
