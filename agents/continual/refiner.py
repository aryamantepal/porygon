"""The Refiner: a text-only model call per pass that edits the harness in place.

Ported from HarnessEvolver in github.com/sethkarten/continual-harness (MIT).
Each pass is independent: it asks for its own reply, validates every edit on its
own, and a failure in one pass or one edit never blocks the others. The paper's
failure case was a malformed call the agent repeated for 1,003 turns, so nothing
here trusts a reply: an edit that does not validate is rejected, logged with its
reason, and shown to the Refiner on its next skills pass.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agents.common import first_json_object
from agents.continual import prompts
from agents.continual.harness import HarnessState
from agents.continual.validate import (
    MAX_CODE,
    MAX_CONTENT,
    MAX_DESCRIPTION,
    MAX_EDITS,
    MAX_GUIDANCE,
    MAX_PATH,
    MAX_TITLE,
    NAME,
    Invalid,
    effectiveness_field,
    importance_field,
    only_fields,
    text_field,
)
from agents.models import ChatModel, Message, ModelCall
from agents.voyager.prompts import EXAMPLE_PROGRAMS, SKILL_LANGUAGE
from agents.voyager.skills import SkillError, parse

_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.IGNORECASE)


@dataclass
class PassResult:
    name: str
    analysis: str = ""
    applied: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""  # the whole pass failed: model error, reply not usable

    def trace(self) -> dict[str, Any]:
        return {"analysis": self.analysis, "applied": self.applied, "rejected": self.rejected, "error": self.error}


@dataclass
class Refinement:
    calls: list[ModelCall] = field(default_factory=list)
    passes: dict[str, PassResult] = field(default_factory=dict)


class Refiner:
    def __init__(
        self, model: ChatModel, *, prompt: bool = True, skills: bool = True, memory: bool = True, example_programs: str = ""
    ) -> None:
        self.model = model
        self.prompt, self.skills, self.memory = prompt, skills, memory
        self.example_programs = example_programs
        self.last_skill_rejections: list[str] = []

    def reset(self) -> None:
        self.last_skill_rejections = []

    def refine(
        self, harness: HarnessState, *, goal: str, window: str, failures: str, count: int,
        fixed: str, keys: Sequence[str], step: int,
    ) -> Refinement:
        refinement = Refinement()
        context = {"goal": goal, "window": window, "count": count}
        passes = [
            (self.prompt, "prompt", lambda: self._prompt_pass(harness, refinement, fixed=fixed, **context)),
            (self.skills, "skills", lambda: self._skills_pass(harness, refinement, keys=keys, failures=failures, **context)),
            (self.memory, "memory", lambda: self._memory_pass(harness, refinement, step=step, **context)),
        ]
        for enabled, name, run in passes:
            if not enabled:
                continue
            try:
                refinement.passes[name] = run()
            except Invalid as problem:  # the reply was unusable as a whole
                refinement.passes[name] = PassResult(name, error=str(problem))
            except Exception as failure:  # one broken pass must not block the others
                refinement.passes[name] = PassResult(name, error=f"{type(failure).__name__}: {failure}")
        return refinement

    def _ask(self, refinement: Refinement, prompt: str) -> str:
        call = self.model.chat([Message("user", prompt)], think=False)
        refinement.calls.append(call)
        if call.error:
            raise Invalid(f"the model call failed: {call.error}")
        return call.output

    # Prompt pass

    def _prompt_pass(self, harness, refinement, *, goal, window, count, fixed) -> PassResult:
        result = PassResult("prompt")
        text = self._ask(
            refinement,
            prompts.PROMPT_PASS.format(
                fixed=fixed, goal=goal, guidance=harness.guidance or "(none yet)", count=count, window=window, cap=MAX_GUIDANCE
            ),
        )
        guidance = _FENCE.sub("", text.strip()).strip()
        guidance = re.sub(r"^IMPROVED GUIDANCE:\s*", "", guidance, flags=re.IGNORECASE).strip()
        edit = {"guidance": guidance}
        try:
            if not guidance:
                raise Invalid("the reply was empty")
            if len(guidance) > MAX_GUIDANCE:
                raise Invalid(f"guidance is {len(guidance)} characters; the limit is {MAX_GUIDANCE}")
            if guidance == harness.guidance:
                raise Invalid("the guidance is unchanged")
        except Invalid as problem:
            result.rejected.append({"op": "rewrite", "edit": edit, "reason": str(problem)})
            return result
        harness.guidance = guidance
        result.applied.append({"op": "rewrite", "edit": edit})
        return result

    # Skills pass

    def _skills_pass(self, harness, refinement, *, goal, window, count, keys, failures) -> PassResult:
        result = PassResult("skills")
        rejected = ""
        if self.last_skill_rejections:
            rejected = "## Your Edits Rejected Last Time\n" + "\n".join(f"- {line}" for line in self.last_skill_rejections) + "\n"
        examples = ""
        if self.example_programs:
            examples = "\n" + EXAMPLE_PROGRAMS.format(programs=self.example_programs) + "\n"
        text = self._ask(
            refinement,
            prompts.SKILLS_PASS.format(
                goal=goal, skills=harness.skill_text(source=True), count=count, window=window, failures=failures,
                rejected=rejected, language=SKILL_LANGUAGE.replace("<ACTION>", "|".join(keys)), keys=", ".join(keys),
                examples=examples, max_edits=MAX_EDITS,
            ),
        )
        adds, updates = _read_edits(text, result)
        for op, edit in _within_cap([("add", e) for e in adds] + [("update", e) for e in updates], result):
            try:
                applied = _add_skill(harness, edit, keys) if op == "add" else _update_skill(harness, edit, keys)
            except Invalid as problem:
                result.rejected.append({"op": op, "edit": edit, "reason": str(problem)})
            else:
                result.applied.append({"op": op, "edit": edit, **applied})
        self.last_skill_rejections = [f"{r['op']} {_label(r['edit'])}: {r['reason']}" for r in result.rejected][:MAX_EDITS]
        return result

    # Memory pass

    def _memory_pass(self, harness, refinement, *, goal, window, count, step) -> PassResult:
        result = PassResult("memory")
        text = self._ask(
            refinement,
            prompts.MEMORY_PASS.format(
                goal=goal, memory=harness.memory_text(len(harness.memory)), count=count, window=window, max_edits=MAX_EDITS
            ),
        )
        adds, updates = _read_edits(text, result)
        for op, edit in _within_cap([("add", e) for e in adds] + [("update", e) for e in updates], result):
            try:
                applied = _add_memory(harness, edit, step) if op == "add" else _update_memory(harness, edit)
            except Invalid as problem:
                result.rejected.append({"op": op, "edit": edit, "reason": str(problem)})
            else:
                result.applied.append({"op": op, "edit": edit, **applied})
        return result


# Reading a JSON reply


def _read_edits(text: str, result: PassResult) -> tuple[list, list]:
    reply = first_json_object(text)
    # A reply cut off mid-object can leave only an inner object, which has none of these keys.
    if reply is None or not ({"analysis", "add", "update"} & set(reply)):
        raise Invalid("the reply was not the requested JSON object")
    adds, updates = reply.get("add", []), reply.get("update", [])
    if not isinstance(adds, list) or not isinstance(updates, list):
        raise Invalid('"add" and "update" must be lists')
    result.analysis = str(reply.get("analysis", ""))[:300]
    return adds, updates


def _within_cap(edits: list[tuple[str, Any]], result: PassResult) -> list[tuple[str, Any]]:
    for op, edit in edits[MAX_EDITS:]:
        result.rejected.append({"op": op, "edit": edit, "reason": f"over the {MAX_EDITS}-edit cap for one pass"})
    return edits[:MAX_EDITS]


def _label(edit: Any) -> str:
    return str(edit.get("name") or edit.get("id") or edit.get("title") or "?") if isinstance(edit, dict) else "?"


def _program(edit: dict, name: str, keys: Sequence[str]):
    code = text_field(edit, "code", MAX_CODE)
    try:
        program = parse(code, keys)
    except SkillError as problem:
        raise Invalid(f"the code does not parse: {problem}") from None
    if program.name != name:
        raise Invalid(f"the code is named `{program.name}`, not `{name}`")
    return program


# Skill edits


def _add_skill(harness: HarnessState, edit: Any, keys: Sequence[str]) -> dict[str, Any]:
    edit = only_fields(edit, {"name", "description", "code", "effectiveness", "importance", "path"}, {"name", "description", "code"})
    name = edit["name"]
    if not isinstance(name, str) or not NAME.match(name):
        raise Invalid('"name" must be a snake_case identifier')
    if name in harness.library.skills:
        raise Invalid(f"`{name}` already exists; use update with its id")
    description = text_field(edit, "description", MAX_DESCRIPTION)
    program = _program(edit, name, keys)
    effectiveness, importance = effectiveness_field(edit), importance_field(edit)
    stored, evicted = harness.add_skill(program, description, effectiveness, importance, "refiner")
    if not stored:
        raise Invalid("the library is full of skills that outrank this one")
    return {"id": name, "evicted": evicted}


def _update_skill(harness: HarnessState, edit: Any, keys: Sequence[str]) -> dict[str, Any]:
    edit = only_fields(edit, {"id", "description", "code", "effectiveness", "importance"}, {"id"})
    name = edit["id"]
    if name not in harness.library.skills:
        raise Invalid(f"no skill with id `{name}`")
    if len(edit) == 1:
        raise Invalid("nothing to update")
    skill, meta = harness.library.skills[name], harness.meta[name]
    description = text_field(edit, "description", MAX_DESCRIPTION) if "description" in edit else skill.description
    program = _program(edit, name, keys) if "code" in edit else skill.program
    effectiveness = effectiveness_field(edit, meta.effectiveness)
    importance = importance_field(edit, meta.importance)
    harness.add_skill(program, description, effectiveness, importance, meta.origin)
    return {"id": name, "evicted": []}


# Memory edits


def _add_memory(harness: HarnessState, edit: Any, step: int) -> dict[str, Any]:
    edit = only_fields(edit, {"path", "title", "content", "importance"}, {"title", "content"})
    title, content = text_field(edit, "title", MAX_TITLE), text_field(edit, "content", MAX_CONTENT)
    path = text_field(edit, "path", MAX_PATH) if "path" in edit else ""
    importance = importance_field(edit)
    if harness.find_memory(title):
        raise Invalid(f"an entry titled {title!r} already exists; update it instead")
    entry, evicted = harness.add_memory(title, content, importance, path=path, step=step, source="refiner")
    if entry is None:
        raise Invalid("memory is full of entries that outrank this one")
    return {"id": entry.id, "evicted": evicted}


def _update_memory(harness: HarnessState, edit: Any) -> dict[str, Any]:
    edit = only_fields(edit, {"id", "content", "importance"}, {"id"})
    entry = harness.memory.get(edit["id"])
    if entry is None:
        raise Invalid(f"no memory entry with id {edit['id']!r}")
    if len(edit) == 1:
        raise Invalid("nothing to update")
    content = text_field(edit, "content", MAX_CONTENT) if "content" in edit else entry.content
    importance = importance_field(edit, entry.importance)
    entry.content, entry.importance = content, importance
    return {"id": entry.id, "evicted": []}
