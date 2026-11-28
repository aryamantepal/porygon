"""The harness state H the Refiner edits: learned guidance, skills, and memory.

Skills are Voyager-language programs (`agents/voyager/skills.py`), stored in a
`SkillLibrary` with the paper's two extra fields per skill: an effectiveness
rating and an importance. Memory entries carry an importance from 1 to 5. Both
are capped, and a full store evicts its least important entry, so a long run
cannot grow the prompt without bound.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from agents.voyager.skills import Program, Skill, SkillLibrary

EFFECTIVENESS = ("low", "medium", "high")
_RANK = {name: rank for rank, name in enumerate(EFFECTIVENESS)}


@dataclass
class SkillMeta:
    effectiveness: str = "medium"
    importance: int = 3
    origin: str = "refiner"
    uses: int = 0
    failures: int = 0


@dataclass
class MemoryEntry:
    id: str
    title: str
    content: str
    importance: int = 3
    path: str = ""
    step: int = 0
    source: str = "agent"


class HarnessState:
    def __init__(self, *, max_memory: int = 30, max_skills: int = 12) -> None:
        self.max_memory = max_memory
        self.max_skills = max_skills
        self.guidance = ""
        self.memory: dict[str, MemoryEntry] = {}
        self.library = SkillLibrary()
        self.meta: dict[str, SkillMeta] = {}
        self._next_memory = 1

    # Memory

    def find_memory(self, title: str) -> MemoryEntry | None:
        return next((e for e in self.memory.values() if e.title.lower() == title.lower()), None)

    def add_memory(
        self, title: str, content: str, importance: int, *, path: str = "", step: int = 0, source: str = "agent"
    ) -> tuple[MemoryEntry | None, list[str]]:
        """Add an entry. Returns (entry, evicted ids); entry is None when the store is full
        of entries that all outrank this one."""
        evicted: list[str] = []
        if len(self.memory) >= self.max_memory:
            victim = min(self.memory.values(), key=lambda e: (e.importance, e.step))
            if victim.importance > importance:
                return None, evicted
            del self.memory[victim.id]
            evicted.append(victim.id)
        entry = MemoryEntry(f"mem_{self._next_memory}", title, content, importance, path, step, source)
        self._next_memory += 1
        self.memory[entry.id] = entry
        return entry, evicted

    def memory_text(self, shown: int) -> str:
        if not self.memory:
            return "No memory entries yet."
        top = sorted(self.memory.values(), key=lambda e: (-e.importance, -e.step))[:shown]
        lines = [f"[{e.id}] importance {e.importance} | {e.title}: {e.content}" for e in top]
        if len(self.memory) > shown:
            lines.append(f"({len(self.memory) - shown} less important entries not shown)")
        return "\n".join(lines)

    # Skills

    def add_skill(self, program: Program, description: str, effectiveness: str, importance: int, source: str) -> tuple[bool, list[str]]:
        """Add or replace a skill. Returns (stored, evicted names); not stored when the
        library is full of skills that all outrank this one."""
        evicted: list[str] = []
        if program.name not in self.library.skills and len(self.library.skills) >= self.max_skills:
            victim = min(self.meta, key=lambda n: self._skill_rank(n))
            if self._skill_rank(victim) > (importance, _RANK[effectiveness], 0):
                return False, evicted
            self.remove_skill(victim)
            evicted.append(victim)
        previous = self.meta.get(program.name)
        self.library.add(program, description, "")
        self.meta[program.name] = SkillMeta(
            effectiveness, importance, source, previous.uses if previous else 0, previous.failures if previous else 0
        )
        return True, evicted

    def remove_skill(self, name: str) -> None:
        self.library.skills.pop(name, None)
        self.meta.pop(name, None)

    def _skill_rank(self, name: str) -> tuple[int, int, int]:
        meta = self.meta[name]
        return (meta.importance, _RANK[meta.effectiveness], meta.uses)

    def skill_text(self, *, source: bool = False) -> str:
        if not self.library.skills:
            return "No skills yet."
        lines = []
        for name in sorted(self.library.skills):
            skill, meta = self.library.skills[name], self.meta[name]
            lines.append(
                f"- {name} (effectiveness: {meta.effectiveness}, importance: {meta.importance}, "
                f"used {meta.uses}, errors {meta.failures}): {skill.description}"
            )
            if source:
                lines.append("    " + skill.program.source.replace("\n", "\n    "))
        return "\n".join(lines)

    def bind(self, legal_keys) -> None:
        """Parse skills loaded from a file against this environment's legal actions."""
        self.library.bind(legal_keys)
        for name in [n for n in self.meta if n not in self.library.skills]:
            del self.meta[name]

    # Serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "guidance": self.guidance,
            "skills": {
                name: {
                    "source": skill.program.source,
                    "description": skill.description,
                    "version": skill.version,
                    **asdict(self.meta[name]),
                }
                for name, skill in sorted(self.library.skills.items())
            },
            "memory": [asdict(e) for e in self.memory.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, max_memory: int = 30, max_skills: int = 12) -> HarnessState:
        """Rebuild a saved state; skills are parsed on the first `bind`. A malformed file is an error."""
        state = cls(max_memory=max_memory, max_skills=max_skills)
        try:
            state.guidance = str(data.get("guidance", ""))
            for name, item in data.get("skills", {}).items():
                program = Program(name, (), str(item["source"]))  # body is parsed in `bind`
                state.library.skills[name] = Skill(program, str(item["description"]), "", int(item.get("version", 1)))
                state.meta[name] = SkillMeta(
                    item.get("effectiveness", "medium"), int(item.get("importance", 3)), item.get("origin", "refiner"),
                    int(item.get("uses", 0)), int(item.get("failures", 0)),
                )
                if state.meta[name].effectiveness not in EFFECTIVENESS:
                    raise ValueError(f"effectiveness {state.meta[name].effectiveness!r}")
            for item in data.get("memory", []):
                entry = MemoryEntry(**item)
                state.memory[entry.id] = entry
                state._next_memory = max(state._next_memory, int(entry.id.removeprefix("mem_")) + 1)
        except (KeyError, TypeError, ValueError, AttributeError) as problem:
            raise ValueError(f"malformed harness state: {problem!r}") from problem
        return state


def dumps(state: HarnessState) -> str:
    return json.dumps(state.to_dict(), indent=2) + "\n"
