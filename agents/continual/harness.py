"""The harness state H the Refiner edits: learned guidance, skills, and memory.

Skills are Voyager-language programs (`agents/voyager/skills.py`), stored in a
`SkillLibrary` with the paper's two extra fields per skill: an effectiveness
rating and an importance. Memory entries carry an importance from 1 to 5. Both
are capped, and a full store evicts its least important entry, so a long run
cannot grow the prompt without bound.

With a `Policy` (`utility_retrieval`, `memory_decay`; `agents/retrieval.py`) the prompt
shows memory and skills in two phases instead. Phase 1 is the order above (this
store has no query, so importance stands for relevance: `importance / 5`, scaled by
retention under decay); phase 2 ranks the top candidates by a learned Q. Every
entry has `Stats` under the key `memory:<id>` or `skill:<name>`, and `clock` is the
player's press count over its life. An entry whose content is rewritten starts over;
a change of importance alone keeps its stats. The Refiner still sees every entry.
`to_dict` carries the stats only while a switch is on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from agents.retrieval import Policy, Ranked, Stats
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
    def __init__(self, *, max_memory: int = 30, max_skills: int = 12, policy: Policy | None = None) -> None:
        self.policy = policy if policy is not None and policy.active else None
        # skills never decay: the passes below that retire a skill are the Refiner's and the full library's
        self.skill_policy = Policy(utility=True, params=policy.params) if self.policy is not None and policy.utility else None
        self.stats: dict[str, Stats] = {}
        self.clock = 0
        self.retrieved: dict[str, list[Ranked]] = {}  # the last ranking of each store shown: "memory" and "skills"
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
            forgotten = self._forgotten_memory()
            if forgotten is not None:  # under decay a forgotten entry goes first, whatever its importance
                victim = self.memory[forgotten]
            else:
                victim = min(self.memory.values(), key=lambda e: (e.importance, e.step))
                if victim.importance > importance:
                    return None, evicted
            self._drop_memory(victim.id)
            evicted.append(victim.id)
        entry = MemoryEntry(f"mem_{self._next_memory}", title, content, importance, path, step, source)
        self._next_memory += 1
        self.memory[entry.id] = entry
        self._begin(f"memory:{entry.id}")
        return entry, evicted

    def revise_memory(self, entry: MemoryEntry, content: str, importance: int, *, step: int | None = None) -> None:
        """Rewrite an entry. New content is a new entry to the policy (its Q and strength start over)."""
        changed = content != entry.content
        entry.content, entry.importance = content, importance
        if step is not None:
            entry.step = step
        if changed:
            self._begin(f"memory:{entry.id}")

    def credit(self, key: str, reward: float) -> tuple[float, float] | None:
        """MemRL's update for the entry `memory:<id>` or `skill:<name>`. Returns (old, new), or None when
        utility_retrieval is off or the entry is gone."""
        if self.policy is None or not self.policy.utility or key not in self.stats:
            return None
        return self.policy.update(self.stats[key], reward)

    def _drop_memory(self, entry_id: str) -> None:
        del self.memory[entry_id]
        self.stats.pop(f"memory:{entry_id}", None)

    def _forgotten_memory(self) -> str | None:
        if self.policy is None:
            return None
        stats = {entry_id: self.stats[f"memory:{entry_id}"] for entry_id in self.memory}
        return self.policy.victim(stats, self.clock)

    def _begin(self, key: str) -> None:
        if self.policy is not None:
            self.stats[key] = self.policy.fresh(self.clock)

    def memory_text(self, shown: int, *, rank: bool = False) -> str:
        """The memory section. `rank` is the agent's prompt: under a policy it retrieves in two phases and
        recalls what it shows. The Refiner passes `shown=len(memory)` and no `rank`, and sees every entry."""
        if not self.memory:
            return "No memory entries yet."
        if rank and self.policy is not None:
            return self._ranked_memory(shown)
        top = sorted(self.memory.values(), key=lambda e: (-e.importance, -e.step))[:shown]
        lines = [f"[{e.id}] importance {e.importance} | {e.title}: {e.content}" for e in top]
        if len(self.memory) > shown:
            lines.append(f"({len(self.memory) - shown} less important entries not shown)")
        return "\n".join(lines)

    def _ranked_memory(self, shown: int) -> str:
        by_importance = sorted(self.memory.values(), key=lambda e: (-e.importance, -e.step))
        entries = [(e.id, e.importance / 5, self.stats[f"memory:{e.id}"]) for e in by_importance]
        rows = self.policy.rank(entries, shown, self.clock)
        self.policy.recall_shown(rows, {e.id: self.stats[f"memory:{e.id}"] for e in by_importance}, self.clock)
        self.retrieved["memory"] = rows
        top = [self.memory[row.key] for row in rows if row.shown]
        lines = [f"[{e.id}] importance {e.importance} | {e.title}: {e.content}" for e in top]
        if len(self.memory) > len(top):
            lines.append(f"({len(self.memory) - len(top)} other entries not shown)")
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
        stored = self.library.skills.get(program.name)
        rewritten = stored is None or stored.program.source != program.source
        self.library.add(program, description, "")
        if rewritten:
            self._begin(f"skill:{program.name}")
        self.meta[program.name] = SkillMeta(
            effectiveness, importance, source, previous.uses if previous else 0, previous.failures if previous else 0
        )
        return True, evicted

    def remove_skill(self, name: str) -> None:
        self.library.skills.pop(name, None)
        self.meta.pop(name, None)
        self.stats.pop(f"skill:{name}", None)

    def _skill_rank(self, name: str) -> tuple[int, int, int]:
        meta = self.meta[name]
        return (meta.importance, _RANK[meta.effectiveness], meta.uses)

    def skill_text(self, *, source: bool = False, rank: int | None = None) -> str:
        """The skill section. `rank` is the agent's prompt: under a policy it shows only the best `rank`
        skills by two-phase retrieval and says how many it left out. The Refiner passes neither and sees all."""
        if not self.library.skills:
            return "No skills yet."
        names = sorted(self.library.skills)
        hidden = 0
        if rank is not None and self.skill_policy is not None:
            names, hidden = self._ranked_skills(rank)
        lines = []
        for name in names:
            skill, meta = self.library.skills[name], self.meta[name]
            label = f"{name}({', '.join(skill.program.params)})" if skill.program.params else name
            lines.append(
                f"- {label} (effectiveness: {meta.effectiveness}, importance: {meta.importance}, "
                f"used {meta.uses}, errors {meta.failures}): {skill.description}"
            )
            if source:
                lines.append("    " + skill.program.source.replace("\n", "\n    "))
        if hidden:
            lines.append(f"({hidden} other skills not shown; run_skill still accepts any name)")
        return "\n".join(lines)

    def _ranked_skills(self, shown: int) -> tuple[list[str], int]:
        def order(name: str) -> tuple[int, int]:
            return (-self.meta[name].importance, -_RANK[self.meta[name].effectiveness])

        by_importance = sorted(sorted(self.library.skills), key=order)
        entries = [(name, self.meta[name].importance / 5, self.stats[f"skill:{name}"]) for name in by_importance]
        rows = self.skill_policy.rank(entries, shown, self.clock)
        self.retrieved["skills"] = rows
        names = [row.key for row in rows if row.shown]
        return names, len(self.library.skills) - len(names)

    def bind(self, legal_keys, *, parameters: bool = False) -> None:
        """Parse skills loaded from a file against this environment's legal actions."""
        self.library.bind(legal_keys, parameters=parameters)
        for name in [n for n in self.meta if n not in self.library.skills]:
            del self.meta[name]
            self.stats.pop(f"skill:{name}", None)

    # Serialization

    def to_dict(self) -> dict[str, Any]:
        data = {
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
        if self.policy is not None:  # only while a switch is on, so a file written with both off is what it was
            data["retrieval"] = {
                "clock": self.clock,
                "stats": {key: stats.to_dict(utility=self.policy.utility, decay=self.policy.decay) for key, stats in sorted(self.stats.items())},
            }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, max_memory: int = 30, max_skills: int = 12, policy: Policy | None = None) -> HarnessState:
        """Rebuild a saved state; skills are parsed on the first `bind`. A malformed file is an error."""
        state = cls(max_memory=max_memory, max_skills=max_skills, policy=policy)
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
            if state.policy is not None:
                saved = data.get("retrieval", {})
                state.clock = int(saved.get("clock", 0))
                for key in [*(f"skill:{n}" for n in state.library.skills), *(f"memory:{i}" for i in state.memory)]:
                    state._begin(key)
                    for field_name, value in saved.get("stats", {}).get(key, {}).items():
                        setattr(state.stats[key], field_name, value)
        except (KeyError, TypeError, ValueError, AttributeError) as problem:
            raise ValueError(f"malformed harness state: {problem!r}") from problem
        return state


def dumps(state: HarnessState) -> str:
    return json.dumps(state.to_dict(), indent=2) + "\n"
