"""Journal (one run) and knowledge graph (persists across runs).

The knowledge graph has a backbone the agent cannot change: nodes and the
relationships between them, written ahead of time as game knowledge. A new
game is a new adapter plus a new backbone file. The agent may only attach
notes and lessons to nodes, and add recipes (short how-to guides).

End-of-run summaries record only what happened, never what it means: the
write-up found an interpretive summary ("unplayable") biased the next run.

With a `Policy` (`utility_retrieval`, `memory_decay`; `agents/retrieval.py`) the notes
and lessons the agent attached to nodes become retrievable entries with `Stats`
(kept in `data["retrieval"]`, written only while a switch is on), and the journal's
leads decay. Backbone nodes, edges, and recipes are not scored: they are written
ahead of time, and nothing teaches the agent which of them helped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.retrieval import Policy, Ranked, Stats


@dataclass
class Journal:
    """One run's entries. Under `memory_decay` an entry's retention is measured in ticks, and `recent`
    skips the ones below the threshold; `recall` strengthens the ones that were shown."""

    entries: list[dict[str, Any]] = field(default_factory=list)
    policy: Policy | None = None
    _stats: dict[int, Stats] = field(default_factory=dict)  # by id(entry); entries are never removed

    def add(self, tick: int, kind: str, text: str) -> dict[str, Any]:
        entry = {"tick": tick, "kind": kind, "text": text}
        self.entries.append(entry)
        if self.policy is not None and self.policy.decay:
            self._stats[id(entry)] = self.policy.fresh(tick)
        return entry

    def recent(self, n: int, kinds: tuple[str, ...] | None = None, *, now: int | None = None) -> list[dict[str, Any]]:
        entries = [e for e in self.entries if kinds is None or e["kind"] in kinds]
        if now is not None and self.policy is not None and self.policy.decay:
            entries = [e for e in entries if not self.policy.forgotten(self._stats[id(e)], now)]
        return entries[-n:]

    def recall(self, entries: list[dict[str, Any]], now: int) -> None:
        if self.policy is not None and self.policy.decay:
            for entry in entries:
                self.policy.recall(self._stats[id(entry)], now)

    def retrieval(self, kinds: tuple[str, ...], shown: list[dict[str, Any]], now: int) -> list[Ranked]:
        """Every entry of `kinds` with its retention and whether `recent` showed it, for the trace.
        A journal has no query, so relevance is recency and there is no similarity; the score is the retention."""
        rows = []
        for e in self.entries:
            if e["kind"] in kinds:
                retention = self.policy.retention(self._stats[id(e)], now)
                hidden = self.policy.forgotten(self._stats[id(e)], now)
                shown_now = any(e is s for s in shown)
                reason = "forgotten" if hidden else "" if shown_now else "older than the entries shown"
                rows.append(Ranked(f"{e['kind']}@{e['tick']}: {e['text'][:40]}", None, None, retention, retention, shown_now, reason))
        return rows


class KnowledgeGraph:
    """{"nodes": {id: {"role", "about", "notes": [], "lessons": []}}, "edges": [[a, relation, b]],
    "recipes": {name: text}, "runs": [summary]}"""

    def __init__(self, backbone: dict[str, Any] | None = None, path: Path | None = None, policy: Policy | None = None) -> None:
        self.path = path
        self.policy = policy if policy is not None and policy.active else None
        self.clock = 0  # the agent's step count over its life; the player sets it
        self.stats: dict[str, Stats] = {}
        self.last_rows: list[Ranked] = []  # the last `relevant` ranking, for the trace
        data = json.loads(path.read_text()) if path is not None and path.exists() else None
        if data is None:
            data = {"nodes": {}, "edges": [], "recipes": {}, "runs": []}
            for node_id, node in (backbone or {}).get("nodes", {}).items():
                data["nodes"][node_id] = {"role": node.get("role", ""), "about": node.get("about", ""), "notes": [], "lessons": []}
            data["edges"] = [list(edge) for edge in (backbone or {}).get("edges", [])]
            data["recipes"] = dict((backbone or {}).get("recipes", {}))
        self.data = data
        self._backbone_edges = [tuple(edge) for edge in data["edges"]]
        if self.policy is not None:
            saved = data.get("retrieval", {})
            self.clock = int(saved.get("clock", 0))
            for node_id, node in data["nodes"].items():
                for kind in ("notes", "lessons"):
                    for text in node[kind]:
                        key = _key(node_id, kind, text)
                        self.stats[key] = self.policy.fresh(self.clock)
                        for field_name, value in saved.get("stats", {}).get(key, {}).items():
                            setattr(self.stats[key], field_name, value)

    @property
    def edges(self) -> list[tuple[str, str, str]]:
        return list(self._backbone_edges)  # read-only: the agent cannot change relationships

    def note(self, node_id: str, text: str, kind: str = "notes") -> bool:
        node = self.data["nodes"].get(node_id)
        if node is None or kind not in ("notes", "lessons"):
            return False
        node[kind].append(text)
        if self.policy is None:
            node[kind] = node[kind][-10:]
        else:
            self.stats[_key(node_id, kind, text)] = self.policy.fresh(self.clock)
            while len(node[kind]) > 10:  # a full list drops a forgotten note first, else the oldest
                stats = {_key(node_id, kind, t): self.stats[_key(node_id, kind, t)] for t in node[kind]}
                forgotten = self.policy.victim(stats, self.clock)
                dropped = next((t for t in node[kind] if _key(node_id, kind, t) == forgotten), node[kind][0])
                node[kind].remove(dropped)
                if dropped not in node[kind]:
                    self.stats.pop(_key(node_id, kind, dropped), None)
        self.save()
        return True

    def credit(self, key: str, reward: float) -> tuple[float, float] | None:
        """MemRL's update for a note or lesson. Returns (old, new), or None when utility_retrieval is off or the note is gone."""
        if self.policy is None or not self.policy.utility or key not in self.stats:
            return None
        return self.policy.update(self.stats[key], reward)

    def add_recipe(self, name: str, text: str) -> None:
        self.data["recipes"][name] = text
        self.save()

    def add_run(self, summary: str) -> None:
        self.data["runs"].append(summary)
        self.save()

    def relevant(self, query: str, limit: int = 6, *, notes_shown: int = 6) -> dict[str, Any]:
        """Nodes, their edges, and recipes whose names or text share words with the query.

        Under a policy the notes and lessons of those nodes are retrieved in two phases (phase 1 by
        words shared with the query, phase 2 by Q), at most `notes_shown` of them, and `last_rows`
        explains the ranking."""
        words = set(re.findall(r"[a-z0-9]+", query.lower().replace("_", " ")))

        def score(text: str) -> int:
            return len(words & set(re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))))

        nodes = sorted(self.data["nodes"].items(), key=lambda item: -score(f"{item[0]} {item[1]['role']} {item[1]['about']}"))
        picked = {node_id: node for node_id, node in nodes[:limit] if score(f"{node_id} {node['role']} {node['about']}")}
        edges = [edge for edge in self.edges if edge[0] in picked or edge[2] in picked]
        recipes = {name: text for name, text in self.data["recipes"].items() if score(f"{name} {text}")}
        if self.policy is not None:
            picked = self._ranked_notes(picked, score, notes_shown)
        return {"nodes": picked, "edges": edges, "recipes": dict(list(recipes.items())[:3]), "last_runs": self.data["runs"][-3:]}

    def _ranked_notes(self, picked: dict[str, Any], score, shown: int) -> dict[str, Any]:
        entries, where = [], {}
        for node_id, node in picked.items():
            for kind in ("notes", "lessons"):
                for text in dict.fromkeys(node[kind]):
                    key = _key(node_id, kind, text)
                    where[key] = (node_id, kind, text)
                    entries.append((key, float(score(f"{node_id} {node['role']} {node['about']} {text}")), self.stats[key]))
        self.last_rows = self.policy.rank(entries, shown, self.clock)
        self.policy.recall_shown(self.last_rows, self.stats, self.clock)
        chosen: dict[tuple[str, str], list[str]] = {}
        for row in self.last_rows:
            if row.shown:
                node_id, kind, text = where[row.key]
                chosen.setdefault((node_id, kind), []).append(text)
        return {node_id: {**node, "notes": chosen.get((node_id, "notes"), []), "lessons": chosen.get((node_id, "lessons"), [])} for node_id, node in picked.items()}

    def save(self) -> None:
        if self.policy is not None:  # only while a switch is on
            self.data["retrieval"] = {
                "clock": self.clock,
                "stats": {key: s.to_dict(utility=self.policy.utility, decay=self.policy.decay) for key, s in sorted(self.stats.items())},
            }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2) + "\n")


def _key(node_id: str, kind: str, text: str) -> str:
    return f"{node_id}/{kind}/{text}"
