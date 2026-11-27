"""Journal (one run) and knowledge graph (persists across runs).

The knowledge graph has a backbone the agent cannot change: nodes and the
relationships between them, written ahead of time as game knowledge. A new
game is a new adapter plus a new backbone file. The agent may only attach
notes and lessons to nodes, and add recipes (short how-to guides).

End-of-run summaries record only what happened, never what it means: the
write-up found an interpretive summary ("unplayable") biased the next run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Journal:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, tick: int, kind: str, text: str) -> dict[str, Any]:
        entry = {"tick": tick, "kind": kind, "text": text}
        self.entries.append(entry)
        return entry

    def recent(self, n: int, kinds: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        entries = [e for e in self.entries if kinds is None or e["kind"] in kinds]
        return entries[-n:]


class KnowledgeGraph:
    """{"nodes": {id: {"role", "about", "notes": [], "lessons": []}}, "edges": [[a, relation, b]],
    "recipes": {name: text}, "runs": [summary]}"""

    def __init__(self, backbone: dict[str, Any] | None = None, path: Path | None = None) -> None:
        self.path = path
        data = json.loads(path.read_text()) if path is not None and path.exists() else None
        if data is None:
            data = {"nodes": {}, "edges": [], "recipes": {}, "runs": []}
            for node_id, node in (backbone or {}).get("nodes", {}).items():
                data["nodes"][node_id] = {"role": node.get("role", ""), "about": node.get("about", ""), "notes": [], "lessons": []}
            data["edges"] = [list(edge) for edge in (backbone or {}).get("edges", [])]
            data["recipes"] = dict((backbone or {}).get("recipes", {}))
        self.data = data
        self._backbone_edges = [tuple(edge) for edge in data["edges"]]

    @property
    def edges(self) -> list[tuple[str, str, str]]:
        return list(self._backbone_edges)  # read-only: the agent cannot change relationships

    def note(self, node_id: str, text: str, kind: str = "notes") -> bool:
        node = self.data["nodes"].get(node_id)
        if node is None or kind not in ("notes", "lessons"):
            return False
        node[kind].append(text)
        node[kind] = node[kind][-10:]
        self.save()
        return True

    def add_recipe(self, name: str, text: str) -> None:
        self.data["recipes"][name] = text
        self.save()

    def add_run(self, summary: str) -> None:
        self.data["runs"].append(summary)
        self.save()

    def relevant(self, query: str, limit: int = 6) -> dict[str, Any]:
        """Nodes, their edges, and recipes whose names or text share words with the query."""
        words = set(re.findall(r"[a-z0-9]+", query.lower().replace("_", " ")))

        def score(text: str) -> int:
            return len(words & set(re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))))

        nodes = sorted(self.data["nodes"].items(), key=lambda item: -score(f"{item[0]} {item[1]['role']} {item[1]['about']}"))
        picked = {node_id: node for node_id, node in nodes[:limit] if score(f"{node_id} {node['role']} {node['about']}")}
        edges = [edge for edge in self.edges if edge[0] in picked or edge[2] in picked]
        recipes = {name: text for name, text in self.data["recipes"].items() if score(f"{name} {text}")}
        return {"nodes": picked, "edges": edges, "recipes": dict(list(recipes.items())[:3]), "last_runs": self.data["runs"][-3:]}

    def save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2) + "\n")
