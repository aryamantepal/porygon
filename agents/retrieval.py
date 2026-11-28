"""Utility-ranked retrieval and memory decay, shared by every agent that keeps a memory.

Two switches, each off by default, each an ablation of a store's existing retrieval:

`utility_retrieval` (MemRL, Zhang et al. 2026, arXiv:2601.03192; github.com/MemTensor/MemRL, Apache-2.0).
Retrieval has two phases. Phase 1 keeps the store's own relevance (word-overlap
cosine for Voyager's library, word overlap for the knowledge graph, importance
for Continual Harness, which has no query) and takes the top `candidates` (the
paper's k1). Phase 2 ranks those by a mix of the z-scored relevance and the
z-scored utility Q of each entry, `(1 - weight) * z(relevance) + weight * z(Q)`,
and shows the top k. Q is learned at run time from a reward the environment
produced, with no weight updates:

    Q <- Q + alpha * (r - Q)        (MemRL's value-driven update, gamma = 0)

The reward is `r` in {0, 1}, defined per store where the store is used. It is
never the model's own claim that something worked. `agents/voyager/player.py`,
`agents/continual/player.py`, and `agents/harness/player.py` say which signal each
store uses. Every Q starts at `q_init`. While every candidate has the same Q the
z-score of Q is 0 and the ranking is the relevance ranking, so a store with no
feedback yet retrieves as it did before.

`memory_decay` (MemoryBank, Zhong et al. 2023, arXiv:2305.10250). An entry's retention is
the Ebbinghaus curve `R = exp(-t / S)`, where `t` is the number of steps since
the entry was written or last shown, and `S` is its strength. Each time an entry
is shown (recalled) its strength is multiplied by `growth` and `t` starts over.
An entry whose retention is below `threshold` is not shown, and a store that is
full evicts a forgotten entry before it evicts anything else. Retention scales the
relevance (`relevance * R`) in both phases.

Deviations from the papers: MemRL samples epsilon-greedily and treats a similarity
below tau as "unknown"; neither is done here, so a run replays. MemRL's Q starts at 0
with rewards of +1 and -1; here the reward is 0 or 1 and Q starts at 0.5, so a
fresh entry is neither rewarded nor punished. MemRL's z-score of similarity uses
statistics over the whole store; here both z-scores use the phase-1 candidates of
one retrieval. MemoryBank counts time in days and adds 1 to S per recall; here
time is in agent steps, S starts at 50, and a recall multiplies it, because a
step is far shorter than a day and +1 would change nothing. MemoryBank summarizes
and reflects on memories; none of that is here.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Q_CLIP = 3.0  # as in MemRL, a z-score is clipped to [-3, 3]


@dataclass(frozen=True)
class MemoryParams:
    """Tuning for both switches; the defaults are ours, not the papers'."""

    alpha: float = 0.1  # Q-learning step size
    q_init: float = 0.5  # a fresh entry's Q, halfway between the rewards 0 and 1
    weight: float = 0.5  # phase 2's weight on utility; 1 - weight goes to relevance
    candidates: int = 10  # phase 1 keeps this many entries (MemRL's k1)
    strength: float = 50.0  # a fresh entry's S, in steps
    growth: float = 2.0  # S is multiplied by this each recall
    threshold: float = 0.3  # an entry below this retention is not shown

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], not {self.alpha}")
        if not 0 <= self.weight <= 1:
            raise ValueError(f"weight must be in [0, 1], not {self.weight}")
        if self.candidates < 1:
            raise ValueError(f"candidates must be at least 1, not {self.candidates}")
        if self.strength <= 0:
            raise ValueError(f"strength must be positive, not {self.strength}")
        if self.growth < 1:
            raise ValueError(f"growth must be at least 1, not {self.growth}")
        if not 0 <= self.threshold < 1:
            raise ValueError(f"threshold must be in [0, 1), not {self.threshold}")

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> MemoryParams:
        """Read the tuning keys of an agent config: `utility_alpha`, `utility_q_init`, `utility_weight`,
        `utility_candidates`, `decay_strength`, `decay_growth`, `decay_threshold`."""
        keys = {
            "alpha": "utility_alpha", "q_init": "utility_q_init", "weight": "utility_weight", "candidates": "utility_candidates",
            "strength": "decay_strength", "growth": "decay_growth", "threshold": "decay_threshold",
        }
        return cls(**{field: spec[key] for field, key in keys.items() if key in spec})


@dataclass
class Stats:
    """What the policy knows about one entry. `last_access` is a clock reading, in steps."""

    q: float
    strength: float
    last_access: int = 0
    visits: int = 0  # utility updates
    recalls: int = 0  # times it was shown

    def to_dict(self, *, utility: bool, decay: bool) -> dict[str, Any]:
        """Only the fields of a switch that is on, so a store written with a switch off has none of them."""
        data: dict[str, Any] = {}
        if utility:
            data |= {"q": self.q, "visits": self.visits}
        if decay:
            data |= {"last_access": self.last_access, "strength": self.strength, "recalls": self.recalls}
        return data


@dataclass(frozen=True)
class Ranked:
    """One candidate of a retrieval, with everything that decided its place."""

    key: str
    similarity: float | None  # the store's own relevance, before retention; None for a store with no query (a journal)
    utility: float | None  # Q, None when utility_retrieval is off
    retention: float | None  # R, None when memory_decay is off
    score: float
    shown: bool
    reason: str = ""  # why it was not shown, when it was not

    def event(self) -> dict[str, Any]:
        row = {
            "id": self.key, "similarity": None if self.similarity is None else round(self.similarity, 4),
            "utility": None if self.utility is None else round(self.utility, 4),
            "retention": None if self.retention is None else round(self.retention, 4),
            "score": round(self.score, 4), "shown": self.shown,
        }
        return row | {"reason": self.reason} if self.reason else row


class Policy:
    """The two switches and their math. Holds no entries; a store keeps one `Stats` per entry."""

    def __init__(self, *, utility: bool = False, decay: bool = False, params: MemoryParams | None = None) -> None:
        self.utility, self.decay = utility, decay
        self.params = params or MemoryParams()

    @property
    def active(self) -> bool:
        return self.utility or self.decay

    def fresh(self, now: int) -> Stats:
        return Stats(self.params.q_init, self.params.strength, now)

    # Utility

    def update(self, stats: Stats, reward: float) -> tuple[float, float]:
        """Q <- Q + alpha * (r - Q). Returns (old, new)."""
        old = stats.q
        stats.q = old + self.params.alpha * (reward - old)
        stats.visits += 1
        return old, stats.q

    # Decay

    def retention(self, stats: Stats, now: int) -> float:
        """R = exp(-t / S), t the steps since the entry was written or last shown."""
        return math.exp(-max(now - stats.last_access, 0) / stats.strength)

    def forgotten(self, stats: Stats, now: int) -> bool:
        return self.decay and self.retention(stats, now) < self.params.threshold

    def recall(self, stats: Stats, now: int) -> None:
        """The entry was shown: S grows and t starts over. Showing it twice in one step counts once."""
        if now == stats.last_access and stats.recalls:
            return
        stats.strength *= self.params.growth
        stats.last_access = now
        stats.recalls += 1

    def recall_shown(self, rows: Sequence[Ranked], stats: Mapping[str, Stats], now: int) -> None:
        """Strengthen every entry a retrieval showed. Nothing happens when memory_decay is off."""
        if self.decay:
            for row in rows:
                if row.shown:
                    self.recall(stats[row.key], now)

    def victim(self, stats: Mapping[str, Stats], now: int) -> str | None:
        """The forgotten entry to evict from a full store (the least retained), or None when nothing is forgotten."""
        if not self.decay:
            return None
        gone = [(self.retention(s, now), key) for key, s in stats.items() if self.forgotten(s, now)]
        return min(gone, key=lambda pair: pair[0])[1] if gone else None

    # Ranking

    def rank(self, entries: Sequence[tuple[str, float, Stats]], k: int, now: int) -> list[Ranked]:
        """Two-phase retrieval over (key, relevance, stats). Returns every entry, best first, the first `k` marked shown.

        Entries below the retention threshold come last, never shown. Equals keep the order `entries` came in, so a store passes them in its own tie-break order.
        """
        p = self.params
        rows: dict[str, tuple[float, float | None, float | None]] = {}  # key -> (relevance, Q, R)
        live: list[tuple[str, float]] = []  # (key, effective relevance)
        forgotten: list[Ranked] = []
        for key, similarity, stats in entries:
            retention = self.retention(stats, now) if self.decay else None
            q = stats.q if self.utility else None
            if retention is not None and retention < p.threshold:
                forgotten.append(Ranked(key, similarity, q, retention, 0.0, False, "forgotten"))
                continue
            rows[key] = (similarity, q, retention)
            live.append((key, similarity * (retention if retention is not None else 1.0)))
        live.sort(key=lambda pair: -pair[1])  # stable
        scores = {key: effective for key, effective in live}
        pool = live[: p.candidates] if self.utility else live  # phase 1
        if self.utility and pool:  # phase 2
            relevance_z = _z([score for _, score in pool])
            q_z = _z([rows[key][1] for key, _ in pool], clip=Q_CLIP)
            for (key, _), rz, qz in zip(pool, relevance_z, q_z):
                scores[key] = (1 - p.weight) * rz + p.weight * qz
        ordered = sorted(pool, key=lambda pair: -scores[pair[0]])  # stable: ties stay in relevance order
        out = [Ranked(key, rows[key][0], rows[key][1], rows[key][2], scores[key], index < k) for index, (key, _) in enumerate(ordered)]
        out += [Ranked(key, rows[key][0], rows[key][1], rows[key][2], scores[key], False, "not in phase 1") for key, _ in live[len(pool) :]]
        return out + sorted(forgotten, key=lambda r: r.retention)


def _z(values: Sequence[float], clip: float | None = None) -> list[float]:
    """z-scores over one retrieval's candidates; all equal values give 0, as MemRL's `std or 1.0` does."""
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    std = std if std > 1e-9 else 1.0
    scores = [(v - mean) / std for v in values]
    return [max(min(z, clip), -clip) for z in scores] if clip is not None else scores


def trace_event(store: str, query: str, rows: Sequence[Ranked]) -> dict[str, Any]:
    """The retrieval event a trace carries: every candidate with the numbers that decided it."""
    return {"type": "retrieval", "store": store, "query": query, "candidates": [row.event() for row in rows]}


def words(text: str) -> Counter:
    """Bag of lowercase words and numbers, underscores as spaces."""
    return Counter(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))


def cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[w] * b[w] for w in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0
