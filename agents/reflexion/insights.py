"""The insight set and ExpeL's operations on it (Zhao et al., AAAI 2024, arXiv:2308.10144).

Ported from `agent/expel.py` (`parse_rules`, `update_rules`; github.com/LeapLabTHU/ExpeL,
Apache-2.0). After a trial the model replies with operations on a numbered list of
insights. Each insight carries an importance count, and one that falls to zero is removed:

    ADD     a new insight, importance 2
    AGREE   +1   (the paper's UPVOTE)
    EDIT    rewrite it, +1
    REMOVE  -1, or -3 when the set is full   (the paper's DOWNVOTE)

Operations apply in the order REMOVE, AGREE, EDIT, ADD, as in the reference code, and the
set is kept sorted by importance, highest first (a stable sort, so ties keep their order).

Differences from the reference, because the setting differs:
- The set has a hard cap, `limit`. When it is over the cap after a reply, the least
  important insight (the one least recently touched among ties) is evicted. The reference
  only lets the set grow and makes REMOVE stronger when it is 5 over its target; here
  "full" (the prompt's REMOVE-first suffix, and REMOVE worth -3) means at the cap.
- A reply is read line by line, and an operation line that breaks the reference's filters
  is rejected and logged with the reason, not dropped silently. At most `max_ops`
  operations apply per reply, and a second operation on the same insight is rejected;
  the reference asks for both in the prompt and enforces neither.
- AGREE and REMOVE find their insight by its text (as in the reference) and fall back to
  the number when no text matches; the reference needs the text. EDIT needs a valid number.
- An operation whose text is not a sentence ending in a period, or that contains an
  operation word (ADD, AGREE, EDIT, REMOVE, UPVOTE, DOWNVOTE), is rejected, as in the reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

ADD, EDIT, AGREE, REMOVE = "ADD", "EDIT", "AGREE", "REMOVE"
_ALIASES = {"UPVOTE": AGREE, "DOWNVOTE": REMOVE}
_WORDS = (ADD, EDIT, AGREE, REMOVE, *_ALIASES)
_LINE = re.compile(rf"^\W*({'|'.join(_WORDS)})\b\s*(\d+)?\s*(:)?\s*(.*)$")  # upper case, as the reference's pattern is
_BANNED = re.compile(rf"\b({'|'.join(_WORDS)})\b")

ADD_IMPORTANCE = 2
VOTE = 1
REMOVE_FULL = 3


@dataclass(frozen=True)
class Insight:
    id: str
    text: str
    importance: int
    added_after_trial: int
    touched_after_trial: int

    def event(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "importance": self.importance}


@dataclass(frozen=True)
class Op:
    kind: str  # ADD, EDIT, AGREE, or REMOVE
    number: int | None
    text: str
    line: str


@dataclass(frozen=True)
class Rejected:
    line: str
    reason: str


def parse_ops(reply: str) -> tuple[list[Op], list[Rejected]]:
    """The operations a reply names, and the operation lines that were unusable. Other lines are prose and are ignored."""
    ops: list[Op] = []
    rejected: list[Rejected] = []
    for raw in reply.splitlines():
        line = raw.strip()
        match = _LINE.match(line)
        if not match:
            continue
        word, number, colon, text = match.group(1), match.group(2), match.group(3), match.group(4).strip()
        kind = _ALIASES.get(word, word)
        if not colon:
            rejected.append(Rejected(line, "no colon after the operation"))
        elif not text:
            rejected.append(Rejected(line, "empty text"))
        elif _BANNED.search(text):
            rejected.append(Rejected(line, "text contains an operation word"))
        elif not text.endswith("."):
            rejected.append(Rejected(line, "text does not end with a period (cut off or malformed)"))
        elif kind == EDIT and number is None:
            rejected.append(Rejected(line, "EDIT needs the number of the insight to rewrite"))
        else:
            ops.append(Op(kind, int(number) if number else None, text, line))
    return ops, rejected


class InsightSet:
    """The insights so far, most important first."""

    def __init__(self, limit: int = 10, max_ops: int = 4) -> None:
        self.limit = limit
        self.max_ops = max_ops
        self.items: list[Insight] = []
        self._next_id = 0

    @property
    def full(self) -> bool:
        return len(self.items) >= self.limit

    def numbered(self) -> str:
        """The list as the model is shown it: `1. text`, most important first."""
        return "\n".join(f"{n}. {i.text}" for n, i in enumerate(self.items, start=1))

    def apply(self, ops: list[Op], trial: int) -> dict[str, Any]:
        """Apply a reply's operations with ExpeL's vote rules; returns what happened, op by op.

        Every operation is resolved against the set as the model saw it, then applied
        in the order REMOVE, AGREE, EDIT, ADD.
        """
        before = list(self.items)
        full = self.full
        results: dict[int, dict[str, Any]] = {}  # by the operation's place in the reply
        valid: list[tuple[int, Op, int | None]] = []
        used: set[int] = set()
        for position, op in enumerate(ops):
            if position >= self.max_ops:
                results[position] = _result(op, "rejected", f"more than {self.max_ops} operations")
                continue
            op, target, why = self._resolve(op, before)
            if why:
                results[position] = _result(op, "rejected", why)
            elif target is not None and target in used:
                results[position] = _result(op, "rejected", "that insight already has an operation")
            else:
                if target is not None:
                    used.add(target)
                valid.append((position, op, target))
        votes = {AGREE: VOTE, REMOVE: -(REMOVE_FULL if full else VOTE), EDIT: VOTE}
        items = {i.id: i for i in before}
        order = [i.id for i in before]
        for kind in (REMOVE, AGREE, EDIT, ADD):
            for position, op, target in valid:
                if op.kind != kind:
                    continue
                if kind == ADD:
                    new = Insight(f"i{self._next_id}", op.text, ADD_IMPORTANCE, trial, trial)
                    self._next_id += 1
                    items[new.id] = new
                    order.append(new.id)
                    results[position] = _result(op, "applied", f"added with importance {ADD_IMPORTANCE}", new.id)
                    continue
                old = items[before[target].id]
                text = op.text if kind == EDIT else old.text
                items[old.id] = replace(old, text=text, importance=old.importance + votes[kind], touched_after_trial=trial)
                results[position] = _result(op, "applied", f"importance {old.importance} -> {old.importance + votes[kind]}", old.id)
        kept = [items[i] for i in order if items[i].importance > 0]
        removed = [items[i].id for i in order if items[i].importance <= 0]
        kept.sort(key=lambda i: -i.importance)  # stable
        evicted = []
        while len(kept) > self.limit:
            worst = min(kept, key=lambda i: (i.importance, i.touched_after_trial))
            kept.remove(worst)
            evicted.append(worst.id)
        self.items = kept
        return {
            "results": [results[n] for n in sorted(results)],
            "removed_at_zero": removed,
            "evicted_over_cap": evicted,
            "full_before": full,
            "insights": [i.event() for i in self.items],
        }

    def _resolve(self, op: Op, before: list[Insight]) -> tuple[Op, int | None, str]:
        """The operation, the index it acts on in `before`, and why it cannot be applied (empty if it can).

        An EDIT whose text is an existing insight changes nothing, so it counts as an AGREE, as in the reference.
        """
        if op.kind == ADD:
            return op, None, "duplicates an existing insight" if any(i.text in op.text for i in before) else ""
        by_text = next((n for n, i in enumerate(before) if i.text in op.text), None)
        if op.kind == EDIT and by_text is not None:
            return replace(op, kind=AGREE), by_text, ""
        if op.kind == EDIT:
            valid = op.number is not None and 1 <= op.number <= len(before)
            return op, (op.number - 1 if valid else None), "" if valid else "no insight with that number"
        if by_text is not None:
            return op, by_text, ""
        if op.number is not None and 1 <= op.number <= len(before):
            return op, op.number - 1, ""
        return op, None, "no existing insight matches the text or the number"


def _result(op: Op, status: str, detail: str, insight: str | None = None) -> dict[str, Any]:
    return {"op": op.kind, "number": op.number, "text": op.text, "status": status, "detail": detail, "insight": insight}
