"""Read a trial step by step: what the agent said, what it pressed, and what actually changed.

    uv run python -m evaluation.trace runs/gridworld-7b-v2-react            # trial 0
    uv run python -m evaluation.trace runs/gridworld-7b-v2-react --trial 2 --width 0

Each line pairs the agent's reasoning with the environment's progress after the
press, so a claim ("I have moved next to the person") sits beside the position
that says whether it happened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def reasoning(decision: dict[str, Any]) -> str:
    """The agent's own words for this decision, whatever architecture wrote them."""
    trace = decision.get("trace") or {}
    if trace.get("thought"):
        return trace["thought"]
    said = []
    for event in trace.get("events") or []:
        for key in ("why", "task", "explain", "error", "reason", "result"):
            if event.get(key):
                said.append(f"[{event['type']}] {event[key]}")
                break
    if said:
        return " / ".join(said)
    calls = decision.get("calls") or []
    return calls[-1].get("output", "") if calls else ""


def describe(records: list[dict[str, Any]], width: int) -> list[str]:
    lines, before = [], None
    for record in records:
        if record["kind"] == "reset":
            before = record.get("progress")
            lines.append(f"start   {json.dumps(before)}")
            continue
        if record["kind"] == "trial_end":
            lines.append(f"end     {json.dumps({k: v for k, v in record.items() if k != 'kind'})}")
            continue
        if record["kind"] != "decision":
            continue
        after = record.get("progress") or {}
        action = record["decision"].get("action")
        pressed = action["key"] if action else "INVALID"
        changed = {k: v for k, v in after.items() if not before or before.get(k) != v}
        effect = json.dumps(changed) if changed else "no change"
        text = " ".join(reasoning(record["decision"]).split())
        if width and len(text) > width:
            text = text[: width - 1] + "…"
        lines.append(f"{record['step']:>4} {pressed:<7} {effect}\n       {text}")
        before = after or before
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--width", type=int, default=160, help="truncate reasoning to this many characters; 0 for all")
    args = parser.parse_args()
    path = args.run_dir / "trials" / f"{args.trial:03d}" / "steps.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    print("\n".join(describe(records, args.width)))
    outcome = path.parent / "outcome.json"
    if outcome.exists():
        o = json.loads(outcome.read_text())
        tokens = o["prompt_tokens"] + o["completion_tokens"]
        print(f"cost    {o['steps']} presses, {o.get('frames', '?')} frames, {o['model_calls']} model calls, {tokens} tokens, {o['wall_seconds']:.0f}s ({o['reason']})")


if __name__ == "__main__":
    main()
