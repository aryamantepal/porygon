"""Compare runs: one table row per run directory, read only from what the runner wrote.

    uv run python -m evaluation.results runs/ladder1-*
    uv run python -m evaluation.results runs/ladder1-* --json

Success counts come from each trial's outcome. Efficiency is reported over
successful trials only (presses, frames, model calls, and tokens to the goal),
so a fast failure never looks efficient. Decision latency is the median over
every decision in the run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summarize(run_dir: Path) -> dict[str, Any]:
    config = json.loads((run_dir / "config.json").read_text())
    trials = sorted((run_dir / "trials").glob("*/outcome.json"))
    outcomes = [json.loads(path.read_text()) for path in trials]
    frames, latencies = [], []
    for path in trials:
        total = 0
        for line in (path.parent / "steps.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["kind"] == "decision":
                total += record.get("frames_advanced", 0)
                latencies.append(record["decision_seconds"])
        frames.append(total)
    won = [(o, f) for o, f in zip(outcomes, frames) if o["success"]]
    switches = [name for name, on in config.get("mechanisms", {}).items() if on]

    def median(values):
        return statistics.median(values) if values else None

    return {
        "run": run_dir.name,
        "task": config.get("task_id"),
        "agent": config.get("agent"),
        "model": (config.get("model") or {}).get("id"),
        "on": switches,
        "successes": len(won),
        "trials": len(outcomes),
        "presses_to_goal": median([o["steps"] for o, _ in won]),
        "frames_to_goal": median([f for _, f in won]),
        "calls_to_goal": median([o["model_calls"] for o, _ in won]),
        "tokens_to_goal": median([o["prompt_tokens"] + o["completion_tokens"] for o, _ in won]),
        "invalid_decisions": sum(o["invalid_decisions"] for o in outcomes),
        "model_calls": sum(o["model_calls"] for o in outcomes),
        "decision_seconds": median(latencies),
        "dirty_code": (config.get("code") or {}).get("dirty"),
    }


def table(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("run", "run"), ("success", None), ("presses", "presses_to_goal"), ("frames", "frames_to_goal"),
        ("calls", "calls_to_goal"), ("tokens", "tokens_to_goal"), ("invalid", "invalid_decisions"),
        ("all calls", "model_calls"), ("s/decision", "decision_seconds"),
    ]

    def cell(row, key):
        if key is None:
            return f"{row['successes']}/{row['trials']}"
        value = row[key]
        if value is None:
            return "-"
        if isinstance(value, float) and not value.is_integer():
            return f"{value:.1f}"
        return str(int(value)) if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(name for name, _ in columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(row, key) for _, key in columns) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    runs = sorted(set(args.runs), key=lambda path: path.name)
    rows = [summarize(path) for path in runs if (path / "summary.json").exists()]
    print(json.dumps(rows, indent=2) if args.json else table(rows))


if __name__ == "__main__":
    main()
