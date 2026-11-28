"""The runner: observe, decide, step, until the goal is met or a budget runs out.

It writes a run directory that replays on its own:

    config.json                     what was run
    trials/NNN/steps.jsonl          one line per decision (plus the reset)
    trials/NNN/frames/NNNNNN.png    the screen the player saw
    trials/NNN/outcome.json         how the trial ended
    summary.json                    per-trial outcomes and totals

Success is decided from `get_progress()`, never from the model's words.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agents.models import ModelCall
from agents.protocol import Decision, Player, TrialOutcome
from environments.protocol import Environment, Observation
from evaluation.goals import Goal


@dataclass(frozen=True)
class Budgets:
    max_steps: int  # environment steps
    max_decisions: int | None = None  # defaults to 2 * max_steps; bounds invalid outputs
    max_model_calls: int | None = None
    max_tokens: int | None = None
    max_wall_seconds: float | None = None

    @property
    def decisions(self) -> int:
        return self.max_decisions if self.max_decisions is not None else 2 * self.max_steps


@dataclass
class TrialResult:
    trial: int
    success: bool
    reason: str
    steps: int = 0
    decisions: int = 0
    invalid_decisions: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    frames: int = 0  # emulator frames the steps advanced
    final_progress: Mapping[str, Any] = field(default_factory=dict)


def run(
    *,
    run_dir: Path,
    env: Environment,
    player: Player,
    goal: Goal,
    budgets: Budgets,
    trials: int = 1,
    config: Mapping[str, Any] | None = None,
) -> list[TrialResult]:
    """Run `trials` trials from the environment's reset state and log them.

    `config` carries what only the caller knows (environment, initial state,
    model id, observation mode); the runner adds the rest.
    """
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "config.json",
        {
            **(config or {}),
            "task_id": goal.task_id,
            "goal": {"text": goal.text, "success": dict(goal.success)},
            "agent": player.name,
            "mechanisms": dict(player.mechanisms()),
            "action_space": [asdict(action) for action in env.legal_actions()],
            "budgets": asdict(budgets),
            "trials": trials,
        },
    )
    results = []
    for index in range(trials):
        result = _run_trial(run_dir / "trials" / f"{index:03d}", index, env, player, goal, budgets)
        results.append(result)
    _write_json(
        run_dir / "summary.json",
        {
            "successes": sum(r.success for r in results),
            "to_goal": _to_goal(results),
            "trials": len(results),
            "results": [asdict(r) for r in results],
        },
    )
    return results


def _run_trial(
    trial_dir: Path, index: int, env: Environment, player: Player, goal: Goal, budgets: Budgets
) -> TrialResult:
    (trial_dir / "frames").mkdir(parents=True)
    result = TrialResult(trial=index, success=False, reason="")
    started = time.perf_counter()
    with (trial_dir / "steps.jsonl").open("w") as log:

        def write(record: dict[str, Any]) -> None:
            log.write(json.dumps(record) + "\n")
            log.flush()

        observation = env.reset()
        player.start_trial(goal.text)
        progress = env.get_progress()
        write({"kind": "reset", "step": 0, "observation": _observation(trial_dir, 0, observation), "progress": dict(progress)})

        while True:
            reason = _stop_reason(goal, progress, result, budgets, time.perf_counter() - started)
            if reason:
                break
            decision_started = time.perf_counter()
            decision = player.decide(observation, env.legal_actions())
            decision_seconds = time.perf_counter() - decision_started
            result.decisions += 1
            _count_calls(result, decision.calls)
            record: dict[str, Any] = {
                "kind": "decision",
                "step": result.steps,
                "decision": _decision(decision),
                "decision_seconds": decision_seconds,
            }
            if decision.action is None:
                result.invalid_decisions += 1
                record["invalid"] = True
                write(record)
                continue
            step = env.step(decision.action)
            result.steps += 1
            result.frames += step.frames_advanced
            observation = step.observation
            progress = env.get_progress()
            record.update(
                {
                    "step": result.steps,
                    "expansion": [asdict(decision.action)],
                    "frames_advanced": step.frames_advanced,
                    "step_seconds": step.wall_seconds,
                    "observation": _observation(trial_dir, result.steps, observation),
                    "progress": dict(progress),
                    "goal_met": goal.is_met(progress),
                }
            )
            write(record)

        result.success = reason == "goal_met"
        result.reason = reason
        result.final_progress = dict(progress)
        review = player.end_trial(TrialOutcome(result.success, reason, result.steps))
        _count_calls(result, review.calls)
        result.wall_seconds = time.perf_counter() - started
        write(
            {
                "kind": "trial_end",
                "success": result.success,
                "reason": reason,
                "review": {
                    "trace": dict(review.trace) if review.trace is not None else None,
                    "calls": [asdict(call) for call in review.calls],
                },
            }
        )
    _write_json(trial_dir / "outcome.json", asdict(result))
    return result


def _stop_reason(goal: Goal, progress, result: TrialResult, budgets: Budgets, elapsed: float) -> str:
    if goal.is_met(progress):
        return "goal_met"
    if result.steps >= budgets.max_steps:
        return "max_steps"
    if result.decisions >= budgets.decisions:
        return "max_decisions"
    if budgets.max_model_calls is not None and result.model_calls >= budgets.max_model_calls:
        return "max_model_calls"
    tokens = result.prompt_tokens + result.completion_tokens
    if budgets.max_tokens is not None and tokens >= budgets.max_tokens:
        return "max_tokens"
    if budgets.max_wall_seconds is not None and elapsed >= budgets.max_wall_seconds:
        return "max_wall_seconds"
    return ""


def _count_calls(result: TrialResult, calls: Sequence[ModelCall]) -> None:
    result.model_calls += len(calls)
    result.prompt_tokens += sum(call.prompt_tokens or 0 for call in calls)
    result.completion_tokens += sum(call.completion_tokens or 0 for call in calls)


def _observation(trial_dir: Path, step: int, observation: Observation) -> dict[str, Any]:
    frame = None
    if observation.frame is not None:
        frame = f"frames/{step:06d}.png"
        (trial_dir / frame).write_bytes(observation.frame.to_png())
    return {
        "mode": observation.mode.value,
        "frame_count": observation.frame_count,
        "frame": frame,
        "state": observation.state,
    }


def _decision(decision: Decision) -> dict[str, Any]:
    return {
        "action": asdict(decision.action) if decision.action is not None else None,
        "trace": dict(decision.trace) if decision.trace is not None else None,
        "calls": [asdict(call) for call in decision.calls],
    }


def _to_goal(results: Sequence[TrialResult]) -> dict[str, float | None]:
    """Median cost of reaching the goal, over successful trials only, so a fast failure never looks efficient."""
    won = [r for r in results if r.success]

    def median(values: list[int]) -> float | None:
        return statistics.median(values) if values else None

    return {
        "presses": median([r.steps for r in won]),
        "frames": median([r.frames for r in won]),
        "model_calls": median([r.model_calls for r in won]),
        "tokens": median([r.prompt_tokens + r.completion_tokens for r in won]),
        "wall_seconds": median([r.wall_seconds for r in won]),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
