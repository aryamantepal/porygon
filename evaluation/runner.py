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
from evaluation.goals import Chain, Goal, Level


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
    # Chains only; None (and left out of the logs) for a plain goal.
    levels_reached: int | None = None  # levels met, in order
    levels: list[dict[str, Any]] | None = None  # per level: task_id, met, steps, decisions, model_calls, tokens, reason


def run(
    *,
    run_dir: Path,
    env: Environment,
    player: Player,
    goal: Goal | Chain,
    budgets: Budgets,
    trials: int = 1,
    config: Mapping[str, Any] | None = None,
) -> list[TrialResult]:
    """Run `trials` trials from the environment's reset state and log them.

    `config` carries what only the caller knows (environment, initial state,
    model id, observation mode); the runner adds the rest.

    A `Chain` is one trial of several goals in order. The runner moves to the next
    level when the current one's success holds in `get_progress()`, tells the player
    through `next_goal` (or `start_trial` if it has none), and stops at the first
    level whose step budget runs out. Budgets other than a level's own are the
    trial's. A trial succeeds when every level is met.
    """
    chain = goal if isinstance(goal, Chain) else None
    task_id = chain.task_id if chain else goal.task_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_dir / "config.json",
        {
            **(config or {}),
            "task_id": task_id,
            "goal": chain.to_dict() if chain else {"text": goal.text, "success": dict(goal.success)},
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
            "results": [_as_dict(r) for r in results],
            **(_chain_summary(chain, results) if chain else {}),
        },
    )
    return results


@dataclass(frozen=True)
class _Spent:
    """What a trial has used so far, so a level can be charged the difference."""

    steps: int
    decisions: int
    model_calls: int
    tokens: int

    @classmethod
    def of(cls, result: TrialResult) -> _Spent:
        return cls(result.steps, result.decisions, result.model_calls, result.prompt_tokens + result.completion_tokens)

    def since(self, start: _Spent) -> _Spent:
        return _Spent(self.steps - start.steps, self.decisions - start.decisions, self.model_calls - start.model_calls, self.tokens - start.tokens)


def _run_trial(
    trial_dir: Path, index: int, env: Environment, player: Player, goal: Goal | Chain, budgets: Budgets
) -> TrialResult:
    (trial_dir / "frames").mkdir(parents=True)
    chain = goal if isinstance(goal, Chain) else None
    levels = chain.levels if chain else (Level(goal),)
    result = TrialResult(trial=index, success=False, reason="", levels_reached=0 if chain else None, levels=[] if chain else None)
    started = time.perf_counter()
    with (trial_dir / "steps.jsonl").open("w") as log:

        def write(record: dict[str, Any]) -> None:
            log.write(json.dumps(record) + "\n")
            log.flush()

        observation = env.reset()
        player.start_trial(levels[0].goal.text)
        progress = env.get_progress()
        write({"kind": "reset", "step": 0, "observation": _observation(trial_dir, 0, observation), "progress": dict(progress)})

        current = 0
        level_start = _Spent.of(result)

        def begin(carried: str | None = None) -> None:
            if chain:
                record: dict[str, Any] = {"kind": "level", "event": "start", "level": current, "task_id": levels[current].goal.task_id, "text": levels[current].goal.text, "step": result.steps, "progress": dict(progress)}
                if carried:
                    record["carried_by"] = carried
                    if carried == "start_trial":
                        record["warning"] = "player has no next_goal: start_trial wiped its per-trial state, so nothing carried between levels"
                write(record)

        def close(met: bool, why: str) -> None:
            if chain:
                spent = _Spent.of(result).since(level_start)
                row = {"task_id": levels[current].goal.task_id, "met": met, "reason": why, "start_step": level_start.steps, "steps": spent.steps, "decisions": spent.decisions, "model_calls": spent.model_calls, "tokens": spent.tokens}
                result.levels.append(row)
                if met:
                    result.levels_reached += 1
                    write({"kind": "level", "event": "met", "level": current, **{k: row[k] for k in ("task_id", "steps", "model_calls")}, "step": result.steps, "progress": dict(progress)})

        begin()
        while True:
            level = levels[current]
            if level.goal.is_met(progress):
                close(True, "goal_met")
                if current == len(levels) - 1:
                    reason = "goal_met"
                    break
                current += 1
                level_start = _Spent.of(result)
                carried = _carry(player, levels[current].goal.text)
                begin(carried)
                continue  # the next level may already hold in the same progress
            reason = _stop_reason(level, result, budgets, time.perf_counter() - started, _Spent.of(result).since(level_start))
            if reason:
                close(False, reason)
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
                    "goal_met": level.goal.is_met(progress),
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
    _write_json(trial_dir / "outcome.json", _as_dict(result))
    return result


def _carry(player: Player, goal: str) -> str:
    """Give the player the next level's goal. `next_goal` keeps its state; `start_trial` is the fallback and wipes it."""
    next_goal = getattr(player, "next_goal", None)
    if next_goal is not None:
        next_goal(goal)
        return "next_goal"
    player.start_trial(goal)
    return "start_trial"


def _stop_reason(level: Level, result: TrialResult, budgets: Budgets, elapsed: float, spent: _Spent) -> str:
    if result.steps >= budgets.max_steps:
        return "max_steps"
    if result.decisions >= budgets.decisions:
        return "max_decisions"
    if level.max_steps is not None:
        if spent.steps >= level.max_steps:
            return "level_max_steps"
        if spent.decisions >= 2 * level.max_steps:
            return "level_max_decisions"
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


def _as_dict(result: TrialResult) -> dict[str, Any]:
    """The result as logged. The chain fields appear only for a chain, so a plain goal's logs keep their shape."""
    data = asdict(result)
    if result.levels is None:
        del data["levels"], data["levels_reached"]
    return data


def _chain_summary(chain: Chain, results: Sequence[TrialResult]) -> dict[str, Any]:
    reached = [r.levels_reached or 0 for r in results]
    return {
        "levels_total": len(chain.levels),
        "levels_reached_mean": statistics.mean(reached) if reached else None,
        "level_successes": [sum(bool(r.levels and len(r.levels) > i and r.levels[i]["met"]) for r in results) for i in range(len(chain.levels))],
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
