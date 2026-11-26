"""Run one experiment config: `python -m experiments.run experiments/<config>.json`.

This is the only place that wires concrete environments, players, and model
clients together. A run writes to runs/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agents.models import ChatModel, OllamaChat
from environments.protocol import Environment, ObservationMode
from evaluation.goals import Goal
from evaluation.runner import Budgets, run


def make_environment(spec: dict[str, Any]) -> Environment:
    mode = ObservationMode(spec.get("mode", "both"))
    if spec["name"] == "lineworld":
        from environments.lineworld import LineWorld, LineWorldConfig

        return LineWorld(LineWorldConfig(width=spec.get("width", 10), start=spec.get("start", 0), mode=mode))
    if spec["name"] == "firered":
        from environments.pokemon_firered import FireRedConfig, FireRedEnv

        return FireRedEnv(FireRedConfig(rom_path=Path(spec["rom"]), initial_state=Path(spec["state"]), mode=mode))
    raise ValueError(f"unknown environment {spec['name']!r}")


def make_model(spec: dict[str, Any]) -> ChatModel:
    if spec.get("provider", "ollama") == "ollama":
        options = {key: spec[key] for key in ("host", "temperature", "max_tokens", "seed") if key in spec}
        return OllamaChat(spec["id"], **options)
    raise ValueError(f"unknown model provider {spec['provider']!r}")


def make_player(spec: dict[str, Any], model: ChatModel):
    name = spec["name"]
    if name == "direct":
        from agents.direct import DirectPolicy

        return DirectPolicy(model)
    raise ValueError(f"unknown agent {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())

    env = make_environment(spec["environment"])
    model = make_model(spec["model"])
    player = make_player(spec["agent"], model)
    goal = Goal(**spec["goal"])
    run_dir = args.run_dir or Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{goal.task_id}-{player.name}"
    results = run(
        run_dir=run_dir,
        env=env,
        player=player,
        goal=goal,
        budgets=Budgets(**spec["budgets"]),
        trials=spec.get("trials", 1),
        config={"config_file": str(args.config), "environment": spec["environment"], "model": spec["model"], "agent_config": spec["agent"]},
    )
    for result in results:
        print(json.dumps({"trial": result.trial, "success": result.success, "reason": result.reason, "steps": result.steps, "invalid": result.invalid_decisions, "model_calls": result.model_calls}))
    print(f"run written to {run_dir}")


if __name__ == "__main__":
    main()
