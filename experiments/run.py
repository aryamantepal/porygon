"""Run one experiment config: `python -m experiments.run experiments/<config>.json`.

This is the only place that wires concrete environments, players, and model
clients together. A run writes to runs/ (gitignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from agents.examples import Examples
from agents.models import ChatModel, OllamaChat
from environments.protocol import Environment, ObservationMode
from evaluation.goals import Goal
from evaluation.runner import Budgets, run


def make_environment(spec: dict[str, Any]) -> Environment:
    mode = ObservationMode(spec.get("mode", "both"))
    if spec["name"] == "lineworld":
        from environments.lineworld import LineWorld, LineWorldConfig

        return LineWorld(LineWorldConfig(width=spec.get("width", 10), start=spec.get("start", 0), mode=mode))
    if spec["name"] == "gridworld":
        from environments.gridworld import GridWorld, GridWorldConfig

        return GridWorld(GridWorldConfig(mode=mode))
    if spec["name"] == "firered":
        from environments.pokemon_firered import FireRedConfig, FireRedEnv

        return FireRedEnv(FireRedConfig(rom_path=Path(spec["rom"]), initial_state=Path(spec["state"]), mode=mode))
    raise ValueError(f"unknown environment {spec['name']!r}")


def make_model(spec: dict[str, Any]) -> ChatModel:
    if spec.get("provider", "ollama") == "ollama":
        options = {key: spec[key] for key in ("host", "temperature", "max_tokens", "seed", "context_length") if key in spec}
        return OllamaChat(spec["id"], **options)
    raise ValueError(f"unknown model provider {spec['provider']!r}")


def make_examples(spec: dict[str, Any]) -> Examples | None:
    """The few-shot file named by the agent config, parsed; no key means zero-shot."""
    return Examples.from_dict(json.loads(Path(spec["examples"]).read_text())) if spec.get("examples") else None


def interface(spec: dict[str, Any]) -> dict[str, Any]:
    """The action-interface switches ReAct, Reflexion, and Voyager share: `"action_interface"` ("buttons" or "tools") and `"specialists"` (tools only)."""
    return {"action_interface": spec.get("action_interface", "buttons"), "specialists": spec.get("specialists", False)}


def make_player(spec: dict[str, Any], model: ChatModel):
    name = spec["name"]
    examples = make_examples(spec)
    if name == "direct":
        from agents.direct import DirectPolicy

        return DirectPolicy(model, examples=examples)
    if name == "react":
        from agents.react import ReAct

        return ReAct(
            model,
            history_turns=spec.get("history_turns", 10),
            retry_illegal_action=spec.get("retry_illegal_action", False),
            examples=examples,
            **interface(spec),
        )
    if name == "reflexion":
        from agents.reflexion import Reflexion

        return Reflexion(
            model,
            history_turns=spec.get("history_turns", 10),
            reflections=spec.get("reflections", True),
            memory_size=spec.get("memory_size", 3),
            retry_illegal_action=spec.get("retry_illegal_action", False),
            examples=examples,
            **interface(spec),
        )
    if name == "voyager":
        from agents.voyager import Voyager

        return Voyager(
            model,
            curriculum=spec.get("curriculum", True),
            skill_library=spec.get("skill_library", True),
            iterative_prompting=spec.get("iterative_prompting", True),
            self_verification=spec.get("self_verification", True),
            max_attempts=spec.get("max_attempts", 4),
            library_path=Path(spec["library_path"]) if spec.get("library_path") else None,
            examples=examples,
            **interface(spec),
        )
    if name == "crystal_harness":
        from agents.harness import CrystalHarness

        return CrystalHarness(
            model,
            backbone=json.loads(Path(spec["backbone"]).read_text()) if spec.get("backbone") else None,
            knowledge_path=Path(spec["knowledge_path"]) if spec.get("knowledge_path") else None,
            summarize_dialogue=spec.get("summarize_dialogue", True),
            stuck_ladder=spec.get("stuck_ladder", True),
            examples=examples,
        )
    if name == "continual":
        from agents.continual import ContinualHarness

        return ContinualHarness(
            model,
            refiner_model=make_model(spec["refiner_model"]) if spec.get("refiner_model") else None,
            refine_prompt=spec.get("refine_prompt", True),
            refine_skills=spec.get("refine_skills", True),
            refine_memory=spec.get("refine_memory", True),
            agent_meta_tools=spec.get("agent_meta_tools", True),
            carry_across_trials=spec.get("carry_across_trials", False),
            refine_warmup=spec.get("refine_warmup", 25),
            refine_early_every=spec.get("refine_early_every", 25),
            refine_early_until=spec.get("refine_early_until", 200),
            refine_late_every=spec.get("refine_late_every", 100),
            refine_window=spec.get("refine_window", 50),
            refine_cooldown=spec.get("refine_cooldown", 10),
            max_agent_refines=spec.get("max_agent_refines", 3),
            repeat_limit=spec.get("repeat_limit", 3),
            max_memory=spec.get("max_memory", 30),
            max_skills=spec.get("max_skills", 12),
            harness_path=Path(spec["harness_path"]) if spec.get("harness_path") else None,
            examples=examples,
        )
    raise ValueError(f"unknown agent {name!r}")


def code_version() -> dict[str, Any]:
    """The commit the run's code came from, and whether the tree had uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def file_hashes(spec: dict[str, Any]) -> dict[str, str | None]:
    """SHA-256 of every file the run starts from (ROM, saved state, examples, agent memory files).

    A path alone does not pin a run: a save can be overwritten. None means the
    file did not exist when the run started (a memory file the run will create).
    """
    paths = [spec["environment"].get(key) for key in ("rom", "state")]
    paths += [spec["agent"].get(key) for key in ("backbone", "knowledge_path", "library_path", "harness_path", "examples")]
    hashes = {}
    for path in filter(None, paths):
        file = Path(path)
        hashes[path] = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    hashes = file_hashes(spec)  # before anything can write to those files

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
        config={"config_file": str(args.config), "code": code_version(), "files": hashes, "environment": spec["environment"], "model": spec["model"], "agent_config": spec["agent"]},
    )
    for result in results:
        print(json.dumps({"trial": result.trial, "success": result.success, "reason": result.reason, "steps": result.steps, "invalid": result.invalid_decisions, "model_calls": result.model_calls}))
    print(f"run written to {run_dir}")


if __name__ == "__main__":
    main()
