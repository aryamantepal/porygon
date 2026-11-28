from __future__ import annotations

import json
from pathlib import Path

from agents.direct import DirectPolicy
from agents.models import ScriptedModel
from environments.lineworld import LineWorld, LineWorldConfig
from environments.protocol import ObservationMode
from evaluation.goals import Goal, goal_from_dict
from evaluation.runner import Budgets, run

GOAL = Goal("reach-end", "Walk to the far right.", {"x": 9})


def test_one_call_per_decision_with_no_history(tmp_path):
    model = ScriptedModel(lambda messages: "RIGHT")
    results = run(run_dir=tmp_path / "run", env=LineWorld(), player=DirectPolicy(model), goal=GOAL, budgets=Budgets(max_steps=20))
    assert results[0].success and results[0].model_calls == 9
    for call in model.calls:
        assert [m["role"] for m in call.messages] == ["system", "user"]
        assert "Goal: Walk to the far right." in call.messages[1]["content"]
    # every request is the same shape: no earlier turns accumulate
    assert len({len(json.dumps(call.messages[0])) for call in model.calls}) == 1


def test_visual_mode_sends_the_frame_and_no_state(tmp_path):
    model = ScriptedModel(lambda messages: "RIGHT")
    env = LineWorld(LineWorldConfig(mode=ObservationMode.VISUAL))
    run(run_dir=tmp_path / "run", env=env, player=DirectPolicy(model), goal=GOAL, budgets=Budgets(max_steps=1))
    user = model.calls[0].messages[1]
    assert user["images"] == 1 and "State:" not in user["content"]


def test_chatty_output_is_invalid_not_guessed(tmp_path):
    model = ScriptedModel(lambda messages: "I think I should press RIGHT")
    results = run(run_dir=tmp_path / "run", env=LineWorld(), player=DirectPolicy(model), goal=GOAL, budgets=Budgets(max_steps=3, max_decisions=3))
    assert results[0].steps == 0 and results[0].invalid_decisions == 3


def test_trace_is_absent_and_thinking_off(tmp_path):
    model = ScriptedModel(lambda messages: "RIGHT")
    run_dir = tmp_path / "run"
    run(run_dir=run_dir, env=LineWorld(), player=DirectPolicy(model), goal=GOAL, budgets=Budgets(max_steps=1))
    record = json.loads((run_dir / "trials/000/steps.jsonl").read_text().splitlines()[1])
    assert record["decision"]["trace"] is None
    config = json.loads((run_dir / "config.json").read_text())
    assert config["mechanisms"] == {"history": False, "thinking": False, "few_shot": False}


def test_experiment_configs_parse():
    from experiments.run import make_environment, make_model, make_player

    for path in Path(__file__).resolve().parents[1].glob("experiments/*.json"):
        spec = json.loads(path.read_text())
        goal_from_dict(spec["goal"])
        Budgets(**spec["budgets"])
        model = make_model(spec["model"])
        make_player(spec["agent"], model)
        if spec["environment"]["name"] in ("lineworld", "gridworld"):
            make_environment(spec["environment"])
