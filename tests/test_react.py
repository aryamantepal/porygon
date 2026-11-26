from __future__ import annotations

import json

from agents.models import ScriptedModel
from agents.react import ReAct
from environments.lineworld import LineWorld
from evaluation.goals import Goal
from evaluation.runner import Budgets, run

GOAL = Goal("reach-end", "Walk to the far right.", {"x": 9})


def test_thought_and_action_each_turn(tmp_path):
    model = ScriptedModel(lambda messages: " The end is to the right.\nAction 1: RIGHT")
    run_dir = tmp_path / "run"
    results = run(run_dir=run_dir, env=LineWorld(), player=ReAct(model), goal=GOAL, budgets=Budgets(max_steps=20))
    assert results[0].success and results[0].model_calls == 9
    record = json.loads((run_dir / "trials/000/steps.jsonl").read_text().splitlines()[1])
    assert record["decision"]["trace"]["thought"] == "The end is to the right."
    assert record["decision"]["trace"]["action_retry"] is False


def test_history_carries_earlier_turns():
    model = ScriptedModel(lambda messages: "Keep going.\nAction: RIGHT")
    player = ReAct(model)
    env = LineWorld()
    player.start_trial(GOAL.text)
    observation = env.reset()
    for _ in range(3):
        decision = player.decide(observation, env.legal_actions())
        observation = env.step(decision.action).observation
    last = model.calls[-1].messages[1]["content"]
    assert "Observation 1:" in last and "Thought 1: Keep going." in last and "Action 2: RIGHT" in last
    assert last.rstrip().endswith("Thought 4:") is False and last.rstrip().endswith("Thought 3:")


def test_history_is_trimmed():
    model = ScriptedModel(lambda messages: "x\nAction: LEFT")
    player = ReAct(model, history_turns=2)
    env = LineWorld()
    player.start_trial(GOAL.text)
    observation = env.reset()
    for _ in range(5):
        observation = env.step(player.decide(observation, env.legal_actions()).action).observation
    last = model.calls[-1].messages[1]["content"]
    assert "(2 earlier turns omitted)" in last and "Thought 2:" not in last and "Thought 3: x" in last


def test_missing_action_triggers_one_action_only_call():
    replies = iter(["I should go right.", "RIGHT"])
    model = ScriptedModel(lambda messages: next(replies))
    player = ReAct(model)
    env = LineWorld()
    player.start_trial(GOAL.text)
    decision = player.decide(env.reset(), env.legal_actions())
    assert decision.action.key == "RIGHT" and len(decision.calls) == 2
    assert decision.trace["thought"] == "I should go right." and decision.trace["action_retry"] is True
    assert model.calls[1].messages[1]["content"].endswith("Thought 1: I should go right.\nAction 1:")


def test_invalid_action_is_recorded_in_history():
    replies = iter(["Jump!\nAction: JUMP", "Go.\nAction: RIGHT"])
    model = ScriptedModel(lambda messages: next(replies))
    player = ReAct(model)
    env = LineWorld()
    player.start_trial(GOAL.text)
    observation = env.reset()
    assert player.decide(observation, env.legal_actions()).action is None
    player.decide(observation, env.legal_actions())
    assert "Action 1: (invalid: JUMP)" in model.calls[-1].messages[1]["content"]
