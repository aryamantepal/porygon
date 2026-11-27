from __future__ import annotations

import json

from agents.models import ScriptedModel
from agents.reflexion import Reflexion
from agents.reflexion.prompts import REFLECTION_PROMPT
from environments.lineworld import LineWorld
from evaluation.goals import Goal
from evaluation.runner import Budgets, run

GOAL = Goal("reach-end", "Walk to the far right.", {"x": 9})


def scripted():
    """Walks LEFT until a reflection says to go right, then walks RIGHT."""

    def reply(messages):
        if messages[0].content.startswith(REFLECTION_PROMPT):
            return "Plan: I walked left into the wall. Press RIGHT every turn."
        system = messages[0].content
        return "Memory says right.\nAction: RIGHT" if "Press RIGHT" in system else "Try left.\nAction: LEFT"

    return ScriptedModel(reply)


def test_fail_reflect_retry(tmp_path):
    run_dir = tmp_path / "run"
    player = Reflexion(scripted())
    results = run(run_dir=run_dir, env=LineWorld(), player=player, goal=GOAL, budgets=Budgets(max_steps=12), trials=2)
    assert [r.success for r in results] == [False, True]
    assert [r.id for r in player.memory] == ["r0"]

    first = [json.loads(line) for line in (run_dir / "trials/000/steps.jsonl").read_text().splitlines()]
    reflection = first[-1]["review"]["trace"]["reflection"]
    assert reflection["failure"] == "max_steps after 12 steps" and reflection["written_after_trial"] == 0
    query = first[-1]["review"]["calls"][0]["messages"][0]["content"]
    assert "Here is the task:" in query and "STATUS: FAIL" in query and query.endswith("New plan:")

    second = [json.loads(line) for line in (run_dir / "trials/001/steps.jsonl").read_text().splitlines()]
    decisions = [r for r in second if r["kind"] == "decision"]
    assert all(r["decision"]["trace"]["reflections_in_context"] == ["r0"] for r in decisions)
    assert "Your memory for the task below:\nTrial 0:\nPlan: I walked left" in decisions[0]["decision"]["calls"][0]["messages"][0]["content"]


def test_reflections_off_is_the_ablation(tmp_path):
    player = Reflexion(scripted(), reflections=False)
    results = run(run_dir=tmp_path / "run", env=LineWorld(), player=player, goal=GOAL, budgets=Budgets(max_steps=12), trials=2)
    assert [r.success for r in results] == [False, False]
    assert player.memory == [] and player.mechanisms()["reflections"] is False


def test_memory_window_keeps_the_last_three():
    player = Reflexion(ScriptedModel(lambda messages: "Plan: x\nAction: LEFT"), memory_size=3)
    env = LineWorld()
    from agents.protocol import TrialOutcome

    for _ in range(5):
        player.start_trial(GOAL.text)
        player.decide(env.reset(), env.legal_actions())
        player.end_trial(TrialOutcome(False, "max_steps", 1))
    player.start_trial(GOAL.text)
    assert [r.id for r in player._in_context] == ["r2", "r3", "r4"]


def test_success_writes_no_reflection():
    from agents.protocol import TrialOutcome

    model = ScriptedModel(lambda messages: "go\nAction: RIGHT")
    player = Reflexion(model)
    player.start_trial(GOAL.text)
    review = player.end_trial(TrialOutcome(True, "goal_met", 9))
    assert review.calls == () and player.memory == []


def test_memory_size_zero_keeps_none():
    player = Reflexion(ScriptedModel(lambda messages: "Plan: x\nAction: LEFT"), memory_size=0)
    env = LineWorld()
    from agents.protocol import TrialOutcome

    for _ in range(2):
        player.start_trial(GOAL.text)
        player.decide(env.reset(), env.legal_actions())
        player.end_trial(TrialOutcome(False, "max_steps", 1))
    player.start_trial(GOAL.text)
    assert player._in_context == []
