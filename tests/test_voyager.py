from __future__ import annotations

import json

import pytest

from agents.models import ScriptedModel
from agents.voyager import Voyager, prompts
from agents.voyager.skills import Run, SkillError, SkillLibrary, execute, parse
from environments.lineworld import LineWorld, LineWorldConfig
from environments.protocol import ObservationMode
from evaluation.goals import Goal
from evaluation.runner import Budgets, run

KEYS = ["LEFT", "RIGHT", "A"]
LAMP = Goal("lamp", "Light the lamp at the far right end.", {"lamp": True})

WALK = "Explain: none\nPlan:\n1) walk\nCode:\n```skill\nskill walk_right\nuntil x == 9 (max 12):\n  RIGHT\n```"
LIGHT = "Plan:\n1) reuse walk\nCode:\n```skill\nskill light_lamp\ncall walk_right\nA\n```"


def scripted(action=None, critic='{"reasoning": "done", "success": true, "critique": ""}'):
    tasks = ["Reasoning: start by walking.\nTask: Walk to the right end", "Reasoning: now the lamp.\nTask: Light the lamp"]
    proposed = []

    def reply(messages):
        system = messages[0].content
        if system.startswith(prompts.CURRICULUM[:40]):
            proposed.append(tasks[min(len(proposed), len(tasks) - 1)])
            return proposed[-1]
        if system == prompts.CRITIC:
            return critic
        if system == prompts.SKILL_DESCRIPTION:
            return "Walks right to the end of the corridor."
        if action:
            return action(messages)
        return LIGHT if "Task: Light the lamp" in messages[1].content else WALK

    return ScriptedModel(reply)


def run_lineworld(tmp_path, player, steps=40, mode=ObservationMode.BOTH):
    run_dir = tmp_path / "run"
    results = run(run_dir=run_dir, env=LineWorld(LineWorldConfig(mode=mode)), player=player, goal=LAMP, budgets=Budgets(max_steps=steps))
    records = [json.loads(line) for line in (run_dir / "trials/000/steps.jsonl").read_text().splitlines()]
    return results[0], records


def events(records, kind):
    return [e for r in records if r["kind"] == "decision" for e in r["decision"]["trace"]["events"] if e["type"] == kind]


# Skill language


def test_parse_and_execute_counts_every_press():
    program = parse("skill demo\nRIGHT x2\nrepeat 2:\n  LEFT\n  A", KEYS)
    steps = execute(program, None, LineWorld().reset(), run := Run())
    assert [a.key for a in steps] == ["RIGHT", "RIGHT", "LEFT", "A", "LEFT", "A"] and run.presses == 6


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("RIGHT", "must start with `skill <name>`"),
        ("skill s\nJUMP", "not a legal action"),
        ("skill s\nrepeat 2:\nRIGHT", "needs an indented body"),
        ("skill s\nuntil x == nine (max 3):\n  RIGHT", "not a JSON value"),
        ("skill s", "no statements"),
    ],
)
def test_parse_errors_are_readable(source, message):
    with pytest.raises(SkillError, match=message):
        parse(source, KEYS)


def test_until_stops_when_condition_holds_and_reports_when_it_does_not():
    env = LineWorld()
    program = parse("skill s\nuntil x >= 2 (max 5):\n  RIGHT", KEYS)
    steps = execute(program, None, env.reset(), Run())
    action = next(steps)
    with pytest.raises(StopIteration):
        while True:
            action = steps.send(env.step(action).observation)
    assert env.get_progress()["x"] == 2

    env = LineWorld()
    steps = execute(parse("skill s\nuntil x == 5 (max 2):\n  LEFT", KEYS), None, env.reset(), Run())
    action = next(steps)
    with pytest.raises(SkillError, match="still false after 2 iterations"):
        while True:
            action = steps.send(env.step(action).observation)


def test_conditions_need_structured_state():
    env = LineWorld(LineWorldConfig(mode=ObservationMode.VISUAL))
    steps = execute(parse("skill s\nuntil x == 1 (max 2):\n  RIGHT", KEYS), None, env.reset(), Run())
    with pytest.raises(SkillError, match="structured snapshot"):
        next(steps)


def test_library_retrieves_by_similarity_and_persists(tmp_path):
    library = SkillLibrary(tmp_path / "skills.json")
    library.add(parse("skill walk_right\nRIGHT", KEYS), "Walks right along the corridor.", "walk right")
    library.add(parse("skill press_a\nA", KEYS), "Presses A once to interact.", "interact")
    assert [s.program.name for s in library.retrieve("walk to the right end", k=1)] == ["walk_right"]
    reloaded = SkillLibrary(tmp_path / "skills.json")
    reloaded.bind(KEYS)
    assert reloaded.skills["press_a"].program.body == library.skills["press_a"].program.body


# The Voyager loop


def test_learns_a_skill_then_reuses_it(tmp_path):
    player = Voyager(scripted())
    result, records = run_lineworld(tmp_path, player)
    assert result.success
    assert [e["task"] for e in events(records, "task")] == ["Walk to the right end", "Light the lamp"]
    assert [e["name"] for e in events(records, "skill_added")] == ["walk_right"]
    assert events(records, "retrieved_skills")[1]["skills"] == ["walk_right"]
    assert player.completed_tasks == ["Walk to the right end"]
    decisions = [r for r in records if r["kind"] == "decision"]
    assert all(r["decision"]["action"] is not None for r in decisions)
    assert decisions[0]["decision"]["trace"]["skill"] == "walk_right"


def test_failed_attempt_feeds_error_and_critique_into_the_rewrite(tmp_path):
    replies = iter(["Code:\n```skill\nskill bad\nJUMP\n```", WALK, LIGHT])
    model = scripted(action=lambda messages: next(replies))
    result, records = run_lineworld(tmp_path, Voyager(model))
    assert result.success
    second_action_prompt = [c for c in model.calls if c.messages[0]["content"].startswith("You are a helpful assistant that writes skill")][1]
    assert "Execution error: line 2: `JUMP` is not a legal action" in second_action_prompt.messages[1]["content"]
    assert "Code from the last round: skill bad" in second_action_prompt.messages[1]["content"]


def test_critic_failure_triggers_retry_with_critique(tmp_path):
    verdicts = iter(['{"reasoning": "no", "success": false, "critique": "Walk further."}'] + ['{"reasoning": "ok", "success": true, "critique": ""}'] * 5)
    model = scripted()
    original = model.reply

    def reply(messages):
        if messages[0].content == prompts.CRITIC:
            return next(verdicts)
        return original(messages)

    model.reply = reply
    result, records = run_lineworld(tmp_path, Voyager(model))
    assert events(records, "critic")[0]["success"] is False
    prompts_seen = [c.messages[1]["content"] for c in model.calls if c.messages[0]["content"].startswith("You are a helpful assistant that writes skill")]
    assert "Critique: Walk further." in prompts_seen[1]


def test_curriculum_off_uses_the_goal(tmp_path):
    model = scripted(action=lambda messages: LIGHT.replace("call walk_right", "RIGHT x9"))
    result, records = run_lineworld(tmp_path, Voyager(model, curriculum=False))
    assert result.success
    assert events(records, "task")[0] == {"type": "task", "task": LAMP.text, "source": "goal (curriculum off)"}


def test_library_off_stores_nothing_and_call_fails(tmp_path):
    player = Voyager(scripted(), skill_library=False)
    result, records = run_lineworld(tmp_path, player, steps=30)
    assert not player.library.skills and not events(records, "skill_added")
    assert any("library is off" in e["error"] for e in events(records, "program_end"))


def test_self_verification_off_makes_no_critic_calls(tmp_path):
    model = scripted()
    result, records = run_lineworld(tmp_path, Voyager(model, self_verification=False))
    assert result.success and not events(records, "critic")
    assert not any(c.messages[0]["content"] == prompts.CRITIC for c in model.calls)


def test_iterative_prompting_off_gives_one_attempt_without_feedback(tmp_path):
    replies = iter(["Code:\n```skill\nskill bad\nJUMP\n```"] * 3 + [WALK, LIGHT])
    model = scripted(action=lambda messages: next(replies))
    player = Voyager(model, iterative_prompting=False)
    run_lineworld(tmp_path, player, steps=30)
    action_prompts = [c.messages[1]["content"] for c in model.calls if c.messages[0]["content"].startswith("You are a helpful assistant that writes skill")]
    assert all("Execution error: No error" in p for p in action_prompts)
    assert player.failed_tasks[:1] == ["Walk to the right end"]


def test_mechanisms_are_recorded():
    assert Voyager(scripted(), curriculum=False).mechanisms()["curriculum"] is False
