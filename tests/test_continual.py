"""Continual Harness tests. A scripted model on LineWorld and GridWorld; no ROM, no model server."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agents.continual import ContinualHarness
from agents.continual.harness import HarnessState
from agents.continual.player import should_refine
from agents.continual.refiner import Refiner
from agents.examples import Examples
from agents.models import ScriptedModel
from agents.voyager.skills import parse
from environments.gridworld import GridWorld
from environments.lineworld import LineWorld, LineWorldConfig
from environments.protocol import ObservationMode
from evaluation.goals import Goal
from evaluation.runner import Budgets, run
from experiments.run import file_hashes, make_player

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["LEFT", "RIGHT", "A"]
REACH_END = Goal("reach-end", "Walk to the far right.", {"x": 9})
NEVER = Goal("lamp", "Light the lamp at the far right end.", {"lamp": True})

PROMPT_PASS = "You are a prompt optimization expert"
SKILLS_PASS = "You are a harness evolution system"
MEMORY_PASS = "You are a memory curator"
WALK = "skill walk_right\nuntil x == 9 (max 12):\n  RIGHT"


def call(tool, **args):
    return json.dumps({"why": "because", "tool": tool, "args": args})


def scripted(agent=lambda messages: call("press", key="RIGHT"), prompt="Walk right until the end.", skills='{"analysis": "none"}', memory='{"analysis": "none"}'):
    """Agent calls have a system message; each Refiner pass is a lone user message."""

    def reply(messages):
        if messages[0].role == "system":
            return agent(messages)
        text = messages[0].content
        if text.startswith(PROMPT_PASS):
            return prompt
        if text.startswith(SKILLS_PASS):
            return skills
        if text.startswith(MEMORY_PASS):
            return memory
        raise AssertionError(f"unexpected prompt: {text[:60]}")

    return ScriptedModel(reply)


def passes_called(model):
    return [name for c in model.calls for name, head in (("prompt", PROMPT_PASS), ("skills", SKILLS_PASS), ("memory", MEMORY_PASS)) if c.messages[0]["content"].startswith(head)]


def run_world(tmp_path, player, goal=REACH_END, steps=40, trials=1, env=None):
    run_dir = tmp_path / "run"
    results = run(run_dir=run_dir, env=env or LineWorld(), player=player, goal=goal, budgets=Budgets(max_steps=steps), trials=trials)
    records = [[json.loads(line) for line in (run_dir / f"trials/{t:03d}/steps.jsonl").read_text().splitlines()] for t in range(trials)]
    return results, records


def events(records, kind):
    return [e for r in records if r["kind"] == "decision" for e in (r["decision"]["trace"] or {}).get("events", []) if e["type"] == kind]


def drive(player, env, presses, goal="Walk to the far right."):
    """Call decide until `presses` presses were made, stepping the environment; returns the decisions."""
    player.start_trial(goal)
    observation = env.reset()
    decisions = []
    while sum(d.action is not None for d in decisions) < presses:
        decision = player.decide(observation, env.legal_actions())
        decisions.append(decision)
        if decision.action is not None:
            observation = env.step(decision.action).observation
    return decisions


# The agent loop


def test_presses_reach_a_simple_goal(tmp_path):
    model = scripted(lambda messages: call("press", key="RIGHT", times=9))
    results, records = run_world(tmp_path, ContinualHarness(model), steps=20)
    assert results[0].success and results[0].steps == 9 and results[0].model_calls == 1
    plan = events(records[0], "plan")[0]
    assert (plan["tool"], plan["args"]) == ("press", {"key": "RIGHT", "times": 9})
    traces = [r["decision"]["trace"] for r in records[0] if r["kind"] == "decision"]
    assert [t["press"] for t in traces] == list(range(1, 10)) and traces[0]["tool"] == "press" and traces[-1]["step"] == 9


def test_prompt_has_the_original_sections():
    model = scripted()
    player = ContinualHarness(model)
    drive(player, LineWorld(), 1)
    messages = model.calls[0].messages
    system, user = messages[0]["content"], messages[1]["content"]
    for tool in ("press", "write_memory", "refine"):
        assert f"**{tool}**" in system
    assert "run_skill" in system  # skills pass on: the Refiner can add skills
    for section in ("## Learned guidance", "### GOAL", "### RECENT ACTIONS", "### CURRENT STATE", "### LONG-TERM MEMORY", "### SKILL LIBRARY"):
        assert section in user
    assert "### GOAL\nWalk to the far right." in user and '"x": 0' in user


def test_strict_validation_feeds_the_reason_back_and_asks_again():
    replies = iter(
        [
            "I will go right.",
            call("press", key="JUMP"),
            call("press", key="RIGHT", times=99),
            call("press", key="RIGHT", times=True),
            call("press", key="RIGHT", speed=2),
            call("dance"),
            call("press", key="right", times=2),
        ]
    )
    model = scripted(lambda messages: next(replies))
    player = ContinualHarness(model)
    decision = drive(player, LineWorld(), 1)[0]
    assert decision.action.key == "RIGHT" and len(decision.calls) == 7
    errors = [e["error"] for e in decision.trace["events"] if e["type"] == "plan" and "error" in e]
    assert "no JSON object" in errors[0]
    assert "'JUMP'" in errors[1] and "from 1 to 10" in errors[2] and "True" in errors[3]
    assert "unknown fields" in errors[4] and "unknown tool 'dance'" in errors[5]
    # each bad reply is shown back, with its reason, on the next call
    assert "Your last reply could not be used: press: " in model.calls[2].messages[1]["content"]
    assert "unknown tool 'dance'" in model.calls[6].messages[1]["content"]


def test_a_model_that_never_gives_a_valid_call_is_bounded(tmp_path):
    model = scripted(lambda messages: "no tool here")
    results, _ = run_world(tmp_path, ContinualHarness(model, max_calls_per_decision=3), steps=5)
    assert results[0].invalid_decisions == 10 and results[0].model_calls == 30 and results[0].reason == "max_decisions"


def test_run_skill_runs_one_press_per_step_and_counts_every_press():
    model = scripted(lambda messages: call("run_skill", name="walk_right") if "walk_right" in messages[1].content else call("press", key="LEFT"))
    player = ContinualHarness(model, carry_across_trials=True)  # so the skill added now survives start_trial
    player.harness.bind(KEYS)
    player.harness.add_skill(parse(WALK, KEYS), "Walks right to the end.", "high", 4, "refiner")
    env = LineWorld()
    decisions = drive(player, env, 9)
    assert [d.action.key for d in decisions] == ["RIGHT"] * 9 and env.get_progress()["x"] == 9
    assert [d.trace["press"] for d in decisions] == list(range(1, 10)) and decisions[0].trace["skill"] == "walk_right"
    assert len(model.calls) == 1 and player.harness.meta["walk_right"].uses == 1


def test_skill_error_is_shown_next_step_and_counted():
    player = ContinualHarness(scripted(lambda m: call("run_skill", name="stuck") if "FAILED" not in m[1].content else call("press", key="A")))
    player.start_trial("g")
    player.harness.bind(KEYS)
    player.harness.add_skill(parse("skill stuck\nuntil x == 5 (max 2):\n  LEFT", KEYS), "Goes left.", "medium", 3, "refiner")
    env = LineWorld()
    observation = env.reset()
    model = player.model
    for _ in range(4):
        decision = player.decide(observation, env.legal_actions())
        observation = env.step(decision.action).observation
    assert player.harness.meta["stuck"].failures == 1
    assert any("run_skill stuck: FAILED after 2 presses: `until x == 5` still false" in c.messages[1]["content"] for c in model.calls)


# The schedule


def test_schedule_matches_the_authors_code():
    fired = [step for step in range(0, 520) if should_refine(step)]
    assert fired == [25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500]
    assert not should_refine(24) and not should_refine(0) and not should_refine(225)
    assert [s for s in range(45) if should_refine(s, warmup=10, early_every=5, early_until=20, late_every=10)] == [10, 15, 20, 30, 40]


def test_schedule_is_configurable_and_never_fires_before_warmup(tmp_path):
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, refine_warmup=5, refine_early_every=1, refine_early_until=7, refine_late_every=2)
    results, records = run_world(tmp_path, player, goal=NEVER, steps=11)
    assert [e["step"] for e in events(records[0], "refine")] == [5, 6, 7, 8, 10]  # 9 is odd, past early_until
    assert all(e["trigger"] == "schedule" for e in events(records[0], "refine"))
    assert passes_called(model) == ["prompt", "skills", "memory"] * 5


def test_no_refinement_before_warmup(tmp_path):
    model = scripted(lambda messages: call("press", key="LEFT"))
    run_world(tmp_path, ContinualHarness(model, refine_warmup=30), goal=NEVER, steps=29)
    assert passes_called(model) == []


def test_refiner_calls_are_counted_in_the_decision():
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2)
    decisions = drive(player, LineWorld(), 3)
    assert [len(d.calls) for d in decisions] == [1, 1, 4]  # step 2 is due before the third press: 3 passes + 1 agent call
    refine = next(e for e in decisions[2].trace["events"] if e["type"] == "refine")
    assert refine["step"] == 2 and set(refine["passes"]) == {"prompt", "skills", "memory"}


def test_runner_budget_counts_refiner_calls(tmp_path):
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2)
    results, _ = run_world(tmp_path, player, goal=NEVER, steps=5)
    assert results[0].model_calls == 5 + 6  # five agent calls, and 3 passes at steps 2 and 4


def test_a_separate_refiner_model_does_the_passes():
    agent, refiner = scripted(lambda messages: call("press", key="LEFT")), scripted()
    player = ContinualHarness(agent, refiner_model=refiner, refine_warmup=2, refine_early_every=2)
    drive(player, LineWorld(), 3)
    assert passes_called(agent) == [] and passes_called(refiner) == ["prompt", "skills", "memory"]
    assert player.mechanisms()["separate_refiner_model"] is True


# Skills written by the Refiner


def add_skill_reply(code, name="walk_right", **extra):
    return json.dumps({"analysis": "walking is repeated", "add": [{"name": name, "description": "Walks right to the end.", "code": code, "effectiveness": "high", "importance": 4, **extra}]})


def test_a_refiner_skill_that_parses_is_added_and_can_be_run(tmp_path):
    def agent(messages):
        return call("run_skill", name="walk_right") if "- walk_right (effectiveness: high" in messages[1].content else call("press", key="LEFT")

    model = scripted(agent, skills=add_skill_reply(WALK))
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2)
    results, records = run_world(tmp_path, player, steps=30)
    assert results[0].success
    refine = events(records[0], "refine")[0]
    applied = refine["passes"]["skills"]["applied"]
    assert applied[0]["op"] == "add" and applied[0]["id"] == "walk_right" and applied[0]["edit"]["code"] == WALK
    assert refine["passes"]["skills"]["rejected"] == []
    runs = [r["decision"]["trace"] for r in records[0] if r["kind"] == "decision" and r["decision"]["trace"]["skill"]]
    assert runs and runs[0]["skill"] == "walk_right" and runs[0]["press"] == 1
    summary = records[0][-1]["review"]["trace"]["harness"]["skills"]["walk_right"]
    assert summary["effectiveness"] == "high" and summary["importance"] == 4 and summary["uses"] == 1


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("skill bad\nJUMP", "not a legal action"),
        ("RIGHT x2", "must start with `skill <name>`"),
        ("skill other_name\nRIGHT", "is named `other_name`, not `walk_right`"),
        ("skill walk_right\nrepeat 2:\nRIGHT", "needs an indented body"),
        ("skill walk_right\nRIGHT\n" * 400, "limit is 1500"),
    ],
)
def test_a_refiner_skill_that_does_not_parse_is_rejected_and_logged(code, reason):
    model = scripted(skills=add_skill_reply(code))
    harness = HarnessState()
    refinement = Refiner(model).refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=1)
    skills = refinement.passes["skills"]
    assert not harness.library.skills and skills.applied == []
    assert reason in skills.rejected[0]["reason"] and skills.rejected[0]["op"] == "add" and skills.rejected[0]["edit"]["name"] == "walk_right"


def test_rejections_are_shown_to_the_refiner_on_its_next_skills_pass():
    model = scripted(skills=add_skill_reply("skill bad\nJUMP"))
    refiner, harness = Refiner(model), HarnessState()
    for _ in range(2):
        refiner.refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=1)
    prompts_seen = [c.messages[0]["content"] for c in model.calls if c.messages[0]["content"].startswith(SKILLS_PASS)]
    assert "Your Edits Rejected Last Time" not in prompts_seen[0]
    assert "add walk_right: the code does not parse: line 2: `JUMP` is not a legal action" in prompts_seen[1]


def test_skill_updates_and_ratings():
    model = scripted(skills=add_skill_reply(WALK))
    harness = HarnessState()
    refiner = Refiner(model)
    refiner.refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=1)
    model.reply = lambda messages: json.dumps({"update": [{"id": "walk_right", "effectiveness": "low"}, {"id": "missing", "effectiveness": "low"}, {"id": "walk_right", "code": "skill walk_right\nJUMP"}, {"id": "walk_right", "effectiveness": "great"}, {"id": "walk_right"}]}) if messages[0].content.startswith(SKILLS_PASS) else '{"analysis": "x"}'
    skills = refiner.refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=2).passes["skills"]
    assert harness.meta["walk_right"].effectiveness == "low" and harness.library.skills["walk_right"].program.source == WALK
    reasons = [r["reason"] for r in skills.rejected]
    assert reasons[0].startswith("no skill with id") and "is not a legal action" in reasons[1]
    assert reasons[2].startswith('"effectiveness" must be one of') and reasons[3] == "nothing to update"


def test_a_duplicate_skill_is_rejected_and_a_full_library_evicts_the_weakest():
    harness = HarnessState(max_skills=2)
    refiner = Refiner(scripted())
    one = lambda name, importance: {"name": name, "description": "d", "code": f"skill {name}\nA", "importance": importance}
    refiner.model.reply = lambda m: json.dumps({"add": [one("a", 2), one("b", 4), one("b", 4), one("c", 3), one("d", 1)]}) if m[0].content.startswith(SKILLS_PASS) else '{"analysis": "x"}'
    skills = refiner.refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=1).passes["skills"]
    assert sorted(harness.library.skills) == ["b", "c"]  # c evicted a; d ranks below both and is refused
    assert [a.get("evicted") for a in skills.applied] == [[], [], ["a"]]
    assert "already exists" in skills.rejected[0]["reason"] and "outrank" in skills.rejected[1]["reason"]


# Memory


def test_memory_add_update_with_importance_and_caps():
    harness = HarnessState(max_memory=3)
    model = scripted(
        memory=json.dumps(
            {
                "analysis": "gaps",
                "add": [
                    {"path": "world/walls", "title": "Walls", "content": "The corridor ends at x=9.", "importance": 4},
                    {"title": "Lamp", "content": "A on the far wall lights the lamp."},
                    {"title": "Walls", "content": "duplicate", "importance": 3},
                    {"title": "Bad", "content": "x", "importance": 9},
                    {"title": "x" * 200, "content": "long title"},
                    {"title": "Extra", "content": "sixth edit is over the cap", "importance": 3},
                    {"title": "Seventh", "content": "x"},
                ],
                "update": [{"id": "mem_1", "content": "Wall at x=9.", "importance": 2}],
            }
        )
    )
    memory = Refiner(model, prompt=False, skills=False).refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=7).passes["memory"]
    assert harness.memory["mem_1"].importance == 4 and harness.memory["mem_1"].content == "The corridor ends at x=9."  # the update was the 8th edit: over the cap
    assert harness.memory["mem_1"].path == "world/walls" and harness.memory["mem_1"].step == 7 and harness.memory["mem_2"].importance == 3
    reasons = [r["reason"] for r in memory.rejected]
    assert sum("over the 5-edit cap" in r for r in reasons) == 3
    assert any("already exists" in r for r in reasons) and any("from 1 to 5" in r for r in reasons) and any("limit is 80" in r for r in reasons)
    assert passes_called(model) == ["memory"]


def test_memory_update_and_eviction_by_importance():
    harness = HarnessState(max_memory=2)
    first, _ = harness.add_memory("A", "a", 2)
    harness.add_memory("B", "b", 4)
    added, evicted = harness.add_memory("C", "c", 3)
    assert evicted == [first.id] and sorted(e.title for e in harness.memory.values()) == ["B", "C"]
    refused, _ = harness.add_memory("D", "d", 1)
    assert refused is None
    model = scripted(memory=json.dumps({"update": [{"id": added.id, "importance": 5, "content": "c2"}, {"id": "mem_99", "content": "x"}]}))
    memory = Refiner(model, prompt=False, skills=False).refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=1).passes["memory"]
    assert (harness.memory[added.id].importance, harness.memory[added.id].content) == (5, "c2") and "no memory entry" in memory.rejected[0]["reason"]


def test_the_agent_writes_memory_during_play_and_sees_it_next_step():
    replies = iter([call("write_memory", title="Wall", content="x=9 is the end", importance=5), call("press", key="RIGHT"), call("write_memory", title="wall", content="x=9 is the wall", importance=2), call("press", key="RIGHT")])
    model = scripted(lambda messages: next(replies))
    player = ContinualHarness(model)
    decisions = drive(player, LineWorld(), 2)
    assert [len(d.calls) for d in decisions] == [2, 2]
    assert "[mem_1] importance 5 | Wall: x=9 is the end" in model.calls[1].messages[1]["content"]
    assert "write_memory: added mem_1" in model.calls[1].messages[1]["content"]
    assert len(player.harness.memory) == 1 and player.harness.memory["mem_1"].content == "x=9 is the wall" and player.harness.memory["mem_1"].importance == 2
    assert [e["op"] for d in decisions for e in d.trace["events"] if e["type"] == "memory"] == ["added", "updated"]


def test_memory_shown_to_the_agent_is_capped_by_importance():
    player = ContinualHarness(scripted(), memory_shown=2)
    for index, importance in enumerate([1, 5, 3, 4]):
        player.harness.add_memory(f"t{index}", "c", importance)
    text = player.harness.memory_text(2)
    assert "t1" in text and "t3" in text and "t0" not in text and "2 less important entries not shown" in text


# Independent passes


def test_malformed_json_in_one_pass_does_not_block_the_others():
    model = scripted(skills="I think the skills are fine.", memory=json.dumps({"add": [{"title": "T", "content": "c"}]}))
    harness = HarnessState()
    refinement = Refiner(model).refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=3)
    assert refinement.passes["skills"].error == "the reply was not the requested JSON object"
    assert harness.guidance == "Walk right until the end." and harness.memory["mem_1"].title == "T"
    assert len(refinement.calls) == 3


def test_a_reply_cut_off_mid_object_is_not_taken_for_an_empty_edit():
    cut = '{"analysis": "x", "add": [{"name": "a", "description": "d", "code": "skill a\\nA", "importance": 3}'
    refinement = Refiner(scripted(skills=cut)).refine(HarnessState(), goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=1)
    assert "not the requested JSON" in refinement.passes["skills"].error


def test_a_model_error_or_a_crash_in_one_pass_leaves_the_rest():
    class Flaky(ScriptedModel):
        def chat(self, messages, *, think, stop=()):
            if messages[0].content.startswith(SKILLS_PASS):
                raise RuntimeError("boom")
            return super().chat(messages, think=think, stop=stop)

    model = Flaky(lambda m: "Guidance." if m[0].content.startswith(PROMPT_PASS) else '{"add": [{"title": "T", "content": "c"}]}')
    harness = HarnessState()
    refinement = Refiner(model).refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=1)
    assert refinement.passes["skills"].error == "RuntimeError: boom" and harness.guidance == "Guidance." and "mem_1" in harness.memory


def test_guidance_must_fit_the_bound_and_change():
    harness = HarnessState()
    refiner = Refiner(scripted(prompt="x" * 1300), skills=False, memory=False)
    result = refiner.refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=1).passes["prompt"]
    assert harness.guidance == "" and "limit is 1200" in result.rejected[0]["reason"]
    refiner.model.reply = lambda m: "```\nIMPROVED GUIDANCE: Try the door.\n```"
    refiner.refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=2)
    assert harness.guidance == "Try the door."
    result = refiner.refine(harness, goal="g", window="w", failures="", count=1, fixed="f", keys=KEYS, step=3).passes["prompt"]
    assert result.rejected[0]["reason"] == "the guidance is unchanged"


def test_learned_guidance_reaches_the_agent_prompt_and_the_fixed_text_is_not_editable():
    model = scripted(prompt="Prefer the east side.")
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2)
    drive(player, LineWorld(), 3)
    last_user = model.calls[-1].messages[1]["content"]
    assert "## Learned guidance\nPrefer the east side." in last_user and "# Strategic Guidance (Seed)" in last_user
    assert "learned guidance below is updated over time" in last_user


@pytest.mark.parametrize(
    ("switch", "head", "others"),
    [
        ("refine_prompt", PROMPT_PASS, ["skills", "memory"]),
        ("refine_skills", SKILLS_PASS, ["prompt", "memory"]),
        ("refine_memory", MEMORY_PASS, ["prompt", "skills"]),
    ],
)
def test_each_switch_off_disables_its_pass_and_shows_in_mechanisms(switch, head, others):
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2, **{switch: False})
    drive(player, LineWorld(), 3)
    assert passes_called(model) == others
    assert player.mechanisms()[switch] is False
    assert all(v for k, v in player.mechanisms().items() if k in {"refine_prompt", "refine_skills", "refine_memory"} - {switch})


def test_all_passes_off_never_refines_and_hides_the_refine_tool():
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, refine_prompt=False, refine_skills=False, refine_memory=False, refine_warmup=1, refine_early_every=1)
    drive(player, LineWorld(), 4)
    assert passes_called(model) == []
    system = model.calls[0].messages[0]["content"]
    assert "refine" not in system and "run_skill" not in system and "**write_memory**" in system


def test_agent_meta_tools_off_removes_write_memory_and_refine():
    replies = iter([call("write_memory", title="a", content="b"), call("refine"), call("press", key="RIGHT")])
    model = scripted(lambda messages: next(replies))
    player = ContinualHarness(model, agent_meta_tools=False)
    decision = drive(player, LineWorld(), 1)[0]
    system = model.calls[0].messages[0]["content"]
    assert "**write_memory**" not in system and "**refine**" not in system and "**press**" in system
    assert len(decision.calls) == 3 and not player.harness.memory
    assert "unknown tool 'write_memory'" in decision.trace["events"][0]["error"]
    assert player.mechanisms()["agent_meta_tools"] is False
    assert ContinualHarness(model).mechanisms()["agent_meta_tools"] is True


def test_mechanisms_default_to_the_paper():
    mechanisms = ContinualHarness(scripted()).mechanisms()
    assert mechanisms == {
        "refine_prompt": True, "refine_skills": True, "refine_memory": True, "agent_meta_tools": True,
        "carry_across_trials": False, "separate_refiner_model": False, "few_shot": False, "thinking": False,
        "skill_parameters": False,
    }


def test_skills_only_ablation_runs_only_the_skills_pass():
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, refine_prompt=False, refine_memory=False, refine_warmup=2, refine_early_every=2)
    drive(player, LineWorld(), 3)
    assert passes_called(model) == ["skills"] and "## Learned guidance" not in model.calls[-1].messages[1]["content"]


# The agent asking for a refinement


def test_agent_triggered_refine_is_rate_limited():
    replies = iter(
        [call("refine", reason="no progress")]  # nothing to review yet: not counted
        + [call("press", key="LEFT")] * 2
        + [call("refine", reason="stuck", num_steps=2), call("refine", reason="again")]  # second: inside the cooldown
        + [call("press", key="LEFT")] * 3
        + [call("refine"), call("refine")]  # allowed, then inside the cooldown
        + [call("press", key="LEFT")] * 3
        + [call("refine")]  # past the cooldown, over the 2-per-trial cap
        + [call("press", key="LEFT")] * 2
    )
    model = scripted(lambda messages: next(replies))
    player = ContinualHarness(model, refine_warmup=500, refine_cooldown=3, max_agent_refines=2)
    decisions = drive(player, LineWorld(), 9)
    kinds = [e["type"] for d in decisions for e in d.trace["events"] if e["type"].startswith("refine")]
    assert kinds == ["refine_skipped", "refine", "refine_refused", "refine", "refine_refused", "refine_refused"]
    reasons = [e["reason"] for d in decisions for e in d.trace["events"] if e["type"] == "refine_refused"]
    assert "wait until step 5" in reasons[0] and "wait until step 8" in reasons[1] and "all 2 refinements" in reasons[2]
    refine = next(e for d in decisions for e in d.trace["events"] if e["type"] == "refine")
    assert refine["trigger"] == "agent" and refine["reason"] == "stuck" and refine["window"] == 2
    agent_prompts = [c.messages[1]["content"] for c in model.calls if c.messages[0]["role"] == "system"]
    assert any("refine: refused; the last refinement" in p for p in agent_prompts)
    assert any("HARNESS REFINEMENT completed" in p for p in agent_prompts)
    assert len(passes_called(model)) == 6  # two real refinements, three passes each


def test_refine_arguments_are_validated():
    replies = iter([call("refine", num_steps=500), call("refine", reason=3), call("press", key="A")])
    model = scripted(lambda messages: next(replies))
    decision = drive(ContinualHarness(model), LineWorld(), 1)[0]
    errors = [e["error"] for e in decision.trace["events"] if "error" in e]
    assert "from 1 to 100" in errors[0] and '"reason" must be text' in errors[1]


# Loops


def test_repeating_one_call_with_no_change_adds_a_note():
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, repeat_limit=3, refine_warmup=500)
    decisions = drive(player, LineWorld(), 5)
    notes = ["in a row and the state did not change" in c.messages[1]["content"] for c in model.calls]
    assert notes == [False, False, False, True, True]
    assert [e["type"] for d in decisions for e in d.trace["events"] if e["type"] == "repeat_note"] == ["repeat_note"] * 2


def test_a_change_in_state_or_in_the_call_resets_the_loop_note():
    keys = iter(["LEFT", "LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])
    model = scripted(lambda messages: call("press", key=next(keys)))
    drive(ContinualHarness(model, repeat_limit=3, refine_warmup=500), LineWorld(), 7)
    notes = ["in a row" in c.messages[1]["content"] for c in model.calls]
    assert notes == [False, False, False, True, False, False, False]  # the note shows on the call after the third repeat; RIGHT then moves the player


# Trials


def test_a_new_trial_starts_from_an_empty_harness_by_default():
    player = ContinualHarness(scripted())
    player.harness.add_memory("old", "lesson", 3)
    player.harness.guidance = "old guidance"
    player.start_trial("g")
    assert not player.harness.memory and player.harness.guidance == ""


def test_carry_across_trials_keeps_the_harness_and_resets_the_trial():
    player = ContinualHarness(scripted(), carry_across_trials=True)
    player.start_trial("g")
    player.harness.add_memory("old", "lesson", 3)
    player.harness.guidance = "old guidance"
    player._steps = 40
    player.start_trial("g")
    assert list(player.harness.memory) == ["mem_1"] and player.harness.guidance == "old guidance"
    assert player._steps == 0 and player._trajectory == [] and player.mechanisms()["carry_across_trials"] is True


def test_trials_in_one_run_are_independent_unless_carried(tmp_path):
    def agent(messages):
        return call("write_memory", title="seen", content="c") if "No memory entries yet." in messages[1].content else call("press", key="RIGHT", times=9)

    for carry in (False, True):
        player = ContinualHarness(scripted(agent), carry_across_trials=carry)
        _, records = run_world(tmp_path / str(carry), player, trials=2)
        memory = [r[-1]["review"]["trace"]["harness"]["memory"] for r in records]
        assert [len(m) for m in memory] == [1, 1]
        # carried: the second trial saw the first trial's note and did not write again
        assert [m[0]["id"] for m in memory] == ["mem_1", "mem_1"]
        assert (len(events(records[1], "memory")) == 1) is (not carry)


def test_end_of_trial_review_holds_the_whole_harness_for_replay(tmp_path):
    model = scripted(
        lambda messages: call("press", key="LEFT"), prompt="Go east.", skills=add_skill_reply(WALK),
        memory=json.dumps({"add": [{"title": "Wall", "content": "x=9", "importance": 5}]}),
    )
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2)
    results, records = run_world(tmp_path, player, goal=NEVER, steps=5)
    review = records[0][-1]["review"]["trace"]
    assert review["steps"] == 5 and [r["step"] for r in review["refinements"]] == [2, 4]
    harness = review["harness"]
    assert harness["guidance"] == "Go east." and harness["skills"]["walk_right"]["source"] == WALK
    assert harness["memory"][0]["importance"] == 5 and harness["skills"]["walk_right"]["effectiveness"] == "high"
    assert json.dumps(harness)  # replayable


def test_the_harness_persists_to_a_path_and_seeds_the_next_run(tmp_path):
    path = tmp_path / "harness.json"
    model = scripted(lambda messages: call("press", key="LEFT"), skills=add_skill_reply(WALK))
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2, harness_path=path)
    run_world(tmp_path, player, goal=NEVER, steps=3)
    saved = json.loads(path.read_text())
    assert saved["skills"]["walk_right"]["source"] == WALK and saved["guidance"]

    seeded = ContinualHarness(scripted(lambda messages: call("run_skill", name="walk_right")), harness_path=path)
    drive(seeded, LineWorld(), 9)
    assert seeded.harness.meta["walk_right"].uses == 1 and seeded.harness.library.skills["walk_right"].program.body
    # a trial that is not carried restarts from the file's state, not from empty
    seeded.start_trial("g")
    assert "walk_right" in seeded.harness.library.skills
    assert file_hashes({"environment": {}, "agent": {"harness_path": str(path)}})[str(path)]


def test_a_malformed_harness_file_is_an_error(tmp_path):
    path = tmp_path / "harness.json"
    path.write_text(json.dumps({"skills": {"a": {"description": "no source"}}}))
    with pytest.raises(ValueError, match="malformed harness state"):
        ContinualHarness(scripted(), harness_path=path)


def test_a_saved_skill_that_no_longer_parses_is_dropped(tmp_path):
    path = tmp_path / "harness.json"
    path.write_text(json.dumps({"skills": {"a": {"source": "skill a\nJUMP", "description": "d"}}}))
    player = ContinualHarness(scripted(), harness_path=path)
    drive(player, LineWorld(), 1)
    assert not player.harness.library.skills and not player.harness.meta


# Other observation modes and games


def test_visual_mode_runs_without_state():
    model = scripted(lambda messages: call("press", key="RIGHT"))
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2)
    env = LineWorld(LineWorldConfig(mode=ObservationMode.VISUAL))
    drive(player, env, 3)
    assert "state not available in this observation mode" in model.calls[-1].messages[1]["content"]
    assert passes_called(model) == ["prompt", "skills", "memory"]


def test_the_window_shows_what_each_press_did_and_flags_blocked_steps():
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, refine_warmup=3, refine_early_every=3)
    drive(player, LineWorld(), 4)
    window = next(c.messages[0]["content"] for c in model.calls if c.messages[0]["content"].startswith(SKILLS_PASS))
    assert 'Step 1: LEFT via press x1 | reasoning: "because" | result: no state change' in window
    assert "Issue: the state did not change after this press (possibly blocked)" in window
    assert "Pattern: same state as 2 presses ago (possible loop)" in window


def test_window_is_limited_to_the_configured_steps():
    model = scripted(lambda messages: call("press", key="RIGHT"))
    player = ContinualHarness(model, refine_warmup=6, refine_early_every=6, refine_window=2)
    drive(player, LineWorld(), 7)
    window = next(c.messages[0]["content"] for c in model.calls if c.messages[0]["content"].startswith(MEMORY_PASS))
    assert "Step 5:" in window and "Step 6:" in window and "Step 4:" not in window and "Last 2 steps" not in window and "last 2 steps" in window


def test_gridworld_press_and_skill_through_the_door(tmp_path):
    down_right = "skill to_door\nDOWN x2\nRIGHT x2"

    def agent(messages):
        return call("run_skill", name="to_door") if "- to_door" in messages[1].content else call("press", key="A")

    model = scripted(agent, skills=json.dumps({"add": [{"name": "to_door", "description": "Goes to the door.", "code": down_right}]}))
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2)
    env = GridWorld()
    drive(player, env, 6)  # two presses of A, then the four presses of the skill
    assert env.get_progress()["position"] == [3, 3] and player.harness.meta["to_door"].uses == 1


# Few-shot, config, and the boundary

EXAMPLES = Examples.from_dict(
    {
        "source": "test",
        "programs": ["skill example_program\nA"],
        "tool_calls": [
            {"situation": {"x": 3}, "goal": "Light the lamp.", "reply": {"why": "walk", "tool": "press", "args": {"key": "RIGHT"}}},
            {"situation": {"x": 3}, "reply": {"why": "other agent's tool", "tool": "go_to", "args": {}}},
        ],
    }
)


def test_few_shot_shows_our_tool_calls_to_the_agent_and_programs_to_the_refiner():
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2, examples=EXAMPLES)
    drive(player, LineWorld(), 3)
    system = model.calls[0].messages[0]["content"]
    assert 'Reply: {"why": "walk", "tool": "press", "args": {"key": "RIGHT"}}' in system and "go_to" not in system
    skills_prompt = next(c.messages[0]["content"] for c in model.calls if c.messages[0]["content"].startswith(SKILLS_PASS))
    assert "skill example_program\nA" in skills_prompt and "do not `call` them" in skills_prompt
    assert not player.harness.library.skills and player.mechanisms()["few_shot"] is True


def test_zero_shot_prompts_have_no_examples():
    model = scripted(lambda messages: call("press", key="LEFT"))
    player = ContinualHarness(model, refine_warmup=2, refine_early_every=2)
    drive(player, LineWorld(), 3)
    assert "Here are example replies" not in model.calls[0].messages[0]["content"] and player.mechanisms()["few_shot"] is False


def test_examples_that_are_not_ours_are_zero_shot():
    examples = Examples.from_dict({"source": "t", "tool_calls": [{"situation": "s", "reply": {"tool": "go_to"}}]})
    assert ContinualHarness(scripted(), examples=examples).mechanisms()["few_shot"] is False


def test_make_player_wires_every_switch_from_config():
    model = scripted()
    player = make_player(
        {
            "name": "continual", "refine_prompt": False, "refine_skills": False, "refine_memory": False, "agent_meta_tools": False,
            "carry_across_trials": True, "refine_warmup": 7, "refine_early_every": 3, "refine_early_until": 20, "refine_late_every": 10,
            "refine_window": 12, "refiner_model": {"provider": "ollama", "id": "other-model"}, "examples": "experiments/examples/gridworld.json",
        },
        model,
    )
    assert player.name == "continual"
    assert player.mechanisms() == {
        "refine_prompt": False, "refine_skills": False, "refine_memory": False, "agent_meta_tools": False,
        "carry_across_trials": True, "separate_refiner_model": True, "few_shot": True, "thinking": False,
        "skill_parameters": False,
    }
    assert player.schedule == {"warmup": 7, "early_every": 3, "early_until": 20, "late_every": 10} and player.refine_window == 12
    assert player.refiner.model.model_id == "other-model"
    assert make_player({"name": "continual"}, model).refiner.model is model


def test_continual_configs_are_valid():
    paths = sorted(ROOT.glob("experiments/*_continual*.json"))
    assert {p.name for p in paths} == {"firered_7b_continual.json", "gridworld_7b_continual.json", "gridworld_7b_continual_params.json", "gridworld_7b_continual_skills_only.json"}
    for path in paths:
        spec = json.loads(path.read_text())
        mechanisms = make_player(spec["agent"], scripted()).mechanisms()
        assert spec["agent"]["name"] == "continual" and mechanisms["few_shot"] is False
    skills_only = make_player(json.loads((ROOT / "experiments/gridworld_7b_continual_skills_only.json").read_text())["agent"], scripted()).mechanisms()
    assert (skills_only["refine_skills"], skills_only["refine_prompt"], skills_only["refine_memory"]) == (True, False, False)
    base = json.loads((ROOT / "experiments/gridworld_7b_continual.json").read_text())
    voyager = json.loads((ROOT / "experiments/gridworld_7b_voyager.json").read_text())
    assert {k: base[k] for k in ("environment", "model", "goal", "budgets", "trials")} == {k: voyager[k] for k in ("environment", "model", "goal", "budgets", "trials")}


GAME_WORDS = re.compile(r"pok[eé]mon|pok[eé]agent|fire\s?red|crystal|emerald|\bgba\b|mgba|game\s?boy", re.IGNORECASE)


def test_the_continual_package_names_no_game():
    paths = sorted((ROOT / "agents" / "continual").rglob("*.py"))
    assert len(paths) >= 5
    for path in paths:
        found = GAME_WORDS.search(path.read_text())
        assert not found, f"{path} names a game: {found.group(0)}"


def test_the_continual_package_imports_no_emulator_or_adapter():
    import ast

    for path in (ROOT / "agents" / "continual").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                assert not name.startswith(("environments.pokemon_firered", "environments.gridworld", "environments.lineworld", "mgba")), f"{path} imports {name}"
