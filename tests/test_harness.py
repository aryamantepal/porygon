from __future__ import annotations

import json

from agents.harness import CrystalHarness
from agents.harness.battle import allowed_options
from agents.harness.memory import KnowledgeGraph
from agents.harness.tools import find_path
from agents.models import ScriptedModel
from environments.gridworld import GridWorld, GridWorldConfig
from environments.protocol import ObservationMode
from evaluation.goals import Goal
from evaluation.runner import Budgets, run

MILESTONE = Goal("milestone", "Reach the milestone in town.", {"milestone": True})
CATCH = Goal("catch", "Catch the creature in the grass.", {"caught": True})
BACKBONE = {
    "nodes": {"town": {"role": "place", "about": "town with grass and a milestone"}, "person": {"role": "person", "about": "lives in the house"}},
    "edges": [["person", "lives_in", "house"]],
    "recipes": {"catching": "Weaken the creature first, then use the catch option."},
}


def state_of(messages):
    for line in messages[1].content.splitlines():
        if line.startswith("State: "):
            return json.loads(line[len("State: "):])
    return {}


def planner_model(plan, text=json.dumps):
    """Planner replies are `text(plan(state))`; dialogue and battle get fixed replies."""

    def reply(messages):
        first = messages[0].content
        if first.startswith("Read this dialogue"):
            return "- The milestone is in the southeast corner of town."
        if first.startswith("You choose the next move"):
            return '{"why": "weaken it first", "option": "tackle"}'
        return text(plan(state_of(messages)))

    return ScriptedModel(reply)


def to_milestone(state):
    if state["map"] == "house":
        return {"why": "the door leads to town", "tool": "go_to", "args": {"place": "door_to_town"}, "intention": "reach the milestone"}
    return {"why": "the lead says southeast", "tool": "go_to", "args": {"place": "milestone"}, "intention": "reach the milestone", "battle_goal": "catch"}


def records_of(run_dir):
    return [json.loads(line) for line in (run_dir / "trials/000/steps.jsonl").read_text().splitlines()]


def events(records, kind):
    return [e for r in records if r["kind"] == "decision" for e in r["decision"]["trace"]["events"] if e["type"] == kind]


def test_path_goes_around_walls():
    grid = ["#####", "#...#", "#.#.#", "#...#", "#####"]
    path = find_path(grid, ".", (1, 1), {(3, 3)}, set())
    assert path[0] == (1, 1) and path[-1] == (3, 3) and len(path) == 5 and (2, 2) not in path


def test_navigator_walks_through_the_door_to_the_milestone(tmp_path):
    harness = CrystalHarness(planner_model(to_milestone))
    results = run(run_dir=tmp_path / "run", env=GridWorld(), player=harness, goal=MILESTONE, budgets=Budgets(max_steps=60))
    assert results[0].success
    records = records_of(tmp_path / "run")
    tool_ends = [e for e in events(records, "tool_end") if e["handler"] == "planner"]
    assert tool_ends[0]["result"] == "went through to town"
    assert all(r["decision"]["action"] for r in records if r["kind"] == "decision")
    # the planner was asked only between tools: far fewer calls than presses
    assert results[0].model_calls < results[0].steps


def test_dialogue_becomes_leads(tmp_path):
    talked = []

    def plan(state):
        if not talked:
            talked.append(True)
            return {"why": "ask around", "tool": "talk_to", "args": {"place": "person"}, "intention": "find the milestone"}
        return to_milestone(state)

    harness = CrystalHarness(planner_model(plan))
    run(run_dir=tmp_path / "run", env=GridWorld(), player=harness, goal=MILESTONE, budgets=Budgets(max_steps=60))
    records = records_of(tmp_path / "run")
    assert events(records, "dialogue")[0]["leads"] == ["The milestone is in the southeast corner of town."]
    later_prompts = [c["messages"][-1]["content"] for r in records if r["kind"] == "decision" for c in r["decision"]["calls"] if c["messages"][-1]["content"].startswith("Goal:")]
    assert "Leads:\n- The milestone is in the southeast corner of town." in later_prompts[-1]


def test_battle_rules_keep_the_catch_alive():
    battle = {
        "options": [
            {"id": "tackle", "kind": "attack", "power": 1, "uses_left": 10},
            {"id": "slam", "kind": "attack", "power": 3, "uses_left": 0},
            {"id": "ball", "kind": "catch", "uses_left": 3},
            {"id": "run", "kind": "flee", "uses_left": 99},
        ],
        "opponent": {"hp": 1, "max_hp": 3},
    }
    allowed, removed = allowed_options(battle, "catch")
    assert [o["id"] for o in allowed] == ["ball"]
    assert any("no uses left" in r for r in removed) and any("knock out" in r for r in removed)
    allowed, _ = allowed_options(battle, "defeat")
    assert [o["id"] for o in allowed] == ["tackle", "ball", "run"]


def test_catches_without_knocking_out(tmp_path):
    def plan(state):
        if state["map"] == "house":
            return {"why": "go to town", "tool": "go_to", "args": {"place": "door_to_town"}, "intention": "catch the creature", "battle_goal": "catch"}
        return {"why": "grass has creatures", "tool": "go_to", "args": {"position": [4, 3]}, "intention": "catch the creature", "battle_goal": "catch"}

    results = run(run_dir=tmp_path / "run", env=GridWorld(), player=CrystalHarness(planner_model(plan)), goal=CATCH, budgets=Budgets(max_steps=80))
    assert results[0].success
    choices = [e["choice"] for e in events(records_of(tmp_path / "run"), "battle")]
    assert choices == ["tackle", "tackle", "ball"]


def test_planner_reads_the_first_object_of_a_run_on_reply(tmp_path):
    # qwen2.5:7b closes the object early and keeps writing; that used to cost every call
    def run_on(plan):
        return json.dumps(plan).replace(', "intention"', '}, "intention"', 1)

    harness = CrystalHarness(planner_model(to_milestone, run_on))
    results = run(run_dir=tmp_path / "run", env=GridWorld(), player=harness, goal=MILESTONE, budgets=Budgets(max_steps=60))
    assert results[0].success


def test_an_unusable_reply_is_shown_back_to_the_planner(tmp_path):
    replies = iter(["go to the door"])
    model = planner_model(to_milestone, lambda plan: next(replies, json.dumps(plan)))
    run(run_dir=tmp_path / "run", env=GridWorld(), player=CrystalHarness(model), goal=MILESTONE, budgets=Budgets(max_steps=60))
    assert "could not be used: go to the door" in model.calls[1].messages[1]["content"]
    assert "could not be used" not in model.calls[2].messages[1]["content"]


def test_stuck_ladder_climbs_to_deep_think_and_help(tmp_path):
    model = planner_model(lambda state: {"why": "try the wall", "tool": "press", "args": {"key": "LEFT"}, "intention": "leave"})
    run(run_dir=tmp_path / "run", env=GridWorld(), player=CrystalHarness(model), goal=MILESTONE, budgets=Budgets(max_steps=14))
    rungs = [e["rung"] for e in events(records_of(tmp_path / "run"), "plan")]
    assert rungs[:2] == ["normal", "remember"] and "nudge" in rungs and "deep_think" in rungs
    assert events(records_of(tmp_path / "run"), "help_requested")
    nudged = [c.messages[1]["content"] for c in model.calls if "going in circles" in c.messages[1]["content"]]
    assert nudged


def test_repeating_a_failed_tool_is_skipped(tmp_path):
    model = planner_model(lambda state: {"why": "go", "tool": "go_to", "args": {"place": "nowhere"}, "intention": "x"})
    run(run_dir=tmp_path / "run", env=GridWorld(), player=CrystalHarness(model, max_calls_per_decision=3), goal=MILESTONE, budgets=Budgets(max_steps=5, max_decisions=1))
    records = records_of(tmp_path / "run")
    assert events(records, "tool_end")[0]["result"] == "no place 'nowhere' on this map"
    assert events(records, "skipped_repeat")


def test_knowledge_backbone_is_fixed_and_notes_persist(tmp_path):
    path = tmp_path / "kg.json"
    graph = KnowledgeGraph(BACKBONE, path)
    assert graph.note("town", "the grass is in the middle")
    assert not graph.note("house_of_cards", "made up")
    graph.edges.append(("person", "is", "villain"))
    assert graph.edges == [("person", "lives_in", "house")]
    reloaded = KnowledgeGraph(None, path)
    assert reloaded.data["nodes"]["town"]["notes"] == ["the grass is in the middle"]
    assert "catching" in reloaded.relevant("catch the creature")["recipes"]


def test_run_summary_is_factual_and_persists(tmp_path):
    path = tmp_path / "kg.json"
    harness = CrystalHarness(planner_model(to_milestone), backbone=BACKBONE, knowledge_path=path)
    run(run_dir=tmp_path / "run", env=GridWorld(), player=harness, goal=MILESTONE, budgets=Budgets(max_steps=60))
    summary = json.loads(path.read_text())["runs"][0]
    assert summary.startswith("Trial 0: goal_met after") and "Maps seen: house, town" in summary


def test_visual_mode_offers_only_press(tmp_path):
    model = planner_model(lambda state: {"why": "move", "tool": "press", "args": {"key": "RIGHT"}, "intention": "explore"})
    env = GridWorld(GridWorldConfig(mode=ObservationMode.VISUAL))
    run(run_dir=tmp_path / "run", env=env, player=CrystalHarness(model), goal=MILESTONE, budgets=Budgets(max_steps=2))
    system = model.calls[0].messages[0]["content"]
    assert "go_to" not in system and "- press" in system


def test_busy_waits_without_calling_the_model(tmp_path):
    from environments.protocol import Observation, Press

    model = planner_model(to_milestone)
    harness = CrystalHarness(model, busy_wait_limit=2)
    harness.start_trial(MILESTONE.text)
    state = {"context": "busy"}
    legal = [Press(k) for k in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B", "WAIT")]
    busy = Observation(ObservationMode.STRUCTURED, 0, None, {**state, "controls": {"wait": "WAIT"}})
    keys = [harness.decide(busy, legal).action.key for _ in range(3)]
    assert keys == ["WAIT", "WAIT", "B"] and model.calls == []


def test_battle_uses_the_adapters_keys_when_given():
    from environments.protocol import Observation, Press

    battle = {
        "options": [
            {"id": "tackle", "kind": "attack", "power": 35, "uses_left": 35, "keys": ["A", "A"]},
            {"id": "growl", "kind": "status", "power": 0, "uses_left": 40, "keys": ["A", "RIGHT", "A"]},
        ],
        "opponent": {"hp": 18, "max_hp": 18, "catchable": False},
    }
    model = ScriptedModel(lambda messages: '{"why": "lower its attack", "option": "growl"}')
    harness = CrystalHarness(model)
    harness.start_trial("win")
    legal = [Press(k) for k in ("UP", "DOWN", "LEFT", "RIGHT", "A", "B")]
    observation = Observation(ObservationMode.STRUCTURED, 0, None, {"context": "battle", "battle": battle})
    assert [harness.decide(observation, legal).action.key for _ in range(3)] == ["A", "RIGHT", "A"]


def test_battle_without_options_just_confirms():
    from environments.protocol import Observation, Press

    model = ScriptedModel(lambda messages: "{}")
    harness = CrystalHarness(model)
    harness.start_trial("win")
    observation = Observation(ObservationMode.STRUCTURED, 0, None, {"context": "battle", "battle": {"moves": [], "options": []}})
    assert harness.decide(observation, [Press("A"), Press("B")]).action.key == "A" and model.calls == []


def test_malformed_args_are_failures_not_crashes(tmp_path):
    bad = iter([
        {"why": "x", "tool": "go_to", "args": "door_to_town"},
        {"why": "x", "tool": "press", "args": {"key": "DOWN", "times": "twice"}},
    ])
    model = planner_model(lambda state: next(bad, to_milestone(state)))
    results = run(run_dir=tmp_path / "run", env=GridWorld(), player=CrystalHarness(model), goal=MILESTONE, budgets=Budgets(max_steps=60))
    assert results[0].success
    records = records_of(tmp_path / "run")
    assert events(records, "plan")[0]["error"].startswith("reply was not")
    assert "whole number" in events(records, "tool_end")[0]["result"]
