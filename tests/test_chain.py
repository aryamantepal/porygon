"""Chains: ordered goals in one trial, each judged from progress. No ROM, no model server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.continual import ContinualHarness
from agents.direct import DirectPolicy
from agents.harness import CrystalHarness
from agents.models import ScriptedModel
from agents.protocol import Decision, Review, TrialOutcome
from agents.react import ReAct
from agents.reflexion import Reflexion
from agents.voyager import Voyager
from environments.gridworld import GridWorld
from environments.lineworld import LineWorld
from environments.protocol import Press
from evaluation import results as results_module
from evaluation import trace
from evaluation.goals import Chain, Goal, Level, goal_from_dict
from evaluation.runner import Budgets, run
from experiments.run import make_player
from tests import test_continual, test_voyager

ROOT = Path(__file__).resolve().parents[1]

# talk to the person, go through the door, walk past the grass to the milestone: 3, 4, and 5 presses
GRIDWORLD_PATH = ["DOWN", "RIGHT", "A", "B", "DOWN", "RIGHT", "DOWN", "RIGHT", "RIGHT", "RIGHT", "DOWN", "DOWN"]


def read_json(path: Path):
    return json.loads(path.read_text())


def read_steps(run_dir: Path, trial: int = 0) -> list[dict]:
    return [json.loads(line) for line in (run_dir / f"trials/{trial:03d}/steps.jsonl").read_text().splitlines()]


def gridworld_chain() -> Chain:
    spec = read_json(ROOT / "experiments/gridworld_7b_chain_crystal_harness.json")
    return goal_from_dict(spec["goal"])


def line_chain(*budgets: int | None) -> Chain:
    """x >= each of 3, 6, 9 in turn, with these per-level step budgets."""
    goals = [Goal(f"x{x}", f"Reach x={x}.", {"x": x}) for x in (3, 6, 9)]
    return Chain("line", tuple(Level(g, b) for g, b in zip(goals, budgets, strict=False)))


class Plain:
    """A player with no `next_goal`: only `start_trial`."""

    name = "plain"

    def __init__(self, key: str = "RIGHT") -> None:
        self.key = key
        self.goals: list[str] = []

    def mechanisms(self):
        return {}

    def start_trial(self, goal: str) -> None:
        self.goals.append(goal)

    def decide(self, observation, legal_actions):
        return Decision(Press(self.key), None)

    def end_trial(self, outcome):
        return Review()


class Carrying(Plain):
    name = "carrying"

    def __init__(self, key: str = "RIGHT") -> None:
        super().__init__(key)
        self.next_goals: list[str] = []

    def next_goal(self, goal: str) -> None:
        self.next_goals.append(goal)


# The runner


def test_chain_advances_through_levels_on_gridworld(tmp_path):
    keys = iter(GRIDWORLD_PATH)
    model = ScriptedModel(lambda messages: next(keys))
    chain = gridworld_chain()
    run_dir = tmp_path / "run"
    result = run(run_dir=run_dir, env=GridWorld(), player=DirectPolicy(model), goal=chain, budgets=Budgets(max_steps=120))[0]

    assert result.success and result.reason == "goal_met" and result.levels_reached == 3 and result.steps == 12
    assert [(r["task_id"], r["met"], r["steps"], r["model_calls"]) for r in result.levels] == [
        ("gridworld-talk", True, 3, 3),
        ("gridworld-town", True, 4, 4),
        ("gridworld-milestone", True, 5, 5),
    ]
    assert [r["start_step"] for r in result.levels] == [0, 3, 7]

    records = read_steps(run_dir)
    levels = [r for r in records if r["kind"] == "level"]
    assert [(r["event"], r["level"], r["step"]) for r in levels] == [
        ("start", 0, 0), ("met", 0, 3), ("start", 1, 3), ("met", 1, 7), ("start", 2, 7), ("met", 2, 12),
    ]
    assert levels[2]["carried_by"] == "next_goal" and "warning" not in levels[2]
    assert levels[1]["progress"]["talked"] is True and levels[3]["progress"]["map"] == "town" and levels[5]["progress"]["milestone"] is True
    assert records[0]["kind"] == "reset" and records[1] == levels[0] and records[-1]["kind"] == "trial_end"

    # the player was told each new goal, with its history untouched by the switch
    goals = [call.messages[1]["content"].split("\n")[0] for call in model.calls]
    assert goals[:3] == ["Goal: Speak with the person who lives in this house."] * 3
    assert goals[3:7] == ["Goal: Go through the door to town."] * 4
    assert goals[7:] == ["Goal: Reach the milestone in town."] * 5

    assert read_json(run_dir / "trials/000/outcome.json")["levels_reached"] == 3
    summary = read_json(run_dir / "summary.json")
    assert (summary["levels_total"], summary["levels_reached_mean"], summary["level_successes"]) == (3, 3, [1, 1, 1])
    config = read_json(run_dir / "config.json")
    assert config["task_id"] == "gridworld-chain" and [l["task_id"] for l in config["goal"]["chain"]] == ["gridworld-talk", "gridworld-town", "gridworld-milestone"]


def test_a_level_already_met_when_the_previous_one_ends_is_passed_at_once(tmp_path):
    chain = Chain("c", (Level(Goal("start", "Be at the start.", {"x": 0})), Level(Goal("three", "Reach x=3.", {"x": 3}))))
    player = Carrying()
    result = run(run_dir=tmp_path / "run", env=LineWorld(), player=player, goal=chain, budgets=Budgets(max_steps=10))[0]
    assert result.success and [(r["task_id"], r["steps"]) for r in result.levels] == [("start", 0), ("three", 3)]
    assert player.goals == ["Be at the start."] and player.next_goals == ["Reach x=3."]


def test_the_model_saying_it_is_done_does_not_advance_a_level(tmp_path):
    model = ScriptedModel(lambda messages: "I have reached x=3, the next goal please. LEFT")
    result = run(run_dir=tmp_path / "run", env=LineWorld(), player=DirectPolicy(model), goal=line_chain(None, None, None), budgets=Budgets(max_steps=4, max_decisions=4))[0]
    assert result.levels_reached == 0 and [r["met"] for r in result.levels] == [False] and result.reason == "max_decisions"


def test_a_level_budget_stops_the_trial_at_that_level(tmp_path):
    run_dir = tmp_path / "run"
    result = run(run_dir=run_dir, env=LineWorld(), player=Carrying(), goal=line_chain(5, 2, 5), budgets=Budgets(max_steps=50))[0]
    # level 0 needs 3 presses and is met; level 1 needs 3 more but is allowed 2
    assert not result.success and result.reason == "level_max_steps" and result.steps == 5
    assert result.levels_reached == 1
    assert [(r["task_id"], r["met"], r["steps"], r["reason"]) for r in result.levels] == [("x3", True, 3, "goal_met"), ("x6", False, 2, "level_max_steps")]
    end = read_steps(run_dir)[-1]
    assert end["kind"] == "trial_end" and end["success"] is False and end["reason"] == "level_max_steps"
    summary = read_json(run_dir / "summary.json")
    assert (summary["levels_reached_mean"], summary["level_successes"]) == (1, [1, 0, 0])


def test_levels_without_a_budget_share_the_trial_budget(tmp_path):
    result = run(run_dir=tmp_path / "run", env=LineWorld(), player=Carrying(), goal=line_chain(None, None, None), budgets=Budgets(max_steps=7))[0]
    assert result.reason == "max_steps" and result.steps == 7 and result.levels_reached == 2
    assert [(r["met"], r["steps"]) for r in result.levels] == [(True, 3), (True, 3), (False, 1)]


def test_level_decision_budget_bounds_invalid_output(tmp_path):
    model = ScriptedModel(lambda messages: "no action here")
    result = run(run_dir=tmp_path / "run", env=LineWorld(), player=DirectPolicy(model), goal=line_chain(2, None, None), budgets=Budgets(max_steps=50))[0]
    assert result.reason == "level_max_decisions" and result.decisions == 4 and result.steps == 0


def test_a_player_without_next_goal_falls_back_to_start_trial_and_the_log_says_so(tmp_path):
    run_dir = tmp_path / "run"
    player = Plain()
    result = run(run_dir=run_dir, env=LineWorld(), player=player, goal=line_chain(10, 10, 10), budgets=Budgets(max_steps=50))[0]
    assert result.success and player.goals == ["Reach x=3.", "Reach x=6.", "Reach x=9."]
    starts = [r for r in read_steps(run_dir) if r["kind"] == "level" and r["event"] == "start"]
    assert "carried_by" not in starts[0]
    assert [r["carried_by"] for r in starts[1:]] == ["start_trial", "start_trial"]
    assert all("no next_goal" in r["warning"] and "wiped" in r["warning"] for r in starts[1:])


def test_trace_shows_level_records(tmp_path):
    run_dir = tmp_path / "run"
    run(run_dir=run_dir, env=LineWorld(), player=Carrying(), goal=line_chain(10, 10, 10), budgets=Budgets(max_steps=50))
    lines = trace.describe(read_steps(run_dir), 0)
    assert [line for line in lines if line.startswith("level")][0].startswith("level 0 start at step 0")
    assert any(line.startswith("level 2 met at step 9") for line in lines)


# Plain goals keep their output shape


def test_single_goal_outputs_keep_their_shape(tmp_path):
    run_dir = tmp_path / "run"
    goal = Goal("reach-end", "Walk to the far right.", {"x": 9})
    result = run(run_dir=run_dir, env=LineWorld(), player=Carrying("RIGHT"), goal=goal, budgets=Budgets(max_steps=50), config={"model": "none"})[0]
    assert result.levels is None and result.levels_reached is None
    records = read_steps(run_dir)
    assert {r["kind"] for r in records} == {"reset", "decision", "trial_end"}
    assert sorted(records[0]) == ["kind", "observation", "progress", "step"]
    assert sorted(records[1]) == ["decision", "decision_seconds", "expansion", "frames_advanced", "goal_met", "kind", "observation", "progress", "step", "step_seconds"]
    assert sorted(records[-1]) == ["kind", "reason", "review", "success"]
    assert sorted(read_json(run_dir / "trials/000/outcome.json")) == sorted(
        ["trial", "success", "reason", "steps", "decisions", "invalid_decisions", "model_calls", "prompt_tokens", "completion_tokens", "wall_seconds", "frames", "final_progress"]
    )
    summary = read_json(run_dir / "summary.json")
    assert sorted(summary) == ["results", "successes", "to_goal", "trials"]
    assert sorted(summary["results"][0]) == sorted(read_json(run_dir / "trials/000/outcome.json"))
    assert sorted(read_json(run_dir / "config.json")) == ["action_space", "agent", "budgets", "goal", "mechanisms", "model", "task_id", "trials"]
    assert read_json(run_dir / "config.json")["goal"] == {"text": "Walk to the far right.", "success": {"x": 9}}


def test_single_goal_results_table_is_unchanged(tmp_path):
    run_dir = tmp_path / "plain"
    run(run_dir=run_dir, env=LineWorld(), player=Carrying(), goal=Goal("reach-end", "Walk.", {"x": 9}), budgets=Budgets(max_steps=50))
    row = results_module.summarize(run_dir)
    assert sorted(row) == sorted(
        ["run", "task", "agent", "model", "on", "successes", "trials", "presses_to_goal", "frames_to_goal", "calls_to_goal", "tokens_to_goal", "invalid_decisions", "model_calls", "decision_seconds", "dirty_code"]
    )
    header = results_module.table([row]).splitlines()[0]
    assert header == "| run | success | presses | frames | calls | tokens | invalid | all calls | s/decision |"


# Results


class SwitchesByTrial(Carrying):
    """Walks right on the first trial and left on the second."""

    name = "switches"

    def __init__(self) -> None:
        super().__init__()
        self.trial = -1

    def start_trial(self, goal: str) -> None:
        self.trial += 1
        self.key = "RIGHT" if self.trial == 0 else "LEFT"


def test_results_table_shows_levels_reached_and_per_level_successes(tmp_path):
    chain = line_chain(10, 10, 10)
    chained = tmp_path / "chain"
    run(run_dir=chained, env=LineWorld(), player=SwitchesByTrial(), goal=chain, budgets=Budgets(max_steps=40), trials=2)
    plain = tmp_path / "plain"
    run(run_dir=plain, env=LineWorld(), player=Carrying(), goal=Goal("reach-end", "Walk.", {"x": 9}), budgets=Budgets(max_steps=50))

    row = results_module.summarize(chained)
    assert (row["levels_total"], row["levels_reached_mean"], row["level_successes"]) == (3, 1.5, [1, 1, 1])
    assert row["successes"] == 1 and row["trials"] == 2

    lines = results_module.table([row]).splitlines()
    assert lines[0].endswith("| levels | per level |")
    assert lines[2].endswith("| 1.5/3 | 1/1/1 |")
    # a table mixing a chain with a plain goal shows "-" for the plain one
    mixed = results_module.table([results_module.summarize(plain), row]).splitlines()
    assert mixed[2].endswith("| - | - |") and mixed[3].endswith("| 1.5/3 | 1/1/1 |")


def test_per_level_successes_count_levels_not_trials(tmp_path):
    run_dir = tmp_path / "run"
    run(run_dir=run_dir, env=LineWorld(), player=Carrying(), goal=line_chain(5, 2, 5), budgets=Budgets(max_steps=50), trials=2)
    row = results_module.summarize(run_dir)
    assert (row["levels_reached_mean"], row["level_successes"]) == (1, [2, 0, 0])
    assert results_module.table([row]).splitlines()[2].endswith("| 1.0/3 | 2/0/0 |")


# next_goal keeps what the player learned


@pytest.mark.parametrize("name", ["direct", "react", "reflexion", "voyager", "crystal_harness", "continual"])
def test_every_player_implements_next_goal(name):
    player = make_player({"name": name}, ScriptedModel(lambda messages: ""))
    player.start_trial("one")
    player.next_goal("two")


def test_direct_and_react_take_the_new_goal_text():
    model = ScriptedModel(lambda messages: "RIGHT")
    env = LineWorld()
    direct = DirectPolicy(model)
    direct.start_trial("one")
    direct.next_goal("two")
    direct.decide(env.reset(), env.legal_actions())
    assert "Goal: two" in model.calls[-1].messages[1]["content"]

    react = ReAct(ScriptedModel(lambda messages: "Thought: go.\nAction: RIGHT"))
    react.start_trial("one")
    observation = env.reset()
    react.decide(observation, env.legal_actions())
    react.next_goal("two")
    react.decide(observation, env.legal_actions())
    assert len(react._turns) == 2  # the history stays
    assert "Goal: two" in react.model.calls[-1].messages[1]["content"]


def test_reflexion_keeps_its_memory_and_history_across_levels():
    player = Reflexion(ScriptedModel(lambda messages: "Plan: press RIGHT.\nAction: RIGHT"))
    env = LineWorld()
    player.start_trial("one")
    player.decide(env.reset(), env.legal_actions())
    player.end_trial(TrialOutcome(False, "max_steps", 1))
    assert [r.id for r in player.memory] == ["r0"]

    player.start_trial("one")
    observation = env.reset()
    player.decide(observation, env.legal_actions())
    player.next_goal("two")
    assert [r.id for r in player.memory] == ["r0"] and [r.id for r in player._in_context] == ["r0"]
    assert len(player._turns) == 1 and player._goal == "two"
    player.decide(observation, env.legal_actions())
    assert "Goal: two" in player.model.calls[-1].messages[1]["content"]

    player.start_trial("two")  # the contrast: a new trial starts the turn history over
    assert player._turns == []


def test_voyager_keeps_its_library_and_completed_tasks_across_levels(tmp_path):
    player = Voyager(test_voyager.scripted())
    test_voyager.run_lineworld(tmp_path, player)
    skills, completed = sorted(player.library.skills), list(player.completed_tasks)
    assert skills and completed
    player.next_goal("a later goal")
    assert sorted(player.library.skills) == skills and player.completed_tasks == completed
    assert player._goal == "a later goal" and player._task is None  # the curriculum picks a task for the new goal


def test_continual_keeps_its_harness_across_levels_but_not_across_trials():
    player = ContinualHarness(test_continual.scripted())
    player.start_trial("one")
    player.harness.guidance = "Walk right."
    player.harness.add_memory("Walls", "The corridor ends at x=9.", 4)
    harness = player.harness
    player.next_goal("two")
    assert player.harness is harness and harness.guidance == "Walk right." and harness.find_memory("Walls")
    assert player._goal == "two"
    player.start_trial("two")  # carry_across_trials is off, so a new trial starts empty
    assert player.harness is not harness and player.harness.guidance == ""


def test_crystal_harness_keeps_its_journal_and_notes_across_levels():
    player = CrystalHarness(ScriptedModel(lambda messages: ""))
    player.start_trial("one")
    player._journal.add(1, "lead", "The milestone is in the southeast corner.")
    player.next_goal("two")
    assert [e["text"] for e in player._journal.entries] == ["The milestone is in the southeast corner."] and player._goal == "two"
    player.start_trial("two")
    assert player._journal.entries == []


# Config


def chain_configs() -> list[Path]:
    return [p for p in sorted((ROOT / "experiments").glob("*.json")) if "chain" in read_json(p)["goal"]]


def test_goal_from_dict_reads_a_plain_goal_or_a_chain():
    assert goal_from_dict({"task_id": "t", "text": "x", "success": {"a": 1}}) == Goal("t", "x", {"a": 1})
    chain = goal_from_dict({"task_id": "c", "chain": [{"task_id": "a", "text": "A", "success": {"x": 1}, "max_steps": 5}, {"task_id": "b", "text": "B", "success": {"x": 2}}]})
    assert chain == Chain("c", (Level(Goal("a", "A", {"x": 1}), 5), Level(Goal("b", "B", {"x": 2}), None)))
    assert goal_from_dict(chain.to_dict() | {"task_id": "c"}) == chain
    with pytest.raises(ValueError):
        Chain("empty", ())


def test_chain_configs_parse_and_budget_every_level():
    paths = chain_configs()
    assert {p.name for p in paths} == {
        "firered_7b_chain_crystal_harness.json",
        "firered_7b_chain_continual.json",
        "firered_7b_chain_reflexion_insights_tools.json",
        "firered_7b_chain_voyager_tools_params.json",
        "gridworld_7b_chain_crystal_harness.json",
        "gridworld_7b_chain_continual.json",
    }
    for path in paths:
        spec = read_json(path)
        chain = goal_from_dict(spec["goal"])
        budgets = Budgets(**spec["budgets"])
        make_player(spec["agent"], ScriptedModel(lambda messages: ""))
        assert spec["trials"] == 3, path.name
        assert all(level.max_steps for level in chain.levels), path.name
        assert sum(level.max_steps for level in chain.levels) <= budgets.max_steps, path.name


def test_firered_chain_levels_use_get_progress_fields_in_order():
    chains = {p.name: goal_from_dict(read_json(p)["goal"]) for p in chain_configs() if p.name.startswith("firered")}
    chain = next(iter(chains.values()))
    assert len(chains) == 4 and all(c == chain for c in chains.values())  # one ladder, four agents
    assert [dict(level.goal.success) for level in chain.levels] == [{"map": [4, 0]}, {"map": [3, 0]}, {"map": [3, 19]}]
    # the map ids after the first are not in this repo's data; the configs must say so
    assert all(level.note.startswith("UNVERIFIED") for level in chain.levels[1:])
    for name, spec in ((p.name, read_json(p)) for p in chain_configs() if p.name.startswith("firered")):
        single = read_json(ROOT / "experiments" / name.replace("_chain", ""))
        assert {k: spec[k] for k in ("environment", "model", "agent")} == {k: single[k] for k in ("environment", "model", "agent")}, name


def test_gridworld_chain_configs_are_the_same_ladder_as_the_milestone_config():
    for name in ("crystal_harness", "continual"):
        chain_spec = read_json(ROOT / f"experiments/gridworld_7b_chain_{name}.json")
        single = read_json(ROOT / f"experiments/gridworld_7b_{name}.json")
        assert {k: chain_spec[k] for k in ("environment", "model", "agent")} == {k: single[k] for k in ("environment", "model", "agent")}
        assert goal_from_dict(chain_spec["goal"]).levels[-1].goal.success == single["goal"]["success"]
