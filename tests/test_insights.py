"""Cross-trial insights and trajectory retrieval on Reflexion (the `insights` and `retrieve_trajectories` switches).

Everything runs on LineWorld or GridWorld with a ScriptedModel: no ROM, no model server.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from agents.examples import render_trajectories
from agents.models import ScriptedModel
from agents.protocol import TrialOutcome
from agents.react.player import Turn
from agents.reflexion import Reflexion, prompts
from agents.reflexion.insights import ADD, AGREE, EDIT, REMOVE, InsightSet, Op, parse_ops
from agents.reflexion.prompts import REFLECTION_PROMPT
from agents.reflexion.trajectories import Trajectory, TrajectoryStore, as_example
from environments.gridworld import GridWorld, GridWorldConfig
from environments.lineworld import LineWorld
from environments.protocol import ObservationMode
from evaluation.goals import Goal
from evaluation.runner import Budgets, run
from experiments.run import make_player

ROOT = Path(__file__).resolve().parents[1]
GOAL = Goal("reach-end", "Walk to the far right.", {"x": 9})
MILESTONE = Goal("milestone", "Reach the milestone in town.", {"milestone": True})


def ops_of(*lines):
    ops, rejected = parse_ops("\n".join(lines))
    assert not rejected, rejected
    return ops


# Parsing


def test_parse_reads_all_four_operations_and_the_papers_names():
    ops, rejected = parse_ops(
        "Some thoughts first.\n"
        "ADD 5: Press RIGHT until the wall.\n"
        "EDIT 2: A better rule.\n"
        "AGREE 1: Keep moving.\n"
        "REMOVE 3: A bad rule.\n"
        "- UPVOTE 4: Look before you step.\n"
        "DOWNVOTE 6: Another bad rule.\n"
        "And a closing remark."
    )
    assert rejected == []
    assert [(o.kind, o.number, o.text) for o in ops] == [
        (ADD, 5, "Press RIGHT until the wall."),
        (EDIT, 2, "A better rule."),
        (AGREE, 1, "Keep moving."),
        (REMOVE, 3, "A bad rule."),
        (AGREE, 4, "Look before you step."),  # UPVOTE
        (REMOVE, 6, "Another bad rule."),  # DOWNVOTE
    ]


def test_parse_rejects_malformed_operation_lines_with_a_reason():
    ops, rejected = parse_ops(
        "ADD 1 no colon here.\n"
        "ADD 2: cut off in the middle of a sen\n"
        "ADD 3:\n"
        "AGREE 1: This text mentions EDIT in it.\n"
        "EDIT: A rewrite with no number.\n"
        "REMOVE 2: A fine rule.\n"
        "Add 4: lower case prose is not an operation.\n"
    )
    assert [(o.kind, o.text) for o in ops] == [(REMOVE, "A fine rule.")]
    assert [r.reason.split(" ")[0:3] for r in rejected] == [
        ["no", "colon", "after"],
        ["text", "does", "not"],
        ["empty", "text"],
        ["text", "contains", "an"],
        ["EDIT", "needs", "the"],
    ]
    assert rejected[0].line == "ADD 1 no colon here."


# Vote rules


def insights(*texts, limit=10):
    s = InsightSet(limit=limit)
    s.apply([Op(ADD, None, t, "") for t in texts], trial=0)
    return s


def importances(s):
    return {i.text: i.importance for i in s.items}


def test_add_starts_at_two_and_votes_move_importance():
    s = insights("Rule a.", "Rule b.", "Rule c.", "Rule d.")
    assert importances(s) == {"Rule a.": 2, "Rule b.": 2, "Rule c.": 2, "Rule d.": 2}
    s = InsightSet()
    s.apply(ops_of("ADD 1: Rule a.", "ADD 2: Rule b.", "ADD 3: Rule c."), trial=0)
    result = s.apply(ops_of("AGREE 1: Rule a.", "REMOVE 2: Rule b.", "EDIT 3: Rule c, improved."), trial=1)
    assert importances(s) == {"Rule a.": 3, "Rule c, improved.": 3, "Rule b.": 1}
    assert [r["status"] for r in result["results"]] == ["applied"] * 3
    assert [i.text for i in s.items] == ["Rule a.", "Rule c, improved.", "Rule b."]  # most important first, ties in order


def test_an_insight_at_zero_is_removed():
    s = InsightSet()
    s.apply(ops_of("ADD 1: Rule a."), trial=0)
    s.apply(ops_of("REMOVE 1: Rule a."), trial=1)
    assert importances(s) == {"Rule a.": 1}
    result = s.apply(ops_of("REMOVE 1: Rule a."), trial=2)
    assert s.items == [] and result["removed_at_zero"] == ["i0"]


def test_remove_costs_three_when_the_set_is_full():
    s = InsightSet(limit=3)
    s.apply(ops_of("ADD 1: Rule a.", "ADD 2: Rule b.", "ADD 3: Rule c."), trial=0)
    s.apply(ops_of("AGREE 1: Rule a.", "AGREE 2: Rule b.", "AGREE 3: Rule c."), trial=1)
    assert s.full and set(importances(s).values()) == {3}
    result = s.apply(ops_of("REMOVE 1: Rule a."), trial=2)
    assert result["full_before"] and "Rule a." not in importances(s)  # 3 - 3 = 0


def test_edit_that_changes_nothing_counts_as_agree_and_text_finds_the_insight():
    s = InsightSet()
    s.apply(ops_of("ADD 1: Rule a.", "ADD 2: Rule b."), trial=0)
    result = s.apply(ops_of("EDIT 2: Rule a.", "AGREE 9: Rule b."), trial=1)  # the number is wrong, the text is right
    assert importances(s) == {"Rule a.": 3, "Rule b.": 3}
    assert [r["op"] for r in result["results"]] == [AGREE, AGREE]


def test_bad_operations_are_rejected_not_applied():
    s = InsightSet(max_ops=6)
    s.apply(ops_of("ADD 1: Rule a.", "ADD 2: Rule b."), trial=0)
    result = s.apply(
        ops_of(
            "ADD 3: Rule a.",  # duplicate
            "EDIT 7: Nothing there.",  # no such number
            "REMOVE 1: Words that match nothing, but the number is valid.",  # applies by number
            "AGREE 1: Rule a.",  # a second operation on insight 1
            "AGREE 99: Not a rule.",  # no text match and no such number
            "ADD 4: Rule c.",
        ),
        trial=1,
    )
    assert [(r["op"], r["status"]) for r in result["results"]] == [
        (ADD, "rejected"),
        (EDIT, "rejected"),
        (REMOVE, "applied"),
        (AGREE, "rejected"),
        (AGREE, "rejected"),
        (ADD, "applied"),
    ]
    assert importances(s) == {"Rule a.": 1, "Rule b.": 2, "Rule c.": 2}


def test_at_most_max_ops_apply_per_reply():
    s = InsightSet(max_ops=2)
    result = s.apply(ops_of("ADD 1: Rule a.", "ADD 2: Rule b.", "ADD 3: Rule c."), trial=0)
    assert [r["status"] for r in result["results"]] == ["applied", "applied", "rejected"]
    assert len(s.items) == 2


def test_the_cap_evicts_the_least_important_then_the_least_recently_touched():
    s = InsightSet(limit=3)
    s.apply(ops_of("ADD 1: Rule a.", "ADD 2: Rule b.", "ADD 3: Rule c."), trial=0)
    s.apply(ops_of("AGREE 1: Rule a."), trial=1)  # a: 3, b: 2, c: 2
    s.apply(ops_of("AGREE 3: Rule c."), trial=2)  # a: 3, b: 2, c: 3 (touched later)
    result = s.apply(ops_of("ADD 4: Rule d."), trial=3)  # d: 2, over the cap by one
    assert len(s.items) == 3 and result["evicted_over_cap"] == ["i1"]  # b: importance 2, the oldest touch among the 2s
    assert "Rule b." not in importances(s) and importances(s)["Rule d."] == 2


# Pairs


def traj(trial, success, steps=5, first="first", goal="goal"):
    turns = tuple(Turn(f"obs {n}", f"thought {n}", "RIGHT") for n in range(steps))
    return Trajectory(trial, goal, turns, success, "goal_met" if success else "max_steps", steps, first)


def test_pair_selection_prefers_the_most_recent_opposite_outcome():
    store = TrajectoryStore()
    for t in (traj(0, False), traj(1, True), traj(2, False), traj(3, True)):
        store.add(t)
    mode, pair = store.pick_pair(traj(4, False))
    assert mode == "compare" and [t.trial for t in pair] == [3, 4] and [t.success for t in pair] == [True, False]  # (success, failure)
    mode, pair = store.pick_pair(traj(4, True))
    assert mode == "compare" and [t.trial for t in pair] == [4, 2] and [t.success for t in pair] == [True, False]


def test_a_success_with_no_failure_is_shown_with_the_recent_successes_and_a_lone_failure_has_nothing():
    store = TrajectoryStore()
    assert store.pick_pair(traj(0, False)) == ("none", [])
    for t in (traj(0, True), traj(1, True), traj(2, True)):
        store.add(t)
    mode, pair = store.pick_pair(traj(3, True))
    assert mode == "successes" and [t.trial for t in pair] == [1, 2, 3]  # the last two stored plus this one: three shown at most
    mode, pair = store.pick_pair(traj(3, False))
    assert mode == "compare" and [t.trial for t in pair] == [2, 3]
    assert TrajectoryStore().pick_pair(traj(0, True))[0] == "successes"


def test_the_store_caps_trials_and_turns_but_keeps_the_first_observation():
    store = TrajectoryStore(capacity=2, max_turns=3)
    for n in range(3):
        store.add(traj(n, True, steps=6, first=f"start {n}"))
    assert [t.trial for t in store.items] == [1, 2]
    assert [t.thought for t in store.items[0].turns] == ["thought 3", "thought 4", "thought 5"]
    assert store.items[0].first_observation == "start 1"


# Retrieval


def test_ranking_is_cosine_over_goal_and_first_observation_successes_only():
    store = TrajectoryStore()
    store.add(traj(0, True, first="you are in a house with a door", goal="reach the milestone"))
    store.add(traj(1, True, first="a long corridor of lamps", goal="reach the milestone"))
    store.add(traj(2, False, first="you are in a house with a door", goal="reach the milestone"))  # a failure is never retrieved
    ranked = store.rank("reach the milestone", "you are in a house with a door", k=5)
    assert [t.trial for t, _ in ranked] == [0, 1]
    assert ranked[0][1] > ranked[1][1] and abs(ranked[0][1] - 1.0) < 1e-9
    assert [t.trial for t, _ in store.rank("reach the milestone", "you are in a house with a door", k=1)] == [0]
    assert store.rank("g", "o", k=0) == []


def test_ties_go_to_the_shorter_trial_then_the_more_recent():
    store = TrajectoryStore()
    store.add(traj(0, True, steps=9))
    store.add(traj(1, True, steps=5))
    store.add(traj(2, True, steps=5))
    assert [t.trial for t, _ in store.rank("goal", "first", k=3)] == [2, 1, 0]


def test_a_retrieved_trial_renders_through_the_few_shot_renderer():
    t = traj(3, True, steps=12)
    text = render_trajectories([as_example(t, max_turns=2, tools=False)], 2)
    assert text == (
        "Goal: goal\nObservation 1: obs 10\nThought 1: thought 10\nAction 1: RIGHT\n"
        "Observation 2: obs 11\nThought 2: thought 11\nAction 2: RIGHT\n"
        "(the first 10 turns of this trial are not shown; the turns above are its last ones)\n"
        "Outcome: reached the goal in 12 steps."
    )
    tools = Trajectory(0, "goal", (Turn("o", "t", "go_to door_0", result="go_to door_0: went through"),), True, "goal_met", 3, "o")
    assert "Result 1: go_to door_0: went through" in render_trajectories([as_example(tools, 8, tools=True)])


# The player and the runner


def system_of(call):
    return call["messages"][0]["content"]


def decisions_of(run_dir, trial):
    records = [json.loads(line) for line in (run_dir / f"trials/{trial:03d}/steps.jsonl").read_text().splitlines()]
    return [r for r in records if r["kind"] == "decision"], records[-1]


def line_model(ops):
    """LEFT for the first 12 decisions (trial 0 fails), RIGHT after (every later trial succeeds); insight calls reply from `ops`."""
    state = {"decisions": 0, "insight_calls": 0}

    def reply(messages):
        if messages[0].content.startswith("You are an advanced reasoning agent"):
            index = state["insight_calls"]
            state["insight_calls"] += 1
            return ops[index] if index < len(ops) else ""
        if messages[0].content.startswith(REFLECTION_PROMPT):
            return "Plan: press RIGHT."
        state["decisions"] += 1
        return "Thinking.\nAction: LEFT" if state["decisions"] <= 12 else "Thinking.\nAction: RIGHT"

    return ScriptedModel(reply)


OPS = [
    "Some prose.\nADD 1: Always press RIGHT to reach the far end.\nADD 2: Never walk into the wall.\nADD 3 no colon.\nADD 4: missing period",
    "AGREE 1: Always press RIGHT to reach the far end.\nREMOVE 2: Never walk into the wall.\nEDIT 9: Nothing at nine.",
    "REMOVE 2: Never walk into the wall.",
]


def test_runner_integration_over_several_trials(tmp_path):
    model = line_model(OPS)
    player = Reflexion(model, insights=True)
    run_dir = tmp_path / "run"
    results = run(run_dir=run_dir, env=LineWorld(), player=player, goal=GOAL, budgets=Budgets(max_steps=12), trials=5)
    assert [r.success for r in results] == [False, True, True, True, True]

    # trial 0 failed with nothing to compare against: no insight call, and the reason is logged
    _, end = decisions_of(run_dir, 0)
    assert end["review"]["trace"]["insights"]["mode"] == "none" and "skipped" in end["review"]["trace"]["insights"]
    assert len(end["review"]["calls"]) == 1  # the reflection only

    # trial 1 succeeded, so it is compared with trial 0's failure; every operation is logged
    _, end = decisions_of(run_dir, 1)
    log = end["review"]["trace"]["insights"]
    assert log["mode"] == "compare" and [t["success"] for t in log["compared_trials"]] == [True, False]
    assert log["raw_reply"] == OPS[0]
    assert [o["op"] for o in log["parsed_ops"]] == [ADD, ADD]
    assert [r["reason"].split(" ")[0] for r in log["rejected_lines"]] == ["no", "text"]
    assert [r["status"] for r in log["results"]] == ["applied", "applied"]
    assert [i["importance"] for i in log["insights"]] == [2, 2]
    query = end["review"]["calls"][0]["messages"]
    assert query[0]["content"].startswith("You are an advanced reasoning agent") and "You failed the trial because the environment ended it with max_steps" in query[0]["content"]
    assert "SUCCESSFUL TRIAL:" in query[1]["content"] and "FAILED TRIAL:" in query[1]["content"] and "STATUS: FAIL (max_steps after 12 steps)" in query[1]["content"]
    assert "(none yet)" in query[1]["content"] and query[1]["content"].endswith(prompts.INSIGHT_SUFFIX_NOT_FULL)

    # trial 2 reads the insights, and its review votes on them
    ds, end = decisions_of(run_dir, 2)
    assert all(d["decision"]["trace"]["insights_in_context"] == ["i0", "i1"] for d in ds)
    system = system_of(ds[0]["decision"]["calls"][0])
    assert prompts.INSIGHTS_HEADER + "\n1. Always press RIGHT to reach the far end.\n2. Never walk into the wall." in system
    assert "Your memory for the task below:\nTrial 0:\nPlan: press RIGHT." in system and system.index("1. Always") < system.index("Your memory")  # both switches' text, insights first
    log = end["review"]["trace"]["insights"]
    assert [(r["op"], r["status"]) for r in log["results"]] == [(AGREE, "applied"), (REMOVE, "applied"), (EDIT, "rejected")]
    assert [(i["id"], i["importance"]) for i in log["insights"]] == [("i0", 3), ("i1", 1)]

    # trial 3 still sees both; the removal at its end takes i1 to zero
    ds, end = decisions_of(run_dir, 3)
    assert ds[0]["decision"]["trace"]["insights_in_context"] == ["i0", "i1"]
    assert end["review"]["trace"]["insights"]["removed_at_zero"] == ["i1"]

    # trial 4 sees only what is left; an empty reply is logged, not applied
    ds, end = decisions_of(run_dir, 4)
    assert ds[0]["decision"]["trace"]["insights_in_context"] == ["i0"]
    assert "2. Never walk" not in system_of(ds[0]["decision"]["calls"][0])
    assert end["review"]["trace"]["insights"]["error"] == "empty reply" and [i.text for i in player.insight_set.items] == ["Always press RIGHT to reach the far end."]
    assert player.mechanisms()["insights"] is True and player.mechanisms()["retrieve_trajectories"] is False


def test_insights_without_reflections_and_the_success_only_comparison(tmp_path):
    # trial 0 succeeds with no failure stored, so it is shown with the successes so far, not compared
    model = ScriptedModel(lambda messages: "ADD 1: Keep pressing RIGHT." if messages[0].content.startswith("You are an advanced") else "x.\nAction: RIGHT")
    player = Reflexion(model, insights=True, reflections=False)
    run_dir = tmp_path / "run"
    run(run_dir=run_dir, env=LineWorld(), player=player, goal=GOAL, budgets=Budgets(max_steps=12), trials=2)
    ds, end = decisions_of(run_dir, 0)
    log = end["review"]["trace"]["insights"]
    assert log["mode"] == "successes" and [t["success"] for t in log["compared_trials"]] == [True]
    assert "Here are the trials:" in end["review"]["calls"][0]["messages"][1]["content"] and "SUCCESSFUL TRIAL" not in end["review"]["calls"][0]["messages"][1]["content"]
    assert end["review"]["calls"][0]["messages"][0]["content"].endswith("tasks to complete.")
    ds, _ = decisions_of(run_dir, 1)
    system = system_of(ds[0]["decision"]["calls"][0])
    assert "1. Keep pressing RIGHT." in system and "Your memory" not in system and player.memory == []  # reflections are off


def test_retrieved_trajectories_go_in_the_prompt_and_are_logged(tmp_path):
    model = line_model(OPS)
    player = Reflexion(model, retrieve_trajectories=True, retrieve_k=2, retrieved_turns=3, reflections=False)
    run_dir = tmp_path / "run"
    run(run_dir=run_dir, env=LineWorld(), player=player, goal=GOAL, budgets=Budgets(max_steps=12), trials=3)
    ds, end = decisions_of(run_dir, 0)
    assert ds[0]["decision"]["trace"]["retrieved_trajectories"] == [] and prompts.RETRIEVED_HEADER not in system_of(ds[0]["decision"]["calls"][0])
    assert "insights" not in end["review"]["trace"] and player.mechanisms()["retrieve_trajectories"] is True
    ds, _ = decisions_of(run_dir, 1)
    assert ds[0]["decision"]["trace"]["retrieved_trajectories"] == []  # trial 0 failed, so there is nothing to retrieve
    ds, _ = decisions_of(run_dir, 2)
    trace = ds[0]["decision"]["trace"]["retrieved_trajectories"]
    assert [(t["trial"], t["success"], t["steps"]) for t in trace] == [(1, True, 9)] and 0 < trace[0]["similarity"] <= 1
    assert all(d["decision"]["trace"]["retrieved_trajectories"] == trace for d in ds)  # fixed for the trial
    system = system_of(ds[0]["decision"]["calls"][0])
    assert prompts.RETRIEVED_HEADER in system and "Goal: Walk to the far right.\nObservation 1:" in system
    assert "(the first 6 turns of this trial are not shown" in system and system.endswith("Outcome: reached the goal in 9 steps.")
    assert system.count("Observation 3:") == 1 and "Observation 4:" not in system  # retrieved_turns=3


def test_retrieval_and_insights_work_over_the_tools_interface(tmp_path):
    def reply(messages):
        if messages[0].content.startswith("You choose the next move"):
            return '{"why": "x", "option": "tackle"}'
        if messages[0].content.startswith("You are an advanced reasoning agent"):
            return "ADD 1: Go through the door first."
        if len(messages) == 1:
            return "Plan: x."
        state = json.loads(messages[1].content.rsplit("State: ", 1)[1].splitlines()[0])
        return "Thought: door.\nAction: go_to door_to_town" if state["map"] == "house" else "Thought: there.\nAction: go_to milestone"

    model = ScriptedModel(reply)
    player = Reflexion(model, insights=True, retrieve_trajectories=True, action_interface="tools", specialists=True)
    run_dir = tmp_path / "run"
    env = GridWorld(GridWorldConfig(mode=ObservationMode.STRUCTURED))
    results = run(run_dir=run_dir, env=env, player=player, goal=MILESTONE, budgets=Budgets(max_steps=80, max_decisions=160), trials=3)
    assert all(r.success for r in results)
    ds, end = decisions_of(run_dir, 0)
    assert end["review"]["trace"]["insights"]["mode"] == "successes"
    ds, end = decisions_of(run_dir, 2)
    system = system_of(next(d for d in ds if d["decision"]["calls"])["decision"]["calls"][0])
    assert "1. Go through the door first." in system and prompts.RETRIEVED_HEADER in system
    assert "go_to door_to_town" in system and "\nResult " in system.split(prompts.RETRIEVED_HEADER)[1]  # tools turns, and how the last tool ended
    assert ds[0]["decision"]["trace"]["insights_in_context"] == ["i0"] and ds[0]["decision"]["trace"]["retrieved_trajectories"]
    insight_call = end["review"]["calls"][0]["messages"][1]["content"]
    assert "Result " in insight_call  # the tools-interface turn lines carry the last tool's result


# The switches are off by default and change nothing then


def _digest(model, actions, reviews):
    payload = json.dumps({"actions": actions, "calls": [[c.messages, c.output] for c in model.calls], "reviews": reviews}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _two_trials(player, env, steps, model):
    actions, reviews = [], []
    for outcome in (TrialOutcome(False, "max_steps", steps), TrialOutcome(True, "goal_met", steps)):
        player.start_trial("goal text")
        obs = env.reset()
        for _ in range(steps):
            d = player.decide(obs, env.legal_actions())
            actions.append(d.action.key if d.action else None)
            if d.action:
                obs = env.step(d.action).observation
        reviews.append(player.end_trial(outcome).trace)
    return _digest(model, actions, reviews)


def _tools_model():
    cycle = itertools.cycle(
        [
            "Thought 1: the door leads to town.\nAction 1: go_to door_to_town",
            "Thought 2: walk.\nAction 2: press RIGHT x2",
            "Thought 3: hmm.\nAction 3: talk_to nobody",
            "no action here",
        ]
    )

    def reply(messages):
        if len(messages) == 1 or messages[0].content.startswith("You choose the next move"):
            return "Plan: go to the door." if len(messages) == 1 else '{"why": "x", "option": "tackle"}'
        return next(cycle)

    return ScriptedModel(reply)


def _buttons_model():
    cycle = itertools.cycle(["Go right.\nAction 1: RIGHT", "Go on.\nAction: DOWN", "Plan: do better"])
    return ScriptedModel(lambda msgs: next(cycle))


# Taken from the commit before these switches existed: two trials (a failure, then a success), every request, decision, and review.
# If one changes, the prompts or the Review trace with the switches off changed: a regression, not a refresh.
OFF_DIGESTS = {"tools": "9d13f581ed2e0f3f", "buttons": "c6bb80d24a0b26ce"}


def test_with_the_switches_off_prompts_decisions_and_reviews_are_unchanged():
    for name, build in (("default", lambda m, **kw: Reflexion(m, **kw)), ("explicit", lambda m, **kw: Reflexion(m, insights=False, retrieve_trajectories=False, **kw))):
        model = _tools_model()
        env = GridWorld(GridWorldConfig(mode=ObservationMode.STRUCTURED))
        assert _two_trials(build(model, action_interface="tools", specialists=True), env, 14, model) == OFF_DIGESTS["tools"], name
        model = _buttons_model()
        assert _two_trials(build(model, memory_size=2), LineWorld(), 7, model) == OFF_DIGESTS["buttons"], name


def test_make_player_wires_the_config_keys_and_reports_both_switches():
    model = ScriptedModel(lambda m: "")
    off = make_player({"name": "reflexion"}, model)
    assert off.mechanisms()["insights"] is False and off.mechanisms()["retrieve_trajectories"] is False
    on = make_player(
        {"name": "reflexion", "insights": True, "retrieve_trajectories": True, "max_insights": 4, "insight_turns": 7, "retrieve_k": 2, "retrieved_turns": 5, "trajectories_kept": 3},
        model,
    )
    assert on.mechanisms()["insights"] is True and on.mechanisms()["retrieve_trajectories"] is True
    assert (on.insight_set.limit, on.insight_turns, on.retrieve_k, on.retrieved_turns, on.store.capacity) == (4, 7, 2, 5, 3)
    tools = make_player({"name": "reflexion", "insights": True, "action_interface": "tools", "specialists": True}, model)
    assert tools.mechanisms()["tools"] is True and tools.mechanisms()["insights"] is True
    assert all(isinstance(v, bool) for v in on.mechanisms().values())


def test_the_insight_configs_are_the_reflexion_configs_with_the_switch_on_and_five_trials():
    pairs = {
        "gridworld_7b_reflexion_insights": ("gridworld_7b_reflexion", {"insights": True}),
        "gridworld_7b_reflexion_insights_only": ("gridworld_7b_reflexion", {"insights": True, "reflections": False}),
        "gridworld_7b_reflexion_insights_tools": ("gridworld_7b_reflexion_tools", {"insights": True}),
        "firered_7b_reflexion_insights_tools": ("firered_7b_reflexion_tools", {"insights": True}),
    }
    for name, (base, changes) in pairs.items():
        new = json.loads((ROOT / f"experiments/{name}.json").read_text())
        old = json.loads((ROOT / f"experiments/{base}.json").read_text())
        assert new["trials"] == 5
        for key in ("environment", "model", "goal", "budgets"):
            assert new[key] == old[key], (name, key)
        assert new["agent"] == {**old["agent"], **changes}, name
        player = make_player(new["agent"], ScriptedModel(lambda m: ""))
        assert player.mechanisms()["insights"] is True and player.mechanisms()["reflections"] is new["agent"].get("reflections", True)
