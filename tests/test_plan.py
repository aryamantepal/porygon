"""Plans with checked steps (`plan`) and explain-and-replan (`explain_replan`).

Toy environments and a ScriptedModel: no ROM, no model server.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.harness import CrystalHarness
from agents.harness import plan as planning
from agents.harness.stuck import StuckLadder
from agents.models import ScriptedModel
from agents.protocol import TrialOutcome
from environments.gridworld import GridWorld
from environments.protocol import Press
from evaluation.runner import Budgets, run
from experiments.run import make_player
from tests.test_harness import (
    BACKBONE,
    MILESTONE,
    events,
    planner_model,
    records_of,
    to_milestone,
)
from tests.test_retrieval import OFF_DIGESTS, _drive

ROOT = Path(__file__).resolve().parents[1]
DOOR = {"tool": "go_to", "args": {"place": "door_to_town"}, "expect": {"map": "town"}}
GOAL_STEP = {"tool": "go_to", "args": {"place": "milestone"}, "expect": {"position": [7, 4]}}
NAMES = ["go_to", "talk_to", "press", "note"]


def plan_reply(steps, **extra):
    return json.dumps({"why": "plan ahead", "plan": steps, "intention": "reach the milestone", **extra})


def is_explain(messages):
    return "A step you chose did not work" in messages[0].content


def scenario(planner, explainer):
    """Planner replies come from `planner(n)` (n counts planner calls), explain replies from `explainer(n)`.
    A function returning None falls back to the normal `to_milestone` single tool call."""
    counts = {"planner": 0, "explain": 0}

    def reply(messages):
        first = messages[0].content
        if first.startswith("Read this dialogue"):
            return "- The milestone is in the southeast corner of town."
        if first.startswith("You choose the next move"):
            return '{"why": "weaken it first", "option": "tackle"}'
        if is_explain(messages):
            counts["explain"] += 1
            return explainer(counts["explain"])
        counts["planner"] += 1
        out = planner(counts["planner"])
        if out is not None:
            return out
        state = next(json.loads(line[len("State: ") :]) for line in messages[1].content.splitlines() if line.startswith("State: "))
        return json.dumps(to_milestone(state))

    model = ScriptedModel(reply)
    model.counts = counts
    return model


def play(model, tmp_path, **options):
    harness = CrystalHarness(model, **options)
    results = run(run_dir=tmp_path / "run", env=GridWorld(), player=harness, goal=MILESTONE, budgets=Budgets(max_steps=80))
    return harness, results, records_of(tmp_path / "run")


def explain_prompts(model):
    return [c.messages[1]["content"] for c in model.calls if c.messages[0]["content"].find("A step you chose did not work") >= 0]


def planner_prompts(model):
    return [c.messages[1]["content"] for c in model.calls if c.messages[1]["content"].startswith("Goal:") and "A step you chose" not in c.messages[0]["content"]]


# Parsing and validation


def test_a_valid_plan_parses_into_steps():
    steps, problems = planning.parse_steps([DOOR, {"tool": "press", "args": {"key": "A"}}], NAMES)
    assert problems == [] and [s.tool for s in steps] == ["go_to", "press"]
    assert steps[0].expect == {"map": "town"} and steps[1].expect == {}


def test_bad_steps_are_rejected_with_every_problem_named():
    bad = [
        {"tool": "fly", "args": {}},
        {"tool": "note", "args": {"node": "town", "text": "x"}},
        {"tool": "go_to", "args": "door"},
        {"tool": "go_to", "args": {}, "expect": {"hp": 3}},
        {"tool": "go_to", "args": {}, "expect": {"map": {"nested": 1}}},
    ]
    steps, problems = planning.parse_steps(bad, NAMES)
    assert steps == []
    text = " | ".join(problems)
    assert "step 1: tool 'fly'" in text and "step 2: note takes no game time" in text
    assert "step 3: args must be" in text and "step 4: expect field 'hp'" in text and "step 5: expect 'map' must be" in text
    assert planning.parse_steps([], NAMES)[1] and planning.parse_steps("go", NAMES)[1] and planning.parse_steps([3], NAMES)[1]


def test_a_plan_over_five_steps_is_rejected():
    steps, problems = planning.parse_steps([DOOR] * 6, NAMES)
    assert steps == [] and "6 steps; the most is 5" in problems[0]
    assert planning.parse_steps([DOOR] * 5, NAMES)[1] == []


def test_one_bad_step_rejects_the_whole_plan():
    steps, problems = planning.parse_steps([DOOR, {"tool": "fly"}], NAMES)
    assert steps == [] and len(problems) == 1


def test_a_reply_without_a_plan_is_one_tool_when_plans_are_off():
    steps, problems = planning.parse_reply({"tool": "go_to", "args": {"place": "x"}}, NAMES, plan_allowed=False)
    assert [s.tool for s in steps] == ["go_to"] and problems == []
    assert planning.parse_reply({"plan": [DOOR]}, NAMES, plan_allowed=False)[1] == ['the reply has no "tool"']
    assert planning.parse_reply({"plan": [DOOR]}, NAMES, plan_allowed=True)[1] == []


# Expect checks read the adapter's state


def test_expect_is_checked_against_state_fields():
    state = {"map": "4.0", "context": "overworld", "position": [3, 4], "facing": "UP"}
    assert planning.check_expect({"map": "4.0", "position": [3, 4], "context": "Overworld", "facing": "up"}, state) == []
    assert planning.check_expect({"map": 4.0}, state) == []  # the id 4.0 is the string "4.0"
    mismatches = planning.check_expect({"map": "5.1", "context": "dialogue", "position": [3, 4]}, state)
    assert mismatches == [{"field": "map", "expected": "5.1", "actual": "4.0"}, {"field": "context", "expected": "dialogue", "actual": "overworld"}]
    assert planning.check_expect({"map": "4.0"}, None) == [{"field": "map", "expected": "4.0", "actual": None}]
    assert planning.check_expect({}, state) == []


def test_the_state_change_is_described_from_the_fields_that_moved():
    assert planning.describe_change({"map": "a", "position": [1, 1]}, {"map": "b", "position": [1, 1]}) == 'map: "a" -> "b"'
    assert "nothing" in planning.describe_change({"map": "a"}, {"map": "a"})


# A plan runs, and each step is checked


def test_a_plan_runs_in_order_with_one_planner_call_and_every_check_logged(tmp_path):
    back = {"tool": "go_to", "args": {"place": "door_to_house"}, "expect": {"map": "house", "context": "overworld"}}
    model = scenario(lambda n: plan_reply([DOOR, back], battle_goal="defeat") if n == 1 else None, lambda n: "{}")
    _, results, records = play(model, tmp_path, plan=True)
    assert results[0].success
    checks = events(records, "plan_check")
    done = [c for c in checks if c["outcome"] == "done"]
    assert [c["index"] for c in done] == [0, 1] and all(c["passed"] for c in done)
    assert done[0]["expect"] == {"map": "town"} and done[0]["actual"] == {"map": "town"}
    assert done[1]["actual"] == {"map": "house", "context": "overworld"}
    assert [e["index"] for e in events(records, "plan_step")][:2] == [0, 1]
    assert events(records, "plan_done") and events(records, "plan")[0]["steps"][1]["expect"] == back["expect"]
    first_plan_ends = [e for e in events(records, "plan_check")]
    assert len(first_plan_ends) == 2 and model.counts["planner"] >= 2  # one call made both steps; the next came after the plan


def test_the_plan_prompt_is_added_only_when_plan_is_on(tmp_path):
    on = scenario(lambda n: None, lambda n: "{}")
    play(on, tmp_path, plan=True)
    assert "up to 5 steps" in on.calls[0].messages[0]["content"] and '"expect"' in on.calls[0].messages[0]["content"]
    off = scenario(lambda n: None, lambda n: "{}")
    play(off, tmp_path / "off")
    assert "expect" not in off.calls[0].messages[0]["content"]


def test_the_check_reads_the_game_not_the_models_claim(tmp_path):
    # the plan says the door leads to "cellar"; the game says "town"
    wrong = {**DOOR, "expect": {"map": "cellar"}}
    model = scenario(lambda n: plan_reply([wrong, GOAL_STEP]) if n == 1 else None, lambda n: "{}")
    _, _, records = play(model, tmp_path, plan=True)
    first = events(records, "plan_check")[0]
    assert first["outcome"] == "done" and first["passed"] is False
    assert first["expect"] == {"map": "cellar"} and first["actual"] == {"map": "town"}
    dropped = events(records, "plan_dropped")
    assert dropped and dropped[0]["at_step"] == 0  # explain_replan is off: no re-plan, the planner is asked again
    assert not events(records, "explanation")


def test_a_step_the_game_interrupts_runs_again_afterward(tmp_path):
    grass = {"tool": "go_to", "args": {"position": [4, 3]}, "expect": {"position": [4, 3]}}
    model = scenario(lambda n: plan_reply([DOOR, grass]) if n == 1 else None, lambda n: "{}")
    _, _results, records = play(model, tmp_path, plan=True)
    interrupted = [c for c in events(records, "plan_check") if c["outcome"] == "interrupted"]
    assert interrupted and interrupted[0]["passed"] is None and interrupted[0]["retries"] == 1
    assert events(records, "battle")
    last = [c for c in events(records, "plan_check") if c["index"] == 1][-1]
    assert last["passed"] is True  # the step was already true when it ran again, so its expect holds
    assert not events(records, "explanation")


def test_a_step_interrupted_too_often_counts_as_failed():
    harness = CrystalHarness(ScriptedModel(lambda m: "{}"), plan=True)
    harness.start_trial("goal")
    harness._plan_run = planning.PlanRun([planning.PlanStep("go_to", {"place": "x"}, {})], "planner")
    turn = type("T", (), {"calls": [], "events": []})()
    state = {"map": "town", "context": "battle"}
    for _ in range(planning.MAX_INTERRUPTS):
        harness._plan_step_ended(harness._plan_run, "interrupted", "the game switched to battle", state, state, turn)
        assert harness._plan_run is not None and harness._plan_run.index == 0
    harness._plan_step_ended(harness._plan_run, "interrupted", "the game switched to battle", state, state, turn)
    assert harness._plan_run is None and harness._ladder.failed_here(state)[0].result.startswith("interrupted 3 times")


# Rejected plans are shown back


def test_a_rejected_plan_is_shown_back_with_its_problems(tmp_path):
    bad = plan_reply([{"tool": "fly", "args": {}, "expect": {"map": "town"}}])
    model = scenario(lambda n: bad if n == 1 else None, lambda n: "{}")
    _, results, records = play(model, tmp_path, plan=True)
    rejected = events(records, "plan")[0]
    assert rejected["error"].startswith("plan rejected") and "tool 'fly'" in rejected["problems"][0]
    second = planner_prompts(model)[1]
    assert "Your last plan could not be used:" in second and "step 1: tool 'fly'" in second and "fly" in second.split("Your reply was:")[1]
    assert "could not be used" not in planner_prompts(model)[2]
    assert results[0].success


# Explain and re-plan


def test_in_plan_refinement_replaces_the_failed_step_and_the_rest(tmp_path):
    wrong = {**DOOR, "expect": {"map": "cellar"}}
    never = {"tool": "press", "args": {"key": "A", "times": 3}}  # the step after the failure; the revision replaces it
    fix = [{"tool": "go_to", "args": {"place": "milestone"}, "expect": {"position": [7, 4]}}]
    model = scenario(
        lambda n: plan_reply([wrong, never]) if n == 1 else None,
        lambda n: json.dumps({"explanation": "the door leads to town, not a cellar", "plan": fix}),
    )
    _, results, records = play(model, tmp_path, plan=True, explain_replan=True)
    assert results[0].success and model.counts["explain"] == 1
    explanation = events(records, "explanation")[0]
    assert explanation["scope"] == "in_plan" and explanation["step"] == 0 and explanation["explanation"] == "the door leads to town, not a cellar"
    assert explanation["revision"] == [{"tool": "go_to", "args": {"place": "milestone"}, "expect": {"position": [7, 4]}}]
    assert events(records, "plan_revised")[0]["scope"] == "in_plan"
    ran = [e["tool"] for e in events(records, "plan_step")]
    assert "press" not in ran  # the replaced suffix never ran
    prompt = explain_prompts(model)[0]
    assert "1. go_to" in prompt and "<- FAILED" in prompt and "2. press" in prompt and "(not run)" in prompt
    assert 'map is "town", expected "cellar"' in prompt  # the failed check, from the game
    assert 'State before the step: {"map": "house"' in prompt and 'State after the step: {"map": "town"' in prompt
    assert 'What changed: map: "house" -> "town"' in prompt
    assert "replace the failed step and every step after it" in model.calls[1].messages[0]["content"]


def test_the_explanation_reaches_the_journal_and_the_next_planner_prompt(tmp_path):
    def planner(n):
        if n == 1:
            return json.dumps({"why": "x", "tool": "go_to", "args": {"place": "nowhere"}, "intention": "reach the milestone"})
        return None

    model = scenario(planner, lambda n: json.dumps({"explanation": "there is no place called nowhere", "tool": "go_to", "args": {"place": "door_to_town"}}))
    harness, results, _records = play(model, tmp_path, explain_replan=True)
    assert results[0].success
    assert "Why recent steps failed:\n- go_to {\"place\": \"nowhere\"} (the tool failed): there is no place called nowhere" in planner_prompts(model)[1]
    assert "Why recent steps failed" not in planner_prompts(model)[0]
    journal = [e for e in harness.end_trial(TrialOutcome(True, "goal_met", 1)).trace["journal"] if e["kind"] == "explanation"]
    assert journal and "no place called nowhere" in journal[0]["text"]


def test_a_single_tool_failure_is_explained_and_fixed_without_plan(tmp_path):
    def planner(n):
        return json.dumps({"why": "x", "tool": "go_to", "args": {"place": "nowhere"}, "intention": "reach the milestone"}) if n == 1 else None

    model = scenario(planner, lambda n: json.dumps({"explanation": "wrong id", "tool": "go_to", "args": {"place": "door_to_town"}}))
    _harness, results, records = play(model, tmp_path, explain_replan=True)
    assert results[0].success and model.counts["explain"] == 1
    # the revised tool ran before the planner was asked again: the next planner prompt already shows it done
    assert 'go_to {"place": "door_to_town"}: done' in planner_prompts(model)[1]
    explanation = events(records, "explanation")[0]
    assert explanation["scope"] == "out_of_plan" and explanation["step"] is None
    assert explanation["revision"] == [{"tool": "go_to", "args": {"place": "door_to_town"}, "expect": {}}]
    prompt = explain_prompts(model)[0]
    assert 'The step that failed: go_to {"place": "nowhere"}' in prompt and "no place 'nowhere' on this map" in prompt
    assert "Reply with one tool call." in model.calls[1].messages[0]["content"]
    assert events(records, "plan_step")[0]["of"] == 1  # plan is off: the revision is a single step


def test_a_failure_after_the_tool_starts_is_explained_too(tmp_path):
    # the door tool ends "failed" mid-walk: pressing a wall for a tile nobody can reach
    model = scenario(
        lambda n: json.dumps({"why": "x", "tool": "go_to", "args": {"position": [0, 0]}, "intention": "x"}) if n == 1 else None,
        lambda n: json.dumps({"explanation": "that tile is a wall", "tool": "go_to", "args": {"place": "door_to_town"}}),
    )
    _, results, records = play(model, tmp_path, explain_replan=True)
    ends = [e for e in events(records, "tool_end") if e["handler"] == "planner"]
    assert ends[0]["outcome"] == "failed"
    assert events(records, "explanation")[0]["failure"] == "the tool failed"
    assert results[0].success


def test_out_of_plan_refinement_returns_a_fresh_plan_when_plans_are_on(tmp_path):
    def planner(n):
        return json.dumps({"why": "x", "tool": "go_to", "args": {"place": "nowhere"}, "intention": "x"}) if n == 1 else None

    fresh = [DOOR, GOAL_STEP]
    model = scenario(planner, lambda n: json.dumps({"explanation": "wrong id", "plan": fresh}))
    _, results, records = play(model, tmp_path, plan=True, explain_replan=True)
    assert results[0].success
    revised = events(records, "plan_revised")[0]
    assert revised["scope"] == "out_of_plan" and len(revised["steps"]) == 2
    assert "Reply with a new plan of up to 5 steps" in model.calls[1].messages[0]["content"]
    assert model.counts["planner"] == 1


def test_an_unusable_revision_drops_the_plan_and_keeps_the_explanation(tmp_path):
    wrong = {**DOOR, "expect": {"map": "cellar"}}
    model = scenario(
        lambda n: plan_reply([wrong, GOAL_STEP]) if n == 1 else None,
        lambda n: json.dumps({"explanation": "wrong map", "plan": [{"tool": "fly", "args": {}}]}),
    )
    _harness, results, records = play(model, tmp_path, plan=True, explain_replan=True)
    explanation = events(records, "explanation")[0]
    assert explanation["error"].startswith("revision rejected") and explanation["explanation"] == "wrong map"
    assert events(records, "plan_dropped")[0]["reason"] == "the revision was not usable"
    assert results[0].success  # the planner took over
    assert "Why recent steps failed:\n- go_to" in planner_prompts(model)[1]


# The cap, and the fall back to the ladder


def _failing_world(max_replans):
    """A planner that always asks for a place that is not there, and an explainer that always suggests another one."""
    model = scenario(
        lambda n: json.dumps({"why": "x", "tool": "go_to", "args": {"place": "nowhere"}, "intention": "x"}),
        lambda n: json.dumps({"explanation": f"try again {n}", "tool": "go_to", "args": {"place": f"nowhere_{n}"}}),
    )
    harness = CrystalHarness(model, explain_replan=True, max_replans=max_replans, max_calls_per_decision=12)
    harness.start_trial("goal")
    env = GridWorld()
    return model, harness, env.reset(), env.legal_actions()


def test_replans_stop_at_the_cap_and_the_ladder_takes_over():
    model, harness, observation, legal = _failing_world(max_replans=2)
    decision = harness.decide(observation, legal)
    assert model.counts["explain"] == 2  # the third failure in this situation got no call
    kinds = [e["type"] for e in decision.trace["events"]]
    assert "replan_cap" in kinds
    cap = next(e for e in decision.trace["events"] if e["type"] == "replan_cap")
    assert cap["cap"] == 2 and cap["fallback"] == "stuck_ladder"
    # the ladder was told of every failure, and the planner (not an explain call) was asked again afterward
    assert [a.args["place"] for a in harness._ladder.failed_here(observation.state)][:3] == ["nowhere", "nowhere_1", "nowhere_2"]
    assert model.counts["planner"] >= 2
    harness.decide(observation, legal)
    assert model.counts["explain"] == 2  # the cap holds across decisions in the same situation
    # the ladder's own rung machinery still climbs on the repeats
    assert harness._ladder.seen[next(iter(harness._ladder.seen))] >= 2


def test_a_cap_of_zero_never_calls_the_explainer():
    model, harness, observation, legal = _failing_world(max_replans=0)
    harness.decide(observation, legal)
    assert model.counts["explain"] == 0 and model.counts["planner"] >= 1


def test_help_resets_the_replan_count_for_that_spot():
    from agents.harness.stuck import signature

    model, harness, observation, legal = _failing_world(max_replans=1)
    harness._replans[signature(observation.state)] = 1
    harness._ladder.seen[signature(observation.state)] = 12  # deep in the ladder: the next planner decision asks for help
    harness.decide(observation, legal)
    assert harness._stats["help_requests"] >= 1 and model.counts["explain"] >= 1


# Selector


def test_the_selector_drops_leading_steps_that_already_failed_here():
    ladder = StuckLadder()
    state = {"map": "house", "position": [1, 1], "context": "overworld"}
    ladder.record_failure(state, "go_to", {"place": "nowhere"}, "no place")
    steps, _ = planning.parse_steps([{"tool": "go_to", "args": {"place": "nowhere"}}, {"tool": "go_to", "args": {"place": "nowhere"}}, DOOR], NAMES)
    kept, dropped = planning.select(steps, ladder, state)
    assert [s.args for s in dropped] == [{"place": "nowhere"}] * 2 and [s.expect for s in kept] == [{"map": "town"}]
    elsewhere = {**state, "position": [2, 2]}
    assert planning.select(steps, ladder, elsewhere) == (steps, [])  # a different situation: nothing failed there


def test_a_plan_step_identical_to_a_failure_here_is_skipped_and_logged():
    model = scenario(lambda n: plan_reply([{"tool": "go_to", "args": {"place": "nowhere"}}, DOOR]), lambda n: "{}")
    harness = CrystalHarness(model, plan=True)
    harness.start_trial("goal")
    env = GridWorld()
    observation = env.reset()
    harness._ladder.record_failure(observation.state, "go_to", {"place": "nowhere"}, "no place")
    decision = harness.decide(observation, env.legal_actions())
    selector = next(e for e in decision.trace["events"] if e["type"] == "plan_selector")
    assert selector["dropped"][0]["args"] == {"place": "nowhere"}
    assert decision.trace["tool"] == "go_to" and decision.trace["args"] == {"place": "door_to_town"}


# Off means off


def test_the_switches_off_leave_the_digest_alone():
    model = planner_model(to_milestone)
    assert _drive(CrystalHarness(model, backbone=BACKBONE, plan=False, explain_replan=False), GridWorld(), 50, model) == OFF_DIGESTS["crystal"]


def test_explain_replan_on_with_no_failure_changes_nothing():
    # 6 steps walk to the door and through it: no tool fails, so no explain call is made and no prompt differs
    # (the end-of-trial summary gains a line about plans when a switch is on, so compare requests and actions)
    off, on = planner_model(to_milestone), planner_model(to_milestone)
    actions = []
    for model, harness in ((off, CrystalHarness(off, backbone=BACKBONE)), (on, CrystalHarness(on, backbone=BACKBONE, explain_replan=True))):
        harness.start_trial("goal text")
        env = GridWorld()
        obs, taken = env.reset(), []
        for _ in range(6):
            decision = harness.decide(obs, env.legal_actions())
            taken.append(decision.action.key)
            obs = env.step(decision.action).observation
        actions.append(taken)
    assert actions[0] == actions[1] and off.calls and [(c.messages, c.output) for c in off.calls] == [(c.messages, c.output) for c in on.calls]


def test_no_plan_or_explain_events_with_the_switches_off(tmp_path):
    model = planner_model(lambda state: {"why": "x", "tool": "go_to", "args": {"place": "nowhere"}, "intention": "x"})
    player = CrystalHarness(model, max_calls_per_decision=3)
    run(run_dir=tmp_path / "run", env=GridWorld(), player=player, goal=MILESTONE, budgets=Budgets(max_steps=5, max_decisions=1))
    types = {e["type"] for r in records_of(tmp_path / "run") if r["kind"] == "decision" for e in r["decision"]["trace"]["events"]}
    assert not types & {"plan_step", "plan_check", "explanation", "plan_revised", "replan_cap", "plan_dropped", "plan_done"}
    assert "plan" not in player.end_trial(TrialOutcome(False, "max_steps", 1)).trace["summary"].lower().replace("planner", "")


# Mechanisms and wiring


def test_mechanisms_report_the_switches():
    model = ScriptedModel(lambda m: "{}")
    off = CrystalHarness(model).mechanisms()
    assert off["plan"] is False and off["explain_replan"] is False
    both = CrystalHarness(model, plan=True, explain_replan=True).mechanisms()
    assert both["plan"] is True and both["explain_replan"] is True
    only = CrystalHarness(model, explain_replan=True).mechanisms()
    assert only["plan"] is False and only["explain_replan"] is True


def test_make_player_wires_plan_explain_replan_and_the_cap():
    model = ScriptedModel(lambda m: "{}")
    off = make_player({"name": "crystal_harness"}, model)
    assert (off.plan, off.explain_replan, off.max_replans) == (False, False, 2)
    on = make_player({"name": "crystal_harness", "plan": True, "explain_replan": True, "max_replans": 4}, model)
    assert (on.plan, on.explain_replan, on.max_replans) == (True, True, 4)
    assert make_player({"name": "crystal_harness", "explain_replan": True}, model).plan is False


def test_the_plan_configs_turn_on_exactly_the_switches_they_name():
    expected = {
        "gridworld_7b_crystal_harness_plan": ("gridworld_7b_crystal_harness", True, True),
        "gridworld_7b_crystal_harness_replan_only": ("gridworld_7b_crystal_harness", False, True),
        "firered_7b_crystal_harness_plan": ("firered_7b_crystal_harness", True, True),
    }
    model = ScriptedModel(lambda m: "")
    for name, (base_name, plan, replan) in expected.items():
        spec = json.loads((ROOT / f"experiments/{name}.json").read_text())
        base = json.loads((ROOT / f"experiments/{base_name}.json").read_text())
        player = make_player(spec["agent"], model)
        assert (player.mechanisms()["plan"], player.mechanisms()["explain_replan"]) == (plan, replan), name
        assert {k for k in spec["agent"] if spec["agent"][k] != base["agent"].get(k)} == ({"plan", "explain_replan"} if plan else {"explain_replan"}), name
        assert {k: v for k, v in spec.items() if k != "agent"} == {k: v for k, v in base.items() if k != "agent"}, name


def test_a_plan_step_that_fails_to_start_is_explained_and_replaced():
    # NOPE is not a legal key, so the step cannot start
    model = scenario(
        lambda n: plan_reply([{"tool": "press", "args": {"key": "NOPE"}, "expect": {}}]),
        lambda n: json.dumps({"explanation": "NOPE is not a button", "plan": [{"tool": "press", "args": {"key": "RIGHT"}}]}),
    )
    harness = CrystalHarness(model, plan=True, explain_replan=True)
    harness.start_trial("goal")
    env = GridWorld()
    decision = harness.decide(env.reset(), env.legal_actions())
    assert [e["type"] for e in decision.trace["events"]].count("explanation") == 1
    assert isinstance(decision.action, Press) and decision.action.key == "RIGHT"
