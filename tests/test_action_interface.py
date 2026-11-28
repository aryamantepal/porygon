"""The action-interface switch (`action_interface`, `specialists`) on ReAct, Reflexion, and Voyager.

Everything runs on GridWorld or LineWorld with a ScriptedModel: no ROM, no model server.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pytest

from agents.examples import Examples
from agents.harness import tools
from agents.harness.calls import ToolCallError, parse_tool_call
from agents.models import ScriptedModel
from agents.protocol import TrialOutcome
from agents.react import ReAct
from agents.reflexion import Reflexion
from agents.voyager import Voyager, prompts
from agents.voyager.skills import Run, SkillError, execute, parse
from environments.gridworld import GridWorld, GridWorldConfig
from environments.lineworld import LineWorld
from environments.protocol import Observation, ObservationMode, Press
from evaluation.goals import Goal
from evaluation.runner import Budgets, run
from experiments.run import make_player

ROOT = Path(__file__).resolve().parents[1]
MILESTONE = Goal("milestone", "Reach the milestone in town.", {"milestone": True})
KEYS = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]
LEGAL = [Press(k) for k in KEYS]


def grid():
    return GridWorld(GridWorldConfig(mode=ObservationMode.STRUCTURED))


def state_of(messages):
    """The State of the newest observation in a ReAct prompt."""
    return json.loads(messages[1].content.rsplit("State: ", 1)[1].splitlines()[0])


def react_model(plan):
    """ReAct replies come from `plan(state, prompt)`; a battle choice gets a fixed reply."""

    def reply(messages):
        if messages[0].content.startswith("You choose the next move"):
            return '{"why": "weaken it first", "option": "tackle"}'
        return plan(state_of(messages), messages[1].content)

    return ScriptedModel(reply)


def to_milestone(state, prompt):
    if state["map"] == "house":
        return "Thought 1: the door leads to town.\nAction 1: go_to door_to_town"
    return "Thought 2: the milestone is here.\nAction 2: go_to milestone"


def records_of(run_dir, trial=0):
    return [json.loads(line) for line in (run_dir / f"trials/{trial:03d}/steps.jsonl").read_text().splitlines()]


def events(records, kind):
    return [e for r in records if r["kind"] == "decision" for e in r["decision"]["trace"].get("events", []) if e["type"] == kind]


def play(player, env, steps):
    player.start_trial("goal text")
    obs = env.reset()
    decisions = []
    for _ in range(steps):
        decision = player.decide(obs, env.legal_actions())
        decisions.append(decision)
        if decision.action:
            obs = env.step(decision.action).observation
    return decisions


# Buttons mode is what it was


def _digest(model, actions, review=None):
    payload = json.dumps({"actions": actions, "calls": [[c.messages, c.output] for c in model.calls], "review": review}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _drive(player, env, steps, model):
    player.start_trial("goal text")
    obs = env.reset()
    actions = []
    for _ in range(steps):
        d = player.decide(obs, env.legal_actions())
        actions.append(d.action.key if d.action else None)
        if d.action:
            obs = env.step(d.action).observation
    review = player.end_trial(TrialOutcome(False, "max_steps", steps))
    return _digest(model, actions, review.trace)


# Digests taken from the commit before the switches existed (every request and every decision).
# If one of these changes, buttons-mode prompts changed: that is a regression, not a refresh.
BUTTONS_DIGESTS = {
    "react_line": "777bdcd32dc0955c",
    "react_retry_grid": "a6ba9f30805917e2",
    "reflexion_grid": "98bc23f0fb6d1a4d",
    "voyager_line": "9a4062b22c26535c",
}


def test_buttons_mode_prompts_and_decisions_are_unchanged():
    cycle = itertools.cycle(["Go right.\nAction 1: RIGHT", "Go on.\nAction: DOWN", "Hmm.\nAction 3: JUMP", "no action here", "RIGHT"])
    model = ScriptedModel(lambda msgs: next(cycle))
    assert _drive(ReAct(model), LineWorld(), 8, model) == BUTTONS_DIGESTS["react_line"]

    cycle = itertools.cycle(["Go right.\nAction 1: RIGHT", "Hmm.\nAction 3: JUMP", "RIGHT", "x"])
    model = ScriptedModel(lambda msgs: next(cycle))
    assert _drive(ReAct(model, retry_illegal_action=True, history_turns=3), GridWorld(), 9, model) == BUTTONS_DIGESTS["react_retry_grid"]

    examples = Examples.from_dict(json.loads((ROOT / "experiments/examples/gridworld.json").read_text()))
    cycle = itertools.cycle(["Go right.\nAction 1: RIGHT", "Go on.\nAction: DOWN", "Plan: do better"])
    model = ScriptedModel(lambda msgs: next(cycle))
    assert _drive(Reflexion(model, examples=examples), GridWorld(), 6, model) == BUTTONS_DIGESTS["reflexion_grid"]

    walk = "Explain: none\nPlan:\n1) walk\nCode:\n```skill\nskill walk_right\nuntil x == 9 (max 12):\n  RIGHT\n```"
    light = "Plan:\n1) reuse walk\nCode:\n```skill\nskill light_lamp\ncall walk_right\nA\n```"
    proposed = []

    def reply(messages):
        system = messages[0].content
        if system.startswith(prompts.CURRICULUM[:40]):
            proposed.append(["Reasoning: r.\nTask: Walk to the right end", "Reasoning: r.\nTask: Light the lamp"][min(len(proposed), 1)])
            return proposed[-1]
        if system == prompts.CRITIC:
            return '{"reasoning": "done", "success": true, "critique": ""}'
        if system == prompts.SKILL_DESCRIPTION:
            return "Walks right."
        return light if "Task: Light the lamp" in messages[1].content else walk

    model = ScriptedModel(reply)
    assert _drive(Voyager(model), LineWorld(), 30, model) == BUTTONS_DIGESTS["voyager_line"]


def test_mechanisms_report_the_switches():
    model = ScriptedModel(lambda m: "")
    for build in (ReAct, Reflexion, Voyager):
        assert build(model).mechanisms()["tools"] is False and build(model).mechanisms()["specialists"] is False
        mechanisms = build(model, action_interface="tools").mechanisms()
        assert mechanisms["tools"] is True and mechanisms["specialists"] is False
        mechanisms = build(model, action_interface="tools", specialists=True).mechanisms()
        assert mechanisms["tools"] is True and mechanisms["specialists"] is True
        assert all(isinstance(v, bool) for v in mechanisms.values())


def test_the_switches_are_validated():
    model = ScriptedModel(lambda m: "")
    for build in (ReAct, Reflexion, Voyager):
        with pytest.raises(ValueError, match="only works with"):
            build(model, specialists=True)
        with pytest.raises(ValueError, match="unknown action_interface"):
            build(model, action_interface="mouse")


def test_make_player_wires_the_config_keys():
    model = ScriptedModel(lambda m: "")
    for name in ("react", "reflexion", "voyager"):
        assert make_player({"name": name}, model).mechanisms()["tools"] is False
        player = make_player({"name": name, "action_interface": "tools", "specialists": True}, model)
        assert player.mechanisms()["tools"] is True and player.mechanisms()["specialists"] is True
        assert make_player({"name": name, "action_interface": "tools"}, model).mechanisms()["specialists"] is False


def test_the_tools_configs_set_both_switches():
    for env in ("gridworld", "firered"):
        for agent in ("react", "reflexion", "voyager"):
            spec = json.loads((ROOT / f"experiments/{env}_7b_{agent}_tools.json").read_text())
            base = json.loads((ROOT / f"experiments/{env}_7b_{agent}.json").read_text())
            assert spec["agent"] == {**base["agent"], "action_interface": "tools", "specialists": True}
            assert {k: v for k, v in spec.items() if k != "agent"} == {k: v for k, v in base.items() if k != "agent"}
    spec = json.loads((ROOT / "experiments/gridworld_7b_react_tools_nospecialists.json").read_text())
    assert spec["agent"]["action_interface"] == "tools" and spec["agent"]["specialists"] is False


# The tool-call grammar


@pytest.mark.parametrize(
    ("text", "tool", "args"),
    [
        ("go_to door_0", "go_to", {"place": "door_0"}),
        ("go_to [4, 2]", "go_to", {"position": [4, 2]}),
        ("`go_to door_0`", "go_to", {"place": "door_0"}),
        ("talk_to person.", "talk_to", {"place": "person"}),
        ("press A", "press", {"key": "A", "times": 1}),
        ("press a x3", "press", {"key": "A", "times": 3}),
        ("GO_TO door_0", "go_to", {"place": "door_0"}),
    ],
)
def test_parse_tool_call_accepts(text, tool, args):
    call = parse_tool_call(text, legal_keys=KEYS, place_ids=["door_0", "person"])
    assert call.tool == tool and dict(call.args) == args


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("fly_to door_0", "not a tool"),
        ("go_to nowhere", "no place 'nowhere'"),
        ("go_to", "needs a place id"),
        ("go_to [4]", "needs a place id"),
        ("press JUMP", "not a legal button"),
        ("press A x0", "out of range"),
        ("press A x99", "out of range"),
        ("press", "press needs a button"),
        ("go_to door_0 and then press A", "needs a place id"),
        ("", "not a tool call"),
    ],
)
def test_parse_tool_call_rejects(text, reason):
    with pytest.raises(ToolCallError, match=reason):
        parse_tool_call(text, legal_keys=KEYS, place_ids=["door_0", "person"])


def test_a_tool_that_is_not_offered_is_rejected():
    with pytest.raises(ToolCallError, match="not a tool"):
        parse_tool_call("go_to door_0", legal_keys=KEYS, offered=["press"])


# ReAct with tools


def test_react_reaches_the_milestone_with_far_fewer_model_calls_than_presses(tmp_path):
    model = react_model(to_milestone)
    player = ReAct(model, action_interface="tools", specialists=True)
    results = run(run_dir=tmp_path / "run", env=grid(), player=player, goal=MILESTONE, budgets=Budgets(max_steps=60))
    assert results[0].success
    asked = [c for c in model.calls if c.messages[0]["content"].startswith("You are playing")]
    assert len(asked) == 3 and results[0].model_calls == 6 and results[0].model_calls * 2 < results[0].steps  # 3 tool calls, 3 battle picks, 13 presses
    records = records_of(tmp_path / "run")
    decisions = [r["decision"] for r in records if r["kind"] == "decision"]
    chosen = [d for d in decisions if "thought" in d["trace"] and d["action"]]
    assert chosen and all(d["trace"]["tool"] in ("go_to", "press") for d in chosen)
    # presses without a model call say which tool they belong to and how far along it is
    pressing = [d["trace"] for d in decisions if not d["calls"] and d["action"] and d["trace"]["handler"] == "model"]
    assert pressing and all(t["tool"] == "go_to" and t["press"] >= 2 for t in pressing)
    # every press is still one runner step
    assert all(d["action"] for d in decisions)


def test_tool_results_open_the_next_observation(tmp_path):
    model = react_model(to_milestone)
    run(run_dir=tmp_path / "run", env=grid(), player=ReAct(model, action_interface="tools", specialists=True), goal=MILESTONE, budgets=Budgets(max_steps=60))
    asks = [c.messages[1]["content"] for c in model.calls if c.messages[0]["content"].startswith("You are playing")]
    # the second turn's Observation starts with how the first tool ended, before the snapshot
    assert "Observation 2: go_to door_to_town: went through to town\nState: " in asks[1]
    assert "Action 1: go_to door_to_town" in asks[1]
    assert "Tools:" in asks[0] and "- go_to <place id>" in asks[0] and "Places you can name: person (person), door_to_town (door)" in asks[0]
    assert "Observation 1: State:" in asks[0]


def test_a_failed_tool_is_reported_not_pressed(tmp_path):
    replies = iter(["Thought 1: try far away.\nAction 1: go_to [10, 10]"])
    model = react_model(lambda state, prompt: next(replies, "Thought 2: ok.\nAction 2: press RIGHT"))
    player = ReAct(model, action_interface="tools")
    decisions = play(player, grid(), 2)
    assert decisions[0].action is None and decisions[0].trace["events"][-1]["outcome"] == "failed"
    follow_up = model.calls[1].messages[1]["content"]
    assert "Observation 2: go_to [10, 10] failed: no path from [1, 1] to [10, 10]\nState: " in follow_up
    assert "Action 1: go_to [10, 10]" in follow_up


def test_unknown_tools_and_places_are_invalid_and_recorded(tmp_path):
    replies = iter(["x.\nAction 1: fly_to door_to_town", "x.\nAction 2: go_to nowhere", "x.\nAction 3: press JUMP"])
    model = react_model(lambda state, prompt: next(replies, "ok.\nAction 4: press RIGHT"))
    player = ReAct(model, action_interface="tools")
    decisions = play(player, grid(), 4)
    assert [d.action is None for d in decisions[:3]] == [True, True, True]
    assert decisions[3].action.key == "RIGHT"
    reasons = [e["reason"] for d in decisions[:3] for e in d.trace["events"] if e["type"] == "invalid_tool_call"]
    assert "not a tool" in reasons[0] and "no place 'nowhere'" in reasons[1] and "not a legal button" in reasons[2]
    last = model.calls[3].messages[1]["content"]
    assert "Action 1: (invalid: fly_to door_to_town)" in last and "Action 2: (invalid: go_to nowhere)" in last


def test_retry_illegal_action_applies_to_tool_calls():
    replies = iter(["x.\nAction 1: go_to nowhere", "go_to door_to_town"])
    model = react_model(lambda state, prompt: next(replies))
    player = ReAct(model, action_interface="tools", retry_illegal_action=True)
    decision = play(player, grid(), 1)[0]
    assert decision.action is not None and decision.action.key in KEYS  # the retry's go_to started walking
    assert decision.trace["tool"] == "go_to" and decision.trace["args"] == {"place": "door_to_town"}
    assert len(decision.calls) == 2 and decision.trace["action_retry"] is True
    retry = model.calls[1].messages[1]["content"]
    assert "'go_to nowhere' is not a valid tool call: no place 'nowhere' on this map" in retry
    assert "Places: person (person), door_to_town (door)" in retry and retry.endswith("Action 1:")


def test_visual_mode_offers_only_press_and_says_so():
    replies = iter(["x.\nAction 1: press RIGHT x2", "x.\nAction 2: go_to door_to_town"])
    model = ScriptedModel(lambda messages: next(replies))
    env = GridWorld(GridWorldConfig(mode=ObservationMode.VISUAL))
    player = ReAct(model, action_interface="tools")
    player.start_trial("goal")
    observation = env.reset()
    decision = player.decide(observation, env.legal_actions())
    assert decision.action.key == "RIGHT" and decision.trace["tools"] == ["press"]
    assert any(e["type"] == "tools_offered" and "no grid" in e["reason"] for e in decision.trace["events"])
    prompt = model.calls[0].messages[1]["content"]
    assert "- press <BUTTON>" in prompt and "- go_to" not in prompt and "not offered" in prompt
    # go_to is then not a tool at all
    again = player.decide(env.step(decision.action).observation, env.legal_actions())  # second press of `press RIGHT x2`
    assert again.action.key == "RIGHT" and not again.calls
    third = player.decide(env.step(again.action).observation, env.legal_actions())
    assert third.action is None and any(e["type"] == "invalid_tool_call" for e in third.trace["events"])


def test_press_times_runs_without_more_model_calls():
    model = react_model(lambda state, prompt: "x.\nAction 1: press DOWN x3")
    decisions = play(ReAct(model, action_interface="tools"), grid(), 3)
    assert [d.action.key for d in decisions] == ["DOWN"] * 3 and [bool(d.calls) for d in decisions] == [True, False, False]
    assert [d.trace["press"] for d in decisions] == [1, 2, 3]


def test_a_tool_that_does_nothing_is_a_failed_result():
    replies = iter(["x.\nAction 1: go_to [1, 1]", "x.\nAction 2: press A"])
    model = react_model(lambda state, prompt: next(replies))
    decisions = play(ReAct(model, action_interface="tools"), grid(), 2)
    assert decisions[0].action is None
    assert "go_to [1, 1] failed: nothing to do: arrived at [1, 1]" in model.calls[1].messages[1]["content"]


def test_without_specialists_the_model_handles_a_battle_itself():
    model = react_model(lambda state, prompt: "Thought: fight.\nAction: press A")
    player = ReAct(model, action_interface="tools", specialists=False)
    player.start_trial("win")
    decision = player.decide(_obs({"context": "battle", "battle": BATTLE}), LEGAL)
    assert decision.action.key == "A" and decision.trace["handler"] == "model" and len(model.calls) == 1
    assert model.calls[0].messages[0]["content"].startswith("You are playing")  # the ReAct prompt, not the battle specialist's


def test_examples_still_show_and_the_system_prompt_says_how_they_differ():
    examples = Examples.from_dict(json.loads((ROOT / "experiments/examples/gridworld.json").read_text()))
    model = react_model(lambda state, prompt: "x.\nAction 1: press A")
    player = ReAct(model, action_interface="tools", examples=examples)
    play(player, grid(), 1)
    system = model.calls[0].messages[0]["content"]
    assert "an Action is a tool call, and `press <BUTTON>` presses one" in system and examples.trajectories[0].goal in system
    assert player.mechanisms()["few_shot"] is True


# Specialists


def _obs(state):
    return Observation(ObservationMode.STRUCTURED, 0, None, state)


def test_specialists_wait_out_busy_without_a_model_call():
    model = react_model(lambda state, prompt: "x.\nAction 1: press A")
    player = ReAct(model, action_interface="tools", specialists=True)
    player.start_trial("goal")
    legal = [Press(k) for k in (*KEYS, "WAIT")]
    busy = _obs({"context": "busy", "controls": {"wait": "WAIT"}})
    keys = [player.decide(busy, legal).action.key for _ in range(7)]
    assert keys == ["WAIT"] * 6 + ["B"] and model.calls == []


def test_specialists_confirm_dialogue_and_hand_the_text_to_the_next_observation():
    model = react_model(lambda state, prompt: "x.\nAction 1: press UP")
    player = ReAct(model, action_interface="tools", specialists=True)
    player.start_trial("goal")
    talking = _obs({"context": "dialogue", "dialogue": {"text": "The milestone is southeast."}})
    decision = player.decide(talking, LEGAL)
    assert decision.action.key == "A" and model.calls == [] and decision.trace["handler"] == "dialogue"
    overworld = _obs({"context": "overworld", "map": "m", "position": [1, 1], "grid": ["..."], "places": []})
    player.decide(overworld, LEGAL)
    assert "Observation 1: Dialogue: The milestone is southeast.\nState: " in model.calls[0].messages[1]["content"]


def test_without_specialists_dialogue_goes_to_the_model():
    model = react_model(lambda state, prompt: "x.\nAction 1: press A")
    player = ReAct(model, action_interface="tools")
    player.start_trial("goal")
    talking = _obs({"context": "dialogue", "dialogue": {"text": "Hello."}})
    decision = player.decide(talking, LEGAL)
    assert decision.action.key == "A" and len(model.calls) == 1


BATTLE = {
    "options": [
        {"id": "tackle", "kind": "attack", "power": 35, "uses_left": 35, "keys": ["A", "A"]},
        {"id": "growl", "kind": "status", "power": 0, "uses_left": 40, "keys": ["A", "RIGHT", "A"]},
    ],
    "opponent": {"hp": 18, "max_hp": 18, "catchable": False},
}


def test_battle_specialist_picks_with_the_harness_prompt_and_the_adapters_keys():
    model = ScriptedModel(lambda messages: '{"why": "lower its attack", "option": "growl"}')
    player = ReAct(model, action_interface="tools", specialists=True)
    player.start_trial("win")
    observation = _obs({"context": "battle", "battle": BATTLE})
    presses = [player.decide(observation, LEGAL) for _ in range(3)]
    assert [d.action.key for d in presses] == ["A", "RIGHT", "A"]
    assert len(model.calls) == 1 and model.calls[0].messages[0]["content"].startswith("You choose the next move")
    assert presses[0].trace["handler"] == "battle" and presses[0].calls and not presses[1].calls
    assert presses[0].trace["events"][0]["choice"] == "growl"


def test_battle_without_options_just_confirms():
    model = ScriptedModel(lambda messages: "{}")
    player = ReAct(model, action_interface="tools", specialists=True)
    player.start_trial("win")
    observation = _obs({"context": "battle", "battle": {"moves": [], "options": []}})
    assert player.decide(observation, LEGAL).action.key == "A" and model.calls == []


def test_a_battle_during_a_press_run_interrupts_it_and_the_specialist_plays(tmp_path):
    def plan(state, prompt):
        if state["map"] == "house":
            return "x.\nAction: go_to door_to_town"
        # in town: walk down into the grass, and once that has been cut short, head for the milestone
        return "x.\nAction: press DOWN x6" if "interrupted" not in prompt else "x.\nAction: go_to milestone"

    model = react_model(plan)
    player = ReAct(model, action_interface="tools", specialists=True)
    results = run(run_dir=tmp_path / "run", env=grid(), player=player, goal=MILESTONE, budgets=Budgets(max_steps=80))
    assert results[0].success
    records = records_of(tmp_path / "run")
    ended = [e for e in events(records, "tool_end") if e["outcome"] == "interrupted"]
    assert ended and ended[0]["result"] == "the game switched to battle" and ended[0]["presses"] == 1
    assert [e["choice"] for e in events(records, "battle")] == ["tackle", "tackle", "tackle"]
    later = [c.messages[1]["content"] for c in model.calls if "interrupted" in c.messages[1]["content"]]
    assert "press DOWN x6 interrupted: the game switched to battle\nBattle: code chose tackle" in later[0]


def test_tool_run_interrupts_on_the_named_context_only_after_pressing():
    def steps():
        yield Press("A")
        yield Press("A")
        return "done"

    battle = _obs({"context": "battle"})
    run_ = tools.ToolRun("model", "press", {}, steps())
    assert isinstance(run_.advance(battle, ("battle",)), Press)  # nothing pressed yet: start it
    end = run_.advance(battle, ("battle",))
    assert end == tools.ToolEnd("interrupted", "the game switched to battle")


# Reflexion with tools


def test_reflexion_with_tools_reflects_on_tool_call_turns():
    def reply(messages):
        if messages[0].content.startswith("You will be given the history"):
            return "Plan: go to the door first, then the milestone."
        return "I will try the door.\nAction 1: go_to door_to_town"

    model = ScriptedModel(reply)
    player = Reflexion(model, action_interface="tools", specialists=True)
    env = grid()
    play(player, env, 7)
    review = player.end_trial(TrialOutcome(False, "max_steps", 4))
    assert review.trace["reflection"]["text"].startswith("Plan: go to the door")
    query = review.calls[0].messages[0]["content"]
    assert "Action 1: go_to door_to_town" in query
    assert "Observation 2: go_to door_to_town: went through to town\nState: " in query
    # and the next trial's prompt carries the reflection
    player.start_trial("again")
    decision = player.decide(env.reset(), env.legal_actions())
    assert "Your memory for the task below:" in model.calls[-1].messages[0]["content"] and decision.trace["reflections_in_context"] == ["r0"]


def test_reflexion_query_gives_the_last_tools_result():
    model = ScriptedModel(lambda messages: "x.\nAction 1: go_to [10, 10]" if messages[0].content.startswith("You are") else "Plan: x")
    player = Reflexion(model, action_interface="tools")
    play(player, grid(), 1)
    review = player.end_trial(TrialOutcome(False, "max_steps", 1))
    assert "Result 1: go_to [10, 10] failed: no path from [1, 1] to [10, 10]" in review.calls[0].messages[0]["content"]


def test_reflexion_query_marks_a_tool_that_was_still_running():
    model = ScriptedModel(lambda messages: "x.\nAction 1: go_to door_to_town" if messages[0].content.startswith("You are") else "Plan: x")
    player = Reflexion(model, action_interface="tools")
    play(player, grid(), 2)
    review = player.end_trial(TrialOutcome(False, "max_steps", 2))
    assert "Result 1: (the trial ended while this tool was still running)" in review.calls[0].messages[0]["content"]


def test_reflexion_buttons_query_has_no_result_lines():
    model = ScriptedModel(lambda messages: "x.\nAction 1: RIGHT" if messages[0].content.startswith("You are") else "Plan: x")
    player = Reflexion(model)
    play(player, grid(), 2)
    review = player.end_trial(TrialOutcome(False, "max_steps", 2))
    assert "Result" not in review.calls[0].messages[0]["content"]


# Voyager with tools

TOWN_SKILL = "Explain: none\nPlan:\n1) walk\nCode:\n```skill\nskill reach_milestone\ngo_to door_to_town\ngo_to [7, 2]\ngo_to milestone\n```"


def voyager_model(code, critic='{"reasoning": "done", "success": true, "critique": ""}', battle=None):
    def reply(messages):
        system = messages[0].content
        if system.startswith(prompts.CURRICULUM[:40]):
            return "Reasoning: go.\nTask: Reach the milestone"
        if system == prompts.CRITIC:
            return critic
        if system == prompts.SKILL_DESCRIPTION:
            return "Walks to the milestone."
        if system.startswith("You choose the next move"):
            return battle or '{"why": "weaken it", "option": "tackle"}'
        return code(messages) if callable(code) else code

    return ScriptedModel(reply)


def test_voyager_go_to_runs_through_the_navigator_and_counts_its_presses(tmp_path):
    model = voyager_model(TOWN_SKILL)
    player = Voyager(model, action_interface="tools", specialists=True)
    results = run(run_dir=tmp_path / "run", env=grid(), player=player, goal=MILESTONE, budgets=Budgets(max_steps=60))
    assert results[0].success
    records = records_of(tmp_path / "run")
    # the goal was met on the program's last press: the runner counted each one, and the trace counts them the same way
    assert results[0].steps == 10 and results[0].model_calls == 2  # the curriculum and the program, no call per press
    pressing = [r["decision"]["trace"] for r in records if r["kind"] == "decision" and r["decision"]["action"]]
    assert [t["press"] for t in pressing] == list(range(1, 11))
    assert [t["statement"] for t in pressing][0] == "go_to door_to_town" and pressing[-1]["statement"] == "go_to milestone"
    # the language lists the tool statements, in tools mode only
    program_prompt = next(c.messages[0]["content"] for c in model.calls if "skill program" in c.messages[0]["content"])
    assert "go_to <place_id>" in program_prompt and "talk_to <place_id>" in program_prompt


def test_voyager_program_end_reports_presses_and_what_each_tool_returned(tmp_path):
    model = voyager_model("Code:\n```skill\nskill reach_town\ngo_to door_to_town\ngo_to [7, 2]\n```")
    player = Voyager(model, action_interface="tools", specialists=True)
    run(run_dir=tmp_path / "run", env=grid(), player=player, goal=MILESTONE, budgets=Budgets(max_steps=9))
    end = events(records_of(tmp_path / "run"), "program_end")[0]
    assert end["error"] == "" and end["presses"] == 8
    assert "tools: go_to door_to_town: went through to town; go_to [7, 2]: arrived at [7, 2]" in end["feedback"]


def test_voyager_buttons_prompt_does_not_list_tool_statements():
    model = voyager_model("Code:\n```skill\nskill s\nRIGHT\n```")
    play(Voyager(model), grid(), 1)
    program_prompt = next(c.messages[0]["content"] for c in model.calls if "skill program" in c.messages[0]["content"])
    assert program_prompt.count("go_to") == 0 and program_prompt.count("talk_to") == 0


def test_go_to_does_not_parse_in_buttons_mode():
    with pytest.raises(SkillError, match="`go_to` is only available with the tools interface"):
        parse("skill s\ngo_to door_0", KEYS)
    with pytest.raises(SkillError, match="`talk_to` is only available"):
        parse("skill s\nrepeat 2:\n  talk_to person", KEYS)
    program = parse("skill s\ngo_to door_0\ntalk_to [3, 2]\ngo_to x3", KEYS, tools=True)
    assert [s.label for s in program.body] == ["go_to door_0", "talk_to [3, 2]", "go_to x3"]
    with pytest.raises(SkillError, match="needs a place id or"):
        parse("skill s\ngo_to", KEYS, tools=True)


def test_voyager_in_buttons_mode_feeds_the_parse_error_back(tmp_path):
    model = voyager_model(TOWN_SKILL)
    player = Voyager(model, max_attempts=1)
    run(run_dir=tmp_path / "run", env=grid(), player=player, goal=MILESTONE, budgets=Budgets(max_steps=5, max_decisions=3))
    program = events(records_of(tmp_path / "run"), "program")[0]
    assert "only available with the tools interface" in program["error"]


def test_a_tool_error_in_a_skill_is_a_skill_error():
    env = grid()
    program = parse("skill s\ngo_to nowhere", KEYS, tools=True)
    with pytest.raises(SkillError, match="`go_to nowhere` failed: no place 'nowhere' on this map"):
        next(execute(program, None, env.reset(), Run(), KEYS))
    program = parse("skill s\ngo_to [10, 10]", KEYS, tools=True)
    with pytest.raises(SkillError, match="no path from"):
        next(execute(program, None, env.reset(), Run(), KEYS))
    with pytest.raises(SkillError, match="buttons interface"):
        next(execute(program, None, env.reset(), Run()))


def test_skill_tool_presses_count_against_the_press_limit():
    env = grid()
    program = parse("skill s\nrepeat 150:\n  go_to door_to_town\n  go_to door_to_house", KEYS, tools=True)
    run_ = Run()
    steps = execute(program, None, env.reset(), run_, KEYS)
    obs = env.observe()
    with pytest.raises(SkillError, match="stopped after 200 presses"):
        press = next(steps)
        while True:
            obs = env.step(press).observation
            press = steps.send(obs)
    assert run_.presses == 200


def test_voyager_failed_tool_reaches_the_next_attempt(tmp_path):
    attempts = iter(["Code:\n```skill\nskill s\ngo_to nowhere\n```", TOWN_SKILL])
    model = voyager_model(lambda messages: next(attempts), critic='{"reasoning": "no", "success": false, "critique": "it failed"}')
    player = Voyager(model, action_interface="tools")
    results = run(run_dir=tmp_path / "run", env=grid(), player=player, goal=MILESTONE, budgets=Budgets(max_steps=60))
    assert results[0].success
    second = [c.messages[1]["content"] for c in model.calls if "Execution error:" in c.messages[1]["content"]][1]
    assert "Execution error: `go_to nowhere` failed: no place 'nowhere' on this map" in second


def test_voyager_specialists_end_a_program_when_a_battle_starts(tmp_path):
    finish = "Code:\n```skill\nskill finish\ngo_to milestone\n```"
    attempts = iter(["Code:\n```skill\nskill s\ngo_to door_to_town\nDOWN x4\n```", finish])
    model = voyager_model(lambda messages: next(attempts))
    player = Voyager(model, action_interface="tools", specialists=True)
    results = run(run_dir=tmp_path / "run", env=grid(), player=player, goal=MILESTONE, budgets=Budgets(max_steps=80))
    assert results[0].success
    records = records_of(tmp_path / "run")
    first_end = events(records, "program_end")[0]
    assert first_end["error"] == "interrupted: battle started" and first_end["interrupted"] is True
    assert not events(records, "critic")  # the interrupted program was not verified; the second ended the trial
    assert [e["choice"] for e in events(records, "battle")] == ["tackle", "tackle", "tackle"]
    second_prompt = [c.messages[1]["content"] for c in model.calls if "Execution error:" in c.messages[1]["content"]][1]
    assert "Execution error: interrupted: battle started" in second_prompt
    specials = [r["decision"]["trace"]["special"] for r in records if r["kind"] == "decision" and r["decision"]["trace"].get("special")]
    assert specials and all(s["handler"] == "battle" for s in specials)


def test_voyager_specialists_handle_busy_and_dialogue_before_any_model_call():
    model = voyager_model("unused")
    player = Voyager(model, action_interface="tools", specialists=True)
    player.start_trial("goal")
    legal = [Press(k) for k in (*KEYS, "WAIT")]
    assert player.decide(_obs({"context": "busy", "controls": {"wait": "WAIT"}}), legal).action.key == "WAIT"
    assert player.decide(_obs({"context": "dialogue", "dialogue": {"text": "Hi."}}), legal).action.key == "A"
    assert model.calls == []


def test_voyager_without_specialists_does_not_route():
    model = voyager_model("Code:\n```skill\nskill s\nA\n```")
    player = Voyager(model, action_interface="tools")
    player.start_trial("goal")
    decision = player.decide(_obs({"context": "dialogue", "dialogue": {"text": "Hi."}}), LEGAL)
    assert decision.calls  # the curriculum is asked even though a dialogue is open


def test_voyager_visual_mode_lists_no_tool_statements_and_says_so():
    model = voyager_model("Code:\n```skill\nskill s\nRIGHT\n```")
    env = GridWorld(GridWorldConfig(mode=ObservationMode.VISUAL))
    player = Voyager(model, action_interface="tools")
    decision = play(player, env, 1)[0]
    assert any(e["type"] == "tools_offered" and not e["tools"] for e in decision.trace["events"])
    program_prompt = next(c.messages[0]["content"] for c in model.calls if "skill program" in c.messages[0]["content"])
    assert "go_to" not in program_prompt
