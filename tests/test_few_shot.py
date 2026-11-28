from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.direct import DirectPolicy
from agents.direct.prompts import SYSTEM as DIRECT_SYSTEM
from agents.examples import MAX_PAIRS, MAX_TURNS, Examples
from agents.harness import CrystalHarness
from agents.models import ScriptedModel
from agents.protocol import TrialOutcome
from agents.react import ReAct
from agents.react.prompts import SYSTEM as REACT_SYSTEM
from agents.reflexion import Reflexion
from agents.reflexion.prompts import REFLECTION_PROMPT
from agents.voyager import Voyager
from environments.gridworld import GridWorld
from environments.lineworld import LineWorld
from experiments.run import file_hashes, make_player

ROOT = Path(__file__).resolve().parents[1]
GOAL = "Walk to the far right."
STATE = {"x": 3}
EXAMPLES = Examples.from_dict(
    {
        "source": "test",
        "trajectories": [
            {
                "goal": "Light the lamp.",
                "turns": [
                    {"observation": STATE, "thought": "The lamp is ahead.", "action": "RIGHT"},
                    {"observation": "free text", "thought": "Press A.", "action": "A"},
                ],
                "outcome": "The lamp is lit.",
            }
        ],
        "reflections": [{"trajectory_summary": "I pressed LEFT forever.", "reflection": "Plan: go RIGHT."}],
        "programs": ["skill example_program\nA"],
        "tool_calls": [{"situation": STATE, "goal": "Light the lamp.", "reply": {"why": "walk", "tool": "press", "args": {"key": "RIGHT"}}}],
    }
)
TRAJECTORY = (
    "Goal: Light the lamp.\n"
    'Observation 1: State: {"x": 3}\nThought 1: The lamp is ahead.\nAction 1: RIGHT\n'
    "Observation 2: free text\nThought 2: Press A.\nAction 2: A\n"
    "The lamp is lit."
)


def first_call(player, model, env=None):
    env = env or LineWorld()
    player.start_trial(GOAL)
    player.decide(env.reset(), env.legal_actions())
    return model.calls[0].messages


# Direct


def test_direct_shows_pairs_before_the_task():
    model = ScriptedModel(lambda messages: "RIGHT")
    player = DirectPolicy(model, examples=EXAMPLES)
    user = first_call(player, model)[1]["content"]
    assert 'Goal: Light the lamp.\nState: {"x": 3}\nReply: RIGHT\n\nGoal: Light the lamp.\nfree text\nReply: A' in user
    assert user.index("Reply: A") < user.index(f"Goal: {GOAL}") and "Thought" not in user
    assert player.mechanisms()["few_shot"] is True


def test_direct_zero_shot_prompt_is_unchanged():
    model = ScriptedModel(lambda messages: "RIGHT")
    player = DirectPolicy(model)
    messages = first_call(player, model)
    assert messages[0]["content"] == DIRECT_SYSTEM
    assert messages[1]["content"].startswith(f"Goal: {GOAL}\n") and "Reply: " not in messages[1]["content"]
    assert player.mechanisms()["few_shot"] is False


# ReAct


def test_react_shows_trajectories_in_the_loops_own_format():
    model = ScriptedModel(lambda messages: "Go.\nAction: RIGHT")
    player = ReAct(model, examples=EXAMPLES)
    messages = first_call(player, model)
    assert TRAJECTORY in messages[0]["content"] and messages[0]["content"].startswith(REACT_SYSTEM)
    assert messages[1]["content"].startswith(f"Goal: {GOAL}")
    assert player.mechanisms()["few_shot"] is True


def test_react_zero_shot_prompt_is_unchanged():
    model = ScriptedModel(lambda messages: "Go.\nAction: RIGHT")
    player = ReAct(model)
    messages = first_call(player, model)
    assert messages[0]["content"] == REACT_SYSTEM and "examples" not in messages[1]["content"]
    assert player.mechanisms()["few_shot"] is False


def test_react_examples_without_trajectories_are_zero_shot():
    model = ScriptedModel(lambda messages: "Go.\nAction: RIGHT")
    player = ReAct(model, examples=Examples.from_dict({"source": "test", "programs": ["skill s\nA"]}))
    assert first_call(player, model)[0]["content"] == REACT_SYSTEM and player.mechanisms()["few_shot"] is False


def test_example_turns_are_capped():
    turns = [{"observation": f"o{i}", "thought": "t", "action": "A"} for i in range(MAX_TURNS + 3)]
    examples = Examples.from_dict({"source": "test", "trajectories": [{"goal": "g", "turns": turns, "outcome": "done"}]})
    text = examples.trajectories_text()
    assert f"Observation {MAX_TURNS}:" in text and f"Observation {MAX_TURNS + 1}:" not in text
    assert "(3 later turns omitted)\ndone" in text
    assert examples.pairs_text().count("Reply:") == min(MAX_PAIRS, MAX_TURNS + 3)


# Reflexion


def test_reflexion_adds_example_reflections_to_the_reflection_prompt():
    model = ScriptedModel(lambda messages: "Go.\nAction: LEFT")
    player = Reflexion(model, examples=EXAMPLES)
    assert TRAJECTORY in first_call(player, model)[0]["content"]
    player.end_trial(TrialOutcome(False, "max_steps", 1))
    query = model.calls[-1].messages[0]["content"]
    example = "I pressed LEFT forever.\nNew plan: Plan: go RIGHT."
    assert example in query and query.index(example) < query.index("Here is the task:") and query.endswith("New plan:")
    assert player.mechanisms()["few_shot"] is True


def test_reflexion_zero_shot_reflection_prompt_is_unchanged():
    model = ScriptedModel(lambda messages: "Go.\nAction: LEFT")
    player = Reflexion(model)
    assert first_call(player, model)[0]["content"] == REACT_SYSTEM
    player.end_trial(TrialOutcome(False, "max_steps", 1))
    assert model.calls[-1].messages[0]["content"].startswith(REFLECTION_PROMPT + "\n\nHere is the task:")
    assert player.mechanisms()["few_shot"] is False


def test_example_reflections_are_unused_when_reflections_are_off():
    examples = Examples.from_dict({"source": "test", "reflections": [{"trajectory_summary": "s", "reflection": "r"}]})
    assert Reflexion(ScriptedModel(lambda m: ""), examples=examples).mechanisms()["few_shot"] is True
    assert Reflexion(ScriptedModel(lambda m: ""), examples=examples, reflections=False).mechanisms()["few_shot"] is False


# Voyager


def test_voyager_shows_example_programs_apart_from_the_library():
    model = ScriptedModel(lambda messages: "Code:\n```skill\nskill walk\nRIGHT\n```")
    player = Voyager(model, curriculum=False, examples=EXAMPLES)
    system = first_call(player, model)[0]["content"]
    assert "(none yet)" in system and "example programs for a different task" in system
    assert system.index("(none yet)") < system.index("skill example_program\nA") < system.index("At each round")
    assert not player.library.skills and player.mechanisms()["few_shot"] is True


def test_voyager_zero_shot_prompt_is_unchanged():
    model = ScriptedModel(lambda messages: "Code:\n```skill\nskill walk\nRIGHT\n```")
    player = Voyager(model, curriculum=False)
    system = first_call(player, model)[0]["content"]
    assert "skill library.\n\n(none yet)\n\nAt each round" in system and "example" not in system
    assert player.mechanisms()["few_shot"] is False


# Crystal harness


def test_harness_shows_example_replies_in_the_planner_prompt():
    reply = '{"why": "x", "tool": "press", "args": {"key": "RIGHT"}}'
    model = ScriptedModel(lambda messages: reply)
    player = CrystalHarness(model, examples=EXAMPLES)
    system = first_call(player, model, GridWorld())[0]["content"]
    assert 'Goal: Light the lamp.\nState: {"x": 3}\nReply: {"why": "walk", "tool": "press", "args": {"key": "RIGHT"}}' in system
    assert player.mechanisms()["few_shot"] is True


def test_harness_zero_shot_prompt_is_unchanged():
    model = ScriptedModel(lambda messages: '{"why": "x", "tool": "press", "args": {"key": "RIGHT"}}')
    player = CrystalHarness(model)
    assert "example" not in first_call(player, model, GridWorld())[0]["content"]
    assert player.mechanisms()["few_shot"] is False


# The file format and the config wiring


def test_malformed_examples_are_an_error():
    with pytest.raises(ValueError, match="source"):
        Examples.from_dict({"trajectories": []})
    with pytest.raises(ValueError, match="malformed"):
        Examples.from_dict({"source": "x", "trajectories": [{"goal": "g"}]})


def test_gridworld_example_file_parses_for_every_agent():
    path = "experiments/examples/gridworld.json"
    model = ScriptedModel(lambda messages: "")
    for name in ("direct", "react", "reflexion", "voyager", "crystal_harness", "continual"):
        player = make_player({"name": name, "examples": path}, model)
        assert player.mechanisms()["few_shot"] is True, name
    # the example task is not the one the gridworld configs evaluate
    goals = set()
    for p in ROOT.glob("experiments/gridworld_7b_*.json"):
        goal = json.loads(p.read_text())["goal"]
        goals |= {level["text"] for level in goal["chain"]} if "chain" in goal else {goal["text"]}
    assert Examples.from_dict(json.loads((ROOT / path).read_text())).trajectories[0].goal not in goals


def test_examples_file_is_hashed_with_the_run():
    spec = {"environment": {}, "agent": {"examples": "experiments/examples/gridworld.json"}}
    assert file_hashes(spec)["experiments/examples/gridworld.json"]


def test_experiment_configs_without_examples_are_zero_shot():
    model = ScriptedModel(lambda messages: "")
    for path in ROOT.glob("experiments/*.json"):
        agent = json.loads(path.read_text())["agent"]
        if agent.get("backbone") and not Path(agent["backbone"]).is_file():
            continue  # a generated backbone; tests/test_brief.py covers its config
        assert make_player(agent, model).mechanisms()["few_shot"] is ("examples" in agent), path.name


def test_retry_illegal_action_is_a_config_switch():
    model = ScriptedModel(lambda messages: "")
    for name in ("react", "reflexion"):
        assert make_player({"name": name}, model).mechanisms()["retry_illegal_action"] is False
        assert make_player({"name": name, "retry_illegal_action": True}, model).mechanisms()["retry_illegal_action"] is True
