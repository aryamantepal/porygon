"""The same contract against real mGBA and a real FireRed ROM.

Skipped unless PORYGON_ROM and PORYGON_STATE point at local files. Neither
file is ever committed. PORYGON_STATE is a state written by `FireRedEnv.save`
with the player standing in the overworld.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from environments.protocol import ObservationMode, Press

ROM = os.environ.get("PORYGON_ROM")
STATE = os.environ.get("PORYGON_STATE")
pytestmark = pytest.mark.skipif(not (ROM and STATE), reason="needs PORYGON_ROM and PORYGON_STATE")


@pytest.fixture
def env():
    from environments.pokemon_firered import FireRedConfig, FireRedEnv

    return FireRedEnv(FireRedConfig(Path(ROM), Path(STATE), ObservationMode.BOTH))


def test_reset_is_repeatable(env):
    first = env.reset()
    env.step(Press("A"))
    assert env.reset() == first


def test_observe_does_not_advance(env):
    env.reset()
    assert env.observe() == env.observe()


def test_symbols_read_plausible_values(env):
    state = env.reset().state
    assert 0x02000000 <= env._save_block1() < 0x02040000
    assert 0 <= len(state["party"]) <= 6
    for mon in state["party"]:
        assert 1 <= mon["level"] <= 100
        assert 0 <= mon["hp"] <= mon["max_hp"]


def test_step_changes_screen_or_state(env):
    before = env.reset()
    changed = False
    for key in ("UP", "DOWN", "LEFT", "RIGHT"):
        after = env.step(Press(key)).observation
        changed = changed or after.frame != before.frame or after.state != before.state
    assert changed


def test_progress_survives_save_and_reload(env, tmp_path):
    env.reset()
    env.step(Press("DOWN"))
    saved = env.get_progress()
    checkpoint = tmp_path / "after.state"
    env.save(checkpoint)
    env.reset()
    env.load(checkpoint)
    assert env.get_progress() == saved


def test_role_snapshot_at_the_bedroom_start(env):
    state = env.reset().state
    assert state["map"] == "4.1" and state["position"] == [6, 6] and state["facing"] == "UP"
    assert len(state["grid"]) == 9 and {len(row) for row in state["grid"]} == {13}
    assert state["grid"][5][6] == "#"  # the console above the start tile blocks the way up
    assert {"id": "door_0", "role": "door", "position": [10, 2], "leads_to": "4.0"} in state["places"]


def test_harness_navigator_takes_the_stairs(env, tmp_path):
    import json

    from agents.harness import CrystalHarness
    from agents.models import ScriptedModel
    from evaluation.goals import Goal
    from evaluation.runner import Budgets, run

    plan = json.dumps({"why": "stairs", "tool": "go_to", "args": {"place": "door_0"}, "intention": "go downstairs"})
    result = run(run_dir=tmp_path / "run", env=env, player=CrystalHarness(ScriptedModel(lambda m: plan)), goal=Goal("down", "Go downstairs.", {"map": [4, 0]}), budgets=Budgets(max_steps=40))[0]
    assert result.success and result.model_calls == 1


def test_context_follows_dialogue_and_locked_input(env):
    assert env.reset().state["context"] == "overworld"
    state = env.step(Press("A")).observation.state  # the console above the start tile
    assert state["context"] == "dialogue"
    assert state["dialogue"]["text"] == "RED played with the NES. …Okay! It's time to go!"
    for _ in range(12):
        if env.step(Press("A")).observation.state["context"] == "overworld":
            break
    assert env.observe().state["context"] == "overworld"
    env.reset()
    env.step(Press("START"))
    assert env.observe().state["context"] == "busy"


BATTLE_STATE = os.environ.get("PORYGON_BATTLE_STATE")


@pytest.mark.skipif(not BATTLE_STATE, reason="needs PORYGON_BATTLE_STATE (a state inside a battle)")
def test_context_reports_battle_until_it_ends(env):
    env.load(Path(BATTLE_STATE))
    assert env.observe().state["context"] == "battle"
    for _ in range(400):
        if env.step(Press("A")).observation.state["context"] != "battle":
            break
    assert env.observe().state["context"] != "battle"


@pytest.mark.skipif(not BATTLE_STATE, reason="needs PORYGON_BATTLE_STATE (a state inside a battle)")
def test_battle_block_describes_the_rival_fight(env):
    env.load(Path(BATTLE_STATE))
    battle = env.observe().state["battle"]
    assert battle["active"] == {"hp": 20, "max_hp": 20, "level": 5}
    assert battle["moves"] == [
        {"id": "tackle", "kind": "attack", "power": 35, "uses_left": 35},
        {"id": "growl", "kind": "status", "power": 0, "uses_left": 40},
    ]
    assert battle["opponent"] == {"hp": 18, "max_hp": 18, "level": 5, "catchable": False}


@pytest.mark.skipif(not BATTLE_STATE, reason="needs PORYGON_BATTLE_STATE (a state inside a battle)")
def test_harness_fights_the_rival_with_adapter_keys(env, tmp_path):
    from agents.harness import CrystalHarness
    from agents.models import ScriptedModel
    from environments.pokemon_firered import FireRedConfig, FireRedEnv
    from evaluation.goals import Goal
    from evaluation.runner import Budgets, run

    in_battle = FireRedEnv(FireRedConfig(Path(ROM), Path(BATTLE_STATE), ObservationMode.BOTH))
    model = ScriptedModel(lambda messages: '{"why": "damage", "option": "tackle"}')
    result = run(run_dir=tmp_path / "run", env=in_battle, player=CrystalHarness(model), goal=Goal("win", "Win the battle.", {"badge_count": 99}), budgets=Budgets(max_steps=400))[0]
    import json

    records = [json.loads(line) for line in (tmp_path / "run/trials/000/steps.jsonl").read_text().splitlines()]
    choices = [e["choice"] for r in records if r["kind"] == "decision" for e in r["decision"]["trace"]["events"] if e["type"] == "battle" and e.get("choice")]
    contexts = [r["observation"]["state"]["context"] for r in records if "observation" in r]
    assert choices and set(choices) == {"tackle"}
    assert "battle" in contexts and contexts[-1] != "battle"  # the fight ended
