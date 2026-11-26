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
