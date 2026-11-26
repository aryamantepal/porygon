"""Contract tests for the FireRed adapter, run against a fake emulator."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from environments.protocol import Environment, ObservationMode, Press
from environments.pokemon_firered import FireRedConfig, FireRedEnv
from environments.recording import RecordingEnvironment
from tests.fake_gba import FakeGBA

PACKAGE = Path(__file__).resolve().parents[1] / "environments"


def make_env(tmp_path: Path, mode: ObservationMode = ObservationMode.BOTH, fake: FakeGBA | None = None):
    fake = fake or FakeGBA(Path("unused.gba"))
    state = tmp_path / "saves" / "start.state"
    config = FireRedConfig(rom_path=Path("unused.gba"), initial_state=state, mode=mode)
    env = FireRedEnv(config, make_emulator=lambda _: fake)
    env.save(state)
    return env, fake


def test_implements_protocol(tmp_path):
    env, _ = make_env(tmp_path)
    assert isinstance(env, Environment)


def test_reset_twice_gives_the_same_world(tmp_path):
    env, _ = make_env(tmp_path)
    first = env.reset()
    progress = env.get_progress()
    env.step(Press("RIGHT"))
    second = env.reset()
    assert second == first
    assert env.get_progress() == progress


def test_observe_does_not_advance(tmp_path):
    env, _ = make_env(tmp_path)
    env.reset()
    assert env.observe() == env.observe()


def test_step_reports_frames_and_changes_state(tmp_path):
    env, _ = make_env(tmp_path)
    before = env.reset()
    result = env.step(Press("RIGHT"))
    assert result.frames_advanced == env.config.hold_frames + env.config.release_frames
    assert result.observation.frame_count == before.frame_count + result.frames_advanced
    assert result.observation.state["x"] == before.state["x"] + 1
    assert result.observation.frame != before.frame


def test_hold_frames_override(tmp_path):
    env, _ = make_env(tmp_path)
    env.reset()
    result = env.step(Press("A", hold_frames=3))
    assert result.frames_advanced == 3 + env.config.release_frames


def test_visual_observation_has_no_memory_fields(tmp_path):
    env, _ = make_env(tmp_path, ObservationMode.VISUAL)
    observation = env.reset()
    assert observation.state is None
    assert observation.frame is not None


def test_structured_observation_has_no_frame(tmp_path):
    env, _ = make_env(tmp_path, ObservationMode.STRUCTURED)
    observation = env.reset()
    assert observation.frame is None
    assert observation.state["party"] == [{"level": 5, "hp": 19, "max_hp": 20}]


def test_role_snapshot_fields(tmp_path):
    env, _ = make_env(tmp_path, ObservationMode.STRUCTURED)
    state = env.reset().state
    assert state["context"] == "overworld"
    assert state["map"] == "3.0" and state["position"] == [5, 7] and state["facing"] == "UP"
    assert state["grid"] == ["#....", ".....", "....."]
    assert state["places"] == [{"id": "door_0", "role": "door", "position": [1, 0], "leads_to": "3.1"}]
    env.step(Press("RIGHT"))
    assert env.observe().state["position"] == [6, 7]


def test_progress_survives_save_and_reload(tmp_path):
    fake = FakeGBA(Path("unused.gba"))
    fake.set_badge(0)
    env, _ = make_env(tmp_path, fake=fake)
    env.reset()
    env.step(Press("RIGHT"))
    saved = env.get_progress()
    checkpoint = tmp_path / "saves" / "after.state"
    env.save(checkpoint)
    env.reset()
    env.load(checkpoint)
    assert env.get_progress() == saved
    assert saved["badge_count"] == 1
    assert saved["map"] == [3, 0]


def test_rejects_state_from_another_revision(tmp_path):
    env, _ = make_env(tmp_path)
    meta = env.config.initial_state.with_name(env.config.initial_state.name + ".json")
    meta.write_text('{"game_code": "BPRE", "revision": 1}')
    with pytest.raises(ValueError, match="saved under"):
        env.reset()


def test_unknown_rom_revision_is_refused():
    with pytest.raises(LookupError):
        FireRedEnv(
            FireRedConfig(Path("unused.gba"), Path("unused.state"), ObservationMode.BOTH),
            make_emulator=lambda _: FakeGBA(Path("unused.gba"), revision=9),
        )


def test_rejects_unknown_action(tmp_path):
    env, _ = make_env(tmp_path)
    env.reset()
    with pytest.raises(ValueError):
        env.step(Press("JUMP"))


def test_legal_actions_are_the_ten_buttons_and_wait(tmp_path):
    env, fake = make_env(tmp_path)
    assert {action.key for action in env.legal_actions()} == {
        "A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L", "WAIT",
    }
    before = env.reset()
    result = env.step(Press("WAIT"))
    assert result.frames_advanced == env.config.hold_frames + env.config.release_frames
    assert result.observation.state["x"] == before.state["x"]


def test_recording_writes_one_line_per_transition(tmp_path):
    env, _ = make_env(tmp_path)
    run_dir = tmp_path / "runs" / "demo"
    recorder = RecordingEnvironment(env, run_dir)
    recorder.reset()
    recorder.step(Press("RIGHT"))
    recorder.close()
    lines = (run_dir / "transitions.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert (run_dir / "frames" / "0000" / "000001.png").read_bytes().startswith(b"\x89PNG")


def test_recording_keeps_frames_from_every_episode(tmp_path):
    env, _ = make_env(tmp_path)
    run_dir = tmp_path / "runs" / "demo"
    first = RecordingEnvironment(env, run_dir)
    first.reset()
    first.step(Press("RIGHT"))
    first.reset()
    first.close()
    reopened = RecordingEnvironment(env, run_dir)
    reopened.reset()
    reopened.close()
    records = [json.loads(line) for line in (run_dir / "transitions.jsonl").read_text().splitlines()]
    assert [record["episode"] for record in records] == [0, 0, 1, 2]
    assert len({record["frame"] for record in records}) == 4
    for record in records:
        assert (run_dir / record["frame"]).exists()


def test_environments_import_no_agents_models_or_evaluation():
    forbidden = {"agents", "evaluation", "ollama", "openai", "anthropic"}
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert name.split(".")[0] not in forbidden, f"{path} imports {name}"


def test_edge_tile_picks_the_walkable_tile_nearest_the_middle():
    from environments.pokemon_firered.adapter import _edge_tile

    grid = ["##..####", "#......#", "#......#", "########"]
    assert _edge_tile(grid, "north") == (3, 0)
    assert _edge_tile(grid, "south") is None  # no walkable tile: the exit is left out
    assert _edge_tile(["#.", "..", "#."], "east") == (1, 1)
