"""Runner, model client, and LineWorld tests. No ROM, no model server."""

from __future__ import annotations

import ast
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agents.common import parse_action, render_observation
from agents.models import Message, OllamaChat, ScriptedModel
from agents.protocol import Decision, Review, TrialOutcome
from environments.lineworld import LineWorld, LineWorldConfig
from environments.protocol import Environment, ObservationMode, Press
from evaluation.goals import Goal
from evaluation.runner import Budgets, run

ROOT = Path(__file__).resolve().parents[1]
REACH_END = Goal("reach-end", "Walk to the far right.", {"x": 9})


class Always:
    """A player that presses one key, or returns no action when key is None."""

    name = "always"

    def __init__(self, key: str | None) -> None:
        self.key = key
        self.outcomes: list[TrialOutcome] = []

    def mechanisms(self):
        return {}

    def start_trial(self, goal: str) -> None:
        pass

    def decide(self, observation, legal_actions):
        action = Press(self.key) if self.key else None
        return Decision(action=action, trace=None)

    def end_trial(self, outcome):
        self.outcomes.append(outcome)
        return Review()


def test_lineworld_implements_protocol():
    env = LineWorld()
    assert isinstance(env, Environment)
    first = env.reset()
    assert env.observe() == first
    env.step(Press("RIGHT"))
    assert env.get_progress()["x"] == 1
    assert LineWorld(LineWorldConfig(mode=ObservationMode.VISUAL)).reset().state is None


def test_runner_stops_when_goal_met(tmp_path):
    player = Always("RIGHT")
    results = run(run_dir=tmp_path / "run", env=LineWorld(), player=player, goal=REACH_END, budgets=Budgets(max_steps=50))
    assert results[0].success and results[0].steps == 9 and results[0].reason == "goal_met"
    assert player.outcomes == [TrialOutcome(True, "goal_met", 9)]


def test_runner_stops_on_step_budget(tmp_path):
    results = run(run_dir=tmp_path / "run", env=LineWorld(), player=Always("LEFT"), goal=REACH_END, budgets=Budgets(max_steps=5))
    assert not results[0].success and results[0].reason == "max_steps" and results[0].steps == 5


def test_invalid_decisions_press_nothing_and_are_counted(tmp_path):
    results = run(run_dir=tmp_path / "run", env=LineWorld(), player=Always(None), goal=REACH_END, budgets=Budgets(max_steps=5, max_decisions=3))
    assert results[0].steps == 0 and results[0].invalid_decisions == 3 and results[0].reason == "max_decisions"


def test_run_log_replays_on_its_own(tmp_path):
    run_dir = tmp_path / "run"
    run(run_dir=run_dir, env=LineWorld(), player=Always("RIGHT"), goal=REACH_END, budgets=Budgets(max_steps=50), trials=2, config={"model": "none"})
    config = json.loads((run_dir / "config.json").read_text())
    assert config["agent"] == "always" and config["model"] == "none" and config["goal"]["success"] == {"x": 9}
    for trial in ("000", "001"):
        records = [json.loads(line) for line in (run_dir / "trials" / trial / "steps.jsonl").read_text().splitlines()]
        assert records[0]["kind"] == "reset"
        assert records[-2]["goal_met"] is True
        assert records[-1]["kind"] == "trial_end" and records[-1]["success"] is True
        for record in records:
            if "observation" in record:
                assert (run_dir / "trials" / trial / record["observation"]["frame"]).read_bytes().startswith(b"\x89PNG")
    assert json.loads((run_dir / "summary.json").read_text())["successes"] == 2


def test_goal_is_decided_by_progress_not_words():
    assert not REACH_END.is_met({"x": 3})
    assert REACH_END.is_met({"x": 9, "lamp": False})
    assert not Goal("lamp", "Light the lamp.", {"lamp": True}).is_met({"x": 9})


@pytest.mark.parametrize(
    ("text", "label", "expected"),
    [
        ("RIGHT", False, "RIGHT"),
        (" right. ", False, "RIGHT"),
        ("Go RIGHT", False, None),
        ("Thought: go on.\nAction: LEFT", True, "LEFT"),
        ("Action: A\nAction: **RIGHT**", True, "RIGHT"),
        ("Thought: hmm", True, None),
        ("Action: JUMP", True, None),
    ],
)
def test_parse_action(text, label, expected):
    legal = LineWorld().legal_actions()
    action = parse_action(text, legal, require_label=label)
    assert (action.key if action else None) == expected


def test_render_respects_mode():
    text, images = render_observation(LineWorld(LineWorldConfig(mode=ObservationMode.STRUCTURED)).reset())
    assert '"x": 0' in text and images == []
    text, images = render_observation(LineWorld(LineWorldConfig(mode=ObservationMode.VISUAL)).reset())
    assert "State" not in text and images[0].startswith(b"\x89PNG")


def test_ollama_request_shape_and_reply():
    seen = {"seeds": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["seeds"].append(seen["body"]["options"]["seed"])
            reply = json.dumps({"message": {"content": "RIGHT", "thinking": ""}, "prompt_eval_count": 12, "eval_count": 1}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=lambda: [server.handle_request() for _ in range(2)], daemon=True).start()
    model = OllamaChat("test-model", host=f"http://127.0.0.1:{server.server_port}", seed=7)
    model.chat([Message("user", "go")], think=False)
    call = model.chat([Message("user", "go", images=[b"png"])], think=False)
    server.server_close()
    # a fixed seed would make every trial of a memoryless player identical, even with temperature > 0
    assert seen["seeds"] == [7, 8]
    assert seen["path"] == "/api/chat"
    assert seen["body"]["think"] is False and seen["body"]["stream"] is False
    assert seen["body"]["options"]["num_ctx"] == 16384 and call.context_length == 16384
    assert seen["body"]["messages"][0]["images"] == ["cG5n"]
    assert call.output == "RIGHT" and call.thinking is None and call.prompt_tokens == 12
    assert call.messages[0]["images"] == 1


def test_ollama_failure_is_logged_not_raised():
    call = OllamaChat("m", host="http://127.0.0.1:9", timeout_seconds=1).chat([Message("user", "go")], think=False)
    assert call.error and call.output == ""


def test_scripted_model_records_calls():
    model = ScriptedModel(lambda messages: "A")
    model.chat([Message("user", "hi")], think=False)
    assert model.calls[0].output == "A"


def test_agents_and_evaluation_import_no_game():
    forbidden = ("environments.pokemon_firered", "mgba")
    for package in ("agents", "evaluation"):
        for path in (ROOT / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    assert not name.startswith(forbidden), f"{path} imports {name}"


def test_run_config_hashes_the_files_it_starts_from(tmp_path):
    import hashlib

    from experiments.run import file_hashes

    state = tmp_path / "start.state"
    state.write_bytes(b"saved")
    spec = {"environment": {"rom": str(tmp_path / "missing.gba"), "state": str(state)}, "agent": {"name": "direct"}}
    assert file_hashes(spec) == {str(tmp_path / "missing.gba"): None, str(state): hashlib.sha256(b"saved").hexdigest()}


def test_run_config_records_the_code_version():
    from experiments.run import code_version

    version = code_version()
    assert set(version) == {"commit", "dirty"}


def test_results_report_efficiency_over_successful_trials_only(tmp_path):
    from evaluation.results import summarize, table

    run(run_dir=tmp_path / "won", env=LineWorld(), player=Always("RIGHT"), goal=REACH_END, budgets=Budgets(max_steps=20), trials=2)
    run(run_dir=tmp_path / "lost", env=LineWorld(), player=Always("LEFT"), goal=REACH_END, budgets=Budgets(max_steps=5))
    won, lost = summarize(tmp_path / "won"), summarize(tmp_path / "lost")
    assert (won["successes"], won["trials"], won["presses_to_goal"]) == (2, 2, 9)
    assert won["frames_to_goal"] > 0
    assert (lost["successes"], lost["presses_to_goal"]) == (0, None)
    assert "| won | 2/2 | 9 |" in table([won, lost])


def test_summary_reports_cost_to_goal_over_successful_trials(tmp_path):
    run(run_dir=tmp_path / "won", env=LineWorld(), player=Always("RIGHT"), goal=REACH_END, budgets=Budgets(max_steps=20), trials=2)
    summary = json.loads((tmp_path / "won" / "summary.json").read_text())
    assert summary["to_goal"]["presses"] == 9 and summary["to_goal"]["frames"] > 0
    outcome = json.loads((tmp_path / "won" / "trials/000/outcome.json").read_text())
    assert outcome["frames"] == summary["to_goal"]["frames"]
    run(run_dir=tmp_path / "lost", env=LineWorld(), player=Always("LEFT"), goal=REACH_END, budgets=Budgets(max_steps=3))
    assert json.loads((tmp_path / "lost" / "summary.json").read_text())["to_goal"]["presses"] is None
