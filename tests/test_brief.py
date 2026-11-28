from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agents.harness.memory import KnowledgeGraph, validate_backbone
from agents.models import ScriptedModel
from evaluation import brief
from environments.gridworld import GridWorld
from evaluation.runner import Budgets, run
from evaluation.brief import QUESTIONS, BriefError, Question, generate, topological_order
from experiments.run import file_hashes, make_player
from tests.test_harness import MILESTONE, planner_model, to_milestone

ROOT = Path(__file__).resolve().parents[1]
BRIEF = ROOT / "experiments/briefs/gridworld.txt"
GENERATED_CONFIG = ROOT / "experiments/gridworld_7b_crystal_harness_generated.json"

GOOD = {
    "nodes": {
        "goal": {"role": "goal", "about": "reach a milestone"},
        "house": {"role": "place", "about": "where you start"},
        "town": {"role": "place", "about": "outside, with grass and a milestone"},
        "person": {"role": "person", "about": "lives in the house and knows the area"},
        "milestone": {"role": "milestone", "about": "marks progress"},
    },
    "edges": [
        ["house", "leads_to", "town"],
        ["milestone", "located_in", "town"],
        ["person", "lives_in", "house"],
        ["goal", "requires", "milestone"],
    ],
    "recipes": {"controls": "A talks.", "when_stuck": "Talk to the person.", "catching": "Weaken, then throw a ball."},
}


def question_id(messages) -> str:
    return re.search(r"Question id: (\w+)", messages[1].content).group(1)


def scripted(backbone_replies):
    """A model that answers each prose question with `answer-<id>` and the backbone question from `backbone_replies`, in order."""
    replies = iter(backbone_replies)

    def reply(messages):
        qid = question_id(messages)
        return next(replies) if qid == "backbone" else f"answer-{qid}"

    return ScriptedModel(reply, model_id="scripted-writer")


def asked(model) -> list[tuple[str, str]]:
    """(question id, user message) for every call the model got, in order."""
    return [(re.search(r"Question id: (\w+)", c.messages[1]["content"]).group(1), c.messages[1]["content"]) for c in model.calls]


# Question order and dependencies


def test_questions_are_asked_in_dependency_order_and_see_only_their_dependencies():
    model = scripted([json.dumps(GOOD)])
    generate(BRIEF, model)
    calls = asked(model)
    assert [qid for qid, _ in calls] == ["goal", "controls", "places", "roles", "order", "stuck", "howto", "backbone"]
    by_id = dict(calls)
    for q in QUESTIONS:
        seen = set(re.findall(r"answer-(\w+)", by_id[q.id]))
        assert seen == set(q.depends_on), q.id
    assert set(re.findall(r"answer-(\w+)", by_id["backbone"])) == {q.id for q in QUESTIONS}


def test_questions_declared_out_of_order_are_sorted_by_their_dependencies():
    declared = [Question("c", "C?", ("a", "b")), Question("b", "B?", ("a",)), Question("a", "A?")]
    assert [q.id for q in topological_order(declared)] == ["a", "b", "c"]
    model = scripted([json.dumps(GOOD)])
    result = generate(BRIEF, model, questions=declared)
    assert [q["id"] for q in result["source"]["questions"]] == ["a", "b", "c", "backbone"]


def test_the_default_dag_is_valid_and_every_question_has_a_dependency_or_is_a_root():
    ids = [q.id for q in topological_order(QUESTIONS)]
    assert set(ids) == {q.id for q in QUESTIONS}
    for q in QUESTIONS:
        assert all(ids.index(dep) < ids.index(q.id) for dep in q.depends_on)


def test_unknown_dependency_cycle_and_duplicate_are_rejected():
    with pytest.raises(ValueError, match="unknown question 'z'"):
        topological_order([Question("a", "A?", ("z",))])
    with pytest.raises(ValueError, match="cycle"):
        topological_order([Question("a", "A?", ("b",)), Question("b", "B?", ("a",))])
    with pytest.raises(ValueError, match="unique"):
        topological_order([Question("a", "A?"), Question("a", "A again?")])


# Output validation and the one retry


def test_validate_backbone_names_every_problem():
    assert validate_backbone(GOOD) == []
    assert validate_backbone(json.loads((ROOT / "experiments/knowledge/gridworld.json").read_text())) == []
    bad = copy.deepcopy(GOOD)
    bad["edges"] += [["town", "leads_to", "nowhere"], ["town", "Leads To", "house"], ["town", "requires"], ["house", "leads_to", "town"], ["town", "requires", "town"]]
    bad["nodes"]["Bad Id"] = {"role": "place", "about": "x"}
    bad["nodes"]["thin"] = {"role": "place"}
    bad["recipes"]["empty"] = " "
    bad["extra"] = 1
    problems = "\n".join(validate_backbone(bad))
    for fragment in ("unknown node 'nowhere'", "relation must be snake_case", "three strings", "appears twice", "itself", "'Bad Id'", "node 'thin'", "recipe 'empty'", "unknown key 'extra'"):
        assert fragment in problems, fragment
    assert validate_backbone([]) and validate_backbone({"nodes": {}, "edges": [], "recipes": {}})


def test_an_invalid_backbone_is_retried_once_with_the_errors():
    broken = copy.deepcopy(GOOD)
    broken["edges"].append(["town", "leads_to", "nowhere"])
    model = scripted([json.dumps(broken), "```json\n" + json.dumps(GOOD) + "\n```"])
    result = generate(BRIEF, model)
    assert result["edges"] == GOOD["edges"]
    retry = model.calls[-1].messages
    assert [m["role"] for m in retry] == ["system", "user", "assistant", "user"]
    assert "unknown node 'nowhere'" in retry[-1]["content"] and "nowhere" in retry[2]["content"]
    attempts = result["source"]["questions"][-1]["attempts"]
    assert [bool(a["errors"]) for a in attempts] == [True, False]


def test_a_goalless_or_unparseable_backbone_fails_after_one_retry_with_a_clear_message():
    no_goal = copy.deepcopy(GOOD)
    del no_goal["nodes"]["goal"], no_goal["edges"][3]
    model = scripted(["not json at all", json.dumps(no_goal)])
    with pytest.raises(BriefError) as error:
        generate(BRIEF, model)
    assert 'no node has role "goal"' in str(error.value) and "2 tries" in str(error.value)
    assert len(model.calls) == len(QUESTIONS) + 2  # no third attempt


def test_a_failed_model_call_is_an_error_not_an_empty_answer():
    with pytest.raises(BriefError, match="empty reply"):
        generate(BRIEF, ScriptedModel(lambda messages: ""))


# The source block


def test_source_block_records_the_brief_hash_model_and_raw_answers():
    model = scripted([json.dumps(GOOD)])
    result = generate(BRIEF, model, model_spec={"provider": "ollama", "id": "scripted-writer"})
    source = result["source"]
    assert source["brief"] == str(BRIEF) and source["brief_sha256"] == hashlib.sha256(BRIEF.read_bytes()).hexdigest()
    assert source["model"] == {"id": "scripted-writer", "spec": {"provider": "ollama", "id": "scripted-writer"}}
    answers = {q["id"]: q["answer"] for q in source["questions"]}
    assert answers["goal"] == "answer-goal" and json.loads(answers["backbone"]) == GOOD
    assert {q["id"]: q["depends_on"] for q in source["questions"]}["roles"] == ["goal", "places"]
    assert source["questions"][0]["question"] == QUESTIONS[0].text
    assert set(result) == {"nodes", "edges", "recipes", "source"}


def test_the_cli_writes_the_file_and_will_not_overwrite_it(tmp_path, monkeypatch, capsys):
    import experiments.run

    monkeypatch.setattr(experiments.run, "make_model", lambda spec: scripted([json.dumps(GOOD)] * 2))
    out = tmp_path / "kg.json"
    monkeypatch.setattr(sys, "argv", ["brief", str(BRIEF), "--model", "some-model", "--out", str(out)])
    brief.main()
    written = json.loads(out.read_text())
    assert written["source"]["model"]["spec"] == {"id": "some-model"} and validate_backbone(written) == []
    with pytest.raises(SystemExit, match="exists"):
        brief.main()
    monkeypatch.setattr(sys, "argv", ["brief", str(BRIEF), "--model", "some-model", "--out", str(out), "--force"])
    brief.main()


def test_model_spec_reads_a_json_block_a_config_or_an_id(tmp_path):
    spec = json.loads(GENERATED_CONFIG.read_text())["model"]
    assert brief.model_spec_from(str(GENERATED_CONFIG)) == spec
    block = tmp_path / "model.json"
    block.write_text(json.dumps(spec))
    assert brief.model_spec_from(str(block)) == spec and brief.model_spec_from("qwen2.5:7b") == {"id": "qwen2.5:7b"}


# The backbone is used like a hand-written one


def generated_config(tmp_path):
    path = tmp_path / "gridworld_generated.json"
    path.write_text(json.dumps(generate(BRIEF, scripted([json.dumps(GOOD)])), indent=2))
    spec = json.loads(GENERATED_CONFIG.read_text())
    spec["agent"]["backbone"] = str(path)
    return spec, path


def test_a_generated_backbone_loads_into_the_graph_and_the_harness_reads_it(tmp_path):
    spec, path = generated_config(tmp_path)
    graph = KnowledgeGraph(json.loads(path.read_text()))
    assert graph.edges[0] == ("house", "leads_to", "town") and graph.data["nodes"]["town"]["notes"] == []
    assert "catching" in graph.relevant("catching the creature")["recipes"]
    spec["agent"]["knowledge_path"] = str(tmp_path / "kg.json")
    model = planner_model(to_milestone)
    player = make_player(spec["agent"], model)
    run(run_dir=tmp_path / "run", env=GridWorld(), player=player, goal=MILESTONE, budgets=Budgets(max_steps=60))
    prompts = [call.messages[1]["content"] for call in model.calls if "What you know" in call.messages[1]["content"]]
    assert prompts and "outside, with grass and a milestone" in prompts[0]
    # notes persist on the generated nodes, and the backbone file itself is never written to by the run
    assert KnowledgeGraph(None, tmp_path / "kg.json").data["nodes"]["goal"]["role"] == "goal"
    assert json.loads(path.read_text())["nodes"]["goal"] == GOOD["nodes"]["goal"]


def test_the_generated_backbone_is_hashed_with_the_run(tmp_path):
    spec, path = generated_config(tmp_path)
    assert file_hashes(spec)[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert file_hashes(json.loads(GENERATED_CONFIG.read_text()))["experiments/knowledge/gridworld_generated.json"] is None


# The config and the missing-file message


def test_the_generated_config_differs_from_the_hand_written_one_only_in_the_backbone():
    hand = json.loads((ROOT / "experiments/gridworld_7b_crystal_harness.json").read_text())
    generated = json.loads(GENERATED_CONFIG.read_text())
    assert generated["agent"].pop("backbone") == "experiments/knowledge/gridworld_generated.json"
    assert hand["agent"].pop("backbone") == "experiments/knowledge/gridworld.json"
    assert generated == hand


def test_a_missing_backbone_names_the_command_that_makes_it():
    agent = json.loads(GENERATED_CONFIG.read_text())["agent"]
    assert not (ROOT / agent["backbone"]).exists(), "the generated backbone is not committed; no real model was run for it"
    with pytest.raises(FileNotFoundError) as error:
        make_player(agent, ScriptedModel(lambda messages: ""))
    message = str(error.value)
    assert "python -m evaluation.brief" in message and "--out experiments/knowledge/gridworld_generated.json" in message


def test_the_gridworld_brief_does_not_say_where_the_milestone_is():
    text = BRIEF.read_text().lower()
    assert "milestone" in text and "controls" in text
    for giveaway in ("southeast", "corner", "past the grass", "(7, 4)"):
        assert giveaway not in text


# Boundaries


def test_the_generator_imports_no_emulator_or_adapter():
    code = (
        "import sys, evaluation.brief, experiments.run\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('mgba', 'cffi') or (m.startswith('environments.') and m not in ('environments.protocol', 'environments.roles'))]\n"
        "print(bad)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]", out
    source = (ROOT / "evaluation/brief.py").read_text()
    imports = re.findall(r"^(?:from|import) ([\w.]+)", source, re.M)
    assert not [m for m in imports if m.startswith("environments") or "mgba" in m or "pokemon" in m]
