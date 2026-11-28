"""Generate a Crystal-harness knowledge backbone from a game brief, as a step of its own.

    python -m evaluation.brief experiments/briefs/<game>.txt --model <spec.json | model id> \\
        --out experiments/knowledge/<name>.json

This is not part of a run. It reads a plain-text brief (the goal and controls a
game would ship with), asks a model a fixed set of questions in dependency
order, and writes a backbone in the format `KnowledgeGraph` loads
(`agents/harness/memory.py`), with a `source` block that records the brief's
SHA-256, the model, and every question with its raw answer. A config then
points `"backbone"` at the file, and the run hashes it like any other.

The method is SPRING (Wu et al., NeurIPS 2023, arXiv:2305.15486): the model
reads the game's text and answers questions, and each question sees only the
answers of the questions it depends on. SPRING's code (github.com/holmeswww/SPRING,
MIT) is one Crafter notebook; nothing here is copied from it. The questions are
ours. Deviations from the paper are listed in the README.

This module names no game, imports no emulator or adapter, and sits in
`evaluation/` because it is a model-calling step whose output is hashed with a
run. It needs only the `ChatModel` interface; the CLI reaches
`experiments.run.make_model` lazily, the one place model clients are wired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.harness.memory import validate_backbone
from agents.models import ChatModel, Message

SYSTEM = (
    "You read a short game brief and answer questions about it, one at a time. "
    "Use only what the brief says. If the brief does not say, answer 'the brief does not say'. "
    "Do not use outside knowledge of any particular game, and do not invent places, people, items, or rules. "
    "Answer in plain text, in a few short lines."
)


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    depends_on: tuple[str, ...] = ()


# The DAG. A question is asked after the ones it depends on and sees only their answers.
QUESTIONS = (
    Question("goal", "What is the goal of the game? Answer in one or two sentences."),
    Question("controls", "What does each control (button or key) do? List each one with its effect, as the brief states it."),
    Question("places", "Which places or areas does the game have, and how are they connected to each other?", ("goal",)),
    Question(
        "roles",
        "Which kinds of people, items, creatures, and milestones does the game have, and in which place does each belong? "
        "Name only those the brief mentions.",
        ("goal", "places"),
    ),
    Question(
        "order",
        "What must happen before what? List each step that must come first and what it makes possible. "
        "Include only requirements the brief states or plainly implies.",
        ("goal", "places", "roles"),
    ),
    Question(
        "stuck",
        "When the player is stuck or does not know what to do next, what should it do? Use only hints the brief gives.",
        ("goal", "controls", "order"),
    ),
    Question(
        "howto",
        "Which short how-to guides does the brief describe (for example for talking to people or for fights)? Give each a short name and its steps.",
        ("controls", "roles", "order"),
    ),
)

BACKBONE_ID = "backbone"
BACKBONE_TEXT = """Using the brief and your earlier answers, write the game's knowledge graph as one JSON object and nothing else.
Shape:
{"nodes": {"<id>": {"role": "<kind of thing>", "about": "<one short sentence>"}},
 "edges": [["<node id>", "<relation>", "<node id>"]],
 "recipes": {"<name>": "<short how-to guide>"}}
Rules:
- Ids, relations, and recipe names are snake_case (a-z, 0-9, _). Every id in an edge must be a node you defined.
- One node for the goal, with id "goal" and role "goal". One node for each place, person, item, creature, or milestone the brief names; role says which kind.
- Relations say how nodes depend on each other, such as requires, leads_to, located_in, contains, lives_in, or unlocks. Use "requires" for what must come first.
- Recipes hold how-to guides: one named "controls" with what each control does, one named "when_stuck" with what to do when stuck (if the brief gives any hint), and one per how-to guide from your earlier answers.
- Use only what the brief says. Do not add nodes, edges, or recipes the brief does not support."""
RETRIES = 1


class BriefError(Exception):
    """The model could not produce a valid backbone."""


def topological_order(questions: Sequence[Question]) -> list[Question]:
    """Questions in an order where each follows everything it depends on; ties keep the declared order."""
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids) or BACKBONE_ID in ids:
        raise ValueError(f"question ids must be unique and not {BACKBONE_ID!r}: {ids}")
    for q in questions:
        for dep in q.depends_on:
            if dep not in ids:
                raise ValueError(f"question {q.id!r} depends on unknown question {dep!r}")
    ordered: list[Question] = []
    done: set[str] = set()
    while len(ordered) < len(questions):
        ready = next((q for q in questions if q.id not in done and set(q.depends_on) <= done), None)
        if ready is None:
            raise ValueError(f"the questions have a dependency cycle among {[q.id for q in questions if q.id not in done]}")
        ordered.append(ready)
        done.add(ready.id)
    return ordered


def request(brief: str, question: Question, answers: dict[str, str], texts: dict[str, str]) -> list[Message]:
    earlier = "\n\n".join(f"[{dep}] {texts[dep]}\n{answers[dep]}" for dep in question.depends_on) or "none"
    user = f"Game brief:\n{brief.strip()}\n\nEarlier answers:\n{earlier}\n\nQuestion id: {question.id}\nQuestion: {question.text}"
    return [Message("system", SYSTEM), Message("user", user)]


def ask(model: ChatModel, messages: list[Message]) -> str:
    call = model.chat(messages, think=False)
    if call.error or not call.output.strip():
        raise BriefError(f"model call failed: {call.error or 'empty reply'}")
    return call.output


def parse_backbone(reply: str) -> tuple[dict[str, Any] | None, list[str]]:
    """The JSON object in `reply` (a code fence or prose around it is tolerated), or the reasons it is not a valid backbone."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        return None, ["the reply holds no JSON object"]
    try:
        data = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as error:
        return None, [f"the reply is not valid JSON: {error}"]
    problems = validate_backbone(data)
    if isinstance(data, dict) and isinstance(data.get("nodes"), dict):
        if not any(isinstance(n, dict) and n.get("role") == "goal" for n in data["nodes"].values()):
            problems.append('no node has role "goal"')
    return (data if not problems else None), problems


def generate(brief_path: Path, model: ChatModel, *, model_spec: dict[str, Any] | None = None, questions: Sequence[Question] = QUESTIONS) -> dict[str, Any]:
    """Ask `questions` in dependency order, then for the backbone. One retry shows the model its errors; then BriefError."""
    raw = brief_path.read_bytes()
    brief = raw.decode()
    ordered = topological_order(questions)
    texts = {q.id: q.text for q in ordered}
    answers: dict[str, str] = {}
    log: list[dict[str, Any]] = []
    for q in ordered:
        answers[q.id] = ask(model, request(brief, q, answers, texts))
        log.append({"id": q.id, "question": q.text, "depends_on": list(q.depends_on), "answer": answers[q.id]})
    final = Question(BACKBONE_ID, BACKBONE_TEXT, tuple(q.id for q in ordered))
    messages = request(brief, final, answers, texts)
    attempts: list[dict[str, Any]] = []
    backbone = None
    for _ in range(RETRIES + 1):
        reply = ask(model, messages)
        backbone, problems = parse_backbone(reply)
        attempts.append({"answer": reply, "errors": problems})
        if backbone is not None:
            break
        listed = "\n".join(f"- {p}" for p in problems[:10])
        messages = [*messages, Message("assistant", reply), Message("user", f"That backbone has problems:\n{listed}\nReturn the corrected JSON object only.")]
    if backbone is None:
        raise BriefError(f"no valid backbone after {len(attempts)} tries; the last errors were:\n" + "\n".join(f"- {p}" for p in attempts[-1]["errors"]))
    log.append({"id": BACKBONE_ID, "question": BACKBONE_TEXT, "depends_on": list(final.depends_on), "answer": attempts[-1]["answer"], "attempts": attempts})
    model_record: dict[str, Any] = {"id": model.model_id}
    if model_spec is not None:
        model_record["spec"] = model_spec
    return {
        "nodes": {node_id: {"role": node["role"], "about": node["about"]} for node_id, node in backbone["nodes"].items()},
        "edges": backbone["edges"],
        "recipes": backbone["recipes"],
        "source": {
            "method": "SPRING (arXiv:2305.15486): questions in dependency order, then the backbone",
            "brief": str(brief_path),
            "brief_sha256": hashlib.sha256(raw).hexdigest(),
            "model": model_record,
            "questions": log,
        },
    }


def model_spec_from(arg: str) -> dict[str, Any]:
    """A JSON file (a model block, or a whole experiment config, whose "model" block is used), or a bare model id."""
    path = Path(arg)
    if path.suffix == ".json" and path.is_file():
        data = json.loads(path.read_text())
        return data["model"] if "model" in data else data
    return {"id": arg}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("brief", type=Path)
    parser.add_argument("--model", required=True, help="a model spec JSON file (or experiment config) or a model id")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} exists; it may be hashed into earlier runs. Use another --out, or --force to replace it.")
    from experiments.run import make_model

    spec = model_spec_from(args.model)
    try:
        backbone = generate(args.brief, make_model(spec), model_spec=spec)
    except BriefError as error:
        sys.exit(f"could not generate a backbone from {args.brief}: {error}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(backbone, indent=2) + "\n")
    print(f"wrote {args.out}: {len(backbone['nodes'])} nodes, {len(backbone['edges'])} edges, {len(backbone['recipes'])} recipes")


if __name__ == "__main__":
    main()
