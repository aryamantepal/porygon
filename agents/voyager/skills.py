"""Skill programs: what a Voyager skill is in this framework, and the library that stores them.

Voyager's skills are JavaScript functions over the Mineflayer API. Here a skill
is a small program over the environment's legal actions, so it runs on any
game. One press is one environment step, which keeps every expansion explicit,
counted, and logged by the runner.

The language, one statement per line, blocks indented under a header:

    skill go_downstairs            # first line names the skill
    UP x4                          # press UP four times (x1 if omitted)
    until x == 10 (max 12):        # repeat the block until the condition holds
      RIGHT
    repeat 3:                      # repeat the block a fixed number of times
      A
    call walk_through_door         # run a skill from the library

Conditions read fields of the structured snapshot (dotted paths, e.g.
`party.0.hp`) and compare them to JSON values (`map == [4, 0]`). In visual
mode there is no structured snapshot, so a condition is an execution error.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from environments.protocol import Action, Observation, Press

MAX_PRESSES = 200  # per program run, so a bad loop cannot eat the whole budget
MAX_CALL_DEPTH = 5


class SkillError(Exception):
    """A parse or execution error, fed back to the model as Voyager feeds back JS errors."""


# Program structure


@dataclass(frozen=True)
class Tap:
    key: str
    times: int


@dataclass(frozen=True)
class Repeat:
    times: int
    body: tuple["Statement", ...]


@dataclass(frozen=True)
class Until:
    field: str
    op: str
    value: Any
    max_iterations: int
    body: tuple["Statement", ...]


@dataclass(frozen=True)
class Call:
    name: str


Statement = Union[Tap, Repeat, Until, Call]


@dataclass(frozen=True)
class Program:
    name: str
    body: tuple[Statement, ...]
    source: str


# Parsing

_NAME = re.compile(r"^skill\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
_TAP = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+x\s*(\d+))?$")
_REPEAT = re.compile(r"^repeat\s+(\d+)\s*:$")
_UNTIL = re.compile(r"^until\s+([A-Za-z0-9_.]+)\s*(==|!=|<=|>=|<|>)\s*(.+?)\s*\(max\s+(\d+)\)\s*:$")
_CALL = re.compile(r"^call\s+([A-Za-z_][A-Za-z0-9_]*)$")


def parse(source: str, legal_keys: Sequence[str]) -> Program:
    lines = []
    for number, raw in enumerate(source.splitlines(), start=1):
        text = raw.split("#", 1)[0].rstrip()
        if text.strip():
            lines.append((number, len(text) - len(text.lstrip()), text.strip()))
    if not lines:
        raise SkillError("the program is empty")
    number, _, first = lines[0]
    match = _NAME.match(first)
    if not match:
        raise SkillError(f"line {number}: a program must start with `skill <name>`")
    body, rest = _parse_block(lines[1:], indent=None, legal_keys=set(legal_keys))
    if rest:
        raise SkillError(f"line {rest[0][0]}: unexpected indentation")
    if not body:
        raise SkillError("the program has no statements")
    return Program(match.group(1), body, source.strip())


def _parse_block(lines, indent, legal_keys):
    statements = []
    while lines:
        number, depth, text = lines[0]
        if indent is None:
            indent = depth
        if depth < indent:
            break
        if depth > indent:
            raise SkillError(f"line {number}: unexpected indentation")
        lines = lines[1:]
        if match := _REPEAT.match(text):
            body, lines = _block_after(number, lines, depth, legal_keys)
            statements.append(Repeat(int(match.group(1)), body))
        elif match := _UNTIL.match(text):
            try:
                value = json.loads(match.group(3))
            except json.JSONDecodeError:
                raise SkillError(f"line {number}: `{match.group(3)}` is not a JSON value") from None
            body, lines = _block_after(number, lines, depth, legal_keys)
            statements.append(Until(match.group(1), match.group(2), value, int(match.group(4)), body))
        elif match := _CALL.match(text):
            statements.append(Call(match.group(1)))
        elif match := _TAP.match(text):
            key = match.group(1).upper()
            if key not in legal_keys:
                raise SkillError(f"line {number}: `{match.group(1)}` is not a legal action ({', '.join(sorted(legal_keys))})")
            statements.append(Tap(key, int(match.group(2) or 1)))
        else:
            raise SkillError(f"line {number}: cannot parse `{text}`")
    return tuple(statements), lines


def _block_after(number, lines, depth, legal_keys):
    if not lines or lines[0][1] <= depth:
        raise SkillError(f"line {number}: a block header needs an indented body")
    return _parse_block(lines, indent=None, legal_keys=legal_keys)


# Execution


@dataclass
class Run:
    """Counts what one program run did, for logging and feedback."""

    presses: int = 0
    notes: list[str] = field(default_factory=list)


Library = Mapping[str, Program]


def execute(
    program: Program, library: Library | None, observation: Observation, run: Run
) -> Generator[Action, Observation, None]:
    """Yield one press at a time; the caller sends back the observation after each."""
    box = [observation]
    yield from _run_block(program.body, library, box, run, depth=0)


def _run_block(body, library, box, run, depth):
    for statement in body:
        if isinstance(statement, Tap):
            for _ in range(statement.times):
                if run.presses >= MAX_PRESSES:
                    raise SkillError(f"stopped after {MAX_PRESSES} presses")
                run.presses += 1
                box[0] = yield Press(statement.key)
        elif isinstance(statement, Repeat):
            for _ in range(statement.times):
                yield from _run_block(statement.body, library, box, run, depth)
        elif isinstance(statement, Until):
            for _ in range(statement.max_iterations):
                if _holds(statement, box[0]):
                    break
                yield from _run_block(statement.body, library, box, run, depth)
            else:
                if not _holds(statement, box[0]):
                    actual = _lookup(box[0], statement.field)
                    raise SkillError(
                        f"`until {statement.field} {statement.op} {json.dumps(statement.value)}` still false after "
                        f"{statement.max_iterations} iterations ({statement.field} is {json.dumps(actual)})"
                    )
        elif isinstance(statement, Call):
            if library is None:
                raise SkillError("the skill library is off, so `call` is unavailable")
            if statement.name not in library:
                raise SkillError(f"no skill named `{statement.name}` in the library")
            if depth >= MAX_CALL_DEPTH:
                raise SkillError(f"calls nested deeper than {MAX_CALL_DEPTH}")
            yield from _run_block(library[statement.name].body, library, box, run, depth + 1)


def _lookup(observation: Observation, path: str) -> Any:
    if observation.state is None:
        raise SkillError("conditions need the structured snapshot, which this observation mode does not provide")
    value: Any = observation.state
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise SkillError(f"the snapshot has no field `{path}`")
    return value


def _holds(statement: Until, observation: Observation) -> bool:
    actual = _lookup(observation, statement.field)
    try:
        return {
            "==": lambda: actual == statement.value,
            "!=": lambda: actual != statement.value,
            "<": lambda: actual < statement.value,
            "<=": lambda: actual <= statement.value,
            ">": lambda: actual > statement.value,
            ">=": lambda: actual >= statement.value,
        }[statement.op]()
    except TypeError:
        raise SkillError(f"cannot compare {statement.field} ({json.dumps(actual)}) {statement.op} {json.dumps(statement.value)}") from None


# Library


@dataclass(frozen=True)
class Skill:
    program: Program
    description: str
    task: str
    version: int = 1


class SkillLibrary:
    """Stores verified skills and retrieves the ones closest to a query.

    Voyager embeds skill descriptions and retrieves by vector similarity. This
    uses bag-of-words cosine over name, description, and task, so it needs no
    embedding model; the interface is the same.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.skills: dict[str, Skill] = {}
        if path is not None and path.exists():
            self._load(path)

    def programs(self) -> dict[str, Program]:
        return {name: skill.program for name, skill in self.skills.items()}

    def add(self, program: Program, description: str, task: str) -> Skill:
        previous = self.skills.get(program.name)
        skill = Skill(program, description, task, version=previous.version + 1 if previous else 1)
        self.skills[program.name] = skill
        if self.path is not None:
            self._save(self.path)
        return skill

    def retrieve(self, query: str, k: int = 5) -> list[Skill]:
        words = _words(query)
        scored = [(_cosine(words, _words(f"{s.program.name} {s.description} {s.task}")), name) for name, s in self.skills.items()]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self.skills[name] for _, name in scored[:k]]

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: {"source": s.program.source, "description": s.description, "task": s.task, "version": s.version}
            for name, s in self.skills.items()
        }
        path.write_text(json.dumps(data, indent=2) + "\n")

    def _load(self, path: Path) -> None:
        for name, data in json.loads(path.read_text()).items():
            program = Program(name, (), data["source"])  # body is re-parsed against legal keys on first use
            self.skills[name] = Skill(program, data["description"], data["task"], data["version"])

    def bind(self, legal_keys: Sequence[str]) -> None:
        """Parse loaded skills against this environment's legal actions."""
        for name, skill in list(self.skills.items()):
            if not skill.program.body:
                try:
                    program = parse(skill.program.source, legal_keys)
                except SkillError:
                    del self.skills[name]
                    continue
                self.skills[name] = Skill(program, skill.description, skill.task, skill.version)


def _words(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[w] * b[w] for w in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0
