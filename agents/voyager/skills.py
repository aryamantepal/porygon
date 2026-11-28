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

With the tools interface (`parse(..., tools=True)`, a player's
`action_interface="tools"`) three statements are added. They expand into
button presses through the Crystal harness's navigator, one press per
decision, and every press counts against MAX_PRESSES:

    go_to door_0                   # walk to a place listed in the snapshot
    go_to [4, 2]                   # walk to a tile
    talk_to person                 # walk next to someone, face them, and talk

A tool that fails (no such place, no path, no conversation) is a SkillError
with the tool's own message. A tool that is interrupted because the game
changed context (a battle started) is a `SkillInterrupted`. In buttons mode
these statements do not parse.

With parameters (`parse(..., parameters=True)`, a player's `skill_parameters`
switch; SkillWeaver, Zheng et al. 2025, arXiv:2504.07079) a skill is a
parameterized API, so one skill can serve any door or target:

    skill walk_through door        # parameter names follow the skill name
    go_to $door                    # `$door` stands where a place id goes
    until map != $start_map (max 10):   # ... or where a JSON value goes
      A
    call open_chest door_0 [4, 0]  # arguments are bound by position

An argument is a place id (`door_0`), an int, a JSON value (`[4, 0]`, `"text"`),
or one of the calling skill's own `$` parameters. There are no expressions.
A call with too few or too many arguments is an execution error, and so is a
top-level run of a skill that has parameters. A `$` reference to a name the skill
did not declare is a parse error. `parse_all` reads a block that holds several
skills, each starting with a `skill` line: the last one runs, with no arguments,
and the earlier ones are definitions it can call. With parameters off, any of
this syntax is rejected with a clear error, as `go_to` is in buttons mode, and a
skill without parameters parses exactly as before.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Collection, Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Union

from agents.harness import tools as navigator
from agents.harness.calls import parse_target
from agents.retrieval import Policy, Ranked, Stats
from environments.protocol import Action, Observation, Press

MAX_PRESSES = 200  # per program run, so a bad loop cannot eat the whole budget
MAX_CALL_DEPTH = 5


class SkillError(Exception):
    """A parse or execution error, fed back to the model as Voyager feeds back JS errors."""


class SkillInterrupted(SkillError):
    """A tool statement stopped because the game changed context (a battle started), not because it failed."""


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
class Param:
    """A `$name` reference, bound to an argument when the skill is called."""

    name: str


@dataclass(frozen=True)
class Until:
    field: str
    op: str
    value: Any  # a JSON value, or a Param
    max_iterations: int
    body: tuple["Statement", ...]


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple[Any, ...] = ()  # literal JSON values or Params


@dataclass(frozen=True)
class Navigate:
    """`go_to` or `talk_to`, with the arguments in the harness's form ({"place": id} or {"position": [x, y]})."""

    tool: str
    args: Mapping[str, Any]
    param: str | None = None  # `go_to $door`: the target comes from this parameter, and `args` is empty

    @property
    def label(self) -> str:
        if self.param is not None:
            return f"{self.tool} ${self.param}"
        target = self.args["place"] if "place" in self.args else str(list(self.args["position"]))
        return f"{self.tool} {target}"


Statement = Union[Tap, Repeat, Until, Call, Navigate]


@dataclass(frozen=True)
class Program:
    name: str
    body: tuple[Statement, ...]
    source: str
    params: tuple[str, ...] = ()


# Parsing

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_NAME = re.compile(rf"^skill\s+({_IDENT})\s*$")
_HEADER = re.compile(rf"^skill\s+({_IDENT})((?:\s+{_IDENT})+)\s*$")
_TAP = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+x\s*(\d+))?$")
_REPEAT = re.compile(r"^repeat\s+(\d+)\s*:$")
_UNTIL = re.compile(r"^until\s+([A-Za-z0-9_.]+)\s*(==|!=|<=|>=|<|>)\s*(.+?)\s*\(max\s+(\d+)\)\s*:$")
_CALL = re.compile(rf"^call\s+({_IDENT})(?:\s+(.+))?$")
_NAVIGATE = re.compile(r"^(go_to|talk_to)(?:\s+(.*))?$")
_PARAM = re.compile(rf"^\$({_IDENT})$")
_PLACE = re.compile(r"^[A-Za-z0-9_.:\-]+$")
_SKILL_LINE = re.compile(r"^skill(\s|$)")
MAX_PARAMS = 4


def _off(what: str) -> str:
    return f"parameters are not enabled for this player ({what}); write the steps for one fixed target"


def parse(source: str, legal_keys: Sequence[str], *, tools: bool = False, parameters: bool = False) -> Program:
    lines = []
    for number, raw in enumerate(source.splitlines(), start=1):
        text = raw.split("#", 1)[0].rstrip()
        if text.strip():
            lines.append((number, len(text) - len(text.lstrip()), text.strip()))
    if not lines:
        raise SkillError("the program is empty")
    number, _, first = lines[0]
    match = _NAME.match(first)
    params: tuple[str, ...] = ()
    if not match and (header := _HEADER.match(first)):
        names = tuple(header.group(2).split())
        if not parameters:
            raise SkillError(f"line {number}: " + _off(f"`skill {header.group(1)}` declares {', '.join(names)}"))
        if len(names) > MAX_PARAMS or len(set(names)) != len(names):
            raise SkillError(f"line {number}: a skill takes at most {MAX_PARAMS} parameters, with different names")
        match, params = header, names
    if not match:
        raise SkillError(f"line {number}: a program must start with `skill <name>`")
    body, rest = _parse_block(lines[1:], None, set(legal_keys), tools, params if parameters else None)
    if rest:
        raise SkillError(f"line {rest[0][0]}: unexpected indentation")
    if not body:
        raise SkillError("the program has no statements")
    if match.group(1) in _calls(body):
        # it would pass verification against the old version, then call itself once stored
        raise SkillError(f"`{match.group(1)}` calls itself; call another skill or repeat the steps")
    return Program(match.group(1), body, source.strip(), params)


def parse_all(source: str, legal_keys: Sequence[str], *, tools: bool = False, parameters: bool = False) -> list[Program]:
    """The skills in a code block: definitions first, the program to run last.

    Without `parameters` a block is one program, as `parse` reads it. With them,
    each `skill` line at column 0 starts a skill. The last one runs with no
    arguments, so it may not declare parameters; the earlier ones are
    definitions it (and the skills after them) can call.
    """
    if not parameters:
        return [parse(source, legal_keys, tools=tools)]
    raw = source.splitlines()
    starts = [i for i, line in enumerate(raw) if _SKILL_LINE.match(line)]
    before = [line for line in raw[: starts[0] if starts else len(raw)] if line.split("#", 1)[0].strip()]
    if not starts or before:
        return [parse(source, legal_keys, tools=tools, parameters=True)]  # raises the usual error
    programs = []
    for start, end in zip(starts, starts[1:] + [len(raw)]):
        # padding keeps the line numbers in errors those of the whole block
        programs.append(parse("\n" * start + "\n".join(raw[start:end]), legal_keys, tools=tools, parameters=True))
    names = [p.name for p in programs]
    if len(set(names)) != len(names):
        raise SkillError(f"the block defines `{next(n for n in names if names.count(n) > 1)}` twice")
    last = programs[-1]
    if last.params:
        raise SkillError(
            f"the last skill in the block runs with no arguments, so `{last.name}` cannot declare parameters "
            f"({', '.join(last.params)}); end the block with a skill that calls it with arguments, as in `call {last.name} <arguments>`"
        )
    return programs


def _calls(body) -> set[str]:
    names = set()
    for statement in body:
        if isinstance(statement, Call):
            names.add(statement.name)
        elif isinstance(statement, (Repeat, Until)):
            names |= _calls(statement.body)
    return names


def _parse_block(lines, indent, legal_keys, tools, params):
    """`params` is the declared parameter names, or None when parameters are off."""
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
            body, lines = _block_after(number, lines, depth, legal_keys, tools, params)
            statements.append(Repeat(int(match.group(1)), body))
        elif match := _UNTIL.match(text):
            value = _until_value(number, match.group(3), params)
            body, lines = _block_after(number, lines, depth, legal_keys, tools, params)
            statements.append(Until(match.group(1), match.group(2), value, int(match.group(4)), body))
        elif match := _CALL.match(text):
            args = ()
            if match.group(2):
                if params is None:
                    raise SkillError(f"line {number}: " + _off(f"`call {match.group(1)}` was given arguments"))
                args = tuple(_argument(number, token, params) for token in _tokens(number, match.group(2)))
            statements.append(Call(match.group(1), args))
        elif match := _NAVIGATE.match(text):
            if not tools:
                raise SkillError(f"line {number}: `{match.group(1)}` is only available with the tools interface; this player presses legal actions ({', '.join(sorted(legal_keys))})")
            statements.append(_navigate_statement(number, match.group(1), match.group(2) or "", params))
        elif match := _TAP.match(text):
            key = match.group(1).upper()
            if key not in legal_keys:
                raise SkillError(f"line {number}: `{match.group(1)}` is not a legal action ({', '.join(sorted(legal_keys))})")
            statements.append(Tap(key, int(match.group(2) or 1)))
        else:
            raise SkillError(f"line {number}: cannot parse `{text}`")
    return tuple(statements), lines


def _block_after(number, lines, depth, legal_keys, tools, params):
    if not lines or lines[0][1] <= depth:
        raise SkillError(f"line {number}: a block header needs an indented body")
    return _parse_block(lines, None, legal_keys, tools, params)


def _reference(number: int, text: str, params) -> Param | None:
    """The Param `text` names, None if `text` is not a `$name`; an error if the name is not declared."""
    match = _PARAM.match(text)
    if not match:
        return None
    if params is None:
        raise SkillError(f"line {number}: " + _off(f"`{text}` is a parameter"))
    if match.group(1) not in params:
        declared = f"this skill declares {', '.join(params)}" if params else "this skill declares no parameters"
        raise SkillError(f"line {number}: `{text}` is not a parameter of this skill ({declared}); declare it as in `skill <name> {match.group(1)}`")
    return Param(match.group(1))


def _until_value(number: int, text: str, params) -> Any:
    if (param := _reference(number, text, params)) is not None:
        return param
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise SkillError(f"line {number}: `{text}` is not a JSON value") from None


def _navigate_statement(number: int, tool: str, text: str, params) -> Navigate:
    if (param := _reference(number, text.strip(), params)) is not None:
        return Navigate(tool, {}, param.name)
    target = parse_target(text)
    if target is None:
        raise SkillError(f"line {number}: `{tool}` needs a place id or [x, y], as in `{tool} door_0` or `{tool} [4, 2]`")
    return Navigate(tool, target)


def _tokens(number: int, text: str) -> list[str]:
    """Split call arguments on spaces outside brackets and quotes, so `[4, 0]` is one token."""
    tokens, current, depth, quoted, escaped = [], "", 0, False, False
    for char in text:
        if quoted:
            current += char
            quoted, escaped = (char != '"' or escaped), (char == "\\" and not escaped)
        elif char == '"':
            quoted, current = True, current + char
        elif char in "[{":
            depth, current = depth + 1, current + char
        elif char in "]}":
            depth, current = depth - 1, current + char
        elif char.isspace() and depth == 0:
            if current:
                tokens.append(current)
            current = ""
        else:
            current += char
    if quoted or depth != 0:
        raise SkillError(f"line {number}: unbalanced quotes or brackets in `{text}`")
    return tokens + [current] if current else tokens


def _argument(number: int, token: str, params) -> Any:
    if (param := _reference(number, token, params)) is not None:
        return param
    try:
        return json.loads(token)
    except json.JSONDecodeError:
        if _PLACE.match(token):
            return token  # a place id
    raise SkillError(f"line {number}: argument `{token}` is not a place id, an int, or a JSON value")


# Execution


@dataclass
class Run:
    """Counts what one program run did, for logging and feedback."""

    presses: int = 0
    notes: list[str] = field(default_factory=list)  # how each tool statement ended, e.g. "go_to door_0: went through to town"
    current: str = ""  # the tool statement being expanded, "" while a plain press runs
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)  # each `call` entered: the skill's name and its bound arguments


Library = Mapping[str, Program]


def execute(
    program: Program,
    library: Library | None,
    observation: Observation,
    run: Run,
    legal_keys: Collection[str] | None = None,
    args: Sequence[Any] = (),
) -> Generator[Action, Observation, None]:
    """Yield one press at a time; the caller sends back the observation after each.

    `legal_keys` turns the tools interface on: a `go_to` or `talk_to` statement
    needs them to find the movement keys. None means a tool statement is an error.
    `args` bind the program's parameters by position; a mismatch is a SkillError
    when the generator first runs.
    """
    box = [observation]
    frame = _bind(program, args)
    yield from _run_block(program.body, library, box, run, 0, None if legal_keys is None else set(legal_keys), frame)


def _bind(program: Program, args: Sequence[Any]) -> dict[str, Any]:
    if len(args) != len(program.params):
        plural = "s" if len(program.params) != 1 else ""
        wanted = f"{len(program.params)} argument{plural} ({', '.join(program.params)})" if program.params else "no arguments"
        raise SkillError(f"`{program.name}` takes {wanted} but got {len(args)}")
    return dict(zip(program.params, args))


def _value(token: Any, frame: Mapping[str, Any]) -> Any:
    return frame[token.name] if isinstance(token, Param) else token


def _run_block(body, library, box, run, depth, legal, frame):
    for statement in body:
        if isinstance(statement, Tap):
            for _ in range(statement.times):
                if run.presses >= MAX_PRESSES:
                    raise SkillError(f"stopped after {MAX_PRESSES} presses")
                run.presses += 1
                box[0] = yield Press(statement.key)
        elif isinstance(statement, Repeat):
            for _ in range(statement.times):
                yield from _run_block(statement.body, library, box, run, depth, legal, frame)
        elif isinstance(statement, Until):
            target = _value(statement.value, frame)
            for _ in range(statement.max_iterations):
                if _holds(statement, box[0], target):
                    break
                yield from _run_block(statement.body, library, box, run, depth, legal, frame)
            else:
                if not _holds(statement, box[0], target):
                    actual = _lookup(box[0], statement.field)
                    raise SkillError(
                        f"`until {statement.field} {statement.op} {json.dumps(target)}` still false after "
                        f"{statement.max_iterations} iterations ({statement.field} is {json.dumps(actual)})"
                    )
        elif isinstance(statement, Call):
            if library is None:
                raise SkillError("the skill library is off, so `call` is unavailable")
            if statement.name not in library:
                raise SkillError(f"no skill named `{statement.name}` in the library")
            if depth >= MAX_CALL_DEPTH:
                raise SkillError(f"calls nested deeper than {MAX_CALL_DEPTH}")
            callee = library[statement.name]
            values = tuple(_value(arg, frame) for arg in statement.args)
            inner = _bind(callee, values)
            run.calls.append((callee.name, values))
            yield from _run_block(callee.body, library, box, run, depth + 1, legal, inner)
        elif isinstance(statement, Navigate):
            yield from _navigate(statement, box, run, legal, frame)


def _target(statement: Navigate, frame: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    """The statement's label and its target in the harness's form; a parameter's value is a place id or [x, y]."""
    if statement.param is None:
        return statement.label, statement.args
    value = frame[statement.param]
    label = f"{statement.tool} {value if isinstance(value, str) else json.dumps(value)}"
    if isinstance(value, str) and _PLACE.match(value):
        return label, {"place": value}
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return label, {"position": value}
    raise SkillError(f"`{statement.label}` needs ${statement.param} to be a place id or [x, y], not {json.dumps(value)}")


def _navigate(statement: Navigate, box, run: Run, legal, frame):
    """Expand `go_to` / `talk_to` through the navigator: one press per yield, each counted."""
    label, args = _target(statement, frame)
    if legal is None:
        raise SkillError(f"`{label}` is unavailable: this player uses the buttons interface")
    observation = box[0]
    if not navigator.navigable(observation.state):
        raise SkillError(f"`{label}` needs a snapshot with a grid and position, which this observation does not have")
    try:
        steps = navigator.tool_steps(statement.tool, args, observation, navigator.controls(observation.state, legal), legal)
    except navigator.ToolError as failure:
        raise SkillError(f"`{label}` failed: {failure}") from None
    run.current = label
    try:
        press = next(steps)
        while True:
            if run.presses >= MAX_PRESSES:
                raise SkillError(f"stopped after {MAX_PRESSES} presses")
            run.presses += 1
            box[0] = yield press
            press = steps.send(box[0])
    except StopIteration as done:
        run.notes.append(f"{label}: {done.value}")
    except navigator.ToolError as failure:
        raise SkillError(f"`{label}` failed: {failure}") from None
    except navigator.Interrupted as interruption:
        raise SkillInterrupted(f"`{label}` interrupted: {interruption}") from None
    finally:
        run.current = ""


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


def _holds(statement: Until, observation: Observation, value: Any) -> bool:
    actual = _lookup(observation, statement.field)
    try:
        return {
            "==": lambda: actual == value,
            "!=": lambda: actual != value,
            "<": lambda: actual < value,
            "<=": lambda: actual <= value,
            ">": lambda: actual > value,
            ">=": lambda: actual >= value,
        }[statement.op]()
    except TypeError:
        raise SkillError(f"cannot compare {statement.field} ({json.dumps(actual)}) {statement.op} {json.dumps(value)}") from None


# Library

CANDIDATE_WEIGHT = 0.8  # a candidate's retrieval score is scaled by this, so a verified skill wins a near tie


@dataclass(frozen=True)
class Skill:
    program: Program
    description: str
    task: str
    version: int = 1
    status: str = "verified"  # "candidate" until it succeeds with arguments it was not first accepted with (`test_skills`)
    uses: int = 0  # programs that called it, counted only under `test_skills`
    successes: int = 0
    proven_with: tuple[str, ...] = ()  # JSON of each argument list a candidate has succeeded with

    @property
    def shown(self) -> str:
        """The text the action agent sees: the source, with a comment line above a candidate."""
        if self.status != "candidate":
            return self.program.source
        tried = ", ".join(self.proven_with)
        return f"# candidate: succeeded {self.successes} of {self.uses} uses, only with arguments {tried}; not yet verified\n{self.program.source}"


class SkillLibrary:
    """Stores verified skills and retrieves the ones closest to a query.

    Voyager embeds skill descriptions and retrieves by vector similarity. This
    uses bag-of-words cosine over name, description, and task, so it needs no
    embedding model; the interface is the same.

    A skill is "verified" unless it was added as a "candidate" (see
    `agents/voyager/player.py`, `test_skills`). `record_use` counts a program's
    use of a skill and promotes a candidate that succeeds with arguments it has
    not succeeded with before. Retrieval scales a candidate's score down.

    With a `policy` (`utility_retrieval`, `memory_decay`; `agents/retrieval.py`) each
    skill also has `Stats`, and `retrieve_ranked` retrieves in two phases and
    explains itself. `clock` is the agent's step count; a library loaded from a
    file starts it at the last access the file recorded, so time passes only while
    the agent plays. Without a policy nothing here changes.
    """

    def __init__(self, path: Path | None = None, policy: Policy | None = None) -> None:
        self.path = path
        self.policy = policy if policy is not None and policy.active else None
        self.skills: dict[str, Skill] = {}
        self.stats: dict[str, Stats] = {}
        self.clock = 0
        if path is not None and path.exists():
            self._load(path)

    def programs(self) -> dict[str, Program]:
        return {name: skill.program for name, skill in self.skills.items()}

    def add(self, program: Program, description: str, task: str, *, status: str = "verified", proven_with: Sequence[Sequence[Any]] = ()) -> Skill:
        """Store a skill (a new version if the name exists; its counters start over).

        A candidate starts with one use and one success, the program run that got it
        accepted; `proven_with` lists the argument lists that run called it with.
        """
        previous = self.skills.get(program.name)
        proven = tuple(dict.fromkeys(json.dumps(list(args)) for args in proven_with))
        tested = int(status == "candidate")
        skill = Skill(
            program, description, task, version=previous.version + 1 if previous else 1,
            status=status, uses=tested, successes=tested, proven_with=proven,
        )
        self.skills[program.name] = skill
        if self.policy is not None:
            self.stats[program.name] = self.policy.fresh(self.clock)  # a new version starts over
        if self.path is not None:
            self._save(self.path)
        return skill

    def record_use(self, name: str, args: Sequence[Any], success: bool) -> tuple[Skill, bool]:
        """Count one use of `name` with `args`. Returns the skill and whether this use promoted it.

        A candidate is promoted by a success with arguments not in `proven_with`.
        A success with arguments it already succeeded with counts as a success but proves nothing new.
        """
        skill = self.skills[name]
        key = json.dumps(list(args))
        promoted = success and skill.status == "candidate" and key not in skill.proven_with
        skill = replace(
            skill,
            uses=skill.uses + 1,
            successes=skill.successes + int(success),
            status="verified" if promoted else skill.status,
            proven_with=skill.proven_with + (key,) if promoted else skill.proven_with,
        )
        self.skills[name] = skill
        if self.path is not None:
            self._save(self.path)
        return skill, promoted

    def retrieve(self, query: str, k: int = 5) -> list[Skill]:
        words = _words(query)
        scored = []
        for name, s in self.skills.items():
            score = _cosine(words, _words(f"{s.program.name} {s.description} {s.task}"))
            scored.append((score * CANDIDATE_WEIGHT if s.status == "candidate" else score, name))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self.skills[name] for _, name in scored[:k]]

    def retrieve_ranked(self, query: str, k: int = 5) -> tuple[list[Skill], list[Ranked]]:
        """`retrieve` under a policy: phase 1 by the cosine below, phase 2 by utility, retention as a multiplier.

        Returns the skills shown and every candidate with its similarity, utility, retention, and score.
        Each skill shown is recalled (its retention restarts and its strength grows).
        """
        words = _words(query)
        entries = []
        for name, s in sorted(self.skills.items()):
            similarity = _cosine(words, _words(f"{s.program.name} {s.description} {s.task}"))
            entries.append((name, similarity * CANDIDATE_WEIGHT if s.status == "candidate" else similarity, self._stats(name)))
        rows = self.policy.rank(entries, k, self.clock)
        self.policy.recall_shown(rows, self.stats, self.clock)
        if self.path is not None and self.policy.decay:
            self._save(self.path)
        return [self.skills[row.key] for row in rows if row.shown], rows

    def credit(self, name: str, reward: float) -> tuple[float, float] | None:
        """Move the skill's Q toward `reward` (0 or 1). Returns (old, new), or None when utility is off or the skill is gone."""
        if self.policy is None or not self.policy.utility or name not in self.skills:
            return None
        change = self.policy.update(self._stats(name), reward)
        if self.path is not None:
            self._save(self.path)
        return change

    def _stats(self, name: str) -> Stats:
        if name not in self.stats:
            self.stats[name] = self.policy.fresh(self.clock)
        return self.stats[name]

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, s in self.skills.items():
            data[name] = {"source": s.program.source, "description": s.description, "task": s.task, "version": s.version}
            if s.status != "verified" or s.uses:  # the extra fields appear only once a skill is being tested
                data[name] |= {"status": s.status, "uses": s.uses, "successes": s.successes, "proven_with": list(s.proven_with)}
            if self.policy is not None:  # only while a switch is on
                data[name]["memory"] = self._stats(name).to_dict(utility=self.policy.utility, decay=self.policy.decay)
        path.write_text(json.dumps(data, indent=2) + "\n")

    def _load(self, path: Path) -> None:
        saved = json.loads(path.read_text())
        for name, data in saved.items():
            program = Program(name, (), data["source"])  # body is re-parsed against legal keys on first use
            self.skills[name] = Skill(
                program, data["description"], data["task"], data["version"],
                data.get("status", "verified"), data.get("uses", 0), data.get("successes", 0), tuple(data.get("proven_with", ())),
            )
        if self.policy is not None:
            self.clock = max((d.get("memory", {}).get("last_access", 0) for d in saved.values()), default=0)
            for name, data in saved.items():
                stats = self.policy.fresh(self.clock)
                for field_name, value in data.get("memory", {}).items():
                    setattr(stats, field_name, value)
                self.stats[name] = stats

    def bind(self, legal_keys: Sequence[str], *, tools: bool = False, parameters: bool = False) -> None:
        """Parse loaded skills against this environment's legal actions."""
        for name, skill in list(self.skills.items()):
            if not skill.program.body:
                try:
                    program = parse(skill.program.source, legal_keys, tools=tools, parameters=parameters)
                except SkillError:
                    del self.skills[name]
                    self.stats.pop(name, None)
                    continue
                self.skills[name] = replace(skill, program=program)


def _words(text: str) -> Counter:
    return Counter(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[w] * b[w] for w in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0
