"""Plans with checked steps, and the pieces the Crystal harness uses to explain a failure and re-plan.

A plan is up to `MAX_PLAN_STEPS` steps. Each step is a tool call plus an `expect`: a
condition over role-snapshot fields (`{"map": "4.0"}`, `{"context": "dialogue"}`) that
code checks against the adapter's state when the step's tool ends. The model's claim
that a step worked is never read. This is AdaPlanner's assertion (Sun et al., NeurIPS
2023, arXiv:2305.16653), with the state predicate in place of generated Python.

Reimplemented from the papers. DEPS (Wang et al., NeurIPS 2023, arXiv:2302.01560;
github.com/CraftJarvis/MC-Planner) publishes no LICENSE file, so none of its text or code
is copied. AdaPlanner's repository is MIT licensed, but its prompts ask for Python
programs, which does not carry over; none of it is copied either.

Pure functions and small dataclasses live here. The harness (player.py) owns when each
runs and the model calls.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from agents.harness.stuck import StuckLadder
from agents.harness.tools import ToolRun

MAX_PLAN_STEPS = 5
MAX_INTERRUPTS = 2  # a step the game interrupts (a battle began) is retried this often before it counts as failed
EXPECT_FIELDS = ("map", "context", "position", "facing")


@dataclass(frozen=True)
class PlanStep:
    tool: str
    args: Mapping[str, Any]
    expect: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args), "expect": dict(self.expect)}


@dataclass
class StepResult:
    index: int
    step: PlanStep
    outcome: str  # "done" | "failed"
    result: str
    mismatches: list[dict[str, Any]]

    @property
    def passed(self) -> bool:
        return self.outcome == "done" and not self.mismatches


@dataclass
class PlanRun:
    steps: list[PlanStep]
    source: str  # "planner" or "revision"
    index: int = 0
    interrupts: int = 0
    current: ToolRun | None = None
    results: list[StepResult] = field(default_factory=list)


# Parsing and validation


def parse_steps(raw: Any, names: Collection[str]) -> tuple[list[PlanStep], list[str]]:
    """The plan a reply named, and every problem with it. A plan with any problem is rejected whole."""
    if not isinstance(raw, list) or not raw:
        return [], ['"plan" must be a non-empty list of steps']
    if len(raw) > MAX_PLAN_STEPS:
        return [], [f"the plan has {len(raw)} steps; the most is {MAX_PLAN_STEPS}"]
    steps, problems = [], []
    game_tools = [name for name in names if name != "note"]
    for number, item in enumerate(raw, 1):
        found = len(problems)
        if not isinstance(item, dict):
            problems.append(f"step {number}: not a JSON object")
            continue
        tool, args, expect = item.get("tool"), item.get("args") or {}, item.get("expect") or {}
        if tool == "note":
            problems.append(f"step {number}: note takes no game time, so it is not a plan step")
        elif tool not in game_tools:
            problems.append(f"step {number}: tool {tool!r} is not one of {game_tools}")
        if not isinstance(args, dict):
            problems.append(f"step {number}: args must be a JSON object")
        if not isinstance(expect, dict):
            problems.append(f"step {number}: expect must be a JSON object")
        else:
            for key, value in expect.items():
                if key not in EXPECT_FIELDS:
                    problems.append(f"step {number}: expect field {key!r} is not one of {list(EXPECT_FIELDS)}")
                elif not _plain(value):
                    problems.append(f"step {number}: expect {key!r} must be a string, number, boolean, or list of those")
        if len(problems) == found:
            steps.append(PlanStep(tool, args, expect))
    return ([], problems) if problems else (steps, [])


def _plain(value: Any) -> bool:
    if isinstance(value, list):
        return all(_plain(v) for v in value)
    return isinstance(value, (str, int, float, bool))


def parse_reply(reply: Mapping[str, Any], names: Collection[str], *, plan_allowed: bool) -> tuple[list[PlanStep], list[str]]:
    """Steps from a model reply: a `plan` list when plans are on, else (or as a fallback) one `tool` and `args`."""
    if plan_allowed and "plan" in reply:
        return parse_steps(reply["plan"], names)
    if "tool" in reply:
        return parse_steps([{"tool": reply["tool"], "args": reply.get("args"), "expect": reply.get("expect")}], names)
    return [], ['the reply has no "plan"' if plan_allowed else 'the reply has no "tool"']


# Checking


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.lower() == expected.lower()
    return actual == expected or str(actual) == str(expected)


def check_expect(expect: Mapping[str, Any], state: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Each expected field that the state does not show, as {field, expected, actual}. Empty means every check passed."""
    state = state or {}
    return [{"field": key, "expected": value, "actual": state.get(key)} for key, value in expect.items() if not _same(state.get(key), value)]


def select(steps: list[PlanStep], ladder: StuckLadder, state: Mapping[str, Any] | None) -> tuple[list[PlanStep], list[PlanStep]]:
    """DEPS's selector, reduced to what the stuck ladder knows (see the deviation in player.py): drop leading
    steps identical to a call that already failed in this exact situation. Returns (kept, dropped)."""
    start = 0
    while start < len(steps) and ladder.is_repeat_failure(state, steps[start].tool, steps[start].args):
        start += 1
    return steps[start:], steps[:start]


# Describing, for the explain prompt


def brief(state: Mapping[str, Any] | None) -> dict[str, Any]:
    return {key: (state or {}).get(key) for key in EXPECT_FIELDS if (state or {}).get(key) is not None}


def describe_change(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> str:
    changes = []
    for key in EXPECT_FIELDS:
        old, new = (before or {}).get(key), (after or {}).get(key)
        if old != new:
            changes.append(f"{key}: {json.dumps(old)} -> {json.dumps(new)}")
    return "; ".join(changes) or "nothing in map, context, position, or facing changed"


def render_progress(run: PlanRun, failed: int) -> str:
    """The plan so far, one line per step: what ran, how it ended, which one failed, which have not run."""
    lines = []
    done = {r.index: r for r in run.results}
    for number, step in enumerate(run.steps):
        head = f"{number + 1}. {step.tool} {json.dumps(dict(step.args))} expect {json.dumps(dict(step.expect))}"
        if number == failed:
            lines.append(f"{head} <- FAILED")
        elif number in done and done[number].passed:
            lines.append(f"{head} (done, checks passed)")
        elif number < failed:
            lines.append(f"{head} (done)")
        else:
            lines.append(f"{head} (not run)")
    return "\n".join(lines)
