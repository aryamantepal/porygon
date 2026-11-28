"""Few-shot examples as data, in one file format that every agent reads.

The ReAct and Reflexion papers put worked example trajectories in the prompt.
Here they are an optional file named by the run config (`"examples": "<path>"`);
`experiments/run.py` parses it and hands an `Examples` to the agent. No file
means zero-shot. Examples come from a different task than the one evaluated, so
they teach the format and the controls without giving away the answer.

    {
      "source": "who wrote these, from which task",          required
      "trajectories": [                                       Direct, ReAct, Reflexion
        {"goal": "...",
         "turns": [{"observation": <state dict or text>, "thought": "...", "action": "RIGHT"}],
         "outcome": "how it ended"}                           optional
      ],
      "reflections": [                                        Reflexion
        {"trajectory_summary": "...", "reflection": "Plan: ..."}
      ],
      "programs": ["skill name\\n<skill program source>"],    Voyager, Continual Harness (Refiner)
      "tool_calls": [                                         crystal harness, Continual Harness
        {"situation": <state dict or text>, "reply": {...}, "goal": "..."}
      ]
    }

A section that is missing or empty means that agent gets no examples and
reports `few_shot: false`. A state dict is shown the way the agent shows a live
snapshot (`State: {json}`); text is shown as written. Each agent shows at most
the first few entries so the prompt stays inside the model's context.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_TRAJECTORIES = 2
MAX_TURNS = 8
MAX_REFLECTIONS = 2
MAX_PROGRAMS = 3
MAX_TOOL_CALLS = 4
MAX_PAIRS = 6


@dataclass(frozen=True)
class ExampleTurn:
    observation: str | Mapping[str, Any]
    thought: str
    action: str


@dataclass(frozen=True)
class ExampleTrajectory:
    goal: str
    turns: tuple[ExampleTurn, ...]
    outcome: str = ""


@dataclass(frozen=True)
class ExampleReflection:
    trajectory_summary: str
    reflection: str


@dataclass(frozen=True)
class ExampleToolCall:
    situation: str | Mapping[str, Any]
    reply: Mapping[str, Any]
    goal: str = ""


@dataclass(frozen=True)
class Examples:
    source: str
    trajectories: tuple[ExampleTrajectory, ...] = ()
    reflections: tuple[ExampleReflection, ...] = ()
    programs: tuple[str, ...] = ()
    tool_calls: tuple[ExampleToolCall, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Examples:
        """Parse a decoded examples file; a malformed one is an error, not silent zero-shot."""
        if not isinstance(data.get("source"), str) or not data["source"]:
            raise ValueError('examples need a "source" saying where they came from')
        try:
            return cls(
                source=data["source"],
                trajectories=tuple(
                    ExampleTrajectory(
                        goal=_text(t["goal"]),
                        turns=tuple(ExampleTurn(_state_or_text(u["observation"]), _text(u["thought"]), _text(u["action"])) for u in t["turns"]),
                        outcome=_text(t.get("outcome", "")),
                    )
                    for t in data.get("trajectories", [])
                ),
                reflections=tuple(ExampleReflection(_text(r["trajectory_summary"]), _text(r["reflection"])) for r in data.get("reflections", [])),
                programs=tuple(_text(p) for p in data.get("programs", [])),
                tool_calls=tuple(
                    ExampleToolCall(_state_or_text(c["situation"]), _mapping(c["reply"]), _text(c.get("goal", "")))
                    for c in data.get("tool_calls", [])
                ),
            )
        except (KeyError, TypeError) as problem:
            raise ValueError(f"malformed examples: {problem!r}") from problem

    # What each agent shows. Empty text means nothing to show.

    def pairs_text(self) -> str:
        """Direct policy: goal, snapshot, and the action taken, with no thought."""
        blocks = []
        for trajectory in self.trajectories:
            for turn in trajectory.turns:
                blocks.append(f"Goal: {trajectory.goal}\n{_observation(turn.observation)}\nReply: {turn.action}")
        return "\n\n".join(blocks[:MAX_PAIRS])

    def trajectories_text(self) -> str:
        """ReAct: whole trajectories in the Thought / Action / Observation form of the live loop."""
        return "\n\n".join(_trajectory(t) for t in self.trajectories[:MAX_TRAJECTORIES])

    def reflections_text(self) -> str:
        """Reflexion: a past attempt, then the plan written after it."""
        return "\n\n".join(f"{r.trajectory_summary}\nNew plan: {r.reflection}" for r in self.reflections[:MAX_REFLECTIONS])

    def programs_text(self) -> str:
        return "\n\n".join(self.programs[:MAX_PROGRAMS])

    def tool_calls_text(self) -> str:
        blocks = []
        for call in self.tool_calls[:MAX_TOOL_CALLS]:
            goal = f"Goal: {call.goal}\n" if call.goal else ""
            blocks.append(f"{goal}{_observation(call.situation)}\nReply: {json.dumps(call.reply)}")
        return "\n\n".join(blocks)


def _trajectory(trajectory: ExampleTrajectory) -> str:
    lines = [f"Goal: {trajectory.goal}"]
    for index, turn in enumerate(trajectory.turns[:MAX_TURNS], start=1):
        lines += [f"Observation {index}: {_observation(turn.observation)}", f"Thought {index}: {turn.thought}", f"Action {index}: {turn.action}"]
    omitted = len(trajectory.turns) - MAX_TURNS
    if omitted > 0:
        lines.append(f"({omitted} later turns omitted)")
    if trajectory.outcome:
        lines.append(trajectory.outcome)
    return "\n".join(lines)


def _observation(observation: str | Mapping[str, Any]) -> str:
    """A state dict as the live prompts show it; text as written."""
    if isinstance(observation, str):
        return observation
    return "State: " + json.dumps(observation, sort_keys=True)


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected text, got {value!r}")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected an object, got {value!r}")
    return value


def _state_or_text(value: Any) -> str | Mapping[str, Any]:
    if not isinstance(value, (str, Mapping)):
        raise TypeError(f"expected a state object or text, got {value!r}")
    return value
