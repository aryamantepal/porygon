"""Tools the planner can call, and how each expands into counted button presses.

The model names a destination or a tool; code does the walking. Each tool is a
generator: it yields one `Press` at a time and receives the observation that
followed, so the runner counts and logs every press. A tool returns a short
result string, or raises `ToolError` (failed) or `Interrupted` (the game
changed context, e.g. a battle started; not a failure).

Everything here reads only role-snapshot fields (environments/roles.py).

`ToolRun` drives a tool one press per decision, and `tool_steps` and
`resolve_target` turn a tool name and arguments into a generator. The Crystal
harness, ReAct, Reflexion, and Voyager all use these, so every architecture
that offers tools expands them the same way.
"""

from __future__ import annotations

import heapq
from collections.abc import Collection, Generator, Mapping
from dataclasses import dataclass
from typing import Any

from environments.protocol import Observation, Press
from environments.roles import DEFAULT_CONTROLS, DEFAULT_WALKABLE

Step = Generator[Press, Observation, str]
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
MAX_TOOL_PRESSES = 150
MAX_PRESS_TIMES = 20


class ToolError(Exception):
    pass


class Interrupted(Exception):
    pass


def controls(state: Mapping[str, Any] | None, legal_keys: set[str]) -> dict[str, str]:
    mapping = dict(DEFAULT_CONTROLS)
    if state and isinstance(state.get("controls"), Mapping):
        mapping.update(state["controls"])
    return {role: key for role, key in mapping.items() if key in legal_keys}


# Path planning


def find_path(grid: list[str], walkable: str, start: tuple[int, int], goals: set[tuple[int, int]], blocked: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """A* over the grid with 4-way moves; returns the tiles from start to a goal, inclusive."""

    def open_tile(tile):
        x, y = tile
        return 0 <= y < len(grid) and 0 <= x < len(grid[y]) and grid[y][x] in walkable and tile not in blocked

    def estimate(tile):
        return min(abs(tile[0] - g[0]) + abs(tile[1] - g[1]) for g in goals)

    frontier = [(estimate(start), 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost = {start: 0}
    while frontier:
        _, spent, tile = heapq.heappop(frontier)
        if tile in goals:
            path = [tile]
            while came_from[path[-1]] is not None:
                path.append(came_from[path[-1]])
            return path[::-1]
        for dx, dy in DELTAS.values():
            nxt = (tile[0] + dx, tile[1] + dy)
            if (open_tile(nxt) or nxt in goals) and spent + 1 < cost.get(nxt, 1 << 30):
                cost[nxt] = spent + 1
                came_from[nxt] = tile
                heapq.heappush(frontier, (spent + 1 + estimate(nxt), spent + 1, nxt))
    return None


def _direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    return {delta: name for name, delta in DELTAS.items()}[(b[0] - a[0], b[1] - a[1])]


# Tools


def go_to(observation: Observation, target: tuple[int, int], keys: dict[str, str], *, adjacent: bool = False, door: bool = False) -> Step:
    """Walk to `target` (or next to it, facing it, when `adjacent`).

    Replans after every press. A press that does not move the player (a turn,
    or something in the way) is retried once; a tile that still will not take
    the player is marked blocked and the route goes around it. Some doors
    (stairs, mats) only lead somewhere when you keep walking into them, so on
    arriving at a `door` the navigator keeps stepping the way it was going.
    """
    state = observation.state
    start_map = state.get("map")
    blocked: set[tuple[int, int]] = set()
    misses = 0
    heading = None
    tried: list[str] = []  # directions already pushed while standing on the door
    for _ in range(MAX_TOOL_PRESSES):
        if state.get("map") != start_map:
            return f"went through to {state.get('map')}"
        context = state.get("context", "overworld")
        if context == "busy" and "wait" in keys:
            state = (yield Press(keys["wait"])).state  # a step or warp still finishing
            continue
        if context != "overworld":
            raise Interrupted(f"the game switched to {context}")
        position = tuple(state["position"])
        if adjacent and abs(position[0] - target[0]) + abs(position[1] - target[1]) == 1:
            facing = _direction(position, target)
            if (state.get("facing") or "").lower() == facing:
                return f"standing next to {list(target)}, facing it"
            state = (yield Press(keys[facing])).state
            continue
        if not adjacent and position == target:
            if not door:
                return f"arrived at {list(target)}"
            push = _next_push(heading, tried)
            if push is None:
                raise ToolError(f"standing on the door at {list(target)}, but no direction led anywhere")
            tried.append(push)
            for _ in range(2):  # the first press may only turn the player
                state = (yield Press(keys[push])).state
                if state.get("map") != start_map or tuple(state["position"]) != position or state.get("context") == "busy":
                    break
            continue
        goals = {(target[0] + dx, target[1] + dy) for dx, dy in DELTAS.values()} if adjacent else {target}
        path = find_path(state["grid"], state.get("walkable", DEFAULT_WALKABLE), position, goals, blocked)
        if path is None or len(path) < 2:
            raise ToolError(f"no path from {list(position)} to {list(target)}")
        nxt = path[1]
        heading = _direction(position, nxt)
        state = (yield Press(keys[heading])).state
        if state.get("map") == start_map and tuple(state["position"]) == position:
            misses += 1
            if misses >= 2:
                blocked.add(nxt)
                misses = 0
        else:
            misses = 0
    raise ToolError(f"gave up after {MAX_TOOL_PRESSES} presses")


OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _next_push(heading: str | None, tried: list[str]) -> str | None:
    """Directions to try on a door: the way we were going first, back the way we came last."""
    order = [heading] if heading else []
    order += [d for d in DELTAS if d != heading and d != OPPOSITE.get(heading)]
    if heading:
        order.append(OPPOSITE[heading])
    return next((d for d in order if d not in tried), None)


def talk_to(observation: Observation, target: tuple[int, int], keys: dict[str, str]) -> Step:
    result = yield from go_to(observation, target, keys, adjacent=True)
    state = (yield Press(keys["confirm"])).state
    if state.get("context") == "dialogue":
        return f"{result}; started a conversation"
    raise ToolError(f"{result}; pressing {keys['confirm']} started no conversation")


def press(key: str, times: int) -> Step:
    for _ in range(times):
        yield Press(key)
    return f"pressed {key} x{times}"


def press_sequence(keys: list[str]) -> Step:
    for key in keys:
        yield Press(key)
    return f"pressed {' '.join(keys)}"


def pick_option(cursor: int, index: int, keys: dict[str, str]) -> Step:
    """Move a menu cursor from `cursor` to `index`, then confirm."""
    direction = "down" if index > cursor else "up"
    for _ in range(abs(index - cursor)):
        yield Press(keys[direction])
    yield Press(keys["confirm"])
    return f"picked option {index}"


# Building tools from a name and arguments


def navigable(state: Mapping[str, Any] | None) -> bool:
    """Whether the snapshot has what `go_to` and `talk_to` need: a grid and a position."""
    return bool(state and state.get("grid") and state.get("position"))


def resolve_target(args: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[tuple[int, int], str]:
    """The tile a tool names, and the role of the place standing there ("" for a bare tile)."""
    if "place" in args:
        for place in state.get("places", []):
            if place.get("id") == args["place"]:
                return tuple(place["position"]), place.get("role", "")
        raise ToolError(f"no place {args['place']!r} on this map")
    position = args.get("position")
    if isinstance(position, list) and len(position) == 2 and all(isinstance(v, int) for v in position):
        # coordinates of a known place mean that place: a door named by its
        # tile still needs the walk-into-it handling
        for place in state.get("places", []):
            if list(place.get("position", [])) == position:
                return tuple(position), place.get("role", "")
        return tuple(position), ""
    raise ToolError('args need "place" or "position": [x, y]')


def tool_steps(tool: str, args: Mapping[str, Any], observation: Observation, keys: Mapping[str, str], legal_keys: Collection[str]) -> Step:
    """The generator for `go_to`, `talk_to`, or `press`; raises `ToolError` if the call is unusable."""
    state = observation.state or {}
    if tool == "press":
        key = str(args.get("key", "")).upper()
        if key not in legal_keys:
            raise ToolError(f"{key!r} is not a legal action")
        times = args.get("times", 1)
        if isinstance(times, bool) or not isinstance(times, int):
            raise ToolError(f'"times" must be a whole number, not {times!r}')
        return press(key, max(1, min(times, MAX_PRESS_TIMES)))
    if tool not in ("go_to", "talk_to"):
        raise ToolError(f"no tool named {tool!r}")
    target, role = resolve_target(args, state)
    if not navigable(state):
        raise ToolError("this observation has no grid and position to walk on")
    needed = ["up", "down", "left", "right"] + (["confirm"] if tool == "talk_to" else [])
    missing = [name for name in needed if name not in keys]
    if missing:
        raise ToolError(f"the action space has no {', '.join(missing)} control")
    if tool == "go_to":
        return go_to(observation, target, keys, door=role == "door")
    return talk_to(observation, target, keys)


# Running a tool one press per decision


@dataclass
class ToolEnd:
    """How a tool run finished: "done", "failed" (`ToolError`), or "interrupted" (the game changed context)."""

    outcome: str
    result: str


@dataclass
class ToolRun:
    """A tool generator driven one press per `decide`.

    `advance` primes the generator with `next`, then sends each observation
    in, and returns the next `Press` or a `ToolEnd`. `handler` says who
    started it ("planner", "model", "battle", "dialogue", "busy", ...).
    `chosen` marks a tool a model picked: if it ends without pressing anything
    that is a failure ("nothing to do"), not a success.
    """

    handler: str
    tool: str
    args: dict[str, Any]
    steps: Step
    start_state: Mapping[str, Any] | None = None
    presses: int = 0
    primed: bool = False
    chosen: bool = False

    def advance(self, observation: Observation, interrupt_on: Collection[str] = ()) -> Press | ToolEnd:
        """`interrupt_on` ends a run that has already pressed when the context becomes one of these."""
        context = (observation.state or {}).get("context", "overworld")
        if self.presses and context in interrupt_on:
            self.steps.close()
            return ToolEnd("interrupted", f"the game switched to {context}")
        try:
            action = self.steps.send(observation) if self.primed else next(self.steps)
        except StopIteration as done:
            if self.chosen and self.presses == 0:
                return ToolEnd("failed", f"nothing to do: {done.value}")
            return ToolEnd("done", done.value)
        except Interrupted as interruption:
            return ToolEnd("interrupted", str(interruption))
        except ToolError as failure:
            return ToolEnd("failed", str(failure))
        self.primed = True
        self.presses += 1
        return action
