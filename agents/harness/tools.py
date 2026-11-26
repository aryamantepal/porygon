"""Tools the planner can call, and how each expands into counted button presses.

The model names a destination or a tool; code does the walking. Each tool is a
generator: it yields one `Press` at a time and receives the observation that
followed, so the runner counts and logs every press. A tool returns a short
result string, or raises `ToolError` (failed) or `Interrupted` (the game
changed context, e.g. a battle started; not a failure).

Everything here reads only role-snapshot fields (environments/roles.py).
"""

from __future__ import annotations

import heapq
from collections.abc import Generator, Mapping
from typing import Any

from environments.protocol import Observation, Press
from environments.roles import DEFAULT_CONTROLS, DEFAULT_WALKABLE

Step = Generator[Press, Observation, str]
DELTAS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
MAX_TOOL_PRESSES = 150


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


def pick_option(cursor: int, index: int, keys: dict[str, str]) -> Step:
    """Move a menu cursor from `cursor` to `index`, then confirm."""
    direction = "down" if index > cursor else "up"
    for _ in range(abs(index - cursor)):
        yield Press(keys[direction])
    yield Press(keys["confirm"])
    return f"picked option {index}"
