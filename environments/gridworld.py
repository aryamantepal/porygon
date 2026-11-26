"""A toy RPG that fills every field of the role snapshot (see roles.py).

Two maps joined by a door, a person to talk to, a patch of grass with one
catchable creature, and a milestone tile. It exists so harness parts
(navigator, battle specialist, dialogue handling, stuck ladder) can be tested
end to end without a ROM. It is not Pokémon and does not try to be.

    house               town
    #######             #########
    #.....#             #...D...#
    #..N..#             #.......#
    #.....#             #..ggg..#
    ###D###             #..ggg.M#
                        #########
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from environments.protocol import Action, Frame, Observation, ObservationMode, Press, Progress, StepResult

MAPS = {
    "house": ["#######", "#.....#", "#..N..#", "#.....#", "###D###"],
    "town": ["#########", "#...D...#", "#.......#", "#..ggg..#", "#..ggg.M#", "#########"],
}
DOORS = {("house", (3, 4)): ("town", (4, 2)), ("town", (4, 1)): ("house", (3, 3))}
PERSON = ("house", (3, 2))
MILESTONE = ("town", (7, 4))
DIRECTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
DIALOGUE = "The milestone is in the southeast corner of town, past the grass."
BATTLE_OPTIONS = (
    {"id": "tackle", "kind": "attack", "power": 1},
    {"id": "slam", "kind": "attack", "power": 3},
    {"id": "ball", "kind": "catch"},
    {"id": "run", "kind": "flee"},
)
TILE = 8
COLORS = {"#": b"\x30\x30\x30\xff", ".": b"\xc8\xc8\xb4\xff", "D": b"\x8b\x5a\x2b\xff", "N": b"\xe0\x40\x40\xff", "g": b"\x40\xa0\x40\xff", "M": b"\xff\xd0\x00\xff"}


@dataclass(frozen=True)
class GridWorldConfig:
    mode: ObservationMode = ObservationMode.BOTH


class GridWorld:
    KEYS = ("UP", "DOWN", "LEFT", "RIGHT", "A", "B")

    def __init__(self, config: GridWorldConfig = GridWorldConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> Observation:
        self._map, self._pos, self._facing = "house", (1, 1), "DOWN"
        self._context = "overworld"
        self._frames = 0
        self._talked = self._milestone = self._caught = self._encountered = False
        self._battle: dict | None = None
        return self.observe()

    def observe(self) -> Observation:
        mode = self.config.mode
        return Observation(mode, self._frames, self._screen() if mode.has_frame else None, self._state() if mode.has_state else None)

    def step(self, action: Action) -> StepResult:
        if not isinstance(action, Press) or action.key not in self.KEYS:
            raise ValueError(f"not a GridWorld action: {action!r}")
        {"overworld": self._overworld, "dialogue": self._dialogue, "battle": self._fight}[self._context](action.key)
        self._frames += 1
        return StepResult(self.observe(), action, frames_advanced=1, wall_seconds=0.0)

    def get_progress(self) -> Progress:
        return {
            "map": self._map,
            "position": list(self._pos),
            "talked": self._talked,
            "caught": self._caught,
            "milestone": self._milestone,
        }

    def legal_actions(self) -> Sequence[Action]:
        return [Press(key) for key in self.KEYS]

    # Rules

    def _overworld(self, key: str) -> None:
        if key in DIRECTIONS:
            self._facing = key
            dx, dy = DIRECTIONS[key]
            target = (self._pos[0] + dx, self._pos[1] + dy)
            if self._tile(target) in "#N":
                return
            self._pos = target
            if (self._map, target) in DOORS:
                self._map, self._pos = DOORS[(self._map, target)]
            elif (self._map, target) == MILESTONE:
                self._milestone = True
            elif self._tile(target) == "g" and not self._encountered:
                self._encountered = True
                self._context = "battle"
                self._battle = {"hp": 3, "cursor": 0, "uses": {"tackle": 10, "slam": 0, "ball": 3, "run": 99}}
        elif key == "A":
            dx, dy = DIRECTIONS[self._facing]
            if (self._map, (self._pos[0] + dx, self._pos[1] + dy)) == PERSON:
                self._context = "dialogue"
                self._talked = True

    def _dialogue(self, key: str) -> None:
        if key in ("A", "B"):
            self._context = "overworld"

    def _fight(self, key: str) -> None:
        battle = self._battle
        if key in ("UP", "DOWN"):
            battle["cursor"] = (battle["cursor"] + (1 if key == "DOWN" else -1)) % len(BATTLE_OPTIONS)
            return
        if key != "A":
            return
        option = BATTLE_OPTIONS[battle["cursor"]]
        if battle["uses"][option["id"]] <= 0:
            return  # nothing happens: the move has no uses left
        battle["uses"][option["id"]] -= 1
        if option["kind"] == "attack":
            battle["hp"] = max(0, battle["hp"] - option["power"])
            if battle["hp"] == 0:
                self._end_battle()
        elif option["kind"] == "catch" and battle["hp"] <= 1:
            self._caught = True
            self._end_battle()
        elif option["kind"] == "flee":
            self._end_battle()

    def _end_battle(self) -> None:
        self._context, self._battle = "overworld", None

    # Snapshot

    def _tile(self, position: tuple[int, int]) -> str:
        x, y = position
        rows = MAPS[self._map]
        return rows[y][x] if 0 <= y < len(rows) and 0 <= x < len(rows[y]) else "#"

    def _state(self) -> dict:
        grid = ["".join("#" if c in "#N" else "." for c in row) for row in MAPS[self._map]]
        places = []
        if self._map == PERSON[0]:
            places.append({"id": "person", "role": "person", "position": list(PERSON[1])})
        for (name, position), (destination, _) in DOORS.items():
            if name == self._map:
                places.append({"id": f"door_to_{destination}", "role": "door", "position": list(position), "leads_to": destination})
        if self._map == MILESTONE[0]:
            places.append({"id": "milestone", "role": "milestone", "position": list(MILESTONE[1])})
        state = {
            "context": self._context,
            "map": self._map,
            "position": list(self._pos),
            "facing": self._facing,
            "grid": grid,
            "places": places,
        }
        if self._context == "dialogue":
            state["dialogue"] = {"text": DIALOGUE}
        if self._context == "battle":
            battle = self._battle
            state["battle"] = {
                "options": [{**option, "uses_left": battle["uses"][option["id"]]} for option in BATTLE_OPTIONS],
                "cursor": battle["cursor"],
                "opponent": {"hp": battle["hp"], "max_hp": 3, "catchable": True},
            }
        return state

    def _screen(self) -> Frame:
        rows = MAPS[self._map]
        width, height = len(rows[0]) * TILE, len(rows) * TILE
        pixels = bytearray(width * height * 4)
        for ty, row in enumerate(rows):
            for tx, char in enumerate(row):
                color = b"\x30\x80\xff\xff" if (tx, ty) == self._pos else COLORS[char]
                for y in range(ty * TILE, ty * TILE + TILE):
                    start = (y * width + tx * TILE) * 4
                    pixels[start : start + TILE * 4] = color * TILE
        return Frame(width, height, bytes(pixels))
