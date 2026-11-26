"""A one-row corridor: the smallest environment that is still a game.

Agents and the runner are tested against it, so they never need a ROM. It is
deliberately not Pokémon: a player that only works on FireRed fails here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from environments.protocol import Action, Frame, Observation, ObservationMode, Press, Progress, StepResult

TILE = 8


@dataclass(frozen=True)
class LineWorldConfig:
    width: int = 10
    start: int = 0
    mode: ObservationMode = ObservationMode.BOTH


class LineWorld:
    KEYS = ("LEFT", "RIGHT", "A")

    def __init__(self, config: LineWorldConfig = LineWorldConfig()) -> None:
        self.config = config
        self._x = config.start
        self._frames = 0
        self._lamp = False

    def reset(self) -> Observation:
        self._x = self.config.start
        self._frames = 0
        self._lamp = False
        return self.observe()

    def observe(self) -> Observation:
        mode = self.config.mode
        return Observation(
            mode=mode,
            frame_count=self._frames,
            frame=self._screen() if mode.has_frame else None,
            state=self._state() if mode.has_state else None,
        )

    def step(self, action: Action) -> StepResult:
        if not isinstance(action, Press) or action.key not in self.KEYS:
            raise ValueError(f"not a LineWorld action: {action!r}")
        if action.key == "LEFT":
            self._x = max(0, self._x - 1)
        elif action.key == "RIGHT":
            self._x = min(self.config.width - 1, self._x + 1)
        elif self._x == self.config.width - 1:
            self._lamp = True  # pressing A at the far wall lights the lamp
        self._frames += 1
        return StepResult(self.observe(), action, frames_advanced=1, wall_seconds=0.0)

    def get_progress(self) -> Progress:
        return {"x": self._x, "lamp": self._lamp}

    def legal_actions(self) -> Sequence[Action]:
        return [Press(key) for key in self.KEYS]

    def _state(self) -> dict:
        return {"x": self._x, "width": self.config.width, "lamp": self._lamp}

    def _screen(self) -> Frame:
        width, height = self.config.width * TILE, TILE
        pixels = bytearray(b"\x20\x20\x20\xff" * (width * height))
        lamp = b"\xff\xd0\x00\xff" if self._lamp else b"\x60\x60\x60\xff"
        for column, color in ((self.config.width - 1, lamp), (self._x, b"\x30\xa0\xff\xff")):
            for y in range(height):
                for x in range(column * TILE, column * TILE + TILE):
                    pixels[(y * width + x) * 4 : (y * width + x) * 4 + 4] = color
        return Frame(width, height, bytes(pixels))
