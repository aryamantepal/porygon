"""The environment interface every game adapter implements.

Nothing here knows which game is running. An adapter narrows these types
to its own controls and memory; agents and evaluation only see this file.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ObservationMode(str, Enum):
    VISUAL = "visual"
    STRUCTURED = "structured"
    BOTH = "both"

    @property
    def has_frame(self) -> bool:
        return self is not ObservationMode.STRUCTURED

    @property
    def has_state(self) -> bool:
        return self is not ObservationMode.VISUAL


@dataclass(frozen=True)
class Frame:
    """One screen, as packed RGBA rows."""

    width: int
    height: int
    rgba: bytes

    def to_png(self) -> bytes:
        def chunk(kind: bytes, data: bytes) -> bytes:
            body = kind + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

        stride = self.width * 4
        scanlines = b"".join(
            b"\x00" + self.rgba[y * stride : (y + 1) * stride] for y in range(self.height)
        )
        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 6, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(scanlines))
            + chunk(b"IEND", b"")
        )


@dataclass(frozen=True)
class Press:
    """Hold one named input, then release it.

    `hold_frames` of None lets the adapter use its default tap length.
    Pointer and key-down/key-up actions join this union when an adapter
    needs them.
    """

    key: str
    hold_frames: int | None = None


Action = Press


@dataclass(frozen=True)
class Observation:
    mode: ObservationMode
    frame_count: int
    frame: Frame | None = None
    state: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    action: Action
    frames_advanced: int
    wall_seconds: float


Progress = Mapping[str, Any]


@runtime_checkable
class Environment(Protocol):
    def reset(self) -> Observation:
        """Restore the configured initial state. Same config, same world."""
        ...

    def observe(self) -> Observation:
        """Return the current snapshot without advancing the game."""
        ...

    def step(self, action: Action) -> StepResult:
        """Apply one action, advance the game, and observe once at the end."""
        ...

    def get_progress(self) -> Progress:
        """Progression read from the game, never from a model."""
        ...

    def legal_actions(self) -> Sequence[Action]:
        """The actions `step` accepts."""
        ...
