"""Transition log: wraps any `Environment` and records every reset and step.

One JSON line per transition in `transitions.jsonl`, plus one PNG per frame.
Game-agnostic; it reads only the protocol types.
"""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from environments.protocol import Action, Environment, Frame, Observation, Progress, StepResult


class RecordingEnvironment:
    def __init__(self, env: Environment, run_dir: Path) -> None:
        self._env = env
        self._run_dir = run_dir
        (run_dir / "frames").mkdir(parents=True, exist_ok=True)
        self._log = (run_dir / "transitions.jsonl").open("a")
        self._index = 0

    def reset(self) -> Observation:
        observation = self._env.reset()
        self._index = 0
        self._write({"kind": "reset", **self._observation(observation)})
        return observation

    def observe(self) -> Observation:
        return self._env.observe()

    def step(self, action: Action) -> StepResult:
        result = self._env.step(action)
        self._index += 1
        self._write(
            {
                "kind": "step",
                "action": asdict(action),
                "frames_advanced": result.frames_advanced,
                "wall_seconds": result.wall_seconds,
                **self._observation(result.observation),
            }
        )
        return result

    def get_progress(self) -> Progress:
        return self._env.get_progress()

    def legal_actions(self) -> Sequence[Action]:
        return self._env.legal_actions()

    def close(self) -> None:
        self._log.close()

    def _observation(self, observation: Observation) -> dict[str, Any]:
        frame_path = None
        if observation.frame is not None:
            frame_path = f"frames/{self._index:06d}.png"
            (self._run_dir / frame_path).write_bytes(encode_png(observation.frame))
        return {
            "step": self._index,
            "mode": observation.mode.value,
            "frame_count": observation.frame_count,
            "frame": frame_path,
            "state": observation.state,
            "progress": dict(self._env.get_progress()),
        }

    def _write(self, record: dict[str, Any]) -> None:
        self._log.write(json.dumps(record) + "\n")
        self._log.flush()


def encode_png(frame: Frame) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    stride = frame.width * 4
    scanlines = b"".join(
        b"\x00" + frame.rgba[y * stride : (y + 1) * stride] for y in range(frame.height)
    )
    header = struct.pack(">IIBBBBB", frame.width, frame.height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
