"""A goal is data: text the player reads, and a progress check the environment decides."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from environments.protocol import Progress


@dataclass(frozen=True)
class Goal:
    task_id: str
    text: str
    success: Mapping[str, Any]  # every key must equal the same key in get_progress()

    def is_met(self, progress: Progress) -> bool:
        return all(key in progress and progress[key] == value for key, value in self.success.items())
