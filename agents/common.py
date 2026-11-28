"""Game-agnostic helpers shared by players: render a snapshot, parse an action."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from environments.protocol import Action, Observation


def render_observation(observation: Observation) -> tuple[str, list[bytes]]:
    """The snapshot as prompt text plus images, using only what the mode allows."""
    parts = []
    images = []
    if observation.frame is not None:
        parts.append("Screen: see the attached image.")
        images.append(observation.frame.to_png())
    if observation.state is not None:
        parts.append("State: " + json.dumps(observation.state, sort_keys=True))
    return "\n".join(parts), images


def first_json_object(text: str):
    """The first complete JSON object in a reply, ignoring anything after it.

    Small models often close the object early and keep writing
    (`{"tool": ...}, "intention": ...`); the first object is what they meant.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(text, start)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def describe_actions(legal_actions: Sequence[Action]) -> str:
    return ", ".join(action.key for action in legal_actions)


_ACTION_LINE = re.compile(r"^\s*action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_action(text: str, legal_actions: Sequence[Action], *, require_label: bool) -> Action | None:
    """Find the chosen action in model output, or None if there is no legal one.

    With `require_label`, the action must follow "Action:" (the last such line
    wins). Without it, the whole reply must be one action name.
    """
    if require_label:
        matches = _ACTION_LINE.findall(text)
        if not matches:
            return None
        candidate = matches[-1]
    else:
        candidate = text
    candidate = candidate.strip().strip("`*.'\"[]() ").upper()
    by_key = {action.key.upper(): action for action in legal_actions}
    return by_key.get(candidate)

