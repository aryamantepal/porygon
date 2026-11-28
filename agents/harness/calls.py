"""The action-interface switch's shared pieces: the text form of a tool call, and how to parse it.

`action_interface` is `"buttons"` (an action is one legal button press) or
`"tools"` (an action is a tool call that code expands into counted presses).
ReAct, Reflexion, and Voyager all take it. The grammar of a tool call, which
ReAct writes after "Action n:":

    go_to <place id>        go_to door_0
    go_to [x, y]            go_to [4, 2]
    talk_to <place id>      talk_to person
    talk_to [x, y]
    press <BUTTON>          press A
    press <BUTTON> x<n>     press A x3        (n from 1 to 20)

Parsing is strict: one call, nothing else on the line. A tool that is not
offered, a place the snapshot does not list, or a button that is not legal is
an error with a reason, and the caller treats it like an illegal action.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from agents.harness.tools import MAX_PRESS_TIMES

ACTION_INTERFACES = ("buttons", "tools")

TOOL_HELP = {
    "go_to": "go_to <place id>  or  go_to [x, y]: walk there; walking onto a door goes through it",
    "talk_to": "talk_to <place id>: walk next to someone, face them, and press the talk key",
    "press": f"press <BUTTON>  or  press <BUTTON> x<n>: press a legal button n times (1 to {MAX_PRESS_TIMES})",
}

_POSITION = re.compile(r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]$")
_PLACE = re.compile(r"^[A-Za-z0-9_.:\-]+$")
_TOOL = re.compile(r"^([A-Za-z_]+)(?:\s+(.*))?$")
_PRESS = re.compile(r"^([A-Za-z0-9_]+)(?:\s+[xX×]\s*(\d+))?$")


class ToolCallError(ValueError):
    """The text is not a usable tool call; the message says why."""


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: Mapping[str, Any]  # the harness's argument form: {"place": id}, {"position": [x, y]}, {"key": K, "times": n}

    def text(self) -> str:
        if self.tool == "press":
            times = self.args["times"]
            return f"press {self.args['key']}" + (f" x{times}" if times != 1 else "")
        target = self.args["place"] if "place" in self.args else str(list(self.args["position"]))
        return f"{self.tool} {target}"


def check_interface(action_interface: str, specialists: bool) -> bool:
    """Whether the interface is tools; rejects an unknown interface and specialists without tools."""
    if action_interface not in ACTION_INTERFACES:
        raise ValueError(f"unknown action_interface {action_interface!r}; use one of {', '.join(ACTION_INTERFACES)}")
    if specialists and action_interface != "tools":
        raise ValueError('"specialists" only works with "action_interface": "tools"')
    return action_interface == "tools"


def parse_target(text: str) -> dict[str, Any] | None:
    """`door_0` or `[4, 2]` as harness arguments; None if it is neither."""
    text = text.strip()
    if match := _POSITION.match(text):
        return {"position": [int(match.group(1)), int(match.group(2))]}
    if _PLACE.match(text):
        return {"place": text}
    return None


def parse_tool_call(
    text: str,
    *,
    legal_keys: Collection[str],
    offered: Collection[str] = ("go_to", "talk_to", "press"),
    place_ids: Collection[str] | None = None,
) -> ToolCall:
    """One tool call from an Action line; raises `ToolCallError` with the reason if it is not usable.

    `place_ids` are the ids the snapshot lists; a named place outside them is
    an error. None skips that check.
    """
    candidate = text.strip().strip("`*'\" ").rstrip(".").strip()
    match = _TOOL.match(candidate)
    if not match:
        raise ToolCallError(f"{text!r} is not a tool call")
    name, rest = match.group(1).lower(), (match.group(2) or "").strip()
    if name not in offered:
        raise ToolCallError(f"{name!r} is not a tool; use one of: {', '.join(offered)}")
    if name == "press":
        pressed = _PRESS.match(rest)
        if not pressed:
            raise ToolCallError(f"press needs a button, as in `press A` or `press A x3`, not {rest!r}")
        key = pressed.group(1).upper()
        if key not in {k.upper() for k in legal_keys}:
            raise ToolCallError(f"{key!r} is not a legal button")
        times = int(pressed.group(2) or 1)
        if not 1 <= times <= MAX_PRESS_TIMES:
            raise ToolCallError(f"press x{times} is out of range (1 to {MAX_PRESS_TIMES})")
        return ToolCall("press", {"key": key, "times": times})
    target = parse_target(rest)
    if target is None:
        raise ToolCallError(f"{name} needs a place id or [x, y], not {rest!r}")
    if "place" in target and place_ids is not None and target["place"] not in place_ids:
        raise ToolCallError(f"no place {target['place']!r} on this map")
    return ToolCall(name, target)
