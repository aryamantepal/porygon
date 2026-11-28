"""The Crystal harness's context specialists, usable by any agent that offers tools.

When the game is not on the overworld, code handles it before a model is asked:

    busy      wait (no model call); back out with cancel if the lock lasts
    dialogue  press confirm (no model call; the text is returned so a caller can show it)
    battle    code removes options that break the battle rules, the model picks
              among the rest with the harness battle prompt, and code presses
              the keys (the adapter's own `keys` for an option when it gave them)

Each handler returns a `Handled`: a `ToolRun` that yields the presses, plus the
events and model calls the handling cost. Every press is still a counted
environment step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agents.common import first_json_object
from agents.harness import battle as battle_rules
from agents.harness import prompts, tools
from agents.models import ChatModel, Message, ModelCall


@dataclass
class Handled:
    run: tools.ToolRun
    events: list[dict[str, Any]] = field(default_factory=list)
    calls: list[ModelCall] = field(default_factory=list)
    note: str = ""  # a line worth showing the agent later (the dialogue text, the battle choice)


class Specialists:
    def __init__(self, model: ChatModel, *, busy_wait_limit: int = 6) -> None:
        self.model = model
        self.busy_wait_limit = busy_wait_limit
        self._busy_streak = 0

    def reset(self) -> None:
        self._busy_streak = 0

    def reset_busy(self) -> None:
        self._busy_streak = 0

    def route(self, state: Mapping[str, Any] | None, keys: Mapping[str, str], battle_goal: str = "defeat") -> Handled | None:
        """The specialist for the current context, or None when the agent should decide (overworld, menu)."""
        context = (state or {}).get("context", "overworld")
        if context != "busy":
            self.reset_busy()
        if context == "battle" and state.get("battle"):
            return self.battle(state, keys, battle_goal)
        if context == "dialogue":
            return self.dialogue(state, keys)
        if context == "busy":
            return self.busy(state, keys)
        return None

    def dialogue(self, state: Mapping[str, Any], keys: Mapping[str, str]) -> Handled:
        text = (state.get("dialogue") or {}).get("text", "")
        run = tools.ToolRun("dialogue", "press", {"key": keys["confirm"]}, tools.press(keys["confirm"], 1), state)
        return Handled(run, note=f"Dialogue: {text}" if text else "")

    def busy(self, state: Mapping[str, Any], keys: Mapping[str, str]) -> Handled:
        """The game ignores movement (a transition, cutscene, or undescribed menu).

        Wait if the adapter offers a do-nothing action; if the lock outlasts
        `busy_wait_limit` decisions, press cancel to back out of a menu.
        """
        self._busy_streak += 1
        if "wait" in keys and self._busy_streak <= self.busy_wait_limit:
            key, reason = keys["wait"], "waiting for the game to take input again"
        else:
            key, reason = keys["cancel"], f"still busy after {self._busy_streak - 1} waits; backing out"
            self._busy_streak = 0
        run = tools.ToolRun("busy", "press", {"key": key}, tools.press(key, 1), state)
        return Handled(run, [{"type": "busy", "press": key, "reason": reason}])

    def battle(self, state: Mapping[str, Any], keys: Mapping[str, str], goal: str = "defeat") -> Handled:
        battle = state["battle"]
        events: list[dict[str, Any]] = []
        calls: list[ModelCall] = []
        if not battle.get("options"):
            # messages or animations: nothing to choose, so move them along
            run = tools.ToolRun("battle", "press", {"key": keys["confirm"]}, tools.press(keys["confirm"], 1), state)
            return Handled(run)
        if goal == "catch" and not any(o.get("kind") == "catch" for o in battle["options"]):
            # nothing on this menu catches, so the catch rules would only stall the fight
            events.append({"type": "battle_goal_unavailable", "goal": goal, "using": "defeat"})
            goal = "defeat"
        allowed, removed = battle_rules.allowed_options(battle, goal)
        if not allowed:
            # Confirming here would take whatever the cursor is on, which may be
            # the option a rule just removed. Under catch, back out instead; otherwise
            # (every option out of uses) confirming lets the game pick its fallback.
            key = keys["cancel"] if goal == "catch" else keys["confirm"]
            events.append({"type": "battle", "goal": goal, "removed": removed, "choice": None, "press": key})
            return Handled(tools.ToolRun("battle", "press", {"key": key}, tools.press(key, 1), state), events)
        why = "only legal option"
        choice = allowed[0]["id"]
        if len(allowed) > 1:
            call = self.model.chat(
                [Message("system", prompts.BATTLE), Message("user", battle_rules.choose_prompt(battle, allowed, goal, removed))],
                think=False,
            )
            calls.append(call)
            reply = first_json_object(call.output) or {}
            if reply.get("option") in {o["id"] for o in allowed}:
                choice, why = reply["option"], reply.get("why", "")
            else:
                why = f"reply did not name an allowed option; took {choice}"
        events.append({"type": "battle", "goal": goal, "allowed": [o["id"] for o in allowed], "removed": removed, "choice": choice, "why": why})
        chosen = next(o for o in allowed if o["id"] == choice)
        given = battle_rules.option_keys(chosen)
        if given:
            steps = tools.press_sequence(given)
        else:
            steps = tools.pick_option(battle.get("cursor", 0), battle_rules.option_index(battle, choice), keys)
        return Handled(tools.ToolRun("battle", "choose", {"option": choice}, steps, state), events, calls, note=f"Battle: code chose {choice} ({why})")
