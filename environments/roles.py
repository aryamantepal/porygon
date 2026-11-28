"""The role snapshot: optional structured-state fields described by role, not by game.

An adapter that fills these lets a harness plan, navigate, and fight without
knowing the game. The harness sees "a door" or "a milestone", never a game's
own names for things. Every field is optional; a player must cope with any of
them missing.

    context    "overworld" | "dialogue" | "menu" | "battle" | "busy"
               what the game is waiting for right now; "busy" means it is not
               taking movement input (a transition, a cutscene, or a menu the
               adapter cannot describe)
    map        an id for the current map; it changes when the player goes through a door
    position   [x, y] of the player on the current map
    facing     "UP" | "DOWN" | "LEFT" | "RIGHT"
    grid       list of equal-length strings, one per row of the current map:
                 "."  walkable     "#"  blocked
               (an adapter may use other characters; `walkable` says which)
    walkable   string of grid characters the player can step on (default ".")
    places     list of {"id", "role", "position", "note"?}, things on the map by role:
                 role "door"       leads elsewhere; "leads_to" names where
                 role "person"     someone to talk to (stand next to, face, press the talk key)
                 role "milestone"  something that marks progress
                 role "item"       something to pick up
                 role "object"     something on the map that blocks the way and may
                                   react to the talk key (the adapter cannot tell more)
               ids are stable within a run so a planner can name them
    dialogue   {"text"} while context is "dialogue"
    menu       {"prompt", "options": [str], "cursor": int} while context is "menu"
    battle     while context is "battle":
                 {"options": [{"id", "kind", "uses_left"?, "power"?}], "cursor": int,
                  "opponent": {"hp", "max_hp", "catchable"?}}
                 kind is "attack", "catch", "item", or "flee"
               an adapter that cannot yet drive the battle menus may instead fill
                 {"active": {"hp", "max_hp", "level"}, "moves": [{"id", "kind", "power", "uses_left"}],
                  "opponent": {...}}
               which describes the fight without `options`; a harness then cannot
               pick a move and only confirms

Keys the harness uses to act come from `legal_actions()`. The adapter says
which button does what through `controls` (all optional):

    controls   {"up", "down", "left", "right", "confirm", "cancel", "wait"} -> action key
               ("wait" is an action that presses nothing, if the adapter has one)
"""

DEFAULT_WALKABLE = "."
DEFAULT_CONTROLS = {
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "confirm": "A",
    "cancel": "B",
}
