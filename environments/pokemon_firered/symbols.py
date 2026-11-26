"""Name -> address tables, one per ROM revision. A table, not logic.

Names follow pret/pokefirered. Struct offsets come from its headers and do
not change between revisions; the addresses do.

The BPRE revision 0 addresses below are the commonly cited values and are
UNVERIFIED until tests/test_firered_rom.py passes against a real ROM. Flip
`verified` only after that test passes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Symbols:
    game_code: str
    revision: int
    verified: bool
    # IWRAM pointers to the save blocks; the blocks move, so read through these.
    save_block1_ptr: int
    save_block2_ptr: int
    # EWRAM
    player_party_count: int
    player_party: int
    # Found by searching memory on BPRE rev 0 and checked against walked
    # positions, the bedroom walls, and the stairs warp (see test_firered_rom.py).
    object_events: int  # gObjectEvents, EWRAM
    map_header: int  # gMapHeader, EWRAM
    backup_map_layout: int  # gBackupMapLayout, IWRAM
    main_flags: int  # the gMain byte holding inBattle (bit 1); checked through the rival battle
    battle_mons: int  # gBattleMons, EWRAM; battler 0 is ours, 1 the opponent
    battle_type_flags: int  # gBattleTypeFlags; only meaningful during a battle (it is not cleared after)
    battle_moves: int  # gBattleMoves, ROM; found by matching Tackle's and Growl's records
    move_names: int  # gMoveNames, ROM; found by matching "TACKLE" and "GROWL"
    string_var4: int  # gStringVar4, EWRAM; the message being printed (checked on two dialogues)
    # Found in the rival battle by moving each cursor through every slot:
    action_cursor: int  # gActionSelectionCursor[0]: 0 FIGHT, 1 BAG, 2 POKéMON, 3 RUN (2x2 grid)
    move_cursor: int  # gMoveSelectionCursor[0]: move slots 0-3 (2x2 grid)
    controller_func: int  # gBattlerControllerFuncs[0]: which input handler our battler is in
    # Unnamed bytes picked by diffing memory between overworld, dialogue, and
    # Start-menu states, then checked across two dialogues in two maps:
    text_window: int  # 2 while a field text box is up, including its last line
    input_locked: int  # 1 while the player cannot move: text, menus, warps, cutscenes


# Offsets inside SaveBlock1 (include/global.h).
SB1_POS_X = 0x0000  # s16
SB1_POS_Y = 0x0002  # s16
SB1_MAP_GROUP = 0x0004  # s8, inside `location`
SB1_MAP_NUM = 0x0005  # s8
SB1_FLAGS = 0x0EE0  # u8[]

# Flag ids (include/constants/flags.h).
FLAG_BADGE01_GET = 0x820
NUM_BADGES = 8

# struct Pokemon is 100 bytes; these fields sit outside the encrypted part.
PARTY_SIZE = 6
MON_SIZE = 100
MON_LEVEL = 0x54  # u8
MON_HP = 0x56  # u16
MON_MAX_HP = 0x58  # u16

# struct ObjectEvent (0x24 bytes): flags u32 (bit 0 active), graphicsId u8 at 0x5,
# localId u8 at 0x8 (0xFF is the player), currentCoords s16 x/y at 0x10,
# facingDirection in the low nibble at 0x18 (1 south, 2 north, 3 west, 4 east).
OBJECT_EVENT_COUNT = 16
OBJECT_EVENT_SIZE = 0x24
OBJ_GRAPHICS = 0x05
OBJ_LOCAL_ID = 0x08
OBJ_COORDS = 0x10
OBJ_FACING = 0x18
PLAYER_LOCAL_ID = 0xFF
FACING = {1: "DOWN", 2: "UP", 3: "LEFT", 4: "RIGHT"}

IN_BATTLE_BIT = 0x02
BATTLE_TYPE_TRAINER = 0x08

# struct BattlePokemon (0x58 bytes)
BATTLER_SIZE = 0x58
BM_MOVES = 0x0C  # u16[4]
BM_PP = 0x24  # u8[4]
BM_HP = 0x28  # u16
BM_LEVEL = 0x2A  # u8
BM_MAX_HP = 0x2C  # u16
# struct BattleMove (12 bytes): effect, power, type, accuracy, pp, ...
MOVE_SIZE = 12
MOVE_POWER = 1
MOVE_NAME_SIZE = 13
MESSAGE_MAX = 1000
# Values of controller_func on BPRE rev 0 (Thumb addresses of the input handlers)
CHOOSE_ACTION = 0x080E763D
CHOOSE_MOVE = 0x080E7989
ACTION_FIGHT, ACTION_RUN = 0, 3
TEXT_WINDOW_OPEN = 2
INPUT_LOCKED = 1

# struct MapHeader: mapLayout ptr at 0x0, events ptr at 0x4.
# struct MapEvents: objectEventCount, warpCount, coordEventCount, bgEventCount (u8),
# then pointers; warps at 0x8. struct WarpEvent (8 bytes): x s16, y s16,
# elevation u8, warpId u8, mapNum u8, mapGroup u8.
MAP_EVENTS = 0x4
EVENTS_WARP_COUNT = 0x1
EVENTS_WARPS = 0x8
WARP_SIZE = 8

# struct BackupMapLayout: width s32, height s32, map u16*. Each u16 is
# metatile (bits 0-9), collision (bits 10-11), elevation (bits 12-15).
# The layout carries a 7-tile border on every side (MAP_OFFSET).
MAP_OFFSET = 7
COLLISION_SHIFT = 10
COLLISION_MASK = 0x3


TABLES: dict[tuple[str, int], Symbols] = {
    ("BPRE", 0): Symbols(
        game_code="BPRE",
        revision=0,
        verified=False,
        save_block1_ptr=0x03005008,
        save_block2_ptr=0x0300500C,
        player_party_count=0x02024029,
        player_party=0x02024284,
        object_events=0x02036E38,
        map_header=0x02036DFC,
        backup_map_layout=0x03005040,
        main_flags=0x03003529,
        battle_mons=0x02023BE4,
        battle_type_flags=0x02022B4C,
        battle_moves=0x08250C04,
        move_names=0x08247094,
        string_var4=0x02021D18,
        action_cursor=0x02023FF8,
        move_cursor=0x02023FFC,
        controller_func=0x03004FE0,
        text_window=0x03000EB1,
        input_locked=0x03000F9C,
    ),
}


def lookup(game_code: str, revision: int) -> Symbols:
    try:
        return TABLES[(game_code, revision)]
    except KeyError:
        raise LookupError(
            f"no symbol table for {game_code} revision {revision}; "
            "resolve one from pret/pokefirered before using structured mode"
        ) from None
