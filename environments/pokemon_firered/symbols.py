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
