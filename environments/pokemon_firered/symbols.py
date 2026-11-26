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


TABLES: dict[tuple[str, int], Symbols] = {
    ("BPRE", 0): Symbols(
        game_code="BPRE",
        revision=0,
        verified=False,
        save_block1_ptr=0x03005008,
        save_block2_ptr=0x0300500C,
        player_party_count=0x02024029,
        player_party=0x02024284,
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
