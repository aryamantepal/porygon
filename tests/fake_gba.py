"""A tiny stand-in for mGBA that lays out FireRed memory the way the adapter reads it."""

from __future__ import annotations

import struct
from pathlib import Path

from environments.protocol import Frame
from environments.pokemon_firered import symbols as sym
from environments.pokemon_firered.buttons import Button

SYMBOLS = sym.TABLES[("BPRE", 0)]
SAVE_BLOCK1 = 0x02025000
MAP_EVENTS = 0x02030000  # struct MapEvents, then one warp
MAP_CELLS = 0x02030100  # collision cells for a 5x3 map plus the 7-tile border
MAP_W, MAP_H = 5, 3
EWRAM, IWRAM = 0x02000000, 0x03000000
WIDTH, HEIGHT = 240, 160


class FakeGBA:
    def __init__(self, rom_path: Path, game_code: str = "BPRE", revision: int = 0) -> None:
        self._header = bytearray(0xC0)
        self._header[0xAC:0xB0] = game_code.encode("ascii")
        self._header[0xBC] = revision
        self._ewram = bytearray(0x40000)
        self._iwram = bytearray(0x8000)
        self._frames = 0
        self._keys = 0
        self._held_last_frame = 0
        self._write(SYMBOLS.save_block1_ptr, struct.pack("<I", SAVE_BLOCK1))
        self._write(SAVE_BLOCK1 + sym.SB1_MAP_GROUP, bytes([3, 0]))
        self._write(SAVE_BLOCK1 + sym.SB1_POS_X, struct.pack("<hh", 5, 7))
        self._write(SYMBOLS.player_party_count, bytes([1]))
        party = SYMBOLS.player_party
        self._write(party + sym.MON_LEVEL, bytes([5]))
        self._write(party + sym.MON_HP, struct.pack("<HH", 19, 20))
        # one warp at (1, 0) to map 3.1
        self._write(SYMBOLS.map_header + sym.MAP_EVENTS, struct.pack("<I", MAP_EVENTS))
        self._write(MAP_EVENTS, bytes([0, 1, 0, 0]) + struct.pack("<II", 0, MAP_EVENTS + 0x20))
        self._write(MAP_EVENTS + 0x20, struct.pack("<hhBBBB", 1, 0, 0, 0, 1, 3))
        # collision: the top-left tile of the map is a wall
        width, height = MAP_W + 2 * sym.MAP_OFFSET, MAP_H + 2 * sym.MAP_OFFSET
        self._write(SYMBOLS.backup_map_layout, struct.pack("<iiI", width, height, MAP_CELLS))
        wall = (sym.MAP_OFFSET * width + sym.MAP_OFFSET) * 2
        self._write(MAP_CELLS + wall, struct.pack("<H", 1 << sym.COLLISION_SHIFT))
        # the player object, mirroring SaveBlock1's position
        self._write(SYMBOLS.object_events, struct.pack("<I", 1))
        self._write(SYMBOLS.object_events + sym.OBJ_LOCAL_ID, bytes([sym.PLAYER_LOCAL_ID]))
        self._sync_player_object()

    def set_badge(self, index: int) -> None:
        flag = sym.FLAG_BADGE01_GET + index
        address = SAVE_BLOCK1 + sym.SB1_FLAGS + flag // 8
        self._write(address, bytes([self.read(address, 1)[0] | 1 << flag % 8]))

    def _sync_player_object(self) -> None:
        x, y = struct.unpack("<hh", self.read(SAVE_BLOCK1 + sym.SB1_POS_X, 4))
        base = SYMBOLS.object_events
        self._write(base + sym.OBJ_COORDS, struct.pack("<hh", x + sym.MAP_OFFSET, y + sym.MAP_OFFSET))
        self._write(base + sym.OBJ_FACING, bytes([2]))

    # Emulator

    @property
    def frame_count(self) -> int:
        return self._frames

    def run_frame(self) -> None:
        pressed = self._keys & ~self._held_last_frame
        if pressed & Button.RIGHT.mask:
            x = struct.unpack("<h", self.read(SAVE_BLOCK1 + sym.SB1_POS_X, 2))[0]
            self._write(SAVE_BLOCK1 + sym.SB1_POS_X, struct.pack("<h", x + 1))
            self._sync_player_object()
        self._held_last_frame = self._keys
        self._frames += 1

    def set_keys(self, mask: int) -> None:
        self._keys = mask

    def read(self, address: int, size: int) -> bytes:
        region, offset = self._region(address)
        return bytes(region[offset : offset + size])

    def screen(self) -> Frame:
        x = self.read(SAVE_BLOCK1 + sym.SB1_POS_X, 1)[0]
        return Frame(WIDTH, HEIGHT, bytes([x, 0, 0, 255]) * (WIDTH * HEIGHT))

    def save_state(self) -> bytes:
        return struct.pack("<QII", self._frames, self._keys, self._held_last_frame) + self._ewram + self._iwram

    def load_state(self, state: bytes) -> None:
        self._frames, self._keys, self._held_last_frame = struct.unpack_from("<QII", state)
        body = state[16:]
        self._ewram[:] = body[: len(self._ewram)]
        self._iwram[:] = body[len(self._ewram) :]

    def _write(self, address: int, data: bytes) -> None:
        region, offset = self._region(address)
        region[offset : offset + len(data)] = data

    def _region(self, address: int) -> tuple[bytearray, int]:
        if address >= 0x08000000:
            return self._header, address - 0x08000000
        if address >= IWRAM:
            return self._iwram, address - IWRAM
        return self._ewram, address - EWRAM
