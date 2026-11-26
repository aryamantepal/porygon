"""A tiny stand-in for mGBA that lays out FireRed memory the way the adapter reads it."""

from __future__ import annotations

import struct
from pathlib import Path

from environments.protocol import Frame
from environments.pokemon_firered import symbols as sym
from environments.pokemon_firered.buttons import Button

SYMBOLS = sym.TABLES[("BPRE", 0)]
SAVE_BLOCK1 = 0x02025000
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

    def set_badge(self, index: int) -> None:
        flag = sym.FLAG_BADGE01_GET + index
        address = SAVE_BLOCK1 + sym.SB1_FLAGS + flag // 8
        self._write(address, bytes([self.read(address, 1)[0] | 1 << flag % 8]))

    # Emulator

    @property
    def frame_count(self) -> int:
        return self._frames

    def run_frame(self) -> None:
        pressed = self._keys & ~self._held_last_frame
        if pressed & Button.RIGHT.mask:
            x = struct.unpack("<h", self.read(SAVE_BLOCK1 + sym.SB1_POS_X, 2))[0]
            self._write(SAVE_BLOCK1 + sym.SB1_POS_X, struct.pack("<h", x + 1))
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
