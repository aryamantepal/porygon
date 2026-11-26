"""Pokémon FireRed as an `Environment`. The only code that knows the game."""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from environments.protocol import Action, Observation, ObservationMode, Press, Progress, StepResult
from environments.pokemon_firered import symbols as sym
from environments.pokemon_firered.buttons import Button
from environments.pokemon_firered.emulator import Emulator, MgbaEmulator, rom_identity


@dataclass(frozen=True)
class FireRedConfig:
    rom_path: Path
    initial_state: Path  # a file written by `FireRedEnv.save`, under saves/
    mode: ObservationMode
    hold_frames: int = 8  # frames a tap holds the button
    release_frames: int = 16  # frames after release, so the game reads the input


class FireRedEnv:
    def __init__(
        self,
        config: FireRedConfig,
        make_emulator: Callable[[Path], Emulator] = MgbaEmulator,
    ) -> None:
        self.config = config
        self._emulator = make_emulator(config.rom_path)
        self._identity = rom_identity(self._emulator)
        self._symbols = sym.lookup(*self._identity)

    # Environment

    def reset(self) -> Observation:
        self.load(self.config.initial_state)
        self._emulator.set_keys(0)
        return self.observe()

    def observe(self) -> Observation:
        mode = self.config.mode
        return Observation(
            mode=mode,
            frame_count=self._emulator.frame_count,
            frame=self._emulator.screen() if mode.has_frame else None,
            state=self._state() if mode.has_state else None,
        )

    def step(self, action: Action) -> StepResult:
        button = self._button(action)
        hold = action.hold_frames if action.hold_frames is not None else self.config.hold_frames
        if hold < 1:
            raise ValueError("hold_frames must be at least 1")

        started = time.perf_counter()
        self._emulator.set_keys(button.mask)
        self._advance(hold)
        self._emulator.set_keys(0)
        self._advance(self.config.release_frames)
        observation = self.observe()
        return StepResult(
            observation=observation,
            action=action,
            frames_advanced=hold + self.config.release_frames,
            wall_seconds=time.perf_counter() - started,
        )

    def get_progress(self) -> Progress:
        location = self._location()
        badges = self._badges()
        return {
            "map": [location["map_group"], location["map_num"]],
            "x": location["x"],
            "y": location["y"],
            "badges": badges,
            "badge_count": sum(badges),
        }

    def legal_actions(self) -> Sequence[Action]:
        return [Press(button.name) for button in Button]

    # Adapter operations, not called by agents

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._emulator.save_state())
        game_code, revision = self._identity
        _meta_path(path).write_text(json.dumps({"game_code": game_code, "revision": revision}))

    # Internals

    def load(self, path: Path) -> None:
        meta = json.loads(_meta_path(path).read_text())
        saved = (meta["game_code"], meta["revision"])
        if saved != self._identity:
            raise ValueError(f"{path} was saved under {saved}, but the running ROM is {self._identity}")
        self._emulator.load_state(path.read_bytes())

    def _advance(self, frames: int) -> None:
        for _ in range(frames):
            self._emulator.run_frame()

    def _button(self, action: Action) -> Button:
        if not isinstance(action, Press) or action.key not in Button.__members__:
            raise ValueError(f"not a FireRed action: {action!r}")
        return Button[action.key]

    def _state(self) -> dict[str, Any]:
        return {**self._location(), "badges": self._badges(), "party": self._party(), **self._roles()}

    def _roles(self) -> dict[str, Any]:
        """Role-snapshot fields (environments/roles.py): what a harness can plan and walk with.

        `context` is not filled yet: dialogue, menu, and battle detection still
        need verified reads, and a wrong context is worse than none.
        """
        location = self._location()
        objects = self._object_events()
        player = next((o for o in objects if o["local_id"] == sym.PLAYER_LOCAL_ID), None)
        grid = self._grid()
        places = [
            {"id": f"door_{index}", "role": "door", "position": [x, y], "leads_to": f"{group}.{num}"}
            for index, (x, y, group, num) in enumerate(self._warps())
        ]
        for obj in objects:
            if obj["local_id"] == sym.PLAYER_LOCAL_ID:
                continue
            places.append({"id": f"object_{obj['local_id']}", "role": "object", "position": obj["position"]})
            x, y = obj["position"]
            if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
                grid[y] = grid[y][:x] + "#" + grid[y][x + 1 :]
        return {
            "map": f"{location['map_group']}.{location['map_num']}",
            "position": player["position"] if player else [location["x"], location["y"]],
            "facing": player["facing"] if player else None,
            "grid": grid,
            "walkable": ".",
            "places": places,
            "controls": {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT", "confirm": "A", "cancel": "B"},
        }

    def _object_events(self) -> list[dict[str, Any]]:
        objects = []
        for index in range(sym.OBJECT_EVENT_COUNT):
            base = self._symbols.object_events + index * sym.OBJECT_EVENT_SIZE
            if not self._read_u32(base) & 1:
                continue
            x, y = struct.unpack("<hh", self._emulator.read(base + sym.OBJ_COORDS, 4))
            objects.append(
                {
                    "local_id": self._read_u8(base + sym.OBJ_LOCAL_ID),
                    "graphics": self._read_u8(base + sym.OBJ_GRAPHICS),
                    "position": [x - sym.MAP_OFFSET, y - sym.MAP_OFFSET],
                    "facing": sym.FACING.get(self._read_u8(base + sym.OBJ_FACING) & 0xF),
                }
            )
        return objects

    def _warps(self) -> list[tuple[int, int, int, int]]:
        events = self._read_u32(self._symbols.map_header + sym.MAP_EVENTS)
        count = self._read_u8(events + sym.EVENTS_WARP_COUNT)
        table = self._read_u32(events + sym.EVENTS_WARPS)
        warps = []
        for index in range(count):
            x, y, _, _, num, group = struct.unpack("<hhBBBB", self._emulator.read(table + index * sym.WARP_SIZE, sym.WARP_SIZE))
            warps.append((x, y, group, num))
        return warps

    def _grid(self) -> list[str]:
        """Collision for the current map, without the layout's border."""
        width, height, pointer = struct.unpack("<iiI", self._emulator.read(self._symbols.backup_map_layout, 12))
        cells = self._emulator.read(pointer, width * height * 2)
        rows = []
        for y in range(sym.MAP_OFFSET, height - sym.MAP_OFFSET):
            row = []
            for x in range(sym.MAP_OFFSET, width - sym.MAP_OFFSET):
                value = struct.unpack_from("<H", cells, (y * width + x) * 2)[0]
                row.append("#" if value >> sym.COLLISION_SHIFT & sym.COLLISION_MASK else ".")
            rows.append("".join(row))
        return rows

    def _read_u8(self, address: int) -> int:
        return self._emulator.read(address, 1)[0]

    def _read_u16(self, address: int) -> int:
        return struct.unpack("<H", self._emulator.read(address, 2))[0]

    def _read_s16(self, address: int) -> int:
        return struct.unpack("<h", self._emulator.read(address, 2))[0]

    def _read_u32(self, address: int) -> int:
        return struct.unpack("<I", self._emulator.read(address, 4))[0]

    def _save_block1(self) -> int:
        return self._read_u32(self._symbols.save_block1_ptr)

    def _location(self) -> dict[str, int]:
        base = self._save_block1()
        return {
            "map_group": struct.unpack("b", self._emulator.read(base + sym.SB1_MAP_GROUP, 1))[0],
            "map_num": struct.unpack("b", self._emulator.read(base + sym.SB1_MAP_NUM, 1))[0],
            "x": self._read_s16(base + sym.SB1_POS_X),
            "y": self._read_s16(base + sym.SB1_POS_Y),
        }

    def _badges(self) -> list[bool]:
        flags = self._save_block1() + sym.SB1_FLAGS
        result = []
        for index in range(sym.NUM_BADGES):
            flag = sym.FLAG_BADGE01_GET + index
            result.append(bool(self._read_u8(flags + flag // 8) >> (flag % 8) & 1))
        return result

    def _party(self) -> list[dict[str, int]]:
        count = min(self._read_u8(self._symbols.player_party_count), sym.PARTY_SIZE)
        party = []
        for slot in range(count):
            base = self._symbols.player_party + slot * sym.MON_SIZE
            party.append(
                {
                    "level": self._read_u8(base + sym.MON_LEVEL),
                    "hp": self._read_u16(base + sym.MON_HP),
                    "max_hp": self._read_u16(base + sym.MON_MAX_HP),
                }
            )
        return party


def _meta_path(state_path: Path) -> Path:
    return state_path.with_name(state_path.name + ".json")
