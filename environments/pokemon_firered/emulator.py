"""The emulator surface the adapter needs, and its mGBA implementation.

The adapter talks only to `Emulator`. Tests swap in a fake; real runs use
`MgbaEmulator`, which drives libmgba's Python bindings in process. The game
advances only when `run_frame` is called, so frame counts are exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from environments.protocol import Frame

ROM_GAME_CODE = 0x080000AC  # 4 ASCII bytes in the cartridge header
ROM_REVISION = 0x080000BC  # u8


class Emulator(Protocol):
    @property
    def frame_count(self) -> int: ...

    def run_frame(self) -> None: ...

    def set_keys(self, mask: int) -> None: ...

    def read(self, address: int, size: int) -> bytes: ...

    def screen(self) -> Frame: ...

    def save_state(self) -> bytes: ...

    def load_state(self, state: bytes) -> None: ...


def rom_identity(emulator: Emulator) -> tuple[str, int]:
    code = emulator.read(ROM_GAME_CODE, 4).decode("ascii")
    revision = emulator.read(ROM_REVISION, 1)[0]
    return code, revision


class MgbaEmulator:
    """libmgba via its Python bindings (mGBA built with -DBUILD_PYTHON=ON)."""

    def __init__(self, rom_path: Path) -> None:
        try:
            import mgba.core
            import mgba.image
        except ImportError as error:
            raise RuntimeError(
                "mGBA Python bindings are not installed; build mGBA with "
                "-DBUILD_PYTHON=ON and install the `mgba` package"
            ) from error

        _silence_mgba_log()
        core = mgba.core.load_path(str(rom_path))
        if core is None:
            raise RuntimeError(f"mGBA could not load {rom_path}")
        width, height = core.desired_video_dimensions()
        self._image = mgba.image.Image(width, height)
        core.set_video_buffer(self._image)
        core.reset()
        self._core = core

    @property
    def frame_count(self) -> int:
        return self._core.frame_counter

    def run_frame(self) -> None:
        self._core.run_frame()

    def set_keys(self, mask: int) -> None:
        self._core.set_keys(*(bit for bit in range(16) if mask >> bit & 1))

    def read(self, address: int, size: int) -> bytes:
        u8 = self._core.memory.u8
        return bytes(u8[address + offset] for offset in range(size))

    def screen(self) -> Frame:
        from mgba._pylib import ffi

        image = self._image
        row_bytes = image.width * 4
        raw = bytes(ffi.buffer(image.buffer, image.stride * image.height * 4))
        rows = bytearray()
        for y in range(image.height):
            row = bytearray(raw[y * image.stride * 4 : y * image.stride * 4 + row_bytes])
            row[3::4] = b"\xff" * image.width  # native format is RGBX
            rows += row
        return Frame(image.width, image.height, bytes(rows))

    # mGBA's raw state leaves out the video buffer, so a load would show
    # whatever was drawn last. The buffer rides at the end of the state.

    def save_state(self) -> bytes:
        return bytes(self._core.save_raw_state()) + bytes(self._video_buffer())

    def load_state(self, state: bytes) -> None:
        video = self._video_buffer()
        split = len(state) - len(video)
        if split <= 0 or not self._core.load_raw_state(state[:split]):
            raise RuntimeError("mGBA rejected the save state")
        video[:] = state[split:]

    def _video_buffer(self):
        from mgba._pylib import ffi

        image = self._image
        return ffi.buffer(image.buffer, image.stride * image.height * 4)


_log_silencer = None


def _silence_mgba_log() -> None:
    """Drop all mGBA log lines with a C-side filter.

    `mgba.log.silence()` routes each line through a Python callback whose
    signature carries a `va_list` cffi cannot express, so it raises on every
    line. A filter with every level off stops lines before that callback.
    """
    global _log_silencer
    if _log_silencer is not None:
        return
    from mgba._pylib import ffi, lib

    log_filter = ffi.new("struct mLogFilter*")
    lib.mLogFilterInit(log_filter)
    log_filter.defaultLevels = 0
    logger = lib.mLoggerPythonCreate(ffi.NULL)
    # struct mLogger is { log, filter }; touching its fields from cffi trips
    # on the va_list in `log`, so set `filter` through a raw pointer.
    ffi.cast("void**", logger)[1] = log_filter
    lib.mLogSetDefaultLogger(logger)
    _log_silencer = (log_filter, logger)
