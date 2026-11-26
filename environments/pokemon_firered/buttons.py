from enum import IntEnum


class Button(IntEnum):
    """GBA KEYINPUT bit positions."""

    A = 0
    B = 1
    SELECT = 2
    START = 3
    RIGHT = 4
    LEFT = 5
    UP = 6
    DOWN = 7
    R = 8
    L = 9

    @property
    def mask(self) -> int:
        return 1 << self.value
