"""Decode FireRed's in-game text encoding (pret: charmap.txt), the subset names and dialogue use."""

END = 0xFF
# 0xFA, 0xFB scroll or start a new box; 0xFE is a line break. All read as a space.
_TABLE = {0xFA: " ", 0xFB: " ", 0xFE: " ", 0x00: " ", 0xAB: "!", 0xAC: "?", 0xAD: ".", 0xAE: "-", 0xB0: "…", 0xB4: "'", 0xB8: ",", 0xBA: "/", 0xF0: ":", 0x1B: "é"}
_TABLE.update({0xA1 + i: str(i) for i in range(10)})
_TABLE.update({0xBB + i: chr(ord("A") + i) for i in range(26)})
_TABLE.update({0xD5 + i: chr(ord("a") + i) for i in range(26)})


def decode(raw: bytes) -> str:
    out = []
    for byte in raw:
        if byte == END:
            break
        out.append(_TABLE.get(byte, "?"))
    return "".join(out)
