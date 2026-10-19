"""Minimal PNG writer so the renderer has no image-library dependency."""

import struct
import zlib


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def write_png(path: str, pixels) -> None:
    """Write an (H, W, 3) uint8 array as a truecolour PNG."""
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("expected an (H, W, 3) RGB array")

    raw = bytearray()
    data = pixels.tobytes()
    stride = width * 3
    for row in range(height):
        raw.append(0)  # filter type 0 (None)
        raw += data[row * stride : (row + 1) * stride]

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    blob = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(blob)
