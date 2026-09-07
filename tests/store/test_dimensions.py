"""Reading how big a picture is out of the front of it."""

from __future__ import annotations

import struct
import zlib

from siatt.store.dimensions import dimensions


def png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        payload = kind + body
        return struct.pack(">I", len(body)) + payload + struct.pack(">I", zlib.crc32(payload))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00" * 10


def jpeg(width: int, height: int, *, marker: bytes = b"\xc0", pad: bool = False) -> bytes:
    out = b"\xff\xd8"
    out += b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
    if pad:
        out += b"\xff\xff"  # fill bytes are legal between segments
    out += b"\xff" + marker + struct.pack(">H", 17) + b"\x08"
    out += struct.pack(">HH", height, width) + b"\x03" + b"\x00" * 9
    return out


def webp_vp8x(width: int, height: int) -> bytes:
    return (
        b"RIFF"
        + b"\x00" * 4
        + b"WEBP"
        + b"VP8X"
        + b"\x00" * 4
        + b"\x00" * 4
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )


def webp_vp8l(width: int, height: int) -> bytes:
    bits = (width - 1) | ((height - 1) << 14)
    return (
        b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8L" + b"\x00" * 4 + b"\x2f" + struct.pack("<I", bits)
    )


def webp_vp8(width: int, height: int) -> bytes:
    return (
        b"RIFF"
        + b"\x00" * 4
        + b"WEBP"
        + b"VP8 "
        + b"\x00" * 4
        + b"\x00" * 3
        + b"\x9d\x01\x2a"
        + struct.pack("<HH", width, height)
    )


def test_the_formats_people_paste_into_a_chat() -> None:
    assert dimensions(png(37, 11)) == (37, 11)
    assert dimensions(gif(640, 480)) == (640, 480)
    assert dimensions(jpeg(200, 100)) == (200, 100)
    assert dimensions(webp_vp8x(1920, 1080)) == (1920, 1080)
    assert dimensions(webp_vp8l(800, 600)) == (800, 600)
    assert dimensions(webp_vp8(320, 240)) == (320, 240)


def test_a_jpeg_whose_size_is_behind_several_segments() -> None:
    """The dimensions are not at a fixed offset: a real photograph carries EXIF,
    a colour profile and thumbnails before its frame header."""
    out = b"\xff\xd8"
    for _ in range(20):
        out += b"\xff\xe1" + struct.pack(">H", 1002) + b"\x00" * 1000
    out += b"\xff\xc2" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", 3024, 4032)
    out += b"\x03" + b"\x00" * 9

    assert dimensions(out) == (4032, 3024)


def test_fill_bytes_between_segments_are_skipped() -> None:
    assert dimensions(jpeg(200, 100, pad=True)) == (200, 100)


def test_a_huffman_table_is_not_read_as_a_frame_header() -> None:
    """0xC4 sits in the middle of the SOF range and is not one. Reading
    dimensions out of it gives a confident wrong answer."""
    out = b"\xff\xd8"
    out += b"\xff\xc4" + struct.pack(">H", 20) + b"\x00" * 18  # DHT
    out += b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", 480, 640)
    out += b"\x03" + b"\x00" * 9

    assert dimensions(out) == (640, 480)


def test_formats_we_do_not_read_come_back_unknown() -> None:
    """Not a failure. The caller charges the token ceiling for these, which is
    the answer that cannot cost a turn."""
    assert dimensions(b"\x00\x00\x00 ftypheic" + b"\x00" * 40) is None
    assert dimensions(b"\x00\x00\x00 ftypmp42" + b"\x00" * 40) is None
    assert dimensions(b"") is None
    assert dimensions(b"nothing recognizable at all") is None


def test_a_truncated_header_does_not_raise() -> None:
    """Every branch is parsing a stranger's file. A malformed header is a
    reason to fall back, never to fail the fetch that produced it."""
    for data in (png(10, 10)[:12], gif(1, 1)[:8], jpeg(1, 1)[:6], webp_vp8l(1, 1)[:22]):
        assert dimensions(data) is None


def test_a_jpeg_with_a_nonsense_segment_length_stops() -> None:
    broken = b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 0) + b"\x00" * 40

    assert dimensions(broken) is None
