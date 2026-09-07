"""What a picture is and how big, read from the front of it.

The packer budgets an image by area, and `siatt/llm/tokens.py` turns what this
returns into a number of tokens. Only the measuring is here; what a pixel costs
is a provider fact and belongs beside the other ones.

Headers only, and only the four formats people actually paste into a chat. This
is not an image library and must not become one: it never decodes a pixel, it
reads at most a few hundred bytes, and anything it does not recognize comes back
`None` so the caller can charge the maximum instead. Guessing small is the one
answer that costs a failed turn.
"""

from __future__ import annotations

import struct

#: Enough for every header below. A JPEG's dimensions live in a segment that can
#: sit some way into the file, so this is generous -- but bounded, because the
#: caller may be holding a video.
HEAD_BYTES = 64 * 1024


#: Magic bytes to mime type, for the same four formats and read the same way.
#: The offset matters for WebP, whose marker is twelve bytes in behind a RIFF
#: header, which is why this is a list of (offset, prefix) rather than a dict.
_MAGIC: tuple[tuple[int, bytes, str], ...] = (
    (0, b"\x89PNG\r\n\x1a\n", "image/png"),
    (0, b"\xff\xd8\xff", "image/jpeg"),
    (0, b"GIF87a", "image/gif"),
    (0, b"GIF89a", "image/gif"),
    (8, b"WEBP", "image/webp"),
)


def mime_for(data: bytes) -> str | None:
    """What kind of picture this is, or None when it is not one we read.

    Here rather than beside the code that needed it, because this is the module
    that already knows what each of these files begins with -- and because the
    two questions are asked about the same bytes at the same moment. Reading the
    header twice, in two places, is how they come to disagree.

    The caller is a generation endpoint that returns an image and does not say
    which kind (`siatt/imagen/openai_images.py`). A guess there is not a wrong
    label in a database; it is a picture a vision model later refuses.
    """
    return next(
        (mime for at, prefix, mime in _MAGIC if data[at : at + len(prefix)] == prefix),
        None,
    )


def dimensions(data: bytes) -> tuple[int, int] | None:
    """`(width, height)`, or None when the format is not one we read.

    Never raises. Every branch here is parsing a stranger's file, and a
    malformed header is a reason to fall back to the ceiling rather than to
    fail the fetch that produced it.
    """
    try:
        return _png(data) or _gif(data) or _webp(data) or _jpeg(data)
    except (struct.error, IndexError, ValueError):
        return None


def _png(data: bytes) -> tuple[int, int] | None:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return (int(width), int(height))


def _gif(data: bytes) -> tuple[int, int] | None:
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    width, height = struct.unpack("<HH", data[6:10])
    return (int(width), int(height))


def _webp(data: bytes) -> tuple[int, int] | None:
    """The three WebP encodings, which do not share a header.

    `VP8 ` is lossy, `VP8L` lossless, `VP8X` the extended container. Their
    dimensions are in three different places and two of them are packed into
    bit fields, which is why this is longer than the others.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    kind = data[12:16]
    if kind == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        width, height = struct.unpack("<HH", data[26:30])
        # The top two bits of each are a scaling factor, not size.
        return (int(width) & 0x3FFF, int(height) & 0x3FFF)
    if kind == b"VP8L":
        if data[20:21] != b"\x2f":
            return None
        bits = struct.unpack("<I", data[21:25])[0]
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if kind == b"VP8X":
        # Two little-endian 24-bit values, each one less than the true size.
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return (width, height)
    return None


#: Frame markers that carry the image's size. Everything else in a JPEG is a
#: segment to be skipped over. `SOF4`/`SOF8`/`SOF12` (0xC4, 0xC8, 0xCC) are
#: excluded on purpose: they are Huffman tables and arithmetic-coding
#: conditioning, not frame headers, and reading dimensions out of one gives a
#: confident wrong answer.
_JPEG_FRAME = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _jpeg(data: bytes) -> tuple[int, int] | None:
    if data[:2] != b"\xff\xd8":
        return None
    at = 2
    end = min(len(data), HEAD_BYTES)
    while at + 9 < end:
        if data[at] != 0xFF:
            # Not on a marker boundary. A JPEG whose segments do not chain is
            # one we decline to guess about.
            return None
        marker = data[at + 1]
        if marker == 0xFF:  # fill byte; markers may be padded
            at += 1
            continue
        if marker in _JPEG_FRAME:
            height, width = struct.unpack(">HH", data[at + 5 : at + 9])
            return (int(width), int(height))
        length = struct.unpack(">H", data[at + 2 : at + 4])[0]
        if length < 2:
            return None
        at += 2 + length
    return None
