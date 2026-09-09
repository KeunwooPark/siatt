"""How much text one Slack message holds, and where to cut one that is longer.

Its own module because it is the one part of egress that is neither the live
rewrite nor the adapter: a rule about Slack, expressible as a string in and a
list of strings out, and testable without a socket or a `LiveMessage`.

**The budget is bytes, not characters**, and that distinction is the whole of
#264. Slack's limits are counted over UTF-8, so a Korean sentence spends three
bytes a character where an English one spends one — and a 2,768-character
Korean answer that a character budget of 3,000 called comfortable arrived at
`chat.update` as 5,543 bytes and came back `msg_too_long`. Nothing was lost
(`stream.py` posts the answer instead) but the thread was left holding a
stranded `thinking…` above it, and the live rewrite spent the rest of the turn
being refused once a second.

The number is deliberately below anything Slack documents. The published
ceiling on `text` is 40,000, but it is not the only one that applies — a
`chat.update` refuses well before it, and the refusal arrives as an error code
rather than as a truncation, which costs the whole turn (#258). Nothing is
gained by discovering each endpoint's real ceiling: an answer that runs past a
few thousand bytes reads better in two messages than in one anyway.

Cutting is not truncation. Every character goes out; what changes is which
message carries it. The break is taken at the largest structure that fits — a
blank line, then a line, then a word — so a digest of five items splits between
items rather than mid-sentence. The *search* for that break stays in characters,
because a byte offset into UTF-8 can land inside one; only the question of what
fits is asked in bytes.
"""

from __future__ import annotations

#: The most one message carries, in UTF-8 bytes. Under Slack's 4,000 with room
#: for whatever it counts that we do not. ASCII is unaffected — 3,500 bytes is
#: 3,500 characters — and Korean lands at around 1,160 characters a message,
#: which is a paragraph rather than a fragment.
MAX_BYTES = 3500

#: Where a break is welcome, best first. A paragraph boundary is invisible in
#: the reading; a word boundary is merely not ugly.
_BREAKS = ("\n\n", "\n", " ")


def split(text: str, limit: int = MAX_BYTES) -> list[str]:
    """Cut `text` into messages of at most `limit` UTF-8 bytes each.

    Empty in, empty out — a caller with nothing to say has nothing to post, and
    Slack will not accept an empty message anyway.
    """
    if limit < 1:
        raise ValueError(f"a message holds at least one byte, not {limit}")
    remaining = text.strip()
    parts: list[str] = []
    while remaining:
        if len(remaining.encode()) <= limit:
            parts.append(remaining)
            break
        cut = _cut(remaining, limit)
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    # A part can still come out empty — a run of whitespace at a cut — and an
    # empty message is a refusal rather than a blank line.
    return [part for part in parts if part]


def _cut(text: str, limit: int) -> int:
    """Where to break `text`, given that it does not fit.

    A *character* offset, because that is what slicing a `str` takes and what
    `rfind` returns. Only the ceiling — how much of the text fits in `limit`
    bytes — is computed in bytes, and it is computed by encoding, cutting, and
    decoding back with `errors="ignore"`: truncating UTF-8 mid-character leaves
    an invalid tail, ignoring it drops exactly that partial character, and what
    is left is the longest character prefix that fits.

    The window is one character wider than that ceiling so a break sitting
    exactly on it is still found: that character starts the next message, and
    everything before it fits in this one.
    """
    fits = len(text.encode()[:limit].decode("utf-8", errors="ignore"))
    window = text[: fits + 1]
    for separator in _BREAKS:
        # Not `> -1`: a break at the very start would produce an empty message
        # and leave the text no shorter, which is how this loops forever.
        if (found := window.rfind(separator)) > 0:
            return found
    # One unbroken run longer than a whole message — a URL, or a language that
    # does not put spaces between words. It is cut where it stops fitting.
    #
    # At least one character, whatever the budget says. A limit smaller than
    # the first character's encoding would otherwise make `fits` zero, and a
    # cut of zero is a message with nothing in it and a remainder no shorter —
    # the same infinite loop the `> 0` above avoids, reached the other way.
    return max(fits, 1)
