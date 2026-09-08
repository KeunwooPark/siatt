"""How much text one Slack message holds, and where to cut one that is longer.

Its own module because it is the one part of egress that is neither the live
rewrite nor the adapter: a rule about Slack, expressible as a string in and a
list of strings out, and testable without a socket or a `LiveMessage`.

The limit is deliberately below anything Slack documents. The published ceiling
on `text` is 40,000 characters, but it is not the only one that applies — a
`chat.update` refuses well before it, and the refusal arrives as an error code
rather than as a truncation, which costs the whole turn (#258). Nothing is
gained by discovering each endpoint's real ceiling: an answer that runs past
3,000 characters reads better in two messages than in one anyway, and this is
the number Slack itself uses for the text of a block.

Cutting is not truncation. Every character goes out; what changes is which
message carries it. The break is taken at the largest structure that fits — a
blank line, then a line, then a word — so a digest of five items splits between
items rather than mid-sentence.
"""

from __future__ import annotations

#: The most one message carries. See the module docstring for why it is not
#: 40,000, and why picking a smaller number costs nothing.
MAX_TEXT = 3000

#: Where a break is welcome, best first. A paragraph boundary is invisible in
#: the reading; a word boundary is merely not ugly.
_BREAKS = ("\n\n", "\n", " ")


def split(text: str, limit: int = MAX_TEXT) -> list[str]:
    """Cut `text` into messages of at most `limit` characters each.

    Empty in, empty out — a caller with nothing to say has nothing to post, and
    Slack will not accept an empty message anyway.
    """
    if limit < 1:
        raise ValueError(f"a message holds at least one character, not {limit}")
    remaining = text.strip()
    parts: list[str] = []
    while remaining:
        if len(remaining) <= limit:
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

    The window is one character wider than the limit so that a break sitting
    exactly at it is still found: that character starts the next message, and
    everything before it fits in this one.
    """
    window = text[: limit + 1]
    for separator in _BREAKS:
        # Not `> -1`: a break at the very start would produce an empty message
        # and leave the text no shorter, which is how this loops forever.
        if (found := window.rfind(separator)) > 0:
            return found
    # One unbroken run longer than a whole message — a URL, or a language that
    # does not put spaces between words. It is cut where it stops fitting.
    return limit
