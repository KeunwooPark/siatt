"""How a memory points at a file, and how a fabricated pointer is caught.

Long-term memory is Markdown a person reads on GitHub, and an attachment does
not belong in it any more than a video belongs in a git repository (§4.1.1). It
belongs the way a link belongs: `[shot.png](siatt://blob/<sha256>)`, an
ordinary Markdown link so the corpus stays a corpus, with a scheme of our own so
nothing mistakes it for a URL to fetch.

**The reference is not trusted because a model wrote it.** Consolidation reads
memory files and writes memory files, so once one of these exists in the corpus
the model has seen the shape -- and a model that has seen the shape will
cheerfully compose a plausible sixty-four hex characters for a picture that was
never sent. That is not a broken link a person can puzzle out; it is a
confident claim about evidence, sitting in a file somebody trusts, pointing at
nothing.

So every reference in a plan is checked against the attachments table before
the patch is applied, and an unknown one is *unwrapped* rather than rejected:
the link becomes its own text and the prose survives. Rejecting the plan would
throw away a memory worth keeping over one hallucinated hash, and the sentence
around it is usually true even when the pointer is not.

There is a second way to point at one, and it is here because it is the same
pointer at a different moment. A model that is going to *cite* an attachment has
first to be able to *name* one, and the digest is sixty-four characters it has
no reason to get right. So a note shows it a short handle, `handle` decides how
short, and `matching` is the only way back -- which means the model names a
candidate and this module resolves it, rather than the model supplying a digest
anything downstream would have to believe.
"""

from __future__ import annotations

import re
from collections.abc import Container, Iterable

SCHEME = "siatt://blob/"

#: How much of a digest names a blob in front of a model.
#:
#: Twelve characters rather than sixty-four, for the reason `_render` gives for
#: numbering transcript lines instead of quoting message ULIDs
#: (`siatt/runner/episodes.py`): a long string a model must copy exactly is a
#: citation that fails on a typo, and a citation that fails silently is worse
#: than no citation at all. Twelve is what `_attachment_note` has always shown,
#: and it does not collide inside one conversation.
HANDLE_CHARS = 12

#: The shortest prefix accepted back. Below this a handle stops naming one
#: attachment and starts naming whichever one sorted first.
MIN_HANDLE_CHARS = 8

#: A Markdown link whose target is one of ours. The digest is matched exactly --
#: sixty-four hex characters and no more -- so a target that merely starts like
#: one is not a reference at all, and never becomes a path.
LINK = re.compile(r"\[(?P<text>[^\]\n]*)\]\(" + re.escape(SCHEME) + r"(?P<sha>[0-9a-f]{64})\)")

#: A handle is hex or it is not a handle. Checked before the prefix match so a
#: name somebody typed can never be answered with "no such attachment" *and*
#: scanned against every digest in the conversation.
_HEX = re.compile(r"[0-9a-f]+")


def uri(sha256: str) -> str:
    return f"{SCHEME}{sha256}"


def link(sha256: str, text: str) -> str:
    """One reference, as it is written into a memory.

    The text is what a person reads and the digest is what resolves, which is
    why both are here: a bare URI in a Markdown file is sixty-four characters of
    noise in the middle of a sentence.
    """
    return f"[{_safe(text)}]({uri(sha256)})"


def handle(sha256: str) -> str:
    """What an attachment is called in front of a model.

    The digest, shortened. A model is asked to *name* an attachment in two
    places -- a `memory_write` argument and an extraction -- and in both the
    thing it is naming was described to it by a note somebody else composed. So
    the short form is fixed here rather than at each of those, and `matching`
    below is the only thing that turns one back into a digest.
    """
    return sha256[:HANDLE_CHARS]


def matching(text: str, known: Iterable[str]) -> list[str]:
    """Every digest `text` could be naming. Empty when it names none.

    A list rather than one digest, because "no such attachment" and "which of
    these two" are different answers and the caller can say so. Nothing here
    decides what to do about either: this resolves, and refusing is the tool's
    job, where there is somebody to tell.

    Tolerant of the shapes the same handle is written in elsewhere. A model that
    read `a1b2c3d4e5f6… - image/jpeg` off an attachment note will sometimes copy
    the ellipsis with it, and one that has seen a reference in a memory file
    will sometimes hand back the whole `siatt://blob/...` URI. Both name the
    blob unambiguously, and rejecting them would be pedantry with a photograph
    at stake.
    """
    wanted = text.strip().lower().removeprefix(SCHEME).rstrip(".\u2026").strip()
    if len(wanted) < MIN_HANDLE_CHARS or not _HEX.fullmatch(wanted):
        return []
    return sorted(sha for sha in known if sha.startswith(wanted))


def referenced(body: str) -> set[str]:
    """Every blob this text points at."""
    return {match["sha"] for match in LINK.finditer(body)}


def prune(body: str, known: Container[str]) -> tuple[str, set[str]]:
    """Unwrap references to blobs that do not exist. Returns the text and what went.

    Unwrapped, not deleted: `[a photo of the whiteboard](siatt://blob/...)`
    becomes `a photo of the whiteboard`. What the model claimed to have seen may
    well be true; what it cannot be allowed to do is leave a pointer to evidence
    that is not there.
    """
    dropped: set[str] = set()

    def unwrap(match: re.Match[str]) -> str:
        if match["sha"] in known:
            return match[0]
        dropped.add(match["sha"])
        return match["text"]

    return LINK.sub(unwrap, body), dropped


def _safe(text: str) -> str:
    """Link text that cannot end the link early.

    The name came off an upload, so somebody else chose it. A `]` in it would
    close the bracket and leave the rest of the URI as prose -- which is a
    broken document rather than an exploit, but a broken document in a file the
    corpus is supposed to be readable as.
    """
    flat = " ".join(text.split())
    return flat.replace("[", "(").replace("]", ")") or "an attachment"
