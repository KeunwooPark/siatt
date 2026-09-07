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
"""

from __future__ import annotations

import re
from collections.abc import Container

SCHEME = "siatt://blob/"

#: A Markdown link whose target is one of ours. The digest is matched exactly --
#: sixty-four hex characters and no more -- so a target that merely starts like
#: one is not a reference at all, and never becomes a path.
LINK = re.compile(r"\[(?P<text>[^\]\n]*)\]\(" + re.escape(SCHEME) + r"(?P<sha>[0-9a-f]{64})\)")


def uri(sha256: str) -> str:
    return f"{SCHEME}{sha256}"


def link(sha256: str, text: str) -> str:
    """One reference, as it is written into a memory.

    The text is what a person reads and the digest is what resolves, which is
    why both are here: a bare URI in a Markdown file is sixty-four characters of
    noise in the middle of a sentence.
    """
    return f"[{_safe(text)}]({uri(sha256)})"


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
