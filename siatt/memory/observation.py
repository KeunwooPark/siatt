"""A candidate fact, before anything durable has happened to it.

An observation is the unit that crosses from short-term to long-term memory:
`episode_close` extracts them from a conversation, `promote` reconciles them
against the corpus and turns them into a patch plan. Between those two it is a
row in SQLite that nobody has committed to believing yet.

The draft type is here rather than in the store because two callers build one —
the `memory_write` tool and the extractor — and both have to agree on what the
fields mean. `scope` in particular: it is inherited from the session and never
chosen by a model, and since #265 that inheritance has exactly one value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

#: What an observation can be. `fact` is the default shape; the others are
#: separate because `promote` has to treat them differently — a `task` ages
#: out, a `decision` supersedes rather than merges.
ObservationKind = Literal["fact", "preference", "decision", "task", "relation"]

#: The same set as a tuple, for the places that need values rather than a type:
#: a JSON Schema `enum`, and a validity check on a tool argument. Derived from
#: the Literal so the two cannot drift.
OBSERVATION_KINDS: tuple[str, ...] = get_args(ObservationKind)


@dataclass(frozen=True, slots=True)
class Cited:
    """One attachment a claim is about.

    Digest and name together, because they answer different questions and only
    one of them is stable. The digest is what resolves and what the patch
    validator checks; the name is what the link text says, and it is a copy
    rather than a lookup — the arrival it came from is deleted with the message
    it arrived on, and `promote` reads this hours later.

    Never built from what a model typed. Both builders resolve a handle against
    the attachments of the conversation they are in, and what they put here is
    what the store said, which is what keeps a claim about a photograph from
    becoming a pointer to somebody else's.
    """

    sha256: str
    name: str


#: `attachment_refs.source` for bytes Siatt drew rather than received
#: (`siatt/imagen/tool.py`). Told apart from every other source because the
#: difference is not which surface delivered the file — it is whether anybody
#: was there when it happened.
GENERATED = "generated"


def named(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """What each attachment of one conversation is called, by digest.

    The fallback for an upload the surface never named is here rather than at
    each caller, so a memory written by the tool reads the same as one written
    by an extraction — and so a file sent back out is called what it was called
    on the way in.
    """
    return {
        str(row["sha256"]): str(row["name"] or "").strip() or f"an attachment ({row['mime']})"
        for row in rows
    }


def citable(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """The same, minus anything Siatt drew itself.

    Both observation builders resolve a handle against this, and a generated
    picture must not be among them. Long-term memory is Markdown a person reads
    and believes, `blobref` exists because a model that has seen the shape of a
    reference will compose one, and a claim citing an image *Siatt invented*
    would be a fabricated exhibit filed as evidence — worse than a broken link,
    because nothing about it looks wrong.

    Only the citation paths filter. A generated file is still a file this
    conversation has: `send_file` can send it again, and the listing that offers
    it by name uses `named` (`siatt/core/file_tools.py`). What it cannot do is
    become a footnote in the corpus.

    Rows must therefore carry `source` — `attachments_for_session` is the query
    that has it, and it is the query both builders use. A row without one counts
    as arrived, because the shape that lacks it (`attachments_for_message`)
    describes files that came in on a message, and nothing draws onto one.

    The `source` it reads is the earliest permitted arrival's, which is the one
    `attachments_for_session` groups to. That is the right one: a picture drawn
    here and later uploaded back out carries a `generated` arrival first, and
    the later delivery cannot launder it.
    """
    return named(row for row in rows if str(row.get("source") or "") != GENERATED)


@dataclass(frozen=True, slots=True)
class ObservationDraft:
    """One candidate fact, ready to be written to `observations`."""

    subject: str
    claim: str
    #: One of `OBSERVATION_KINDS`, checked by whoever built the draft. Not the
    #: `ObservationKind` Literal: both builders start from a plain string — a
    #: tool argument, or a model's JSON — and a `cast` at each of them would
    #: assert exactly what the check beside it has just established.
    kind: str
    scope: str
    confidence: float = 0.7
    source_refs: Sequence[str] = field(default_factory=tuple)
    #: The attachments this claim is about, if any. Distinct from `source_refs`,
    #: which says where the claim was said: a message can be the source of five
    #: observations and the photograph belongs to one of them.
    attachments: Sequence[Cited] = field(default_factory=tuple)
