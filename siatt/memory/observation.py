"""A candidate fact, before anything durable has happened to it.

An observation is the unit that crosses from short-term to long-term memory:
`episode_close` extracts them from a conversation, `promote` reconciles them
against the corpus and turns them into a patch plan. Between those two it is a
row in SQLite that nobody has committed to believing yet.

The draft type is here rather than in the store because two callers build one —
the `memory_write` tool and the extractor — and both have to agree on what the
fields mean. `scope` in particular: it is inherited from the session, never
chosen, because an observation made in a DM must not become workspace knowledge
by way of a model deciding it would be more useful there.
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


def citable(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """What each attachment of one conversation is called, by digest.

    Both builders resolve a handle against this, and both write the name it
    gives them onto the observation — so the fallback for an upload the surface
    never named is here rather than at each of them, and a memory written by
    the tool reads the same as one written by an extraction.
    """
    return {
        str(row["sha256"]): str(row["name"] or "").strip() or f"an attachment ({row['mime']})"
        for row in rows
    }


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
