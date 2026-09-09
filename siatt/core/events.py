"""What a surface delivered, in the one shape the rest of Siatt understands.

An adapter's whole job at ingress is to turn a provider's event into an
`InboundEvent` and enqueue it. Everything downstream — the dispatcher, the
session router, the agent loop — sees only this type, so adding a surface never
means teaching the core about another payload shape.

It is also what sits in `inbox.payload`, which makes the queue self-describing:
a row that replays after a restart carries everything needed to answer it, and
a row that cannot be parsed is a row this version does not know how to answer
rather than one it will half-answer.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from siatt.errors import SiattError


class EventError(SiattError):
    """A queued payload is not an event this build can deliver."""


class Attached(BaseModel):
    """A file that came with a message, before anything has fetched it.

    A descriptor rather than the bytes, because this rides inside `inbox.payload`
    and a queue row is not a place to put a video. It is also what keeps the
    decision at ingress pure: saying *that* a file arrived is a judgement about a
    payload, and going and getting it is a network call that must not sit in
    front of the ack.

    Every field is somebody else's claim. `mime` and `size` are what the surface
    said, not what arrived, and nothing downstream may treat them as measured.
    """

    model_config = ConfigDict(frozen=True)

    #: Where to fetch it. Whether that address may be fetched *with a token* is
    #: not this type's business; see `siatt/adapters/slack/files.py`.
    #:
    #: None when the surface named a file but gave nowhere to get it — a
    #: tombstoned upload, or one this install may not read. That is still an
    #: attachment the answer has to account for, so it is a descriptor with no
    #: URL rather than an entry dropped on the floor.
    url: str | None = None
    #: What the surface called it. Shown; never joined to a path.
    name: str | None = None
    mime: str = ""
    #: Claimed, in bytes. Zero when the surface did not say.
    size: int = 0

    #: Filled in once the bytes are in the attachment store, and empty for
    #: anything that was not kept. It is what turns a descriptor into something
    #: the turn can act on: the block that goes in the transcript references
    #: this, and the agent reads the bytes back through it.
    #:
    #: Set after the queue rather than at ingress, so a payload sitting in the
    #: inbox never carries one — which is correct, because nothing has been
    #: fetched yet at that point.
    sha256: str | None = None


class InboundEvent(BaseModel):
    """One message from one surface, normalized.

    Frozen, because a row in the inbox is a record of what arrived. Anything
    the agent decides about it belongs beside it, not inside it.
    """

    model_config = ConfigDict(frozen=True)

    #: Which surface this came from: 'slack', 'cli', 'http'. Also the session's
    #: `surface`, since a session belongs to exactly one of them.
    source: str = Field(min_length=1)

    #: The provider's own id for this event, and the dedupe key. A surface with
    #: no such id has to mint one that is stable across the provider's retries —
    #: a fresh ULID per delivery attempt would defeat the whole table.
    external_id: str = Field(min_length=1)

    #: The actor key: one serialized conversation. On Slack this is the thread.
    session_id: str = Field(min_length=1)

    text: str = ""

    #: Ingress persisted a scrubbed form and retained the original only in
    #: process memory for this immediate turn.
    credential_scrubbed: bool = False

    #: The visibility anything learned here is recorded with. One value, since
    #: Siatt keeps one pool of memory for one person (#265); the column stays
    #: because it says what a row was written under.
    scope: str = "workspace"

    #: The platform's id for whoever spoke, not a display name.
    author: str | None = None

    #: The IANA zone the author is in, when the surface knows it. What makes
    #: "yesterday" in their message mean their yesterday and not Greenwich's
    #: (#223) — the author's, per message, because in a channel with several
    #: people there is no one workspace answer.
    #:
    #: Defaulted, like `origin`, so that an event queued before this field
    #: existed still parses after an upgrade.
    tz: str | None = None

    #: Where a reply goes. Opaque to the core — Slack puts `channel` and the
    #: `thread_ts` to reply in, and only the Slack adapter reads them back.
    channel: str | None = None
    reply_to: str | None = None

    #: What put this in the queue. `message` is somebody speaking; `scheduled`
    #: is a standing task firing (#179), where nobody said anything just now.
    #: The turn needs to know the difference — an answer that opens "as you
    #: asked" when the thread has been quiet since Tuesday reads as a
    #: hallucination.
    #:
    #: Defaulted rather than required so that every payload written before this
    #: field existed still parses. A row in the inbox is a record of what
    #: arrived, and a queue that stops being able to read its own backlog after
    #: an upgrade is a queue that drops messages.
    origin: str = "message"

    #: Files that came with the message, not yet fetched. Defaulted, like
    #: `origin`, so a payload queued before this field existed still parses
    #: after an upgrade — a queue that cannot read its own backlog is a queue
    #: that drops messages.
    attachments: tuple[Attached, ...] = ()

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Self:
        try:
            return cls.model_validate_json(payload)
        except ValidationError as exc:
            raise EventError(f"queued payload is not a deliverable event: {exc}") from exc
