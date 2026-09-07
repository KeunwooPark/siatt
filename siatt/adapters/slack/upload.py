"""Putting the file in the thread, once the answer is already in it.

The inbound half of this is `files.py`, and it is careful about one thing: a URL
out of an event payload, and a bearer token that must not follow it anywhere.
Outbound the risk is the mirror image and just as quiet — bytes leaving the
conversation they belong to. A photograph sent in a DM is `private:<author>`
(`scope_for`), and the only thing standing between it and a public channel is
that the bytes are read through `Attachments.read` under the scope the turn ran
under. That one call is the check. There is no second path to a blob here, and
there must not be: `BlobStore` will hand back anything it is asked for, which is
why nothing outside its own module is given one.

Three more decisions, all of them about what happens when this goes wrong.

**The answer goes first, the files follow.** A Slack outage then costs the
picture rather than the reply, which is the same bargain the fetcher makes. It
also has to be this order for the live path: `LiveMessage` is still repainting
until `finish` returns, and a file uploaded mid-stream lands above a message
that is still growing.

**A failed upload says so.** The answer has already said "here it is", so
silence would read as a file that was sent and lost. One line into the thread,
naming the file and the reason, is something a person can act on -- and the
reason is usually `files:write`, which is something they can fix.

**Sent once, across restarts.** An inbox row can be redelivered and the whole
turn re-run; `_unfinished` keeps that from posting a second answer, and the
equivalent for a file is the arrival key that already exists in
`attachment_refs`. Recording the upload as an arrival under the request's own
`external_id` makes "have I already sent this here" a question the database
answers, rather than one a dictionary in this process answers until it restarts.
The record is written *after* the upload: a crash in between re-sends a file,
and the other order loses one for good.
"""

from __future__ import annotations

import logging
from typing import Protocol

from siatt.core.agent import AgentResult
from siatt.errors import SiattError
from siatt.store import Store
from siatt.store.blobs import Attachment, AttachmentError, Attachments

log = logging.getLogger(__name__)

#: `attachment_refs.source` for a file Siatt sent. Told apart from `slack`
#: because the two are different events about the same bytes -- one is somebody
#: uploading a photograph, the other is Siatt handing it back -- and because the
#: arrival key is scoped by source, so an answer in the thread the picture
#: arrived on does not collide with the arrival itself.
SOURCE = "slack-out"

#: What a file with no name of its own is called, by kind.
_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif", "video/mp4": "mp4"}


class UploadRefused(SiattError):
    """Slack would not take the file, in words somebody can act on."""


class Uploader(Protocol):
    """The one Slack call this needs, kept behind a protocol.

    Same split as `Poster` in `stream.py`, and for the same reason: what this
    module needs from Slack is "put these bytes in that thread", and keeping
    `slack_sdk` on the adapter's side is what lets the ordering and the failure
    handling be tested without it.
    """

    async def upload(
        self, *, channel: str, thread_ts: str | None, filename: str, title: str, data: bytes
    ) -> None: ...


class Poster(Protocol):
    async def post(self, *, channel: str, thread_ts: str | None, text: str) -> str: ...


class SlackUploads:
    """The files on one answer, sent into the thread that answer is in."""

    def __init__(
        self,
        *,
        uploader: Uploader,
        poster: Poster,
        store: Store,
        attachments: Attachments | None,
    ) -> None:
        self._uploader = uploader
        self._poster = poster
        self._store = store
        self._attachments = attachments

    async def send(
        self,
        result: AgentResult,
        *,
        session_id: str,
        external_id: str,
        channel: str | None,
        thread_ts: str | None,
    ) -> None:
        """Upload what this turn asked to send. Never raises.

        The turn is over and the answer is posted; anything that fails here is a
        line in the thread, because failing the turn would re-run a model call
        to re-post an answer that is already up.
        """
        if not result.attachments or self._attachments is None or not channel:
            return
        scope = await self._scope(session_id)
        trouble: list[str] = []
        for held in result.attachments:
            try:
                await self._one(
                    held, scope=scope, external_id=external_id, channel=channel, thread_ts=thread_ts
                )
            except (AttachmentError, UploadRefused) as exc:
                log.info("could not send %s on %s: %s", held.sha256[:12], external_id, exc)
                trouble.append(f"{_filename(held)} — {exc}")
            except Exception:
                # A bug here is still not a reason to lose the answer, and the
                # traceback belongs in the log rather than in somebody's thread.
                log.exception("could not send %s on %s", held.sha256[:12], external_id)
                trouble.append(f"{_filename(held)} — something went wrong sending it")
        if trouble:
            await self._say(channel, thread_ts, trouble)

    async def _one(
        self,
        held: Attachment,
        *,
        scope: str,
        external_id: str,
        channel: str,
        thread_ts: str | None,
    ) -> None:
        assert self._attachments is not None
        if await self._store.attachment_ref_id(
            source=SOURCE, external_id=external_id, sha256=held.sha256
        ):
            # A redelivery of a turn that already sent this. The answer is being
            # rewritten; the file is not sent twice.
            log.info("already sent %s on %s", held.sha256[:12], external_id)
            return
        # The scope check, and the only one. Not the event's scope: the session
        # is what the turn ran under, and an event cannot widen a session that
        # was opened private.
        data = await self._attachments.read(held.sha256, scope=scope)
        name = _filename(held)
        await self._uploader.upload(
            channel=channel, thread_ts=thread_ts, filename=name, title=name, data=data
        )
        await self._store.add_attachment_ref(
            sha256=held.sha256,
            source=SOURCE,
            # The scope it was read under, which is the scope it went out to.
            # Widening here would be widening it for everything downstream.
            scope=scope,
            external_id=external_id,
            session_id=None,
            name=held.name,
        )

    async def _scope(self, session_id: str) -> str:
        """What this conversation may read, as the session recorded it.

        The session rather than the event, for the reason the runtime prefers it
        when it starts a turn: they are derived the same way and normally agree,
        and when they do not, the record of what this conversation has been all
        along is the one to trust.
        """
        row = await self._store.get_session(session_id)
        return str(row["scope"]) if row and row.get("scope") else "workspace"

    async def _say(self, channel: str, thread_ts: str | None, trouble: list[str]) -> None:
        head = "I could not send the file:" if len(trouble) == 1 else "I could not send:"
        try:
            await self._poster.post(
                channel=channel,
                thread_ts=thread_ts,
                text="\n".join([f"_{head}_", *(f"- {line}" for line in trouble)]),
            )
        except Exception:
            # Two failures in a row is Slack being down, not something a third
            # call will fix. The log is where this one lands.
            log.exception("could not say that a file was not sent on %s", channel)


def _filename(held: Attachment) -> str:
    """What the file is called on the way out.

    Somebody else chose it -- it came off an upload -- so it is flattened before
    it is shown or sent: whitespace collapsed so it cannot forge a second line
    of the note above, and separators dropped so a name is never something that
    reads as a path. A file with no name at all is named after its kind, because
    Slack shows the filename and "untitled" tells nobody anything.
    """
    flat = held.shown.replace("/", "-").replace("\\", "-").strip(". ")
    if flat:
        return flat
    suffix = _EXTENSIONS.get(held.mime) or held.mime.partition("/")[2] or "bin"
    return f"attachment.{suffix}"
