"""Bytes on disk, and the scope that says who may ask for them.

Two things live here because they are two halves of one guarantee.

`BlobStore` is the filesystem: content-addressed files, written under a
temporary name and renamed into place, with a cap enforced on the way past
rather than after arrival. It knows nothing about conversations, sessions or
visibility, and it should not -- given a hash it will hand back bytes.

`Attachments` is the half that knows who is asking. Every read it offers takes
a scope and refuses what that scope may not see, and a blob is only reachable
through it. The database is where visibility is decided (`attachment_refs`),
so the composition is the point: a caller holding only a `BlobStore` can read
anything, and no caller outside this module is given one.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

from siatt.errors import StoreError
from siatt.store.dimensions import HEAD_BYTES, dimensions

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, not at type time
    from siatt.store.db import Store

#: Bytes read from the source in one go. Small enough that the cap below stops
#: a large file early rather than after it is all in memory.
CHUNK = 64 * 1024

#: What an install gets without saying otherwise. Slack's own upload ceiling is
#: 1GB, which is not a number to hold in memory or to hand a model; this is the
#: size of a long phone video and already far past anything useful.
DEFAULT_MAX_BYTES = 25_000_000

#: Prefixes, matched against the mime type. Prefixes rather than exact types
#: because `image/png`, `image/jpeg`, `image/heic` and whatever comes next are
#: one decision, and an allowlist that has to be extended per codec is one that
#: will be wrong on the day it matters.
DEFAULT_ALLOWED_MIME: tuple[str, ...] = ("image/", "video/")


class AttachmentError(StoreError):
    """An attachment could not be stored or read."""


class AttachmentTooLarge(AttachmentError):
    """More bytes arrived than the install allows to be kept."""


class AttachmentRejected(AttachmentError):
    """The bytes are not of a kind this install keeps."""


@dataclass(frozen=True, slots=True)
class Blob:
    """What was written, as measured rather than as promised."""

    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class Attachment:
    """One blob, as seen from a conversation allowed to see it."""

    sha256: str
    mime: str
    size: int
    #: What the surface called the file. Somebody else's text: it is shown, and
    #: never joined to a path.
    name: str | None = None
    #: Pixels, for an image whose header we could read. None otherwise, and the
    #: packer charges the ceiling rather than guessing small.
    width: int | None = None
    height: int | None = None

    @property
    def is_image(self) -> bool:
        return self.mime.startswith("image/")


class BlobStore:
    """Content-addressed bytes under one directory.

    The directory is created on first write, never on construction. An install
    that never enables attachments should not grow an empty `blobs/` beside its
    database, and a `siatt doctor` run should not be what creates it.
    """

    def __init__(self, root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.root = root
        self.max_bytes = max_bytes

    @classmethod
    def beside(cls, db_path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> Self:
        """The blob directory for a database at `db_path`.

        A sibling of the database on purpose. One directory is then the whole of
        Siatt's local state, so a backup that copies `siatt.db` and misses the
        blobs is visibly incomplete rather than silently so.
        """
        return cls(Path(db_path).expanduser().parent / "blobs", max_bytes=max_bytes)

    def path(self, sha256: str) -> Path:
        """Where `sha256` lives. Fanned out by its first byte, because a flat
        directory of tens of thousands of files is slow to list and unpleasant
        to look at."""
        digest = _checked(sha256)
        return self.root / digest[:2] / digest

    def has(self, sha256: str) -> bool:
        return self.path(sha256).exists()

    async def write(self, source: bytes | AsyncIterable[bytes]) -> Blob:
        """Store `source` and return what was actually written.

        The hash is computed from the bytes as they go past, not taken from the
        caller: a store that trusted a promised digest would file a truncated
        download under the name of the whole one.

        Writing goes to a temporary name in the same directory and is renamed
        into place, so an interrupted write -- a crash, a cap tripping, a socket
        dying mid-download -- never leaves a partial file under a name that
        claims to be a hash. `os.replace` is atomic within a filesystem, and the
        temporary directory is inside the blob root to keep it one filesystem.
        """
        tmp_dir = self.root / "tmp"
        await asyncio.to_thread(tmp_dir.mkdir, parents=True, exist_ok=True)
        tmp = tmp_dir / f"{secrets.token_hex(16)}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            handle = await asyncio.to_thread(tmp.open, "wb")
            try:
                async for chunk in _chunks(source):
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise AttachmentTooLarge(
                            f"the attachment is larger than the {self.max_bytes} byte cap "
                            "(attachments.max_bytes); it was not kept"
                        )
                    digest.update(chunk)
                    await asyncio.to_thread(handle.write, chunk)
            finally:
                await asyncio.to_thread(handle.close)
            sha256 = digest.hexdigest()
            final = self.path(sha256)
            await asyncio.to_thread(final.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(os.replace, tmp, final)
        finally:
            await asyncio.to_thread(tmp.unlink, True)
        return Blob(sha256=sha256, size=size)

    async def read(self, sha256: str) -> bytes:
        try:
            return await asyncio.to_thread(self.path(sha256).read_bytes)
        except OSError as exc:
            raise AttachmentError(f"attachment {sha256[:12]} is not on disk ({exc})") from exc

    async def measure(self, sha256: str) -> tuple[int, int] | None:
        """How big the picture is, or None for a format we do not parse.

        Reads the head of the file rather than all of it: a dimension lives in
        the first few hundred bytes of every format `dimensions` knows, and
        loading a 20MB photograph to look at its header is a waste this runs on
        every upload.
        """
        try:
            head = await asyncio.to_thread(_head, self.path(sha256), HEAD_BYTES)
        except OSError:
            return None
        return dimensions(head)

    async def remove(self, sha256: str) -> bool:
        """Delete the bytes. True when there was something to delete.

        Nothing in this build calls it: collecting a blob needs to know what the
        Markdown still points at, which is the collector's business rather than
        the store's. It is here so that the deletion is written once.
        """
        path = self.path(sha256)
        try:
            await asyncio.to_thread(path.unlink)
        except FileNotFoundError:
            return False
        return True


class Attachments:
    """Blob and row together, with a scope on every way out.

    The only route to an attachment's bytes. `put` writes the file and the rows
    in an order that survives a crash between them, and every read takes the
    scope of whoever is asking and applies the same rule retrieval applies to a
    memory: workspace is visible everywhere, anything narrower only from inside
    itself.
    """

    def __init__(
        self,
        store: Store,
        blobs: BlobStore,
        *,
        allowed_mime: tuple[str, ...] = DEFAULT_ALLOWED_MIME,
    ) -> None:
        self._store = store
        self._blobs = blobs
        self._allowed = allowed_mime

    @property
    def max_bytes(self) -> int:
        return self._blobs.max_bytes

    def accepts(self, mime: str) -> bool:
        return any(mime.startswith(prefix) for prefix in self._allowed)

    async def put(
        self,
        source: bytes | AsyncIterable[bytes],
        *,
        mime: str,
        source_name: str,
        scope: str,
        external_id: str | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
        author: str | None = None,
        name: str | None = None,
    ) -> str:
        """Keep these bytes, and record that they arrived. Returns the sha256.

        Idempotent on content: the same file twice writes one blob and one row
        in `attachments`. It is *not* idempotent on arrival unless the caller
        gives an `external_id` -- two people sending the same picture are two
        refs, because they are two arrivals with two scopes.

        The blob is written before the rows. The other order would leave a row
        promising bytes that are not there, which every reader would have to
        handle; this order leaves at worst an unreferenced file, which is dead
        space the collector reclaims and nothing has to know about.
        """
        if not self.accepts(mime):
            raise AttachmentRejected(
                f"{mime} is not a kind of file this install keeps "
                f"(attachments.allowed_mime: {', '.join(self._allowed)})"
            )
        blob = await self._blobs.write(source)
        # Measured here because here is where the file is: the header is a
        # filesystem read away, and doing it in the packer would put one inside
        # the loop that runs on every turn.
        size = await self._blobs.measure(blob.sha256) if mime.startswith("image/") else None
        await self._store.record_attachment(
            sha256=blob.sha256,
            mime=mime,
            size=blob.size,
            width=size[0] if size else None,
            height=size[1] if size else None,
        )
        await self._store.add_attachment_ref(
            sha256=blob.sha256,
            source=source_name,
            scope=scope,
            external_id=external_id,
            session_id=session_id,
            message_id=message_id,
            author=author,
            name=name,
        )
        return blob.sha256

    async def get(self, sha256: str, *, scope: str) -> Attachment | None:
        row = await self._store.attachment(sha256, scope=scope)
        return _attachment(row) if row else None

    async def read(self, sha256: str, *, scope: str) -> bytes:
        """The bytes, if `scope` is allowed to have them.

        The permission check is a database read before a filesystem read, and
        not the other way around: the answer to "may I see this" must not depend
        on whether the file happens to be there.
        """
        if await self._store.attachment(sha256, scope=scope) is None:
            raise AttachmentError(f"attachment {sha256[:12]} is not visible from {scope}")
        return await self._blobs.read(sha256)

    async def for_message(self, message_id: str, *, scope: str) -> list[Attachment]:
        rows = await self._store.attachments_for_message(message_id, scope=scope)
        return [_attachment(row) for row in rows]


def _attachment(row: dict[str, object]) -> Attachment:
    name = row.get("name")
    return Attachment(
        sha256=str(row["sha256"]),
        mime=str(row["mime"]),
        size=int(str(row["bytes"])),
        name=str(name) if name is not None else None,
        width=_int(row.get("width")),
        height=_int(row.get("height")),
    )


def _int(value: object) -> int | None:
    return int(str(value)) if value is not None else None


async def _chunks(source: bytes | AsyncIterable[bytes]) -> AsyncIterator[bytes]:
    """One shape for a payload that arrives whole or a piece at a time.

    Bytes already in hand are still cut up, so that the cap trips at the same
    place either way rather than being a check in one branch and a stream in
    the other.
    """
    if isinstance(source, bytes):
        for start in range(0, len(source), CHUNK):
            yield source[start : start + CHUNK]
        return
    async for chunk in source:
        yield chunk


def _head(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def _checked(sha256: str) -> str:
    """A digest, or an error -- never a path.

    A hash arrives from a database row and, in time, from a model that wrote a
    memory. `../../etc/passwd` is a string like any other until something joins
    it to a directory, and this is the join.
    """
    digest = sha256.strip().lower()
    if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        raise AttachmentError(f"{sha256!r} is not a sha256 digest")
    return digest
