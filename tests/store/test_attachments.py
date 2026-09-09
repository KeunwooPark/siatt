"""The attachment store: bytes on disk, rows in SQLite, scope on every way out."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from siatt.config import AttachmentSettings, Config
from siatt.llm.types import Message
from siatt.store import Store
from siatt.store.blobs import (
    DEFAULT_ALLOWED_MIME,
    DEFAULT_MAX_BYTES,
    AttachmentError,
    AttachmentRejected,
    Attachments,
    AttachmentTooLarge,
    BlobStore,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 100


async def a_message(store: Store, session: str = "s1") -> str:
    """A real `messages` row, because `attachment_refs.message_id` has a foreign
    key to one — which is what makes `clear_session` take the refs with it."""
    await store.ensure_session(session, surface="slack")
    return await store.append_message(session, Message.user("here you go"))


def attachments(
    store: Store,
    tmp_path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allowed_mime: tuple[str, ...] = DEFAULT_ALLOWED_MIME,
) -> Attachments:
    settings = AttachmentSettings(enabled=True, max_bytes=max_bytes, allowed_mime=allowed_mime)
    return settings.build(store, tmp_path / "siatt.db")


async def test_migration_applies_and_is_idempotent(tmp_path: Path) -> None:
    store = await Store.open(tmp_path / "k.db")
    try:
        assert await store.migrate() == []
        tables = {
            row["name"]
            for row in await store.raw("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"attachments", "attachment_refs"} <= tables
    finally:
        await store.close()


async def test_put_stores_bytes_and_returns_their_hash(store: Store, tmp_path: Path) -> None:
    files = attachments(store, tmp_path)
    sha = await files.put(PNG, mime="image/png", source_name="slack", scope="workspace")

    assert sha == hashlib.sha256(PNG).hexdigest()
    assert await files.read(sha) == PNG
    held = await files.get(sha)
    assert held is not None
    assert (held.mime, held.size) == ("image/png", len(PNG))


async def test_the_same_bytes_twice_are_one_blob_and_two_refs(store: Store, tmp_path: Path) -> None:
    files = attachments(store, tmp_path)
    first = await files.put(PNG, mime="image/png", source_name="slack", scope="workspace")
    second = await files.put(PNG, mime="image/png", source_name="slack", scope="workspace")

    assert first == second
    blobs = await store.raw("SELECT COUNT(*) AS n FROM attachments")
    refs = await store.raw("SELECT COUNT(*) AS n FROM attachment_refs")
    assert (blobs[0]["n"], refs[0]["n"]) == (1, 2)


async def test_a_redelivery_of_one_arrival_is_one_ref(store: Store, tmp_path: Path) -> None:
    """At-least-once delivery plus the arrival key is at-most-once in effect."""
    files = attachments(store, tmp_path)
    arrival = "slack:T1:C1:1700000000.1"
    for _ in range(2):
        await files.put(
            PNG,
            mime="image/png",
            source_name="slack",
            scope="workspace",
            external_id=arrival,
        )

    refs = await store.raw("SELECT COUNT(*) AS n FROM attachment_refs")
    assert refs[0]["n"] == 1


# -- one pool (#265) ---------------------------------------------------------


async def test_a_blob_that_arrived_anywhere_is_readable(store: Store, tmp_path: Path) -> None:
    """One person, one pool. A picture sent in a DM is the same person's picture
    in a channel, and reading it is not a permission question (#265)."""
    files = attachments(store, tmp_path)
    sha = await files.put(PNG, mime="image/png", source_name="slack", scope="private:U123")

    assert await files.get(sha) is not None
    assert await files.read(sha) == PNG


async def test_a_blob_nothing_points_at_is_not_readable(store: Store, tmp_path: Path) -> None:
    """The one refusal left: the row is what says these bytes are the store's."""
    files = attachments(store, tmp_path)
    with pytest.raises(AttachmentError, match="not in the store"):
        await files.read("0" * 64)


async def test_for_message_returns_what_arrived_on_it(store: Store, tmp_path: Path) -> None:
    files = attachments(store, tmp_path)
    message = await a_message(store)
    await files.put(
        PNG,
        mime="image/png",
        source_name="slack",
        scope="workspace",
        message_id=message,
        name="shot.png",
    )

    assert [a.name for a in await files.for_message(message)] == ["shot.png"]


# -- what a conversation may cite --------------------------------------------


async def test_a_session_sees_what_arrived_in_it(store: Store, tmp_path: Path) -> None:
    """What `memory_write` resolves a handle against. A ref carries the session
    from the moment the file is fetched, which is before the turn has appended
    the message it came on."""
    files = attachments(store, tmp_path)
    await files.put(
        PNG,
        mime="image/png",
        source_name="slack",
        scope="workspace",
        session_id="s1",
        name="shot.png",
    )

    rows = await store.attachments_for_session("s1")

    assert [(r["sha256"], r["name"]) for r in rows] == [
        (hashlib.sha256(PNG).hexdigest(), "shot.png")
    ]
    assert await store.attachments_for_session("s2") == []


async def test_a_ref_is_found_through_the_message_it_arrived_on(
    store: Store, tmp_path: Path
) -> None:
    """The other half of the same question: a ref written with a message and no
    session belongs to that message's conversation."""
    files = attachments(store, tmp_path)
    message = await a_message(store, "s1")
    await files.put(PNG, mime="image/png", source_name="cli", scope="workspace", message_id=message)

    rows = await store.attachments_for_session("s1")
    assert len(rows) == 1


async def test_what_a_session_may_cite_is_what_arrived_in_it(store: Store, tmp_path: Path) -> None:
    """The line is the session, not the scope: a model may cite a file somebody
    sent where it is being asked, and nothing else (#265)."""
    files = attachments(store, tmp_path)
    await files.put(PNG, mime="image/png", source_name="slack", scope="workspace", session_id="s1")

    assert await store.attachments_for_session("s1") != []
    assert await store.attachments_for_session("s2") == []


async def test_one_picture_sent_twice_is_one_thing_to_cite(store: Store, tmp_path: Path) -> None:
    """Two arrivals, one blob, one line in the note. The name shown is the one
    it arrived under first."""
    files = attachments(store, tmp_path)
    for name in ("first.png", "second.png"):
        await files.put(
            PNG,
            mime="image/png",
            source_name="slack",
            scope="workspace",
            session_id="s1",
            name=name,
        )

    rows = await store.attachments_for_session("s1")

    assert [r["name"] for r in rows] == ["first.png"]


# -- caps and kinds ----------------------------------------------------------


async def test_a_file_over_the_cap_is_refused_and_the_cap_is_named(
    store: Store, tmp_path: Path
) -> None:
    files = attachments(store, tmp_path, max_bytes=1_024)
    with pytest.raises(AttachmentTooLarge, match="1024 byte cap"):
        await files.put(b"x" * 2_000, mime="image/png", source_name="cli", scope="workspace")

    assert await store.raw("SELECT COUNT(*) AS n FROM attachments") == [{"n": 0}]


async def test_the_cap_trips_mid_stream(store: Store, tmp_path: Path) -> None:
    """The whole point of a cap: it stops the download rather than measuring it
    once it is already in memory."""
    read = 0

    async def endless() -> AsyncIterator[bytes]:
        nonlocal read
        while True:
            read += 1
            yield b"x" * 4_096

    files = attachments(store, tmp_path, max_bytes=8_192)
    with pytest.raises(AttachmentTooLarge):
        await files.put(endless(), mime="video/mp4", source_name="slack", scope="workspace")

    assert read <= 3


async def test_a_kind_the_install_does_not_keep_is_refused(store: Store, tmp_path: Path) -> None:
    files = attachments(store, tmp_path)
    with pytest.raises(AttachmentRejected, match="allowed_mime"):
        await files.put(b"MZ", mime="application/x-dosexec", source_name="slack", scope="workspace")


async def test_video_is_kept_by_default(store: Store, tmp_path: Path) -> None:
    """Stored and unread is the promise; refusing to store it is not."""
    files = attachments(store, tmp_path)
    sha = await files.put(
        b"\x00\x00\x00 ftypmp42", mime="video/mp4", source_name="slack", scope="workspace"
    )
    assert await files.get(sha) is not None


# -- what is on disk ---------------------------------------------------------


async def test_the_stored_hash_matches_the_file_on_disk(store: Store, tmp_path: Path) -> None:
    settings = AttachmentSettings(enabled=True)
    blobs = settings.blobs(tmp_path / "siatt.db")
    files = settings.build(store, tmp_path / "siatt.db")
    sha = await files.put(PNG, mime="image/png", source_name="cli", scope="workspace")

    on_disk = blobs.path(sha)
    assert on_disk.parent.name == sha[:2]
    assert hashlib.sha256(on_disk.read_bytes()).hexdigest() == sha


async def test_an_interrupted_write_leaves_nothing_behind(tmp_path: Path) -> None:
    async def dies() -> AsyncIterator[bytes]:
        yield b"half a file"
        raise ConnectionResetError("the socket went away")

    blobs = BlobStore(tmp_path / "blobs")
    with pytest.raises(ConnectionResetError):
        await blobs.write(dies())

    written = [p for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert written == []


async def test_a_digest_that_is_not_a_digest_never_becomes_a_path(tmp_path: Path) -> None:
    blobs = BlobStore(tmp_path / "blobs")
    with pytest.raises(AttachmentError, match="not a sha256"):
        blobs.path("../../etc/passwd")


async def test_reading_bytes_that_are_not_there_says_so(store: Store, tmp_path: Path) -> None:
    files = attachments(store, tmp_path)
    settings = AttachmentSettings(enabled=True)
    sha = await files.put(PNG, mime="image/png", source_name="cli", scope="workspace")
    settings.blobs(tmp_path / "siatt.db").path(sha).unlink()

    with pytest.raises(AttachmentError, match="not on disk"):
        await files.read(sha)


# -- doing nothing by default ------------------------------------------------


async def test_off_by_default(tmp_path: Path) -> None:
    assert Config().attachments.enabled is False


async def test_no_directory_is_created_until_something_is_written(tmp_path: Path) -> None:
    AttachmentSettings().blobs(tmp_path / "siatt.db")
    assert not (tmp_path / "blobs").exists()


async def test_clearing_a_session_takes_its_refs_with_it(store: Store, tmp_path: Path) -> None:
    """A ref exists because of the message it arrived on. The blob row survives:
    deciding that bytes nothing points at may go is the collector's job, and it
    needs to know what long-term memory still references."""
    files = attachments(store, tmp_path)
    message = await a_message(store)
    await files.put(
        PNG,
        mime="image/png",
        source_name="slack",
        scope="workspace",
        message_id=message,
    )

    await store.clear_session("s1")

    assert await store.raw("SELECT COUNT(*) AS n FROM attachment_refs") == [{"n": 0}]
    assert await store.raw("SELECT COUNT(*) AS n FROM attachments") == [{"n": 1}]
