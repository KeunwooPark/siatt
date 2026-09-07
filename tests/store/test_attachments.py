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
    assert await files.read(sha, scope="workspace") == PNG
    held = await files.get(sha, scope="workspace")
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


# -- scope -------------------------------------------------------------------


async def test_a_dm_blob_is_invisible_from_a_channel(store: Store, tmp_path: Path) -> None:
    """The failure this table exists to prevent: a picture from a DM turning up
    in a public channel because somebody knew its hash."""
    files = attachments(store, tmp_path)
    sha = await files.put(PNG, mime="image/png", source_name="slack", scope="private:U123")

    assert await files.get(sha, scope="private:U123") is not None
    assert await files.get(sha, scope="channel:C456") is None
    with pytest.raises(AttachmentError, match="not visible"):
        await files.read(sha, scope="channel:C456")


async def test_a_workspace_blob_is_visible_everywhere(store: Store, tmp_path: Path) -> None:
    files = attachments(store, tmp_path)
    sha = await files.put(PNG, mime="image/png", source_name="slack", scope="workspace")
    assert await files.get(sha, scope="private:U123") is not None


async def test_a_public_arrival_does_not_widen_a_private_one(store: Store, tmp_path: Path) -> None:
    """One set of bytes, two arrivals. The narrow one must stay narrow even
    though the wide one deduplicated onto the same blob."""
    files = attachments(store, tmp_path)
    public = await a_message(store)
    sha = await files.put(PNG, mime="image/png", source_name="slack", scope="private:U123")
    await files.put(
        PNG,
        mime="image/png",
        source_name="slack",
        scope="channel:C456",
        message_id=public,
    )

    # The blob is now reachable from the channel — because it genuinely arrived
    # there — but the DM's own arrival is not what granted that.
    private_refs = await store.raw(
        "SELECT scope FROM attachment_refs WHERE sha256 = ? ORDER BY scope", (sha,)
    )
    assert [row["scope"] for row in private_refs] == ["channel:C456", "private:U123"]
    # And a third conversation that saw neither arrival still sees nothing.
    assert await files.get(sha, scope="private:U999") is None


async def test_for_message_filters_before_it_returns(store: Store, tmp_path: Path) -> None:
    files = attachments(store, tmp_path)
    message = await a_message(store)
    await files.put(
        PNG,
        mime="image/png",
        source_name="slack",
        scope="private:U123",
        message_id=message,
        name="secret.png",
    )

    seen = await files.for_message(message, scope="private:U123")
    assert [a.name for a in seen] == ["secret.png"]
    assert await files.for_message(message, scope="channel:C456") == []


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
    assert await files.get(sha, scope="workspace") is not None


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
        await files.read(sha, scope="workspace")


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
