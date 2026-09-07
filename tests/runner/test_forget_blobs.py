"""Reclaiming stored attachments: the one deletion here that git cannot undo."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from siatt.config import AttachmentSettings
from siatt.memory.blobref import link
from siatt.memory.layout import ARCHIVE_DIR
from siatt.store import Store
from siatt.store.blobs import Attachments
from tests.runner.test_forget import (
    NOW,
    aged,
    collector_for,
    memory,
    write,
)
from tests.runner.test_forget import clone as clone  # the fixture, reused here

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 50


def files(store: Store, tmp_path: Path) -> Attachments:
    return AttachmentSettings(enabled=True).build(store, tmp_path / "siatt.db")


async def held(
    attachments: Attachments, data: bytes = PNG, *, days_ago: float = 30, name: str = "shot.png"
) -> str:
    """One stored attachment, aged by rewriting its row's timestamp."""
    sha = await attachments.put(
        data, mime="image/png", source_name="slack", scope="workspace", name=name
    )
    return sha


async def age_blob(store: Store, sha: str, days: float) -> None:
    when = (NOW - timedelta(days=days)).isoformat(timespec="milliseconds")
    await store.write("UPDATE attachments SET created_at = ? WHERE sha256 = ?", (when, sha))


async def drop_refs(store: Store, sha: str) -> None:
    """What a cleared session or a purged transcript leaves behind."""
    await store.write("DELETE FROM attachment_refs WHERE sha256 = ?", (sha,))


async def test_an_attachment_nothing_points_at_is_collected(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    sha = await held(attachments)
    await age_blob(store, sha, 30)
    await drop_refs(store, sha)

    outcome = await collector_for(clone, store, attachments=attachments).run()

    assert [item.sha256 for item in outcome.reclaimed] == [sha]
    assert await store.raw("SELECT COUNT(*) AS n FROM attachments") == [{"n": 0}]
    assert not AttachmentSettings(enabled=True).blobs(tmp_path / "siatt.db").path(sha).exists()


async def test_the_row_goes_before_the_file(clone: Path, store: Store, tmp_path: Path) -> None:
    """A row promising bytes that are not there is a dangling reference every
    reader has to handle. A file whose row is gone is invisible and harmless."""
    attachments = files(store, tmp_path)
    sha = await held(attachments)
    await age_blob(store, sha, 30)
    await drop_refs(store, sha)
    order: list[str] = []

    original_delete = store.delete_attachment
    original_remove = attachments._blobs.remove

    async def note_row(sha256: str) -> bool:
        order.append("row")
        return await original_delete(sha256)

    async def note_file(sha256: str) -> bool:
        order.append("file")
        return await original_remove(sha256)

    store.delete_attachment = note_row  # type: ignore[method-assign]
    attachments._blobs.remove = note_file  # type: ignore[method-assign]

    await collector_for(clone, store, attachments=attachments).run()

    assert order == ["row", "file"]


async def test_a_conversation_that_still_holds_it_protects_it(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    """A ref means a message somewhere still has it attached, and short-term
    memory is a transcript of what was actually said."""
    attachments = files(store, tmp_path)
    sha = await held(attachments)
    await age_blob(store, sha, 30)

    outcome = await collector_for(clone, store, attachments=attachments).run()

    assert outcome.reclaimed == []


async def test_a_live_memory_that_cites_it_protects_it(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    sha = await held(attachments)
    await age_blob(store, sha, 30)
    await drop_refs(store, sha)
    write(clone, aged(memory("The whiteboard", f"We agreed {link(sha, 'shot.png')}."), 400))

    outcome = await collector_for(clone, store, attachments=attachments).run()

    assert outcome.reclaimed == []


async def test_an_archived_memory_protects_it_too(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    """Wider than `_linked_from_live`, on purpose. That rule ignores links out
    of the archive so a corpus of dead references can shrink; an archived
    memory is still readable, and taking its picture away leaves it saying
    "see the photograph" beside nothing. Deleting a memory is a `git rm` that
    history keeps — deleting a blob is not."""
    attachments = files(store, tmp_path)
    sha = await held(attachments)
    await age_blob(store, sha, 30)
    await drop_refs(store, sha)
    doc = aged(memory("Old board", f"It said {link(sha, 'shot.png')}."), 400)
    write(clone, doc, path=f"{ARCHIVE_DIR}/old-board.md")

    outcome = await collector_for(clone, store, attachments=attachments).run()

    assert outcome.reclaimed == []


async def test_something_stored_this_week_is_never_collected(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    """The hazard is not a race with one turn. A memory that will cite a picture
    is written by `promote`, hours after the conversation ended."""
    attachments = files(store, tmp_path)
    sha = await held(attachments)
    await age_blob(store, sha, 2)
    await drop_refs(store, sha)

    outcome = await collector_for(clone, store, attachments=attachments, blob_grace_days=7).run()

    assert outcome.reclaimed == []
    assert await store.raw("SELECT COUNT(*) AS n FROM attachments") == [{"n": 1}]


async def test_the_per_run_budget_is_enforced_and_reported(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    for index in range(5):
        sha = await held(attachments, b"\x89PNG\r\n\x1a\n" + bytes([index]) * 40)
        await age_blob(store, sha, 30)
        await drop_refs(store, sha)

    outcome = await collector_for(clone, store, attachments=attachments, max_blobs_per_run=2).run()

    assert len(outcome.reclaimed) == 2
    assert "2 attachment(s) reclaimed" in outcome.summary()
    assert await store.raw("SELECT COUNT(*) AS n FROM attachments") == [{"n": 3}]


async def test_the_blob_budget_is_separate_from_the_memory_budget(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    """Two transitions a person reads in a pull request, and a sweep of files
    nobody diffs. Sharing one budget would let a week of uploads crowd out the
    forgetting that matters."""
    attachments = files(store, tmp_path)
    sha = await held(attachments)
    await age_blob(store, sha, 30)
    await drop_refs(store, sha)
    write(clone, aged(memory("Cold", salience=0.0), 400))

    outcome = await collector_for(
        clone, store, attachments=attachments, max_per_run=1, **{"archive_below": 1.0}
    ).run()

    assert len(outcome.archived) == 1
    assert len(outcome.reclaimed) == 1


async def test_an_install_that_keeps_nothing_sweeps_nothing(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    outcome = await collector_for(clone, store).run()

    assert outcome.reclaimed == []
    assert "reclaimed" not in outcome.summary()


async def test_a_sweep_that_fails_does_not_fail_the_run(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    """The memories are already committed by then, and the bytes will still be
    there next week."""
    attachments = files(store, tmp_path)

    async def explodes(**kwargs: Any) -> list[Any]:
        raise OSError("the disk is gone")

    attachments.collect = explodes  # type: ignore[method-assign]

    outcome = await collector_for(clone, store, attachments=attachments).run()

    assert outcome.reclaimed == []
