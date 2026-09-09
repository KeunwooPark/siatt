"""A picture from ingress to the provider and back onto disk."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from siatt.config import AttachmentSettings
from siatt.llm.tokens import Tokenizer
from siatt.store import Store
from siatt.store.blobs import Attachments
from tests.core.test_agent import build, says


def png(width: int = 800, height: int = 600) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        payload = kind + body
        return struct.pack(">I", len(body)) + payload + struct.pack(">I", zlib.crc32(payload))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def files(store: Store, tmp_path: Path) -> Attachments:
    return AttachmentSettings(enabled=True).build(store, tmp_path / "siatt.db")


async def stored(
    attachments: Attachments, *, mime: str = "image/png", scope: str = "workspace"
) -> str:
    return await attachments.put(
        png(), mime=mime, source_name="slack", scope=scope, name="shot.png"
    )


async def test_an_image_reaches_the_provider_with_its_bytes(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    agent, provider = build(store, tokenizer, [says("a photograph")], attachments=attachments)

    await agent.respond("s1", "what's this?", attachments=[sha])

    sent = provider.requests[0].messages[-1]
    assert sent.text == "what's this?"
    assert sent.images[0].data == png()
    # Read from the header when it was stored, not guessed at pack time.
    assert (sent.images[0].width, sent.images[0].height) == (800, 600)


async def test_the_transcript_keeps_a_reference_not_the_bytes(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    agent, _ = build(store, tokenizer, [says("ok")], attachments=attachments)

    await agent.respond("s1", "look", attachments=[sha])

    rows = await store.raw("SELECT content FROM messages WHERE role = 'user'")
    content = str(rows[0]["content"])
    assert sha in content
    assert "data" not in content


async def test_an_image_from_an_earlier_turn_is_rehydrated(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """The block on disk has no bytes. Every later turn in the thread has to put
    them back, or the model loses the picture it was just shown."""
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    agent, provider = build(
        store, tokenizer, [says("a photograph"), says("still one")], attachments=attachments
    )

    await agent.respond("s1", "what's this?", attachments=[sha])
    await agent.respond("s1", "and now?")

    second = provider.requests[1].messages
    assert any(b.data == png() for m in second for b in m.images)


async def test_a_video_is_not_put_in_the_turn(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """Kept and referenced, and there is nothing that reads one. The note told
    the model so; the turn must not then quietly contain it."""
    attachments = files(store, tmp_path)
    sha = await attachments.put(
        b"\x00\x00\x00 ftypmp42", mime="video/mp4", source_name="slack", scope="workspace"
    )
    agent, provider = build(store, tokenizer, [says("ok")], attachments=attachments)

    await agent.respond("s1", "watch this", attachments=[sha])

    assert provider.requests[0].messages[-1].images == ()


async def test_an_image_that_is_not_in_the_store_is_not_invented(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """A hash is a guessable-looking string. Naming one nothing points at must
    produce nothing, not a picture."""
    attachments = files(store, tmp_path)
    agent, provider = build(store, tokenizer, [says("ok")], attachments=attachments)

    await agent.respond("s1", "look", attachments=["0" * 64])

    assert provider.requests[0].messages[-1].images == ()


async def test_bytes_gone_from_disk_still_answer(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """The compat layer turns an unhydrated block into a sentence. Losing a file
    should cost the picture, not the turn."""
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    agent, provider = build(store, tokenizer, [says("ok"), says("gone")], attachments=attachments)
    await agent.respond("s1", "look", attachments=[sha])

    AttachmentSettings(enabled=True).blobs(tmp_path / "siatt.db").path(sha).unlink()
    result = await agent.respond("s1", "still there?")

    assert result.text == "gone"
    assert all(b.data is None for m in provider.requests[1].messages for b in m.images)


async def test_no_attachment_store_is_not_an_error(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """An install that keeps nothing still answers; the hashes resolve to
    nothing rather than to a failure."""
    agent, provider = build(store, tokenizer, [says("ok")])

    result = await agent.respond("s1", "look", attachments=["a" * 64])

    assert result.text == "ok"
    assert provider.requests[0].messages[-1].images == ()


async def test_the_packer_charges_for_the_picture_not_for_its_hash(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """800x600 costs 640 tokens; the block serializes to about thirty. Counting
    it as text is how a context that looked like it fit arrives as a 400."""
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    agent, _ = build(store, tokenizer, [says("ok"), says("ok")], attachments=attachments)

    plain = await agent.respond("s2", "look")
    withimage = await agent.respond("s1", "look", attachments=[sha])

    assert plain.trace is not None and withimage.trace is not None
    assert withimage.trace.used - plain.trace.used > 600
