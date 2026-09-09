"""`send_file`: what a turn may send back, and what it must not promise.

Two questions, and the second is the one with teeth. Can the model send a file
somebody actually put in front of it — and can it be talked into sending one
that was never here, or into telling somebody a photograph is on its way to a
terminal that cannot receive one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from siatt.config import AttachmentSettings
from siatt.core.file_tools import MAX_SENT, file_tools
from siatt.core.tools import Tool, ToolContext, ToolRegistry
from siatt.llm.tokens import Tokenizer
from siatt.llm.types import ChatResponse, Message, ToolUseBlock, Usage
from siatt.memory.blobref import handle
from siatt.store import Store
from siatt.store.blobs import Attachments
from tests.core.test_agent import build, says
from tests.core.test_agent_images import png


def files(store: Store, tmp_path: Path) -> Attachments:
    return AttachmentSettings(enabled=True).build(store, tmp_path / "siatt.db")


async def stored(
    attachments: Attachments,
    *,
    name: str = "shot.png",
    scope: str = "workspace",
    session_id: str | None = "s1",
    mime: str = "image/png",
    width: int = 800,
) -> str:
    """One attachment, arrived in a conversation the way Slack's do."""
    return await attachments.put(
        png(width=width),
        mime=mime,
        source_name="slack",
        scope=scope,
        session_id=session_id,
        external_id=f"ts-{width}-{name}",
        name=name,
    )


def tools(store: Store, attachments: Attachments) -> list[Tool]:
    return file_tools(store=store, attachments=attachments)


async def call(store: Store, attachments: Attachments, ids: list[str], context: ToolContext) -> str:
    registry = ToolRegistry(tools(store, attachments))
    result = await registry.dispatch(
        ToolUseBlock(id="t1", name="send_file", input={"ids": ids}), context
    )
    return result.content


def asking(**kwargs: Any) -> ToolContext:
    """A turn on a surface that can send files, unless a test says otherwise."""
    return ToolContext(session_id="s1", can_send_files=True, **kwargs)


# -- what may be sent --------------------------------------------------------


async def test_a_file_from_this_conversation_goes_on_the_answer(
    store: Store, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    context = asking()

    said = await call(store, attachments, [handle(sha)], context)

    assert context.outgoing == [sha]
    assert "shot.png" in said
    # The model is mid-answer, and the file has not moved yet.
    assert "has been sent yet" in said


async def test_a_photograph_a_memory_surfaced_this_turn_can_be_sent(
    store: Store, tmp_path: Path
) -> None:
    """The case the feature is for: "send me the photo from that memory".

    The blob belongs to no session here — it was reached through a memory — and
    it is sendable because `memory_read` already resolved it under this scope
    before putting it in front of the model.
    """
    attachments = files(store, tmp_path)
    sha = await stored(attachments, session_id=None, name="whiteboard.jpg")
    context = asking(surfaced=[sha])

    said = await call(store, attachments, [handle(sha)], context)

    assert context.outgoing == [sha]
    assert "whiteboard.jpg" in said


async def test_a_video_can_be_sent_even_though_nothing_can_read_it(
    store: Store, tmp_path: Path
) -> None:
    """Readability is a question about the model, not about the person waiting."""
    attachments = files(store, tmp_path)
    sha = await stored(attachments, mime="video/mp4", name="clip.mp4")
    context = asking()

    await call(store, attachments, [handle(sha)], context)

    assert context.outgoing == [sha]


async def test_two_files_go_in_one_call(store: Store, tmp_path: Path) -> None:
    attachments = files(store, tmp_path)
    first = await stored(attachments, name="one.png", width=800)
    second = await stored(attachments, name="two.png", width=801)
    context = asking()

    await call(store, attachments, [handle(first), handle(second)], context)

    assert context.outgoing == [first, second]


# -- what may not ------------------------------------------------------------


async def test_a_digest_from_nowhere_sends_nothing(store: Store, tmp_path: Path) -> None:
    """A model that has seen the shape of a handle will compose one."""
    attachments = files(store, tmp_path)
    await stored(attachments)
    context = asking()

    said = await call(store, attachments, ["deadbeefcafe"], context)

    assert context.outgoing == []
    assert "nothing was sent" in said
    # And it says what there actually is, so the retry has somewhere to go.
    assert "shot.png" in said


async def test_a_file_from_another_conversation_is_not_reachable(
    store: Store, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    elsewhere = await stored(attachments, session_id="s2", name="theirs.png")
    context = asking()

    said = await call(store, attachments, [handle(elsewhere)], context)

    assert context.outgoing == []
    assert "no files in this conversation" in said


async def test_a_digest_nothing_points_at_is_not_reachable(store: Store, tmp_path: Path) -> None:
    """A hash is a guessable-looking string, and one the store has never held
    resolves to nothing rather than to an error the model can probe."""
    context = asking()

    said = await call(store, files(store, tmp_path), ["0" * 12], context)

    assert context.outgoing == []
    assert "no files in this conversation" in said


async def test_an_ambiguous_prefix_is_refused_rather_than_guessed(
    store: Store, tmp_path: Path
) -> None:
    """Two real digests never collide for eight characters, so the rows are
    written by hand. What is under test is the branch, not the arithmetic."""
    attachments = files(store, tmp_path)
    for digest, name in (("a" * 64, "one.png"), ("a" * 63 + "b", "two.png")):
        await store.record_attachment(sha256=digest, mime="image/png", size=45)
        await store.add_attachment_ref(
            sha256=digest, source="slack", scope="workspace", session_id="s1", name=name
        )
    context = asking()

    said = await call(store, attachments, ["a" * 8], context)

    assert context.outgoing == []
    assert "more than one file" in said


async def test_one_bad_id_sends_none_of_them(store: Store, tmp_path: Path) -> None:
    """Half a request answered as a success is how "I sent both" gets said."""
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    context = asking()

    await call(store, attachments, [handle(sha), "abcdef123456"], context)

    assert context.outgoing == []


async def test_past_the_bound_nothing_more_is_sent(store: Store, tmp_path: Path) -> None:
    attachments = files(store, tmp_path)
    shas = [await stored(attachments, name=f"{i}.png", width=800 + i) for i in range(MAX_SENT + 1)]
    context = asking()

    await call(store, attachments, [handle(sha) for sha in shas[:MAX_SENT]], context)
    said = await call(store, attachments, [handle(shas[MAX_SENT])], context)

    assert context.outgoing == shas[:MAX_SENT]
    assert f"at most {MAX_SENT} files" in said


async def test_naming_the_same_file_twice_sends_it_once(store: Store, tmp_path: Path) -> None:
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    context = asking()

    await call(store, attachments, [handle(sha)], context)
    said = await call(store, attachments, [handle(sha)], context)

    assert context.outgoing == [sha]
    assert "already going out" in said


async def test_a_blob_collected_mid_turn_is_not_claimed_to_have_been_sent(
    store: Store, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    context = asking(surfaced=[sha])
    await store.raw("DELETE FROM attachment_refs")

    said = await call(store, attachments, [handle(sha)], context)

    assert context.outgoing == []
    assert "no files in this conversation" in said


# -- a surface that cannot send ----------------------------------------------


async def test_a_surface_that_cannot_send_files_refuses_before_it_resolves(
    store: Store, tmp_path: Path
) -> None:
    """The failure this feature exists to avoid is an answer saying "here it
    is" where nothing can be sent. So the refusal comes first, and it does not
    depend on the id naming anything."""
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    context = ToolContext(session_id="s1")

    said = await call(store, attachments, [handle(sha)], context)

    assert context.outgoing == []
    assert "cannot carry files back" in said


# -- and out through the turn ------------------------------------------------


async def test_the_result_carries_the_row_not_the_digest(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """A surface about to upload needs the mime type and the name, and must not
    have to look either of them up."""
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    agent, _ = build(
        store,
        tokenizer,
        [_sends(sha), says("here it is")],
        tools=tools(store, attachments),
        attachments=attachments,
    )

    result = await agent.respond("s1", "send me that shot", can_send_files=True)

    assert [(a.sha256, a.mime, a.name) for a in result.attachments] == [
        (sha, "image/png", "shot.png")
    ]


async def test_a_turn_on_a_mute_surface_ends_with_no_attachments(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    agent, _ = build(
        store,
        tokenizer,
        [_sends(sha), says("I cannot send it, but it showed the whiteboard")],
        tools=tools(store, attachments),
        attachments=attachments,
    )

    result = await agent.respond("s1", "send me that shot")

    assert result.attachments == ()


async def test_a_blob_that_goes_away_before_the_answer_is_dropped(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """The answer is written and about to be posted. Losing the picture is the
    small half of that, and it must not raise."""
    attachments = files(store, tmp_path)
    sha = await stored(attachments)
    agent, _ = build(
        store,
        tokenizer,
        [_sends(sha), says("here it is")],
        tools=tools(store, attachments),
        attachments=attachments,
    )
    await store.raw("DELETE FROM attachment_refs")

    result = await agent.respond("s1", "send me that shot", can_send_files=True)

    assert result.text == "here it is"
    assert result.attachments == ()


def _sends(sha: str) -> ChatResponse:
    """The model asking for one file, as a scripted turn."""
    return ChatResponse(
        message=Message(
            role="assistant",
            content=(ToolUseBlock(id="t1", name="send_file", input={"ids": [handle(sha)]}),),
        ),
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5),
        model="m",
    )
