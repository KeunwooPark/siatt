"""Ending a message without ending the turn.

The tool records; nothing here posts. What is worth testing is the order of
its checks — the surface before anything else — and that the bound is a bound.
"""

from __future__ import annotations

from siatt.core.message_tools import CANNOT_SEND, MAX_PARTS, message_tools
from siatt.core.tools import ToolContext
from siatt.llm.tokens import Tokenizer
from siatt.llm.types import ChatResponse, Message, ToolUseBlock, Usage
from siatt.store import Store
from tests.core.test_agent import build, says


def tool():
    (send,) = message_tools()
    return send


def context(**kwargs) -> ToolContext:
    return ToolContext(session_id="S1", sends_messages=True, **kwargs)


async def test_a_queued_message_is_recorded_in_order() -> None:
    ctx = context()

    await tool().handler({"text": "section ①"}, ctx)
    await tool().handler({"text": "section ②"}, ctx)

    assert ctx.parts == ["section ①", "section ②"]


async def test_the_surface_is_asked_before_anything_is_recorded() -> None:
    """`send_file`'s rule, for the same reason: the failure worth preventing is
    not an unsent message, it is a turn that stopped believing one was coming."""
    ctx = ToolContext(session_id="S1", sends_messages=False)

    assert await tool().handler({"text": "section ①"}, ctx) == CANNOT_SEND
    assert ctx.parts == [], "nothing was recorded on a surface that cannot send it"


async def test_a_refusal_says_what_to_do_instead() -> None:
    """Mid-answer, the useful next move is to keep writing."""
    assert "Write the whole answer at once" in CANNOT_SEND


async def test_an_empty_message_is_not_queued() -> None:
    ctx = context()

    assert "needs something in it" in await tool().handler({"text": "   "}, ctx)
    assert ctx.parts == []


async def test_the_budget_is_a_budget() -> None:
    ctx = context()
    for n in range(MAX_PARTS):
        await tool().handler({"text": f"part {n}"}, ctx)

    answer = await tool().handler({"text": "one too many"}, ctx)

    assert f"at most {MAX_PARTS}" in answer
    assert len(ctx.parts) == MAX_PARTS, "and the last one was not quietly dropped in"


async def test_the_turn_is_told_the_answer_is_sent_too() -> None:
    """The failure this prevents is a turn that queues every section and then
    repeats the last one as its answer."""
    answer = await tool().handler({"text": "section ①"}, context())

    assert "do not repeat it there" in answer


async def test_the_turn_is_told_how_much_room_is_left() -> None:
    ctx = context()
    for n in range(MAX_PARTS - 1):
        await tool().handler({"text": f"part {n}"}, ctx)

    assert "That was the last one" in await tool().handler({"text": "last"}, ctx)


# -- through a turn -----------------------------------------------------------


def _queues(*texts: str) -> ChatResponse:
    """The model ending one message and meaning to write another."""
    return ChatResponse(
        message=Message(
            role="assistant",
            content=tuple(
                ToolUseBlock(id=f"t{n}", name="send_message", input={"text": text})
                for n, text in enumerate(texts)
            ),
        ),
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5),
        model="m",
    )


async def test_a_turn_carries_out_the_messages_it_queued(
    store: Store, tokenizer: Tokenizer
) -> None:
    agent, _ = build(
        store,
        tokenizer,
        [_queues("section ①", "section ②"), says("section ③")],
        tools=message_tools(),
    )

    result = await agent.respond("s1", "the digest please", sends_messages=True)

    assert result.parts == ("section ①", "section ②")
    assert result.text == "section ③", "and the answer is the last message, not the only one"


async def test_a_surface_that_shows_one_answer_carries_none(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The default. A new surface is mute about splitting rather than promising
    something nobody wired up."""
    agent, _ = build(
        store, tokenizer, [_queues("section ①"), says("all of it")], tools=message_tools()
    )

    result = await agent.respond("s1", "the digest please")

    assert result.parts == ()
    assert result.text == "all of it"


async def test_a_turn_that_said_everything_it_meant_to_is_not_reported_as_silent(
    store: Store, tokenizer: Tokenizer
) -> None:
    """`note` reads the answer to decide whether the turn produced one. A turn
    that queued its sections and had nothing left to add ends with no text, and
    saying "the model returned nothing" under three messages of digest would
    contradict what the reader can see."""
    agent, _ = build(store, tokenizer, [_queues("all of it"), says("")], tools=message_tools())

    result = await agent.respond("s1", "the digest please", sends_messages=True)

    assert result.parts == ("all of it",)
    assert result.note is None
