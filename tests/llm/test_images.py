"""An image through the canonical types and out to both provider families."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import pytest
from PIL import Image

from siatt.llm.anthropic_compat import AnthropicCompatProvider
from siatt.llm.images import DEFAULT_IMAGE_POLICY
from siatt.llm.openai_compat import OpenAICompatProvider
from siatt.llm.tokens import HeuristicTokenizer, count_message, image_tokens
from siatt.llm.types import ChatRequest, ImageBlock, Message, TextBlock, ToolResultBlock

_output = BytesIO()
Image.new("RGB", (800, 600)).save(_output, "PNG")
PIXELS = _output.getvalue()


def block(**kwargs: Any) -> ImageBlock:
    fields: dict[str, Any] = {
        "sha256": "a" * 64,
        "mime": "image/png",
        "width": 800,
        "height": 600,
        "data": PIXELS,
    }
    return ImageBlock(**{**fields, **kwargs})


def anthropic(*, vision: bool = True) -> AnthropicCompatProvider:
    return AnthropicCompatProvider(model="claude-opus-5", api_key="k", supports_images=vision)


def openai(*, vision: bool = True) -> OpenAICompatProvider:
    return OpenAICompatProvider(model="gpt-4o", api_key="k", supports_images=vision)


def turn(*blocks: Any) -> ChatRequest:
    return ChatRequest(messages=(Message(role="user", content=tuple(blocks)),))


# -- the wire ----------------------------------------------------------------


def test_anthropic_sends_an_image_block() -> None:
    payload = anthropic()._payload(turn(TextBlock(text="what's this?"), block()), stream=False)

    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what's this?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(PIXELS).decode(),
                    },
                },
            ],
        }
    ]


def test_openai_sends_a_data_uri_part() -> None:
    payload = openai()._payload(turn(TextBlock(text="what's this?"), block()), stream=False)

    encoded = base64.b64encode(PIXELS).decode()
    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what's this?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ],
        }
    ]


def test_a_turn_with_no_image_is_unchanged_on_both() -> None:
    """The wire format must not move for installs that never send one."""
    plain = turn(TextBlock(text="hello"))

    assert openai()._payload(plain, stream=False)["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    assert anthropic()._payload(plain, stream=False)["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_tool_results_still_serialize_alongside_an_image() -> None:
    """A `tool` role message and a picture are the two things that make an
    OpenAI user turn unusual, and they must not collide."""
    mixed = ChatRequest(
        messages=(
            Message(
                role="user",
                content=(ToolResultBlock(tool_use_id="t1", content="42"), TextBlock(text="and?")),
            ),
        )
    )

    messages = openai()._payload(mixed, stream=False)["messages"]

    assert messages[0] == {"role": "tool", "tool_call_id": "t1", "content": "42"}
    assert messages[1] == {"role": "user", "content": "and?"}


def test_a_picture_rides_out_with_the_tool_result_it_came_with() -> None:
    """What #244 rests on. A tool result is text, so an image a tool found is
    carried by the turn holding that result — and each family already has a
    place to put it. Anthropic: one user turn, results first. OpenAI: the
    results become `tool` messages and the picture a user turn after them."""
    carried = ChatRequest(
        messages=(
            Message.tool_results(
                [ToolResultBlock(tool_use_id="t1", content="the memory")], images=[block()]
            ),
        )
    )

    anthropic_turn = anthropic()._payload(carried, stream=False)["messages"][0]
    assert [part["type"] for part in anthropic_turn["content"]] == ["tool_result", "image"]

    openai_messages = openai()._payload(carried, stream=False)["messages"]
    assert openai_messages[0] == {"role": "tool", "tool_call_id": "t1", "content": "the memory"}
    assert openai_messages[1]["role"] == "user"
    assert [part["type"] for part in openai_messages[1]["content"]] == ["image_url"]


# -- degrading rather than 400ing --------------------------------------------


@pytest.mark.parametrize("vision", [True, False])
def test_an_unhydrated_block_becomes_words(vision: bool) -> None:
    """A block off disk always has `data is None`. If nothing filled it in, the
    file could not be read — and the model should be told that in a sentence
    rather than the turn ending in a provider error."""
    payload = anthropic(vision=vision)._payload(turn(block(data=None)), stream=False)

    said = payload["messages"][0]["content"][0]
    assert said["type"] == "text"
    assert "image" in said["text"]


def test_a_model_that_cannot_see_is_told_in_words() -> None:
    payload = anthropic(vision=False)._payload(turn(block()), stream=False)
    openai_payload = openai(vision=False)._payload(turn(block()), stream=False)

    assert "cannot be shown images" in payload["messages"][0]["content"][0]["text"]
    assert "cannot be shown images" in openai_payload["messages"][0]["content"][0]["text"]


def test_a_format_no_model_takes_is_described(store: None = None) -> None:
    """A HEIC straight off a phone is worth storing and cannot be sent."""
    heic = block(mime="image/heic")

    payload = anthropic()._payload(turn(heic), stream=False)

    assert "cannot be shown to a model" in payload["messages"][0]["content"][0]["text"]


# -- budgeting ---------------------------------------------------------------


def test_an_image_is_counted_by_area_not_by_its_json() -> None:
    """A picture serializes to a hash and a mime type. Counted as text it costs
    about thirty tokens and actually costs hundreds or thousands, which is how
    a context that looked like it fit arrives as a 400."""
    counted = count_message(Message(role="user", content=(block(),)), HeuristicTokenizer())

    assert counted > 600  # 800*600/750 = 640, plus message overhead


def test_dimensions_nobody_could_read_cost_the_ceiling() -> None:
    assert image_tokens(None, None) == DEFAULT_IMAGE_POLICY.tokens(None, None)
    assert image_tokens(0, 100) == DEFAULT_IMAGE_POLICY.tokens(None, None)


def test_large_image_estimate_uses_local_bounds() -> None:
    assert image_tokens(6000, 4000) <= DEFAULT_IMAGE_POLICY.tokens(None, None)


# -- persistence -------------------------------------------------------------


def test_the_bytes_are_never_written_to_the_transcript() -> None:
    """`messages.content` is JSON in SQLite. A base64 payload in there puts the
    blob back in the database through the back door."""
    stored = Message.user("look", images=[block()]).model_dump_json()

    assert base64.b64encode(PIXELS).decode() not in stored
    assert "data" not in stored
    assert "a" * 64 in stored


def test_a_block_read_back_has_no_data() -> None:
    message = Message.user("look", images=[block()])

    reloaded = Message.model_validate_json(message.model_dump_json())

    assert reloaded.images[0].data is None
    assert (reloaded.images[0].width, reloaded.images[0].height) == (800, 600)


def test_text_comes_before_the_picture() -> None:
    """Both families read a turn in order, and an image with the question after
    it reads as an image with a caption."""
    message = Message.user("what's this?", images=[block()])

    assert [b.type for b in message.content] == ["text", "image"]
