"""The image endpoint: what is sent, what is believed, and what is refused.

The provider's job is small and the failures are all of the quiet kind. It must
send only what this install configured — one endpoint behind this shape rejects
a parameter another requires — and it must not take the response's word for
anything it can read off the bytes.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from siatt.errors import AuthError, ContentFilterError, ProviderProtocolError, RateLimitError
from siatt.imagen.openai_images import OpenAIImages
from tests.conftest import mock_client
from tests.core.test_agent_images import png

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 32


def replying(
    body: dict[str, Any] | None = None,
    *,
    status: int = 200,
    text: str | None = None,
    seen: list[httpx.Request] | None = None,
) -> OpenAIImages:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=body or {})

    return OpenAIImages(
        model="openai/gpt-image-2", api_key="k", client=mock_client(handler), size="1024x1024"
    )


def drawn(raw: bytes, **extra: Any) -> dict[str, Any]:
    return {
        "data": [{"b64_json": base64.b64encode(raw).decode()}],
        "usage": {"input_tokens": 12, "output_tokens": 272},
        **extra,
    }


# -- what goes out -----------------------------------------------------------


async def test_only_the_parameters_this_install_configured_are_sent() -> None:
    """`quality` is unset here, and an endpoint without tiers rejects it."""
    seen: list[httpx.Request] = []
    provider = replying(drawn(png()), seen=seen)

    await provider.generate("a red circle")

    sent = json.loads(seen[0].content)
    assert sent == {
        "model": "openai/gpt-image-2",
        "prompt": "a red circle",
        "n": 1,
        "size": "1024x1024",
    }
    assert seen[0].url.path.endswith("/v1/images/generations")


async def test_quality_is_sent_when_it_is_configured() -> None:
    seen: list[httpx.Request] = []
    provider = OpenAIImages(
        model="m",
        api_key="k",
        quality="medium",
        client=mock_client(lambda r: (seen.append(r), httpx.Response(200, json=drawn(png())))[1]),
    )

    await provider.generate("x")

    assert json.loads(seen[0].content)["quality"] == "medium"


# -- what comes back ---------------------------------------------------------


async def test_the_picture_and_its_usage_come_back() -> None:
    provider = replying(drawn(png(), model="openai/gpt-image-2-2026-01-01"))

    made = await provider.generate("a red circle")

    assert made.data == png()
    assert made.usage.input_tokens == 12
    assert made.usage.output_tokens == 272
    # What actually ran, not what config asked for: a gateway may answer with a
    # more specific name and the meter should record that one.
    assert made.model == "openai/gpt-image-2-2026-01-01"


@pytest.mark.parametrize(("raw", "mime"), [(png(), "image/png"), (JPEG, "image/jpeg")])
async def test_the_type_is_read_from_the_bytes_not_from_the_response(raw: bytes, mime: str) -> None:
    """One endpoint answered PNG and another JPEG, neither saying which."""
    provider = replying(drawn(raw, output_format="png"))

    assert (await provider.generate("x")).mime == mime


async def test_bytes_that_are_not_a_picture_are_refused() -> None:
    provider = replying(drawn(b"not an image at all, just some text"))

    with pytest.raises(ProviderProtocolError, match="not a PNG"):
        await provider.generate("x")


async def test_a_url_in_place_of_the_image_is_refused_rather_than_fetched() -> None:
    """A second request to a host the response chose is not an image."""
    provider = replying({"data": [{"url": "https://example.invalid/a.png"}]})

    with pytest.raises(ProviderProtocolError, match="does not fetch"):
        await provider.generate("x")


async def test_an_empty_response_is_refused() -> None:
    provider = replying({"data": []})

    with pytest.raises(ProviderProtocolError, match="no image"):
        await provider.generate("x")


async def test_unreadable_base64_is_refused() -> None:
    provider = replying({"data": [{"b64_json": "not base64 !!!"}]})

    with pytest.raises(ProviderProtocolError, match="unreadable base64"):
        await provider.generate("x")


async def test_a_missing_usage_block_is_zeros_rather_than_a_failure() -> None:
    """The picture was drawn and paid for; a missing counter is not worth it."""
    provider = replying({"data": [{"b64_json": base64.b64encode(png()).decode()}]})

    made = await provider.generate("x")

    assert made.usage.input_tokens == 0
    assert made.usage.output_tokens == 0


# -- failures, mapped by the transport it shares with the chat providers ------


@pytest.mark.parametrize(
    ("status", "text", "expected"),
    [
        (401, "nope", AuthError),
        (429, "slow down", RateLimitError),
        (400, "your request was rejected by our content policy", ContentFilterError),
    ],
)
async def test_the_status_becomes_the_error_the_rest_of_siatt_knows(
    status: int, text: str, expected: type[Exception]
) -> None:
    provider = replying(status=status, text=text)

    with pytest.raises(expected):
        await provider.generate("x")
