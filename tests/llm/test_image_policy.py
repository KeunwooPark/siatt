"""Local media limits and endpoint-specific budgeting (#250)."""

import base64
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from PIL import Image
from pydantic import ValidationError

from siatt.config import AttachmentSettings, Config, ProviderConfig, load_config, write_config
from siatt.core.backoff import Backoff
from siatt.core.inbox import Inbox
from siatt.errors import LLMError
from siatt.llm.anthropic_compat import AnthropicCompatProvider
from siatt.llm.images import DEFAULT_IMAGE_POLICY, ImagePolicy, prepare
from siatt.llm.openai_compat import OpenAICompatProvider
from siatt.llm.tokens import HeuristicTokenizer, count_message, image_tokens
from siatt.llm.types import ChatRequest, ImageBlock, Message
from siatt.store import Store
from tests.core.test_inbox import event


def picture(size: tuple[int, int], fmt: str = "JPEG") -> bytes:
    output = BytesIO()
    Image.new("RGB", size, "navy").save(output, fmt)
    return output.getvalue()


@pytest.mark.parametrize("provider_type", [OpenAICompatProvider, AnthropicCompatProvider])
async def test_large_portrait_is_bounded_without_changing_storage(
    store: Store,
    tmp_path: Path,
    provider_type: type[OpenAICompatProvider | AnthropicCompatProvider],
) -> None:
    original = picture((3213, 5712))
    attachments = AttachmentSettings(enabled=True).build(store, tmp_path / "siatt.db")
    sha = await attachments.put(original, mime="image/jpeg", source_name="slack", scope="workspace")
    block = ImageBlock(sha256=sha, mime="image/jpeg", data=await attachments.read(sha))
    provider = provider_type(model="test", api_key="k")
    try:
        payload = provider._payload(
            ChatRequest(messages=(Message.user("look", images=[block]),)), stream=False
        )
        part = payload["messages"][0]["content"][1]
        encoded = (
            part["source"]["data"] if "source" in part else part["image_url"]["url"].split(",")[1]
        )
        with Image.open(BytesIO(base64.b64decode(encoded))) as outgoing:
            w, h = outgoing.size
            assert max(w, h) <= 1568
            assert w * h <= 1_000_000
            assert w / h == pytest.approx(3213 / 5712, abs=0.002)
            assert count_message(
                Message.user("", images=[block]), HeuristicTokenizer()
            ) == 4 + image_tokens(w, h)
        assert await attachments.read(sha) == original
        assert block.data == original
    finally:
        await provider.aclose()


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "GIF", "WEBP"])
def test_supported_still_images(fmt: str) -> None:
    data = picture((80, 60), fmt)
    block = ImageBlock(sha256="a" * 64, mime=f"image/{fmt.lower()}", data=data)
    assert prepare(block, DEFAULT_IMAGE_POLICY).data == data


def test_custom_policy_and_unknown_dimensions() -> None:
    policy = ImagePolicy(max_edge=800, max_pixels=200_000, pixels_per_token=100, token_overhead=50)
    block = ImageBlock(
        sha256="a" * 64, mime="image/jpeg", data=picture((3213, 5712)), width=1, height=1
    )
    prepared = prepare(block, policy)
    assert prepared.width is not None and prepared.height is not None
    assert prepared.width * prepared.height <= 200_000
    expected = policy.tokens(prepared.width, prepared.height)
    assert (
        count_message(Message.user("", images=[block]), HeuristicTokenizer(), (policy,))
        == 4 + expected
    )
    assert image_tokens(None, 12, policy) == 2050
    assert image_tokens(0, 12, policy) == 2050
    assert ImagePolicy(max_pixels=10_000_000, max_edge=5000).tokens(3000, 3000) > 1600


def test_config_builds_policy_for_each_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "k")
    for kind in ("openai", "anthropic"):
        config = ProviderConfig.model_validate(
            {
                "kind": kind,
                "model": "test",
                "key_env": "TEST_KEY",
                "image_policy": {"max_edge": 512},
            }
        )
        provider = config.build()
        assert isinstance(provider, OpenAICompatProvider | AnthropicCompatProvider)
        assert provider.image_policy.max_edge == 512
    with pytest.raises(ValidationError):
        ImagePolicy(max_pixels=0)
    with pytest.raises(ValidationError):
        ImagePolicy(pixels_per_token=0)


def test_corrupt_image_has_actionable_permanent_error() -> None:
    with pytest.raises(LLMError, match="Send a smaller, valid still image") as failure:
        prepare(ImageBlock(sha256="a" * 64, mime="image/png", data=b"broken"), DEFAULT_IMAGE_POLICY)
    assert not failure.value.retryable


@pytest.mark.parametrize("status,attempts", [(400, 1), (503, 5), (429, 5)])
async def test_http_failure_at_inbox_boundary(store: Store, status: int, attempts: int) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status,
            json={
                "error": "A single image or video occupies 16587 prompt tokens, "
                "which exceeds the activation-safe prefill limit of 16384 tokens."
            },
        )

    inbox = Inbox(store, backoff=Backoff(max_attempts=5, base=0, cap=0))
    await inbox.enqueue(event("image"))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://example.test"
    ) as client:
        provider = OpenAICompatProvider(model="test", api_key="k", client=client)
        while leased := await inbox.lease():
            with pytest.raises(LLMError) as failure:
                await provider.complete(ChatRequest(messages=(Message.user("look"),)))
            await inbox.fail(leased[0], failure.value)
    assert calls == attempts
    assert await inbox.counts() == {"failed": 1}
    if status == 400:
        assert "image_policy.max_edge" in (await inbox.dead_letters())[0]["last_error"]


def test_estimate_covers_all_fallback_policies() -> None:
    policies = (ImagePolicy(), ImagePolicy(pixels_per_token=50))
    block = ImageBlock(sha256="a" * 64, mime="image/png", width=800, height=600)
    assert count_message(
        Message.user("", images=[block]), HeuristicTokenizer(), policies
    ) == 4 + policies[1].tokens(800, 600)


def test_thin_image_still_respects_pixel_limit() -> None:
    policy = ImagePolicy(max_edge=1000, max_pixels=10)
    w, h = policy.dimensions(1, 10000)
    assert w * h <= 10


def test_exif_orientation_is_applied() -> None:
    original = Image.new("RGB", (120, 80))
    exif = Image.Exif()
    exif[274] = 6
    output = BytesIO()
    original.save(output, "JPEG", exif=exif)
    result = prepare(
        ImageBlock(sha256="a" * 64, mime="image/jpeg", data=output.getvalue()),
        ImagePolicy(max_edge=60),
    )
    assert (result.width, result.height) == (40, 60)


def test_animated_image_is_not_silently_reduced_to_one_frame() -> None:
    output = BytesIO()
    Image.new("RGB", (20, 20), "red").save(
        output, "GIF", save_all=True, append_images=[Image.new("RGB", (20, 20), "blue")]
    )
    with pytest.raises(LLMError, match="still-frame export"):
        prepare(
            ImageBlock(sha256="a" * 64, mime="image/gif", data=output.getvalue()),
            DEFAULT_IMAGE_POLICY,
        )


def test_custom_primary_and_fallback_policies_round_trip(tmp_path: Path) -> None:
    cfg = Config(
        llm={
            "chat": ProviderConfig(
                kind="openai",
                model="primary",
                image_policy=ImagePolicy(max_edge=800),
                fallbacks=[
                    ProviderConfig(
                        kind="anthropic",
                        model="fallback",
                        image_policy=ImagePolicy(max_pixels=200000, pixels_per_token=250),
                    )
                ],
            )
        }
    )
    path = tmp_path / "config.toml"
    write_config(cfg, path)
    assert load_config(path) == cfg
