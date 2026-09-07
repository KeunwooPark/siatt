"""`POST /v1/images/generations`, as OpenAI shaped it and as gateways copy it.

Built on `HTTPProvider` (`siatt/llm/base.py`) rather than on bare httpx, for one
thing that is worth more than the transport: `_error_for` already turns a status
into the failure the rest of Siatt understands. A rejected key is an `AuthError`,
a 429 is a `RateLimitError` carrying `retry-after`, and a refused prompt is a
`ContentFilterError` -- which the tool needs, because "draw that" refused for
content is a sentence to say and not a fault to retry.

Two of the base class's attributes mean nothing here (`supports_images`,
`image_policy`: they describe sending a picture *to* a model). They are left at
their defaults rather than worked around, because the alternative is copying the
error mapping into a second place where it can drift.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

import httpx

from siatt.errors import ProviderProtocolError
from siatt.imagen.base import GeneratedImage
from siatt.llm.base import HTTPProvider
from siatt.llm.types import Usage
from siatt.store.dimensions import mime_for

GENERATIONS_PATH = "/images/generations"

#: Generation is slower than a chat turn by an order of magnitude, and the read
#: timeout has to allow for the slowest tier of the most expensive model rather
#: than the median. The tool's own timeout is what actually bounds the turn.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)


class OpenAIImages(HTTPProvider):
    """One endpoint, one model, and the parameters this install settled on."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        name: str = "openai",
        size: str | None = None,
        quality: str | None = None,
        timeout: httpx.Timeout | float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout or DEFAULT_TIMEOUT,
            client=client,
        )
        self._size = size
        self._quality = quality

    async def generate(self, prompt: str) -> GeneratedImage:
        # Only what config named. `quality` is not a parameter every endpoint
        # behind this shape has -- measured, `google/gemini-3.1-flash-image`
        # rejects it outright rather than ignoring it -- so a default sent on
        # its behalf would be an install that cannot use half its own models.
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "n": 1}
        if self._size:
            payload["size"] = self._size
        if self._quality:
            payload["quality"] = self._quality

        body = await self._post(GENERATIONS_PATH, payload)
        raw = self._bytes(body)
        return GeneratedImage(
            data=raw,
            mime=self._mime(raw),
            model=str(body.get("model") or self.model),
            usage=_usage(body.get("usage")),
        )

    # -- reading the response ------------------------------------------------

    def _bytes(self, body: dict[str, Any]) -> bytes:
        data = body.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ProviderProtocolError("the image endpoint returned no image", provider=self.name)
        encoded = data[0].get("b64_json")
        if not isinstance(encoded, str) or not encoded:
            # The other documented shape is a url to fetch, and Siatt does not
            # take it: a second request, to a host chosen by the response,
            # bypassing `siatt/fetch/guard.py`, is a larger thing than an image.
            raise ProviderProtocolError(
                "the image endpoint returned a reference rather than the image itself, "
                "and Siatt does not fetch it",
                provider=self.name,
            )
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProviderProtocolError(
                f"the image endpoint returned unreadable base64: {exc}", provider=self.name
            ) from exc

    def _mime(self, raw: bytes) -> str:
        """Read from the bytes, because the response does not say.

        `output_format` comes back from one endpoint and not from the other, and
        the one that omits it returned a JPEG while the default everywhere says
        PNG. A type taken on trust here is one that surfaces much later, as a
        picture a vision model refuses or a file Slack renders as a download.
        """
        mime = mime_for(raw)
        if mime is None:
            raise ProviderProtocolError(
                "the image endpoint returned bytes that are not a PNG, JPEG, GIF or WebP",
                provider=self.name,
            )
        return mime


def _usage(raw: object) -> Usage:
    """Tokens, when the endpoint reports them, and zeros when it does not.

    Tolerant on purpose, like `siatt/search/brave.py:_results`: an unpriced or
    uncounted call is a gap in `siatt cost`, and failing the turn over a missing
    counter would throw away a picture that was already drawn and already paid
    for.
    """
    if not isinstance(raw, dict):
        return Usage()
    return Usage(
        input_tokens=_count(raw.get("input_tokens")),
        output_tokens=_count(raw.get("output_tokens")),
    )


def _count(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0
