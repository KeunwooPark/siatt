"""What an image backend is, in terms that name no vendor.

One method and one return type. Everything that varies between endpoints --
what a size string may say, whether there are quality tiers at all -- is
configuration held by the provider, not an argument the caller has to know how
to fill in. The tool asks for a picture of something; how this install draws one
is settled before the turn starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from siatt.llm.types import Usage


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """One picture, as measured rather than as promised.

    `mime` is read from the bytes and never from the response: measured, one
    endpoint answered with a PNG and another with a JPEG, and neither labelled
    which. A stored image whose declared type came from a hopeful default is one
    that reaches a vision model as a 400.
    """

    data: bytes
    mime: str
    #: What actually drew it, as the endpoint reported it. Config names a model
    #: and a gateway may answer with a more specific one, and the meter should
    #: record what ran.
    model: str
    #: In tokens, which is why this meters through `CostMeter` beside every
    #: other call rather than needing a per-call price of its own the way
    #: `[search]` does.
    usage: Usage


class ImageProvider(Protocol):
    """An image generation backend."""

    name: str
    model: str

    async def generate(self, prompt: str) -> GeneratedImage: ...

    async def aclose(self) -> None: ...
