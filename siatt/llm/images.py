"""Turning an `ImageBlock` into whatever the endpoint on the far side takes.

Both provider families accept a picture inline as base64 and describe it
differently, so the encoding is shared and the shape is not. What *is* shared is
the failure: three separate things can mean "there is no image to send here",
and all three have to come out as words rather than as a 400.

- The block was never hydrated, or the bytes are gone from disk. The agent fills
  `data` in on the way past; a block that arrives here without it is one whose
  file could not be read.
- The endpoint does not take images. An install can point `base_url` at a proxy
  or a local server that is text-only, and `supports_images = false` says so.
- The mime type is one no vision model accepts. PNG, JPEG, GIF and WebP are the
  four both families document; a HEIC straight off a phone is not among them.

A placeholder is always better than an error. The model can say "you sent me a
picture I cannot look at", which is true and useful; a 400 ends the turn with
nothing, and the person who sent the photograph is told only that something
broke.
"""

from __future__ import annotations

import base64
import math
from io import BytesIO

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from siatt.errors import LLMError
from siatt.llm.types import ImageBlock

#: What both families document as accepted. Anything else is described rather
#: than sent -- including everything the *store* is happy to keep, which is a
#: wider set on purpose: a HEIC is worth holding on to even while nothing can
#: read it.
SENDABLE = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


def sendable(block: ImageBlock, supports_images: bool) -> bool:
    return supports_images and block.data is not None and block.mime.lower() in SENDABLE


def described(block: ImageBlock, supports_images: bool) -> str:
    """What to say instead of the picture, and why.

    Named for what it is: this text goes to the model, so it reads as a fact
    about the conversation rather than as an error string.
    """
    what = f"an image ({block.mime})" if block.mime else "an image"
    if not supports_images:
        return f"[{what} was attached; this model cannot be shown images]"
    if block.mime.lower() not in SENDABLE:
        return f"[{what} was attached; that format cannot be shown to a model]"
    return f"[{what} was attached, but its data could not be read]"


def encoded(block: ImageBlock) -> str:
    assert block.data is not None
    return base64.b64encode(block.data).decode("ascii")


class ImagePolicy(BaseModel):
    """Local bounds and a configurable estimate, not a provider-side promise."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_edge: int = Field(default=1568, ge=1)
    max_pixels: int = Field(default=1_000_000, ge=1)
    pixels_per_token: int = Field(default=500, ge=1)
    token_overhead: int = Field(default=256, ge=0)

    def dimensions(self, width: int, height: int) -> tuple[int, int]:
        scale = min(
            1.0,
            min(self.max_edge, self.max_pixels) / max(width, height),
            math.sqrt(self.max_pixels / (width * height)),
        )
        return max(1, int(width * scale)), max(1, int(height * scale))

    def tokens(self, width: int | None, height: int | None) -> int:
        if width is None or height is None or width <= 0 or height <= 0:
            area = min(self.max_pixels, self.max_edge**2)
        else:
            w, h = self.dimensions(width, height)
            area = w * h
        return self.token_overhead + math.ceil(area / self.pixels_per_token)


DEFAULT_IMAGE_POLICY = ImagePolicy()


def prepare(block: ImageBlock, policy: ImagePolicy) -> ImageBlock:
    """Decode and bound a request copy. Never write to the attachment store.

    Animated inputs are rejected explicitly: silently keeping only frame zero
    would remove image access that the user reasonably expects.
    """
    assert block.data is not None
    try:
        with Image.open(BytesIO(block.data)) as original:
            if getattr(original, "is_animated", False):
                raise ValueError("animated images require a still-frame export")
            original.load()
            oriented = ImageOps.exif_transpose(original)
            size = policy.dimensions(*oriented.size)
            if size == original.size and not original.getexif().get(274):
                return block.model_copy(update={"width": size[0], "height": size[1]})
            mode = (
                "RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB"
            )
            resized = oriented.convert(mode).resize(size, Image.Resampling.LANCZOS)
            output = BytesIO()
            resized.save(output, "PNG")
            return block.model_copy(
                update={
                    "data": output.getvalue(),
                    "mime": "image/png",
                    "width": size[0],
                    "height": size[1],
                }
            )
    except (
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise LLMError(
            f"Cannot prepare attached image {block.sha256[:12]}: {exc}. "
            "Send a smaller, valid still image as PNG or JPEG."
        ) from exc


def dimensions(block: ImageBlock) -> tuple[int | None, int | None]:
    """Use the actual header when hydrated; stored dimensions are only a hint."""
    if block.data is not None and block.mime.lower() in SENDABLE:
        try:
            with Image.open(BytesIO(block.data)) as picture:
                return picture.size
        except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            return None, None
    return block.width, block.height
