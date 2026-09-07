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
