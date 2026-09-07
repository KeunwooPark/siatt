"""`image_generate`, and the three things it settles before it draws anything.

The order in this handler is the design. Generation is the first tool in Siatt
that spends real money on one call, produces bytes that have to be kept, and
then depends on a *surface* to deliver what it made -- so each of those is
checked before the request goes out, not after:

- **The surface can carry a file.** `send_file`'s rule, and its reason: the
  failure worth engineering against is not a missing picture, it is an answer
  that says *"here it is"* where nothing can be sent.
- **The answer has room for it.** `MAX_SENT` is shared with `send_file` and
  counted off the same notebook, because they are one budget: what one answer
  may carry.
- **The ceiling has not been reached.** `web_search`'s order, for a larger
  version of its reason. A request that has been billed cannot be unspent, and
  an image is worth many turns rather than a fraction of one.

What it makes goes out with the answer rather than waiting to be named again. A
model that has drawn a picture on request has already decided it should be sent,
and the alternative invents a class of turn that generates an image and forgets
to deliver it.

What it makes is also **not evidence**, and the store is told so. See
`siatt/memory/observation.py`: a generated file carries `source = "generated"`
and nothing that writes a memory is offered one.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from siatt.core.file_tools import CANNOT_SEND, MAX_SENT
from siatt.core.tools import Tool, ToolContext
from siatt.errors import LLMError
from siatt.imagen.base import ImageProvider
from siatt.llm.cost import CostMeter
from siatt.llm.types import Usage
from siatt.memory import blobref
from siatt.memory.observation import GENERATED
from siatt.store.blobs import AttachmentError, Attachments

log = logging.getLogger(__name__)

OVER_BUDGET = (
    "Drawing is unavailable: today's spend ceiling has been reached. Nothing was "
    "drawn. Describe the picture in words, and say that you could not make one."
)

FULL = (
    f"An answer may carry at most {MAX_SENT} files and this one is already full; "
    "nothing was drawn. Send the picture it is actually about."
)

NO_PROMPT = "A picture needs a description of what to draw; nothing was drawn."

_DESCRIPTION = (
    "Draw a picture from a description and send it back with this answer. Use it "
    "when a picture is what was asked for — a diagram, an illustration, a mockup — "
    "not to reproduce something that exists, which it cannot do. It invents the "
    "image, so nothing it draws is evidence of anything, and it must never be "
    "offered as a photograph, a screenshot or a record. One call makes one picture."
)

#: What a generated file is called, before its extension. It comes out of the
#: prompt so that a person receiving it can tell one from another, and it is
#: rebuilt out of `[a-z0-9]` words rather than used as written: the prompt is
#: the model's own text, and a filename is somewhere text gets shown.
_WORDS = re.compile(r"[A-Za-z0-9]+")
_STEM_WORDS = 6
_STEM_CHARS = 48

_SUFFIX = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


def image_generate_tool(
    *,
    provider: ImageProvider,
    attachments: Attachments,
    meter: CostMeter | None = None,
    timeout: float = 120.0,
) -> Tool:
    """The tool, bound to one endpoint and one blob store.

    `meter` is the same one the model calls go through, so a drawing lands in
    `llm_calls` beside them and the existing daily ceiling covers both. Unlike
    search, this is priced per token rather than per call: the endpoint reports
    usage, so `[pricing]` prices it on the model prefix like everything else.
    """

    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        # First, and before anything is spent: on a surface with nowhere to put
        # a file, every other answer this could give would be a promise.
        if not context.can_send_files:
            return CANNOT_SEND
        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return NO_PROMPT
        if len(context.outgoing) >= MAX_SENT:
            return FULL
        if meter is not None and await meter.daily_ceiling_reached():
            return OVER_BUDGET

        started = time.monotonic()
        try:
            made = await provider.generate(prompt)
        except LLMError as exc:
            await _record(meter, provider, context, started, Usage(), error=str(exc))
            # Raised rather than returned, so the result is marked `is_error`
            # and the model can tell a refused prompt from an ugly picture. The
            # registry catches it; nothing reaches the turn loop.
            raise

        await _record(meter, provider, context, started, made.usage, model=made.model)

        name = _named(prompt, made.mime)
        try:
            sha = await attachments.put(
                made.data,
                mime=made.mime,
                # Not the surface it will go out on. This is where the bytes
                # came from, and it is the whole of how anything downstream
                # tells a picture Siatt drew from one somebody took.
                source_name=GENERATED,
                scope=context.scope,
                session_id=context.session_id,
                author=context.author,
                name=name,
            )
        except AttachmentError as exc:
            # An install that narrowed `allowed_mime`, or a cap smaller than
            # what the endpoint drew. The picture exists and has been paid for,
            # and there is nowhere to keep it -- which the model should say
            # rather than claim a file is coming.
            log.info("could not keep a generated image: %s", exc)
            return f"The picture was drawn but could not be kept, so nothing was sent: {exc}"

        context.outgoing.append(sha)
        # Deliberately not put in front of the model as well. It costs tokens by
        # area on every drawing turn, and a model looking at what it just asked
        # for learns close to nothing -- the case for showing a picture is a
        # memory that cites one (`_shows`), where the model has not seen it.
        return (
            f"Drew {name} — {made.mime}, {len(made.data):,} bytes, "
            f"{blobref.handle(sha)} — going out with this answer.\n\n"
            "Nothing has been sent yet: the file goes out when you finish this answer, "
            "so write it as something arriving with the message. It is a picture you "
            "made up, so do not describe it as a photograph or as a record of anything."
        )

    return Tool(
        name="image_generate",
        description=_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "What to draw, described in full. Nothing else in the "
                        "conversation reaches the endpoint, so say everything that "
                        "matters: subject, style, composition, and any text to appear."
                    ),
                }
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
        handler=handler,
        # Comfortably above the provider's own read timeout, so a slow drawing
        # fails with the provider's message rather than the dispatcher's
        # stopwatch. Far above `DEFAULT_TOOL_TIMEOUT`, which no image endpoint
        # meets.
        timeout=timeout,
    )


def _named(prompt: str, mime: str) -> str:
    """A filename for a picture nobody uploaded.

    Slack shows the filename and the terminal prints it, so `attachment.png` --
    what `_filename` falls back to with no name at all -- makes four drawings in
    a thread indistinguishable. The stem is rebuilt from the prompt's words
    rather than sliced out of it, which is what keeps a separator, a newline or
    a leading dot from ever reaching a name.
    """
    words = _WORDS.findall(prompt.lower())[:_STEM_WORDS]
    stem = "-".join(words)[:_STEM_CHARS].strip("-") or "picture"
    return f"{stem}.{_SUFFIX.get(mime) or mime.partition('/')[2] or 'bin'}"


async def _record(
    meter: CostMeter | None,
    provider: ImageProvider,
    context: ToolContext,
    started: float,
    usage: Usage,
    *,
    model: str | None = None,
    error: str | None = None,
) -> None:
    if meter is None:
        return
    await meter.record(
        role="image",
        provider=provider.name,
        model=model or provider.model,
        usage=usage,
        latency_ms=int((time.monotonic() - started) * 1000),
        tag="image_generate",
        ok=error is None,
        error=error,
        session_id=context.session_id,
    )
