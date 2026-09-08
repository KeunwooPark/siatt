"""Ending a message without ending the turn.

Every answer used to be one message, and nothing said so. A turn that decided
its answer read better in three — *"이번엔 섹션별로 나눠서 짧게 보낼게요"* — wrote the
first section, ended the turn expecting to continue, and the other two were
never written at all (#259). Not a failure anything could report: the model was
not refused, it was simply mistaken about what a turn is.

#258 made a long answer arrive as several messages, cut wherever the character
count fell. This is the other half of the same problem, and the difference is
whose choice the break is. A digest of three sections wants three messages
because it *has* three sections, not because the third one crossed 3,000
characters; and knowing it may have them is what stops the turn ending after
the first.

The shape is `send_file`'s, deliberately, down to the order of the checks:

- **The surface is asked first.** Before anything is queued, before the model
  is told anything. On a terminal an answer is one stream of text and there is
  no second message to send, so the honest answer is a refusal with the reason
  in it rather than a promise nothing will keep.
- **The model names, the surface sends.** Nothing here posts. A part is
  recorded on the turn and delivered with the answer, so a turn that fails
  before it finishes has posted nothing, and the redelivery that follows is the
  same turn rather than half of one plus another whole one.
- **Bounded, like `MAX_SENT`.** A turn that has decided to send eleven messages
  has misunderstood the request.
"""

from __future__ import annotations

import logging
from typing import Any

from siatt.core.tools import Tool, ToolContext

log = logging.getLogger(__name__)

#: Messages one answer may send before the one it ends with. Four, so a digest
#: of three sections and a preamble fits, and the fifth is a sentence the model
#: can read rather than a thread nobody wanted. The same bound and the same
#: argument as `MAX_SENT` in `siatt/core/file_tools.py`.
MAX_PARTS = 4

#: What the model is told where an answer is one continuous piece of text. A
#: refusal with the alternative in it, because the model is mid-answer and the
#: useful next move is to keep writing rather than to apologise for a
#: mechanism.
CANNOT_SEND = (
    "This conversation shows one continuous answer, so there are no separate "
    "messages to send and nothing was queued. Write the whole answer at once; "
    "it will not be cut off."
)


def message_tools() -> list[Tool]:
    """The one tool that ends a message early. It needs nothing to do it."""
    return [_send_tool()]


def _send_tool() -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        # First, and before anything is recorded: on a surface that shows one
        # answer, every other reply this could give would be a promise.
        if not context.sends_messages:
            return CANNOT_SEND

        text = str(args.get("text") or "").strip()
        if not text:
            return "A message needs something in it; nothing was queued."

        if len(context.parts) >= MAX_PARTS:
            return (
                f"One answer may send at most {MAX_PARTS} messages before the one it ends "
                f"with, and this turn has queued {len(context.parts)}; nothing was queued. "
                "Put the rest in the answer you finish with."
            )

        context.parts.append(text)
        remaining = MAX_PARTS - len(context.parts)
        log.debug("queued message %d of %d", len(context.parts), MAX_PARTS)
        return (
            f"Queued as message {len(context.parts)}. It goes out ahead of the answer you "
            f"end this turn with, which is sent too — do not repeat it there. "
            + (
                f"{remaining} more may be queued."
                if remaining
                else "That was the last one that may be queued."
            )
        )

    return Tool(
        name="send_message",
        description=(
            "End one message of a multi-part answer and keep going. The text is sent as "
            "its own message, ahead of the answer you finish the turn with — which is "
            "sent as well, so it must not repeat what you queued here.\n\n"
            "For an answer that has parts: a digest with three sections, or a long piece "
            "of work you want to hand over as you go rather than in one wall of text. Use "
            "it when the break is one the reader would want, not to avoid a length limit "
            "— a long answer is already delivered in as many messages as it needs.\n\n"
            f"At most {MAX_PARTS} before the one you end with. On a surface that shows a "
            "single continuous answer this does nothing and says so."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The message to send now. Markdown, as an answer is.",
                }
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        handler=handler,
    )
