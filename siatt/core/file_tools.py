"""Sending a file back, which until now was the direction that did not exist.

A photograph arrives, is fetched without handing out the bot token
(`siatt/adapters/slack/files.py`), is hydrated into the turn (`Agent._hydrate`)
and can be cited by a memory and looked at again a week later
(`siatt/memory/blobref.py`). Nothing could send one out: every egress path in
the build ends in a string, so "send me that whiteboard shot again" got a
paragraph about a file sitting on Siatt's own disk.

This is the verb, and it is deliberately only half of one. `send_file` records
*which* file goes back; the surface does the sending, and a surface that cannot
is the first thing this checks. That order matters more than it looks: the
failure worth engineering against here is not an unsent file, it is an answer
that says "here it is" when nothing was sent, and the only place to prevent that
is before the model is told the file is on its way.

Two rules it shares with `memory_write`, for the same reasons:

- **The model names, it does not address.** A handle in, `blobref.matching` out.
  A model that has seen a digest will compose a plausible one, and a digest
  taken on its word is a picture chosen by a hallucination.
- **What it may name is this conversation's.** Files attached here, plus what a
  `memory_read` put in front of it this turn — the second scope-checked before
  it was ever shown. A digest from anywhere else resolves to nothing, because
  the alternative is a way to post a stranger's photograph into a channel by
  guessing twelve characters.
"""

from __future__ import annotations

import logging
from typing import Any

from siatt.core.tools import Tool, ToolContext
from siatt.memory import blobref
from siatt.memory.observation import citable
from siatt.store import Store
from siatt.store.blobs import Attachment, Attachments

log = logging.getLogger(__name__)

#: Files one turn may send. The same kind of bound as `MAX_CITED` and
#: `MAX_SHOWN` (`siatt/core/memory_tools.py`) and the same argument: a turn that
#: has decided to send eleven photographs has misunderstood the request, and the
#: eleventh should be a sentence it can read rather than a thread nobody wanted.
MAX_SENT = 4

#: What the model is told on a surface that ends in a string. It is a refusal
#: with an alternative in it, because the model is mid-answer and the useful
#: next move is to describe the file rather than to apologise for a mechanism.
CANNOT_SEND = (
    "This conversation cannot carry files back, so nothing was sent. "
    "Describe the file instead, or say where it came from."
)


def file_tools(*, store: Store, attachments: Attachments) -> list[Tool]:
    """The one outbound-file tool, bound to one database and one blob store."""
    return [_send_tool(store, attachments)]


def _send_tool(store: Store, attachments: Attachments) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        # First, and before anything is resolved: on a surface with nowhere to
        # put a file, every other answer this could give would be a promise.
        if not context.can_send_files:
            return CANNOT_SEND

        handles = [str(h) for h in args.get("ids") or []]
        if not handles:
            return "Name at least one file id; nothing was sent."

        known = await _sendable(store, attachments, context)
        if not known:
            return (
                "There are no files in this conversation to send, and nothing was sent. "
                "Only a file somebody attached here, or one a memory you read this turn "
                "points at, can go back."
            )

        chosen: dict[str, str] = {}
        for given in handles:
            matches = blobref.matching(given, known)
            if len(matches) != 1:
                trouble = (
                    "matches more than one file here" if matches else "is not a file from here"
                )
                # Nothing is sent, rather than the ones that did resolve. A
                # request for two photographs answered with one is the same
                # half-success `memory_write` refuses (#79): the model reports
                # what it asked for, not what arrived.
                return (
                    f"{given!r} {trouble}; nothing was sent. "
                    f"The files available here are: {_offered(known)}."
                )
            chosen[matches[0]] = known[matches[0]]

        room = MAX_SENT - len(context.outgoing)
        fresh = [sha for sha in chosen if sha not in context.outgoing]
        if len(fresh) > room:
            return (
                f"One answer may send at most {MAX_SENT} files, and this turn has already "
                f"sent {len(context.outgoing)}; nothing was sent. Send the ones it is "
                "actually about."
            )

        lines = []
        for sha in chosen:
            if sha in context.outgoing:
                # Already on the answer. Saying so is what stops a model from
                # concluding the first call failed and asking again.
                lines.append(f"- {chosen[sha]} — already going out with this answer")
                continue
            held = await attachments.get(sha, scope=context.scope)
            if held is None:
                # Between the listing and here: collected, or never visible
                # from this scope in the first place. Either way there are no
                # bytes to send and the model should not say there were.
                lines.append(f"- {chosen[sha]} — no longer stored, and not sent")
                continue
            context.outgoing.append(sha)
            lines.append(f"- {chosen[sha]} — {_described(held)}, going out with this answer")

        return (
            "\n".join(lines)
            + "\n\nNothing has been sent yet: the files go out when you finish this "
            "answer, so write it as something arriving with the message."
        )

    return Tool(
        name="send_file",
        description=(
            "Send a file back with this answer. Use it for a file somebody attached "
            "in this conversation, or one a memory you read this turn points at: pass "
            "the id the attachment note gave it. It does not make files — only "
            "something already here can be sent."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_SENT,
                    "items": {"type": "string"},
                    "description": (
                        "Ids of the files to send, as an attachment note gave them. "
                        "An id you did not read here is not one."
                    ),
                }
            },
            "required": ["ids"],
            "additionalProperties": False,
        },
        handler=handler,
    )


async def _sendable(store: Store, attachments: Attachments, context: ToolContext) -> dict[str, str]:
    """Every file this turn may send, by digest, with what to call it.

    Two sets, and the union is the whole permission model. The conversation's
    own attachments are `memory_write`'s rule (`_cited`), scoped in the query.
    The blobs a `memory_read` surfaced this turn are the reason the feature is
    worth having — "send me the photo from that memory" — and they are safe on
    somebody else's check: `_shows` resolved each one under this scope before it
    was ever put in front of the model.

    The name always comes off the row. What the surface called the file is a
    fact about the upload; what the model calls it is a guess.
    """
    rows = await store.attachments_for_session(context.session_id, scope=context.scope)
    known = citable(rows)
    for sha in context.surfaced:
        if sha in known:
            continue
        held = await attachments.get(sha, scope=context.scope)
        if held is not None:
            known.update(citable([{"sha256": sha, "name": held.name, "mime": held.mime}]))
    return known


def _described(held: Attachment) -> str:
    return f"{held.mime}, {held.size:,} bytes"


#: How many files a refusal lists back, on `_offered`'s argument in
#: `memory_tools`: a long conversation holds dozens, and naming all of them
#: spends the context window telling the model what it has already read.
_OFFERED = 8


def _offered(known: dict[str, str]) -> str:
    shown = list(known.items())[:_OFFERED]
    listed = ", ".join(f"{blobref.handle(sha)} ({name})" for sha, name in shown)
    rest = len(known) - len(shown)
    return f"{listed}, and {rest} more" if rest else listed
