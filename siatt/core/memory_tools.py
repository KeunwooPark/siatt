"""The agent's own access to long-term memory.

Pre-injected retrieval handles the common case at no extra latency, and it will
still miss — someone asks about a memory whose wording shares nothing with the
question, or asks a second question the first retrieval did not anticipate.
These three tools are the tail, and shipping both is the point: injection covers
the ninety percent, tools cover the rest.

Two constraints shape all of them:

- **Scope comes from the session, never from the model.** A tool argument that
  could widen visibility would make every other scope check decorative.
- **`memory_write` does not write.** It appends an observation, and the
  `promote` job turns observations into a validated patch plan. The interactive
  path and the background path therefore share one write path — the one that
  has been through the patch validator.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from siatt.core.tools import Tool, ToolContext
from siatt.memory import blobref
from siatt.memory.document import MemoryDoc, MemoryError_, is_memory_id
from siatt.memory.ltm import MemoryStore, MemoryStoreError
from siatt.memory.observation import OBSERVATION_KINDS, Cited, citable
from siatt.memory.retrieve import Retriever, permits, render_snippet
from siatt.memory.subject import normalize_subject
from siatt.store import Store
from siatt.store.blobs import Attachment, Attachments

log = logging.getLogger(__name__)


MAX_SEARCH_LIMIT = 20

WROTE = (
    "Noted. This is queued as a candidate for long-term memory; it is reviewed "
    "and written by the consolidation job, not immediately."
)

#: Attachments one observation may cite. A memory is a claim, not an album: the
#: number here is enough for "the whiteboard, and the sketch we drew after it"
#: and small enough that a turn cannot quietly attach a conversation's worth of
#: photographs to one sentence.
MAX_CITED = 4

#: Pictures one turn may put in front of the model by reading memories.
#:
#: A bound rather than a policy: an image costs tokens by area, a memory may
#: cite several, and a turn may read several memories. Past this the note still
#: names the file and says it was not shown, which is a model that knows what it
#: has not seen rather than one that thinks the picture is gone.
MAX_SHOWN = 4


def memory_tools(
    *,
    retriever: Retriever,
    memory: MemoryStore,
    store: Store,
    attachments: Attachments | None = None,
) -> list[Tool]:
    """The three memory tools, bound to one repo and one database."""
    return [
        _search_tool(retriever),
        _read_tool(memory, attachments),
        _write_tool(store),
    ]


# -- memory_search -----------------------------------------------------------


def _search_tool(retriever: Retriever) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        query = str(args["query"]).strip()
        limit = min(int(args.get("limit", 5)), MAX_SEARCH_LIMIT)

        scope = context.scope
        if hint := args.get("scope_hint"):
            # A hint may narrow the search, never widen it. Asking to search a
            # scope this session cannot see is refused rather than downgraded,
            # so the model is told what happened instead of quietly getting
            # results from somewhere else.
            if not permits(context.scope, str(hint)):
                return f"This conversation cannot search the scope {hint!r}."
            scope = str(hint)

        # Pinned memories are excluded from the pool and the results are read
        # in rank order. Both matter here and nowhere else: an explicit search
        # asked for what matches, and every pinned memory is already in the
        # prompt under `# Pinned memory`. Leading with one answered a question
        # nobody asked, and spent a result slot doing it. A pinned memory that
        # does match the query still ranks, and still gets its pinned bonus.
        # The limit is passed down rather than applied to the result. The
        # retriever's own limit is how many memories fit in a prompt beside the
        # conversation; slicing what it had already packed to that bound meant
        # `limit` could only ever shrink a list of eight, and a request for
        # twenty was answered with eight and no indication of it (#61).
        retrieval = await retriever.retrieve(
            query,
            scope=scope,
            include_pinned=False,
            limit=limit,
            # The same zone the pre-injected recall used. A model that searches
            # for "what did they do yesterday" mid-turn must not land on a
            # different day from the one the turn opened with.
            tz=context.tz,
        )
        # Noted on the turn, so that feedback on the answer reaches what the
        # model went and found as well as what it was handed (#36). A search
        # mid-turn is often where the memory that actually answered the
        # question comes from.
        context.recalled.extend(retrieval.memory_ids)
        if not retrieval.kept:
            return f"No memories matched {query!r}."
        return "\n\n".join(render_snippet(c) for c in retrieval.kept)

    return Tool(
        name="memory_search",
        description=(
            "Search long-term memory. Use it when the working context does not "
            "already contain what you need, or to check a second thing the "
            "conversation has moved on to. Returns ranked snippets, each headed "
            "by its memory id."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in words likely to appear in the memory.",
                },
                "scope_hint": {
                    "type": "string",
                    "description": (
                        "Optional visibility scope to restrict the search to, e.g. "
                        "'channel:C0123'. Can only narrow what this conversation "
                        "may already see."
                    ),
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_LIMIT},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )


# -- memory_read -------------------------------------------------------------


async def _attachment_note(body: str, attachments: Attachments | None, context: ToolContext) -> str:
    """What the store knows about the files this memory points at.

    The link text already carries a name -- that is why references are written
    as `[shot.png](siatt://blob/...)` rather than as a bare URI -- so this adds
    the two things prose cannot: what kind of file it is, and whether it is
    still there. A memory outlives the conversation it came from and can outlive
    the attachment too, and "the photograph this cites is gone" is a fact the
    model should have before it describes one.

    It also *shows* the pictures. A memory that cites a photograph could say its
    mime type and its size and never put it in front of the model, so a
    photograph sent this morning was looked at and the same photograph reached
    through the memory that cites it was a byte count -- which reads, to whoever
    asked, as Siatt having forgotten something it is still holding. What this
    writes into `context.surfaced` is what the agent loop turns into image
    blocks on the turn carrying this result.

    Scoped, like every attachment read. A memory visible from here may cite a
    blob that is not, and the honest answer for that one is the same as for a
    blob that has been collected: it cannot be shown.
    """
    wanted = sorted(blobref.referenced(body))
    if not wanted or attachments is None:
        return ""
    lines = []
    for sha in wanted:
        held = await attachments.get(sha, scope=context.scope)
        if held is None:
            lines.append(f"- {blobref.handle(sha)}… — no longer stored")
            continue
        line = f"- {blobref.handle(sha)}… — {held.mime}, {held.size:,} bytes"
        lines.append(line + _shows(held, context))
    return "\n\n[attachments this memory points at]\n" + "\n".join(lines)


def _shows(held: Attachment, context: ToolContext) -> str:
    """Whether this one is being put in the turn, and what to say if not.

    The cap is per turn rather than per call, counted off the turn's own
    notebook: a model that reads four memories citing ten photographs each is
    the case this protects the context window from, and a limit on one call
    would not see the other three.

    Said out loud either way. "Shown below" is what stops the model describing
    a picture it was only told the size of, and the sentence for one held back
    is what stops it concluding the file is gone.
    """
    if not held.is_image:
        # Stored, referenced, and unreadable by anything -- the same answer the
        # attachment note at ingress gives for a video, for the same reason.
        return ", which nothing can read"
    if held.sha256 in context.surfaced:
        # Already in this turn, on the message that carried the result which
        # first reached it. "Below" would be a lie by one message, and the
        # model looking for a picture that is above it is the model concluding
        # it was not sent one.
        return ", shown earlier in this turn"
    if len(context.surfaced) >= MAX_SHOWN:
        return f", not shown — this turn has already shown {MAX_SHOWN} pictures"
    context.surfaced.append(held.sha256)
    return ", shown below"


def _read_tool(memory: MemoryStore, attachments: Attachments | None = None) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        memory_id = str(args["memory_id"]).strip()
        if not is_memory_id(memory_id):
            return f"{memory_id!r} is not a memory id. Ids look like mem_01K8XQ…"

        entry = memory.manifest().resolve(memory_id)
        if entry is None:
            return f"No memory with id {memory_id}."

        # The scope check happens on the document, not the manifest entry: the
        # entry is a copy, and a copy is a thing that can go stale.
        try:
            raw = memory.read(entry.path)
            doc = MemoryDoc.parse(raw, source=entry.path)
        except (MemoryStoreError, MemoryError_) as exc:
            return f"Could not read {entry.path}: {exc}"

        if not permits(context.scope, doc.frontmatter.visibility):
            log.info("refused a cross-scope memory_read of %s", memory_id)
            return f"No memory with id {memory_id}."

        # After the scope check, never before it: a memory this conversation
        # may not see did not contribute to its answer, and recording it here
        # would put its id in front of whoever reads the feedback.
        context.recalled.append(memory_id)
        return raw + await _attachment_note(doc.body, attachments, context)

    return Tool(
        name="memory_read",
        description=(
            "Read one long-term memory in full, by its id. Use it after "
            "memory_search when a snippet is not enough."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "A mem_… id from a search result."}
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        },
        handler=handler,
    )


# -- memory_write ------------------------------------------------------------


class _Unresolved(Exception):
    """A handle named no attachment this conversation has, or named two.

    An exception rather than a returned pair because there is exactly one thing
    to do about it and one place that knows how to say it. The message is what
    the model reads.
    """


async def _cited(store: Store, handles: Sequence[str], context: ToolContext) -> list[Cited]:
    """The attachments these handles name, or an explanation of why they name none.

    Resolved against this conversation, scoped in the query. That is the whole
    permission model here and it is deliberately narrow: the model can cite a
    file somebody sent *where it is being asked*, and a digest copied from
    anywhere else — an older thread, a memory it read a moment ago, its own
    invention — resolves to nothing. Widening it to every attachment the scope
    can see would make "remember this photo" a way to attach any picture in the
    workspace to any claim.

    The name comes from the row, never from the model. What a memory's link text
    says is what the surface called the file.
    """
    if not handles:
        return []
    rows = await store.attachments_for_session(context.session_id, scope=context.scope)
    known = citable(rows)
    if not known:
        raise _Unresolved(
            "Nothing was attached in this conversation, so there is nothing to cite; "
            "nothing was recorded. Send the claim again with no attachments."
        )

    if len(handles) > MAX_CITED:
        # Refused rather than truncated. The schema says the same thing, and a
        # silent slice here would be the one path where a request to remember
        # five photographs comes back saying four were kept as though it had
        # been asked for four.
        raise _Unresolved(
            f"A memory may cite at most {MAX_CITED} files; nothing was recorded. "
            "Send the claim again with the ones it is actually about."
        )

    cited: dict[str, Cited] = {}
    for given in handles:
        matches = blobref.matching(given, known)
        if len(matches) != 1:
            trouble = "matches more than one file here" if matches else "is not a file from here"
            raise _Unresolved(
                f"{given!r} {trouble}; nothing was recorded. "
                f"The files in this conversation are: {_offered(known)}. "
                "Use one of those ids, or record the claim with no attachments."
            )
        sha = matches[0]
        cited[sha] = Cited(sha256=sha, name=known[sha])
    return list(cited.values())


#: How many files a refusal lists back. A long conversation can hold dozens,
#: and a tool result that names all of them is a context window spent telling a
#: model what it already read in the notes above.
_OFFERED = 8


def _offered(known: Mapping[str, str]) -> str:
    shown = list(known.items())[:_OFFERED]
    listed = ", ".join(f"{blobref.handle(sha)} ({name})" for sha, name in shown)
    rest = len(known) - len(shown)
    return f"{listed}, and {rest} more" if rest else listed


def _write_tool(store: Store) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        kind = str(args["kind"])
        if kind not in OBSERVATION_KINDS:
            return f"kind must be one of {', '.join(OBSERVATION_KINDS)}."

        # Normalized here as well as in the store, because the store cannot
        # answer the model. A subject of "???" is a grouping key of "", and an
        # observation nothing can ever be grouped with is one nobody will read.
        subject = normalize_subject(str(args["subject"]))
        claim = str(args["claim"]).strip()
        # `minLength` in the schema rejects the empty string upstream, the way
        # the `kind` enum does; it cannot see a string of spaces. Both used to
        # get through, and the observation sat `pending` forever with nothing
        # in it — after the model had been told the write succeeded, which is
        # the one answer a write tool must not give for a no-op (#79).
        if not subject or not claim:
            return "subject and claim must each say something; nothing was recorded."

        handles = [str(h) for h in args.get("attachments") or []]
        try:
            cited = await _cited(store, handles, context)
        except _Unresolved as exc:
            # Refused rather than written without them. The claim and the
            # picture were one request, and recording half of it under a
            # success message is how a memory ends up describing a photograph
            # nothing points at — the same no-op-reported-as-a-write #79 fixed
            # for an empty claim. The reply says what *is* citable here, so the
            # retry has somewhere to go.
            return str(exc)

        await store.add_observation(
            subject=subject,
            claim=claim,
            kind=kind,
            # Inherited, never supplied: an observation from a DM stays private
            # even if the model would rather it were general knowledge.
            scope=context.scope,
            session_id=context.session_id,
            attachments=cited,
        )
        if cited:
            return f"{WROTE} It will cite {', '.join(c.name for c in cited)}."
        return WROTE

    return Tool(
        name="memory_write",
        description=(
            "Record something worth remembering beyond this conversation. This "
            "queues a candidate fact for review; it does not write to memory "
            "directly, and nothing you record here is visible to a later "
            "conversation until it has been consolidated. A file somebody sent "
            "in this conversation can be recorded with it: pass the id the "
            "attachment note gave it, and the memory will point at the file "
            "itself rather than only at your description of it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(OBSERVATION_KINDS)},
                "subject": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Who or what this is about, e.g. a person or project name.",
                },
                "claim": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The thing to remember, as one self-contained sentence.",
                },
                "attachments": {
                    "type": "array",
                    "maxItems": MAX_CITED,
                    "items": {"type": "string"},
                    "description": (
                        "Ids of files from this conversation the claim is about, as the "
                        "attachment note gave them. Only what somebody actually sent here; "
                        "an id you did not read is not one."
                    ),
                },
            },
            "required": ["kind", "subject", "claim"],
            "additionalProperties": False,
        },
        handler=handler,
    )
