"""The agent's memory tools, and the two acceptance criteria for #16.

The interesting cases are not "does search return results". They are: can the
agent recover when pre-injection missed, and can `memory_write` be talked into
touching the repository. The answers have to be yes and no.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from siatt.config import AttachmentSettings
from siatt.core.agent import Agent
from siatt.core.context import RETRIEVED_HEADER, ContextPacker
from siatt.core.memory_tools import MAX_CITED, MAX_SHOWN, memory_tools
from siatt.core.tools import ToolContext, ToolRegistry
from siatt.llm.registry import ModelRole, ProviderRegistry
from siatt.llm.tokens import HeuristicTokenizer, Tokenizer
from siatt.llm.types import (
    ChatRequest,
    ChatResponse,
    Delta,
    MessageStop,
    TextDelta,
    ToolResultBlock,
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
    ToolUseStop,
    Usage,
)
from siatt.memory import blobref
from siatt.memory.blobref import handle
from siatt.memory.bootstrap import bootstrap
from siatt.memory.document import MemoryDoc, new_memory_id
from siatt.memory.gitcmd import GitRepo
from siatt.memory.index import MemoryIndex
from siatt.memory.ltm import MemoryStore
from siatt.memory.manifest import Manifest
from siatt.memory.retrieve import Retriever
from siatt.store import Store
from siatt.store.blobs import Attachments


@pytest.fixture
def tokenizer() -> Tokenizer:
    return HeuristicTokenizer()


class Memory:
    """A memory repo, its index, the attachment store, and the tools on all three."""

    def __init__(
        self,
        root: Path,
        store: Store,
        registry: ToolRegistry,
        repo: GitRepo,
        attachments: Attachments,
    ) -> None:
        self.root = root
        self.store = store
        self.registry = registry
        self.repo = repo
        self.attachments = attachments

    async def call(self, name: str, args: dict[str, Any], **context: str) -> str:
        result = await self.registry.dispatch(
            ToolUseBlock(id="t1", name=name, input=args), ToolContext(**context)
        )
        return result.content


@pytest.fixture
async def memory(tmp_path: Path, store: Store, tokenizer: Tokenizer) -> AsyncIterator[Memory]:
    root = tmp_path / "ltm"
    repo = GitRepo.init(root, branch="main")
    bootstrap(root)

    for doc in (
        MemoryDoc.new(
            type="person",
            title="Jane Okafor",
            tags=["infra"],
            body="Jane owns the deploy pipeline and reviews every change to it.",
        ),
        MemoryDoc.new(
            type="fact",
            title="Salary review outcome",
            body="They were moved to band 5 in July.",
            visibility="private:U01",
        ),
    ):
        target = root / doc.suggested_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.render())
    Manifest.rebuild(root)[0].save(root)
    repo.commit("memory: seed")
    await MemoryIndex(store, root).reindex()
    ltm = MemoryStore(repo, store, branch="main", push=False)
    attachments = AttachmentSettings(enabled=True).build(store, tmp_path / "siatt.db")
    registry = ToolRegistry(
        memory_tools(
            retriever=Retriever(store, tokenizer=tokenizer),
            memory=ltm,
            store=store,
            attachments=attachments,
        )
    )
    yield Memory(root, store, registry, repo, attachments)


def id_of(memory: Memory, path: str) -> str:
    return MemoryDoc.parse((memory.root / path).read_text()).id


# -- memory_search -----------------------------------------------------------


async def test_search_returns_snippets_with_ids(memory: Memory) -> None:
    result = await memory.call("memory_search", {"query": "who owns the deploy pipeline"})

    assert "Jane" in result
    assert f"[[{id_of(memory, 'memory/people/jane-okafor.md')}]]" in result


async def test_search_says_so_when_nothing_matches(memory: Memory) -> None:
    result = await memory.call("memory_search", {"query": "zygomorphic quinquagenarian"})
    assert "No memories matched" in result


async def test_search_respects_the_session_scope(memory: Memory) -> None:
    """A DM-scoped memory must not surface in a workspace conversation."""
    secret = id_of(memory, "memory/facts/salary-review-outcome.md")

    public = await memory.call("memory_search", {"query": "salary band"})
    assert secret not in public

    private = await memory.call("memory_search", {"query": "salary band"}, scope="private:U01")
    assert secret in private


async def test_a_scope_hint_cannot_widen_access(memory: Memory) -> None:
    """The one argument a model could use to escape its own scope."""
    result = await memory.call(
        "memory_search", {"query": "salary band", "scope_hint": "private:U01"}
    )

    assert id_of(memory, "memory/facts/salary-review-outcome.md") not in result
    assert "cannot search the scope" in result


async def test_a_scope_hint_may_narrow(memory: Memory) -> None:
    result = await memory.call(
        "memory_search",
        {"query": "salary band", "scope_hint": "private:U01"},
        scope="private:U01",
    )
    assert id_of(memory, "memory/facts/salary-review-outcome.md") in result


async def test_the_search_limit_is_capped(memory: Memory) -> None:
    result = await memory.call("memory_search", {"query": "deploy", "limit": 9999})
    assert result  # the schema caps it; the handler clamps it too


async def stock(memory: Memory, count: int, term: str) -> None:
    """Add `count` memories that all match `term`, and reindex."""
    for n in range(count):
        doc = MemoryDoc.new(
            type="fact",
            title=f"{term.title()} runbook step {n}",
            body=f"Step {n} of the {term} runbook is owned by the platform team.",
        )
        target = memory.root / doc.suggested_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.render())
    await MemoryIndex(memory.store, memory.root).reindex()


def snippet_count(result: str) -> int:
    return result.count("[[mem_")


async def test_search_returns_as_many_results_as_it_asked_for(memory: Memory) -> None:
    """#61. The schema advertises 20; the retriever's packing limit capped it at 8."""
    await stock(memory, 15, "escalation")

    twelve = await memory.call("memory_search", {"query": "escalation runbook", "limit": 12})
    assert snippet_count(twelve) == 12


async def test_search_returns_five_when_it_asks_for_nothing(memory: Memory) -> None:
    """The default is the tool's, and passing it down must not have changed it."""
    await stock(memory, 15, "escalation")

    default = await memory.call("memory_search", {"query": "escalation runbook"})
    assert snippet_count(default) == 5


async def test_a_limit_below_one_is_refused_before_the_handler_sees_it(memory: Memory) -> None:
    """Why the handler needs no floor: `limit` now bounds packing, and a zero
    there would report an empty pack as "nothing matched"."""
    result = await memory.call("memory_search", {"query": "deploy pipeline", "limit": 0})
    assert "less than the minimum" in result


async def pin(memory: Memory, **fields: Any) -> str:
    """Add a pinned memory to the corpus and reindex, returning its id."""
    doc = MemoryDoc.new(pinned=True, **fields)
    target = memory.root / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    await MemoryIndex(memory.store, memory.root).reindex()
    return doc.id


async def test_search_does_not_lead_with_an_unrelated_pinned_memory(memory: Memory) -> None:
    """#57. The tool promises ranked snippets, so the top hit must be the best one."""
    standing = await pin(
        memory,
        type="fact",
        title="House style",
        body="Answer in British English and keep replies under four sentences.",
    )

    result = await memory.call("memory_search", {"query": "who owns the deploy pipeline"})

    assert result.startswith(f"[[{id_of(memory, 'memory/people/jane-okafor.md')}]]")
    assert standing not in result


async def test_a_pinned_memory_still_ranks_when_it_matches(memory: Memory) -> None:
    """Excluded from the pool, not from the results: the query can still find it."""
    standing = await pin(
        memory,
        type="fact",
        title="Deploy freeze",
        body="No deploys go out between the 20th of December and the 2nd of January.",
    )

    result = await memory.call("memory_search", {"query": "deploy freeze over christmas"})
    assert standing in result


async def test_pinned_memories_do_not_crowd_out_the_answer(memory: Memory) -> None:
    """Five standing instructions used to fill the default limit exactly."""
    for n in range(5):
        await pin(
            memory,
            type="fact",
            title=f"House style {n}",
            body=f"Standing instruction number {n}, about nothing in particular.",
        )

    result = await memory.call("memory_search", {"query": "who owns the deploy pipeline"})
    assert "Jane" in result


# -- memory_read -------------------------------------------------------------


async def test_read_returns_the_whole_file(memory: Memory) -> None:
    jane = id_of(memory, "memory/people/jane-okafor.md")
    result = await memory.call("memory_read", {"memory_id": jane})

    assert result.startswith("---")
    assert "reviews every change" in result


async def test_reading_a_private_memory_from_a_public_scope_is_refused(memory: Memory) -> None:
    """And refused the same way a missing one is, so the id itself leaks nothing."""
    secret = id_of(memory, "memory/facts/salary-review-outcome.md")

    refused = await memory.call("memory_read", {"memory_id": secret})
    missing = await memory.call("memory_read", {"memory_id": new_memory_id()})

    assert "band 5" not in refused
    assert refused.split()[:-1] == missing.split()[:-1], "indistinguishable but for the id"


async def test_reading_a_private_memory_from_its_own_scope_works(memory: Memory) -> None:
    secret = id_of(memory, "memory/facts/salary-review-outcome.md")
    result = await memory.call("memory_read", {"memory_id": secret}, scope="private:U01")
    assert "band 5" in result


@pytest.mark.parametrize(
    "memory_id", ["../../etc/passwd", "memory/people/jane.md", "", "not-an-id", "mem_short"]
)
async def test_read_only_accepts_memory_ids(memory: Memory, memory_id: str) -> None:
    result = await memory.call("memory_read", {"memory_id": memory_id})
    assert "is not a memory id" in result


async def test_read_follows_the_supersedes_chain(memory: Memory, store: Store) -> None:
    """A snippet quoted from an old conversation still resolves after a merge."""
    old_id = new_memory_id()
    successor = MemoryDoc.new(type="topic", title="Merged", body="Now here.", supersedes=[old_id])
    target = memory.root / successor.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(successor.render())
    Manifest.rebuild(memory.root)[0].save(memory.root)

    assert "Now here." in await memory.call("memory_read", {"memory_id": old_id})


# -- memory_write ------------------------------------------------------------


async def test_write_enqueues_an_observation(memory: Memory, store: Store) -> None:
    result = await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "Bob", "claim": "Bob runs the incident rota."},
        session_id="cli:1",
    )

    assert "queued" in result
    pending = await store.pending_observations()
    # The subject is stored normalized: it is the key `promote` groups by, and
    # "Bob" typed here has to meet "Bob's" extracted from a conversation. The
    # claim is not — it is prose a person will read in a memory file.
    assert [(o["subject"], o["claim"], o["kind"]) for o in pending] == [
        ("bob", "Bob runs the incident rota.", "fact")
    ]


async def test_write_never_produces_a_commit(memory: Memory) -> None:
    """The acceptance criterion. The agent proposes; `promote` disposes."""
    before = memory.repo.head()

    await memory.call(
        "memory_write", {"kind": "decision", "subject": "Deploys", "claim": "Nightly at 02:00."}
    )

    assert memory.repo.head() == before
    assert not memory.repo.is_dirty(), "nothing was written to the working copy either"


async def test_an_observation_inherits_the_session_scope(memory: Memory, store: Store) -> None:
    """Something said in a DM must not become general knowledge."""
    await store.ensure_session("slack:U01", surface="slack", scope="private:U01")
    await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "Pay", "claim": "They asked for a raise."},
        scope="private:U01",
        session_id="slack:U01",
    )

    observation = (await store.pending_observations())[0]
    assert observation["scope"] == "private:U01"
    assert observation["session_id"] == "slack:U01"


async def test_the_model_cannot_choose_the_scope(memory: Memory, store: Store) -> None:
    """`scope` is not an argument, so a plan to widen one has nowhere to land."""
    result = await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "X", "claim": "Y", "scope": "workspace"},
        scope="private:U01",
    )

    assert "invalid arguments" in result, "the schema forbids unknown properties"
    assert await store.pending_observations() == []


async def test_an_unknown_kind_is_rejected(memory: Memory, store: Store) -> None:
    result = await memory.call(
        "memory_write", {"kind": "instruction", "subject": "X", "claim": "Y"}
    )
    assert "invalid arguments" in result
    assert await store.pending_observations() == []


async def test_an_observation_survives_an_unknown_session(memory: Memory, store: Store) -> None:
    """Losing the fact because the bookkeeping link is missing is the wrong trade."""
    await memory.call(
        "memory_write", {"kind": "fact", "subject": "X", "claim": "Y"}, session_id="never-existed"
    )

    observation = (await store.pending_observations())[0]
    assert observation["claim"] == "Y"
    assert observation["session_id"] is None


async def test_observations_start_pending_with_no_episode(memory: Memory, store: Store) -> None:
    """#27 attaches the episode; the interactive path has only a session."""
    await memory.call("memory_write", {"kind": "fact", "subject": "X", "claim": "Y"})

    observation = (await store.pending_observations())[0]
    assert observation["state"] == "pending"
    assert observation["episode_id"] is None
    assert json.loads(observation["source_refs"]) == []


# -- memory_read: showing what a memory cites --------------------------------


def citing(memory: Memory, *shas: str, name: str = "whiteboard.png") -> str:
    """A memory in the corpus that points at those blobs. Returns its id."""
    links = " and ".join(blobref.link(sha, name) for sha in shas)
    doc = MemoryDoc.new(type="fact", title="The whiteboard", body=f"We drew the plan on {links}.")
    target = memory.root / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    Manifest.rebuild(memory.root)[0].save(memory.root)
    return doc.id


async def test_reading_a_memory_puts_the_picture_it_cites_in_the_turn(memory: Memory) -> None:
    """The gap #244 closes. A photograph sent this morning is looked at; the
    same photograph reached through the memory that cites it was a byte count."""
    sha = await sent(memory)
    context = ToolContext(session_id="cli:1")

    result = await memory.registry.dispatch(
        ToolUseBlock(id="t1", name="memory_read", input={"memory_id": citing(memory, sha)}),
        context,
    )

    assert "shown below" in result.content
    assert context.surfaced == [sha]


async def test_a_memory_that_cites_nothing_shows_nothing(memory: Memory) -> None:
    context = ToolContext(session_id="cli:1")

    await memory.registry.dispatch(
        ToolUseBlock(
            id="t1",
            name="memory_read",
            input={"memory_id": id_of(memory, "memory/people/jane-okafor.md")},
        ),
        context,
    )

    assert context.surfaced == []


async def test_a_collected_attachment_stays_a_sentence(memory: Memory) -> None:
    """The bytes are gone and the memory outlived them. Saying so is the answer;
    showing nothing and explaining nothing is how a model describes a
    photograph it has never seen."""
    context = ToolContext(session_id="cli:1")

    result = await memory.registry.dispatch(
        ToolUseBlock(id="t1", name="memory_read", input={"memory_id": citing(memory, "a" * 64)}),
        context,
    )

    assert "no longer stored" in result.content
    assert context.surfaced == []


async def test_an_attachment_out_of_scope_is_not_shown(memory: Memory) -> None:
    """A memory this conversation may read can cite a blob it may not. The
    honest answer is the one a collected blob gets."""
    private = await sent(memory, scope="private:U01")
    memory_id = citing(memory, private)

    result = await memory.registry.dispatch(
        ToolUseBlock(id="t1", name="memory_read", input={"memory_id": memory_id}),
        context := ToolContext(session_id="cli:1"),
    )

    assert "no longer stored" in result.content
    assert context.surfaced == []


async def test_a_video_is_named_and_never_shown(memory: Memory) -> None:
    """Stored and referenced, and nothing reads one — the same answer the note
    at ingress gives, for the same reason."""
    clip = await memory.attachments.put(
        b"\x00\x00\x00\x18ftypmp42", mime="video/mp4", source_name="slack", scope="workspace"
    )

    result = await memory.registry.dispatch(
        ToolUseBlock(id="t1", name="memory_read", input={"memory_id": citing(memory, clip)}),
        context := ToolContext(session_id="cli:1"),
    )

    assert "which nothing can read" in result.content
    assert context.surfaced == []


async def test_a_turn_shows_only_so_many_pictures(memory: Memory) -> None:
    """An image costs tokens by area, a memory may cite several, and a turn may
    read several memories. The ones past the bound are still named."""
    shas = [await sent(memory, payload=f"picture {n}".encode()) for n in range(MAX_SHOWN + 2)]
    context = ToolContext(session_id="cli:1")

    result = await memory.registry.dispatch(
        ToolUseBlock(id="t1", name="memory_read", input={"memory_id": citing(memory, *shas)}),
        context,
    )

    assert len(context.surfaced) == MAX_SHOWN
    assert result.content.count("shown below") == MAX_SHOWN
    assert result.content.count("not shown") == 2, "and the rest say so rather than going quiet"


async def test_the_bound_is_the_turn_and_not_the_call(memory: Memory) -> None:
    """Two reads in one turn share one budget. A per-call limit would not see
    the other three memories the same turn opened."""
    shas = [await sent(memory, payload=f"picture {n}".encode()) for n in range(MAX_SHOWN)]
    context = ToolContext(session_id="cli:1")
    for sha in shas:
        await memory.registry.dispatch(
            ToolUseBlock(id="t1", name="memory_read", input={"memory_id": citing(memory, sha)}),
            context,
        )

    extra = await sent(memory, payload=b"one too many")
    result = await memory.registry.dispatch(
        ToolUseBlock(id="t1", name="memory_read", input={"memory_id": citing(memory, extra)}),
        context,
    )

    assert "not shown" in result.content
    assert len(context.surfaced) == MAX_SHOWN


async def test_reading_the_same_memory_twice_shows_it_once(memory: Memory) -> None:
    """The budget is spent on distinct pictures, not on distinct reads."""
    sha = await sent(memory)
    memory_id = citing(memory, sha)
    context = ToolContext(session_id="cli:1")

    for _ in range(3):
        result = await memory.registry.dispatch(
            ToolUseBlock(id="t1", name="memory_read", input={"memory_id": memory_id}), context
        )

    assert context.surfaced == [sha]
    assert "shown earlier in this turn" in result.content, "and it says where to look"


# -- memory_write: citing what somebody sent ---------------------------------


async def cited(memory: Memory) -> list[dict[str, str]]:
    observation = (await memory.store.pending_observations())[0]
    return list(json.loads(observation["attachments"]))


async def sent(
    memory: Memory,
    *,
    session_id: str = "cli:1",
    scope: str = "workspace",
    name: str = "IMG_3604.jpg",
    payload: bytes = b"\xff\xd8\xff-not-really-a-jpeg",
) -> str:
    """A file that arrived in a conversation. Returns its digest."""
    return await memory.attachments.put(
        payload,
        mime="image/jpeg",
        source_name="slack",
        scope=scope,
        session_id=session_id,
        name=name,
    )


async def test_a_write_can_cite_a_file_from_this_conversation(memory: Memory) -> None:
    """The whole point of #243: the photograph and the sentence about it arrive
    together, and the observation carries both to `promote`."""
    sha = await sent(memory)

    result = await memory.call(
        "memory_write",
        {
            "kind": "fact",
            "subject": "Keunwoo",
            "claim": "Keunwoo photographed the Sejong Arts Center on 5 September 2026.",
            "attachments": [handle(sha)],
        },
        session_id="cli:1",
    )

    assert "queued" in result
    assert "IMG_3604.jpg" in result, "the answer names what it kept, not a digest"
    assert await cited(memory) == [{"sha256": sha, "name": "IMG_3604.jpg"}]


async def test_the_name_in_a_citation_comes_from_the_store(memory: Memory) -> None:
    """The link text a person will read in a year. The model supplies a handle
    and nothing else — it cannot relabel somebody's upload on the way past."""
    sha = await sent(memory, name="whiteboard.png")

    await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "X", "claim": "Y", "attachments": [sha]},
        session_id="cli:1",
    )

    assert await cited(memory) == [{"sha256": sha, "name": "whiteboard.png"}]


async def test_an_invented_handle_records_nothing(memory: Memory) -> None:
    """A model that has seen one handle can compose another. Writing the claim
    without the picture would report success for half the request."""
    await sent(memory)

    result = await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "X", "claim": "Y", "attachments": ["dead" * 3]},
        session_id="cli:1",
    )

    assert "nothing was recorded" in result
    assert "IMG_3604.jpg" in result, "and it says what it could have cited instead"
    assert await memory.store.pending_observations() == []


async def test_a_file_from_another_conversation_cannot_be_cited(memory: Memory) -> None:
    """The narrow rule: what is citable is what was sent *here*. A digest
    carried in from somewhere else resolves to nothing."""
    elsewhere = await sent(memory, session_id="cli:2")

    result = await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "X", "claim": "Y", "attachments": [handle(elsewhere)]},
        session_id="cli:1",
    )

    assert "nothing was recorded" in result
    assert await memory.store.pending_observations() == []


async def test_a_handle_cannot_reach_across_the_scope_line(memory: Memory) -> None:
    """Same session id, narrower arrival. The scope check is in the query that
    resolves the handle, so there is no answer to be had here at all."""
    private = await sent(memory, scope="private:U01")

    result = await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "X", "claim": "Y", "attachments": [handle(private)]},
        session_id="cli:1",
    )

    assert "nothing to cite" in result
    assert await memory.store.pending_observations() == []


async def test_citing_nothing_is_the_ordinary_case(memory: Memory) -> None:
    """Most claims are not about a file, and one query per write to discover
    that is one query too many."""
    await memory.call("memory_write", {"kind": "fact", "subject": "X", "claim": "Y"})

    assert await cited(memory) == []


async def test_more_attachments_than_a_memory_may_hold_are_refused(memory: Memory) -> None:
    """A memory is a claim, not an album."""
    sha = await sent(memory)

    result = await memory.call(
        "memory_write",
        {
            "kind": "fact",
            "subject": "X",
            "claim": "Y",
            "attachments": [handle(sha)] * (MAX_CITED + 1),
        },
        session_id="cli:1",
    )

    assert "invalid arguments" in result
    assert await memory.store.pending_observations() == []


# -- acceptance: the picture reaches the model -------------------------------


class Reading:
    """A provider that reads one memory, then answers. Keeps what it was sent."""

    name = "scripted"
    model = "m"

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        self.requests: list[ChatRequest] = []
        self._turn = 0

    async def complete(self, req: ChatRequest) -> ChatResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:
        self.requests.append(req)
        self._turn += 1
        if self._turn == 1:
            yield ToolUseStart(id="t1", name="memory_read")
            yield ToolUseArgsDelta(id="t1", partial_json=json.dumps({"memory_id": self.memory_id}))
            yield ToolUseStop(id="t1")
            yield MessageStop(stop_reason="tool_use", usage=Usage(), model="m")
            return
        yield TextDelta(text="A whiteboard with the release plan on it.")
        yield MessageStop(stop_reason="end_turn", usage=Usage(), model="m")

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        return [[0.0] for _ in texts]

    async def aclose(self) -> None:
        return None


def reading_agent(memory: Memory, store: Store, tokenizer: Tokenizer, provider: Reading) -> Agent:
    return Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=memory.registry,
        packer=ContextPacker(tokenizer=tokenizer),
        retriever=Retriever(store, tokenizer=tokenizer),
        attachments=memory.attachments,
    )


async def test_the_picture_a_memory_cites_reaches_the_model(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """The acceptance criterion for #244. A tool result is text, so the image
    rides on the turn carrying it — and it arrives with its bytes."""
    payload = b"\xff\xd8\xff-not-really-a-jpeg"
    sha = await sent(memory, payload=payload)
    provider = Reading(citing(memory, sha))

    await reading_agent(memory, store, tokenizer, provider).respond("cli:1", "what did we draw?")

    answering = provider.requests[1].messages[-1]
    assert answering.tool_results_in, "the results and the picture are one turn"
    assert [b.sha256 for b in answering.images] == [sha]
    assert answering.images[0].data == payload


async def test_the_results_lead_the_turn_that_carries_a_picture(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """The order is the contract: an Anthropic user turn must open with its
    `tool_result` blocks, and the OpenAI shape splits this message on it."""
    sha = await sent(memory)
    provider = Reading(citing(memory, sha))

    await reading_agent(memory, store, tokenizer, provider).respond("cli:1", "what did we draw?")

    kinds = [block.type for block in provider.requests[1].messages[-1].content]
    assert kinds == ["tool_result", "image"]


async def test_the_transcript_keeps_the_reference_and_not_the_bytes(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """Same bargain as an upload: re-reading this conversation next week costs
    what it cost today."""
    payload = b"\xff\xd8\xff-not-really-a-jpeg"
    sha = await sent(memory, payload=payload)
    provider = Reading(citing(memory, sha))

    await reading_agent(memory, store, tokenizer, provider).respond("cli:1", "what did we draw?")

    rows = await store.raw("SELECT content FROM messages WHERE role = 'user'")
    stored_content = "".join(str(row["content"]) for row in rows)
    assert sha in stored_content
    assert "data" not in stored_content


async def test_a_picture_out_of_scope_never_reaches_the_provider(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """Two checks, and both have to hold: the note refuses to surface it, and
    hydration would refuse to load it."""
    private = await sent(memory, scope="private:U01")
    provider = Reading(citing(memory, private))

    await reading_agent(memory, store, tokenizer, provider).respond("cli:1", "what did we draw?")

    assert provider.requests[1].messages[-1].images == ()


# -- acceptance: the agent recovers from a pre-injection miss ----------------


class Scripted:
    """A provider that calls memory_search, then answers from what came back."""

    name = "scripted"
    model = "m"

    def __init__(self) -> None:
        self.tool_results: list[str] = []
        self.contexts: list[str | None] = []
        self._turn = 0

    async def complete(self, req: ChatRequest) -> ChatResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:
        self._turn += 1
        self.contexts.append(req.context)
        for message in req.messages:
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    self.tool_results.append(block.content)

        if self._turn == 1:
            yield ToolUseStart(id="t1", name="memory_search")
            yield ToolUseArgsDelta(id="t1", partial_json='{"query": "deploy pipeline owner"}')
            yield ToolUseStop(id="t1")
            yield MessageStop(stop_reason="tool_use", usage=Usage(), model="m")
            return

        answer = "Jane owns it." if any("Jane" in r for r in self.tool_results) else "I don't know."
        yield TextDelta(text=answer)
        yield MessageStop(stop_reason="end_turn", usage=Usage(), model="m")

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        return [[0.0] for _ in texts]

    async def aclose(self) -> None:
        return None


async def test_the_agent_can_answer_from_tools_when_injection_missed(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """The acceptance criterion for #16.

    The question shares no words with the memory, so pre-injection returns
    nothing. The agent has to find it by searching, in its own words.
    """
    provider = Scripted()
    retriever = Retriever(store, tokenizer=tokenizer)
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=memory.registry,
        packer=ContextPacker(tokenizer=tokenizer),
        retriever=retriever,
    )

    result = await agent.respond("cli:1", "Who should I nudge about shipping?")

    # The turn's own status is always there (#201); retrieved memory is not.
    assert RETRIEVED_HEADER not in (provider.contexts[0] or ""), (
        "pre-injection found nothing to inject"
    )
    assert result.tool_calls == 1
    assert result.text == "Jane owns it."


async def test_pre_injection_puts_memory_in_the_context(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """The other ninety percent: no tool call needed."""
    provider = Scripted()
    provider._turn = 1  # skip straight to answering
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=memory.registry,
        packer=ContextPacker(tokenizer=tokenizer),
        retriever=Retriever(store, tokenizer=tokenizer),
    )

    await agent.respond("cli:2", "Who owns the deploy pipeline?")

    assert provider.contexts[0] is not None
    assert "Jane" in provider.contexts[0]


async def test_a_broken_retriever_degrades_the_turn_rather_than_ending_it(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """Answering without memory beats refusing to answer."""

    class Broken(Retriever):
        async def retrieve(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("index is on fire")

    provider = Scripted()
    provider._turn = 1
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=memory.registry,
        packer=ContextPacker(tokenizer=tokenizer),
        retriever=Broken(store, tokenizer=tokenizer),
    )

    result = await agent.respond("cli:3", "Who owns the deploy pipeline?")
    assert result.text == "I don't know."


async def test_pre_injection_respects_the_session_scope(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    provider = Scripted()
    provider._turn = 1
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=memory.registry,
        packer=ContextPacker(tokenizer=tokenizer),
        retriever=Retriever(store, tokenizer=tokenizer),
    )

    await agent.respond("cli:4", "what was the salary band outcome?")
    injected = provider.contexts[0] or ""
    assert id_of(memory, "memory/facts/salary-review-outcome.md") not in injected


# -- a write tool must not confirm a no-op (#79) ------------------------------


async def test_whitespace_is_not_a_claim(memory: Memory, store: Store) -> None:
    """#79. Both fields were stripped and written, and the model was told the
    write had succeeded — leaving an observation with nothing in it `pending`
    for the promote job to deal with.

    `minLength` in the schema catches the empty string upstream; it cannot see
    a string of spaces, so the handler checks the stripped value too."""
    result = await memory.call("memory_write", {"kind": "fact", "subject": " ", "claim": "\t\n"})

    assert "must each say something" in result
    assert "queued" not in result
    assert await store.pending_observations() == []


@pytest.mark.parametrize(
    "args",
    [
        {"kind": "fact", "subject": "", "claim": ""},
        {"kind": "fact", "subject": "Jane", "claim": ""},
        {"kind": "fact", "subject": "", "claim": "Jane owns deploys."},
    ],
)
async def test_an_empty_field_is_refused_by_the_schema(
    memory: Memory, store: Store, args: dict[str, Any]
) -> None:
    """Upstream, the way the `kind` enum is — so it arrives as an error rather
    than as an answer."""
    result = await memory.call("memory_write", args)

    assert "invalid arguments" in result
    assert await store.pending_observations() == []


# -- memory_write: what a memory may not cite --------------------------------


async def test_a_picture_siatt_drew_itself_cannot_be_cited(memory: Memory) -> None:
    """The corpus is Markdown a person reads and believes, and a claim citing an
    image Siatt invented is a fabricated exhibit filed as evidence. It is not a
    broken link somebody can puzzle out — nothing about it looks wrong."""
    drawn = await memory.attachments.put(
        b"\xff\xd8\xff-not-really-a-jpeg",
        mime="image/jpeg",
        source_name="generated",
        scope="workspace",
        session_id="cli:1",
        name="a-red-circle.jpg",
    )

    result = await memory.call(
        "memory_write",
        {
            "kind": "fact",
            "subject": "Keunwoo",
            "claim": "Keunwoo photographed the Sejong Arts Center.",
            "attachments": [handle(drawn)],
        },
        session_id="cli:1",
    )

    assert "nothing to cite" in result
    assert "queued" not in result
    assert await memory.store.pending_observations() == []


async def test_a_drawing_does_not_hide_the_photographs_beside_it(memory: Memory) -> None:
    """The filter drops one row, not the conversation. A thread with a drawing
    in it must still be able to cite what somebody actually sent."""
    await memory.attachments.put(
        b"\xff\xd8\xff-drawn",
        mime="image/jpeg",
        source_name="generated",
        scope="workspace",
        session_id="cli:1",
        name="a-red-circle.jpg",
    )
    photograph = await sent(memory)

    result = await memory.call(
        "memory_write",
        {
            "kind": "fact",
            "subject": "Keunwoo",
            "claim": "Keunwoo photographed the Sejong Arts Center.",
            "attachments": [handle(photograph)],
        },
        session_id="cli:1",
    )

    assert "queued" in result
    assert await cited(memory) == [{"sha256": photograph, "name": "IMG_3604.jpg"}]
