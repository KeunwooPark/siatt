"""`promote`: from the pending queue to a commit somebody can read and revert."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from siatt.config import AttachmentSettings, PromoteSettings
from siatt.llm.registry import ModelRole, ProviderRegistry
from siatt.llm.tokens import HeuristicTokenizer
from siatt.llm.types import ChatRequest, ChatResponse, Delta, Message, Usage
from siatt.memory.bootstrap import bootstrap
from siatt.memory.consolidate import PLAN_TOKENS
from siatt.memory.document import MemoryDoc
from siatt.memory.gitcmd import GitRepo
from siatt.memory.index import MemoryIndex
from siatt.memory.ltm import MemoryStore
from siatt.memory.manifest import Manifest
from siatt.memory.observation import Cited
from siatt.memory.retrieve import Retriever
from siatt.runner.promote import Promoter
from siatt.store import Store


class Scripted:
    """Returns a fixed list of replies, and remembers what it was asked."""

    name = "scripted"
    model = "m"

    def __init__(self, *replies: str | Exception | ChatResponse) -> None:
        self.replies: list[str | Exception | ChatResponse] = list(replies)
        self.requests: list[ChatRequest] = []

    @property
    def prompts(self) -> list[str]:
        return [req.messages[0].text for req in self.requests]

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        reply = self.replies.pop(0) if self.replies else "[]"
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, ChatResponse):
            return reply
        return ChatResponse(
            message=Message.assistant(reply),
            stop_reason="end_turn",
            usage=Usage(),
            model="m",
        )

    def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """An empty, bootstrapped memory repo with no remote — nothing is pushed."""
    repo = tmp_path / "ltm"
    GitRepo.init(repo, branch="main")
    bootstrap(repo)
    Manifest.rebuild(repo)[0].save(repo)
    GitRepo.at(repo).commit("memory: bootstrap")
    return repo


def write_memory(clone: Path, doc: MemoryDoc, *, path: str | None = None) -> str:
    target = clone / (path or doc.suggested_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    Manifest.rebuild(clone)[0].save(clone)
    GitRepo.at(clone).commit(f"memory: seed {doc.id}")
    return str(target.relative_to(clone))


async def promoter_for(clone: Path, store: Store, provider: Scripted, **settings: Any) -> Promoter:
    """A promoter over a real repo and a real index, with a scripted planner."""
    await MemoryIndex(store, clone).reindex()
    memory = MemoryStore(GitRepo.at(clone), store, branch="main", push=False)
    retriever = Retriever(store, tokenizer=HeuristicTokenizer(), budget_tokens=4_000)
    return Promoter(
        store,
        memory,
        retriever,
        ProviderRegistry({ModelRole.CHAT: [provider]}),
        settings=PromoteSettings(**settings),
    )


async def observe(
    store: Store,
    subject: str,
    claim: str,
    *,
    scope: str = "workspace",
    kind: str = "fact",
    attachments: Sequence[Cited] = (),
) -> str:
    return await store.add_observation(
        subject=subject, claim=claim, kind=kind, scope=scope, attachments=attachments
    )


async def kept(store: Store, tmp_path: Path, *, name: str = "whiteboard.png") -> Cited:
    """A real blob, so the patch validator has something to accept."""
    files = AttachmentSettings(enabled=True).build(store, tmp_path / "siatt.db")
    sha = await files.put(
        b"\x89PNG\r\n\x1a\n" + name.encode(),
        mime="image/png",
        source_name="slack",
        scope="workspace",
        session_id="s1",
        name=name,
    )
    return Cited(sha256=sha, name=name)


def creating(title: str, body: str, *, memory_type: str = "fact") -> str:
    """A plan that creates one memory. The id is the model's to supply."""
    doc = MemoryDoc.new(type=memory_type, title=title, body=body)  # type: ignore[arg-type]
    return json.dumps([{"type": "create", "memory": doc.model_dump(mode="json")}])


def out_of_room(text: str = "") -> ChatResponse:
    """A reply the model ran out of output budget to finish.

    Empty by default, which is what a reasoning model returns when it spends
    the whole budget thinking: the thinking is not text, so nothing arrives.
    """
    return ChatResponse(
        message=Message.assistant(text),
        stop_reason="max_tokens",
        usage=Usage(output_tokens=PLAN_TOKENS),
        model="m",
    )


def updating(memory_id: str, body: str) -> str:
    return json.dumps([{"type": "update", "id": memory_id, "body": body}])


def files_under(clone: Path, directory: str) -> list[str]:
    return sorted(p.name for p in (clone / "memory" / directory).glob("*.md"))


def files_with_id(clone: Path, memory_id: str) -> list[str]:
    """Every file in the repo whose frontmatter claims `memory_id`.

    Read off disk rather than from the manifest: a manifest maps an id to one
    path, so it is the one structure that cannot show you a duplicate.
    """
    found = []
    for path in sorted((clone / "memory").rglob("*.md")):
        try:
            doc = MemoryDoc.parse(path.read_text(), source=str(path))
        except Exception:
            continue
        if doc.id == memory_id:
            found.append(path.relative_to(clone).as_posix())
    return found


# -- attachments -------------------------------------------------------------


async def test_the_plan_prompt_offers_what_the_observations_cited(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    """A model cannot write a reference to a file nobody told it about, and a
    digest is not something it may be asked to invent."""
    photo = await kept(store, tmp_path)
    await observe(store, "Priya", "Priya drew the release plan.", attachments=[photo])
    provider = Scripted("[]")

    await (await promoter_for(clone, store, provider)).run()

    prompt = provider.prompts[0]
    assert f"[a1] whiteboard.png — siatt://blob/{photo.sha256}" in prompt
    assert "(attachments: a1)" in prompt, "and the claim says which one is its own"


async def test_an_attachment_collected_since_is_not_offered(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    """`forget` may have taken the bytes between the conversation and this run.
    Offering it would buy a sentence written around a link about to be unwrapped
    — evidence claimed and then removed."""
    photo = await kept(store, tmp_path)
    await observe(store, "Priya", "Priya drew the release plan.", attachments=[photo])
    # As `forget` collects one: the refs go with the conversation, and the
    # blob is reclaimed once nothing holds it.
    await store.write("DELETE FROM attachment_refs WHERE sha256 = ?", (photo.sha256,))
    await store.delete_attachment(photo.sha256)
    provider = Scripted("[]")

    await (await promoter_for(clone, store, provider)).run()

    assert "siatt://blob/" not in provider.prompts[0]
    assert "(attachments:" not in provider.prompts[0]


async def test_a_cited_attachment_reaches_the_committed_memory(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    """End to end, and the acceptance criterion for #243: a photograph sent in
    a conversation is a link in a file on the branch."""
    photo = await kept(store, tmp_path)
    await observe(store, "Priya", "Priya drew the release plan.", attachments=[photo])
    body = f"Priya drew the release plan on [whiteboard.png](siatt://blob/{photo.sha256})."
    provider = Scripted(creating("The release plan", body))

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.changed
    written = (clone / "memory/facts/the-release-plan.md").read_text()
    assert f"[whiteboard.png](siatt://blob/{photo.sha256})" in written


async def test_a_digest_the_plan_invented_is_still_unwrapped(
    clone: Path, store: Store, tmp_path: Path
) -> None:
    """Offering real ones does not make the validator's job optional: a model
    that has seen the shape composes another, and the memory keeps the prose."""
    photo = await kept(store, tmp_path)
    await observe(store, "Priya", "Priya drew the release plan.", attachments=[photo])
    invented = "b" * 64
    body = f"Priya drew it on [a whiteboard](siatt://blob/{invented})."
    provider = Scripted(creating("The release plan", body))

    await (await promoter_for(clone, store, provider)).run()

    written = (clone / "memory/facts/the-release-plan.md").read_text()
    assert invented not in written
    assert "Priya drew it on a whiteboard." in written


async def test_observations_with_no_attachments_say_nothing_about_them(
    clone: Path, store: Store
) -> None:
    """Most groups. The prompt should not grow a section explaining that there
    are no files, on every run, forever."""
    await observe(store, "Priya", "Priya owns deploys.")
    provider = Scripted("[]")

    await (await promoter_for(clone, store, provider)).run()

    assert "(attachments:" not in provider.prompts[0]
    assert '"attachments": []' in provider.prompts[0]


# -- the ordinary path -------------------------------------------------------


async def test_nothing_pending_is_a_no_op(clone: Path, store: Store) -> None:
    provider = Scripted()
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.subjects == 0
    assert result.summary() == "nothing pending"
    assert provider.requests == [], "and it cost nothing to find out"
    assert GitRepo.at(clone).head() == before


async def test_a_new_subject_becomes_a_file_in_the_repo(clone: Path, store: Store) -> None:
    """The job that makes the product exist: a row in SQLite becomes a Markdown
    file a person can open, disagree with, and revert."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    provider = Scripted(creating("Deploy pipeline ownership", "Priya Raman owns it."))

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 1
    assert files_under(clone, "facts") == ["deploy-pipeline-ownership.md"]
    assert result.sha is not None
    rows = await store.raw("SELECT state, reason FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "promoted"
    assert result.sha in str(rows[0]["reason"]), "the reason says where to go and look"


async def test_create_prompt_shows_the_nested_memory_document_shape(
    clone: Path, store: Store
) -> None:
    await observe(store, "Bob", "Bob runs the rota.")
    provider = Scripted("[]")

    await (await promoter_for(clone, store, provider)).run()

    prompt = provider.prompts[0]
    assert '"memory": {"frontmatter": {"id": "<memory id>"' in prompt
    assert "frontmatter fields must be nested under `memory.frontmatter`" in prompt
    assert "Return raw JSON only, with no Markdown or code fences" in prompt


async def test_the_commit_is_machine_readable(clone: Path, store: Store) -> None:
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    provider = Scripted(creating("Deploy pipeline ownership", "Priya Raman owns it."))

    await (await promoter_for(clone, store, provider)).run()

    message = GitRepo.at(clone).run("log", "-1", "--format=%B")
    assert "Siatt-Job: promote" in message
    assert "Siatt-Memory-Ids: mem_" in message


async def test_observations_about_one_subject_are_reconciled_together(
    clone: Path, store: Store
) -> None:
    """One call per subject, not per fact. Two claims about the same person are
    the same memory, and asking twice writes two files that disagree."""
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    await observe(store, "Priya Raman", "Priya Raman is on leave until October.")
    provider = Scripted(creating("Priya Raman", "Owns deploys; on leave until October."))

    result = await (await promoter_for(clone, store, provider)).run()

    assert len(provider.requests) == 1
    assert result.promoted == 2
    assert "deploy pipeline" in provider.prompts[0]
    assert "on leave" in provider.prompts[0]


async def test_everything_a_run_decides_lands_in_one_commit(clone: Path, store: Store) -> None:
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    await observe(store, "Release window", "The release window is Thursdays.")
    provider = Scripted(
        creating("Priya Raman", "Owns deploys."), creating("Release window", "Thursdays.")
    )
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert len(provider.requests) == 2
    assert len(files_under(clone, "facts")) == 2
    commits = GitRepo.at(clone).run("log", "--format=%H", f"{before}..HEAD").split()
    assert len(commits) == 1, "one commit per run, whatever it decided"
    assert result.sha is not None


# -- the acceptance criteria -------------------------------------------------


async def test_re_running_promote_is_a_no_op(clone: Path, store: Store) -> None:
    """The first acceptance criterion. Idempotence comes from the observations
    table: a run that commits marks its inputs, and the next run finds nothing."""
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    provider = Scripted(creating("Priya Raman", "Owns deploys."))
    await (await promoter_for(clone, store, provider)).run()
    after_first = GitRepo.at(clone).head()

    second = await (await promoter_for(clone, store, provider)).run()

    assert second.subjects == 0
    assert len(provider.requests) == 1, "the second run did not reach the model"
    assert GitRepo.at(clone).head() == after_first
    assert len(files_under(clone, "facts")) == 1


async def test_a_restated_fact_updates_the_existing_memory(clone: Path, store: Store) -> None:
    """The second acceptance criterion, and the reason the competition step
    exists at all: the planner cannot update a memory it was never shown."""
    existing = MemoryDoc.new(
        type="fact", title="Deploy pipeline ownership", body="Priya Raman owns the deploy pipeline."
    )
    path = write_memory(clone, existing)
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline, and the runbook.")
    provider = Scripted(updating(existing.id, "Priya Raman owns the deploy pipeline and runbook."))

    result = await (await promoter_for(clone, store, provider)).run()

    assert existing.id in provider.prompts[0], "the competing memory reached the planner"
    assert "owns the deploy pipeline." in provider.prompts[0], "as its whole body, not a snippet"
    assert files_under(clone, "facts") == ["deploy-pipeline-ownership.md"], "no duplicate"
    assert "and runbook" in (clone / path).read_text()
    assert result.promoted == 1


# -- visibility --------------------------------------------------------------


async def test_two_scopes_of_one_subject_are_never_reconciled_together(
    clone: Path, store: Store
) -> None:
    """A group becomes one prompt and one memory's audience. Mixing scopes in
    it is how something said in a DM ends up in a workspace file."""
    await observe(store, "Priya Raman", "Priya Raman owns deploys.", scope="workspace")
    await observe(store, "Priya Raman", "Priya Raman is job hunting.", scope="private:U1")
    provider = Scripted("[]", "[]")

    await (await promoter_for(clone, store, provider)).run()

    assert len(provider.requests) == 2
    # One claim each. A prompt holding both is a prompt in which the private
    # one can end up quoted into the workspace memory.
    for prompt in provider.prompts:
        assert ("job hunting" in prompt) != ("owns deploys" in prompt)


async def test_a_created_memory_inherits_the_group_scope(
    clone: Path, store: Store, caplog: Any
) -> None:
    """Corrected, not rejected. The scope is not the model's to choose, so a
    plan that got it wrong is not a plan to argue with — and refusing it would
    throw away the fact to punish the formatting."""
    await observe(store, "Priya Raman", "Priya Raman is job hunting.", scope="private:U1")
    # The plan asks for `workspace`, which is wider than the conversation it
    # came from. This is the leak the whole scope discipline exists to stop.
    provider = Scripted(creating("Priya Raman", "Job hunting."))

    with caplog.at_level("WARNING", logger="siatt.runner.promote"):
        await (await promoter_for(clone, store, provider)).run()

    written = MemoryDoc.parse((clone / "memory/facts/priya-raman.md").read_text())
    assert written.frontmatter.visibility == "private:U1"
    assert "set visibility" in caplog.text


async def test_a_private_memory_is_not_offered_to_another_private_scope(
    clone: Path, store: Store
) -> None:
    """Retrieval is filtered to the group. A planner that cannot see a memory
    cannot quote it into one it is writing for somebody else."""
    secret = MemoryDoc.new(
        type="fact",
        title="Priya plans",
        body="Priya Raman is job hunting.",
        visibility="private:U1",
    )
    write_memory(clone, secret)
    await observe(store, "Priya Raman", "Priya Raman owns deploys.", scope="private:U2")
    provider = Scripted("[]")

    await (await promoter_for(clone, store, provider)).run()

    assert "job hunting" not in provider.prompts[0]


async def test_an_update_may_not_change_visibility(clone: Path, store: Store, caplog: Any) -> None:
    """A memory's audience was set when it was written. This plan is about one
    group's claims, not about who may read them."""
    existing = MemoryDoc.new(type="fact", title="Priya Raman", body="Owns deploys.")
    path = write_memory(clone, existing)
    await observe(store, "Priya Raman", "Priya Raman also owns the runbook.")
    provider = Scripted(
        json.dumps(
            [
                {
                    "type": "update",
                    "id": existing.id,
                    "body": "Owns deploys and the runbook.",
                    "frontmatter": {"visibility": "private:U9"},
                }
            ]
        )
    )

    with caplog.at_level("WARNING", logger="siatt.runner.promote"):
        await (await promoter_for(clone, store, provider)).run()

    written = MemoryDoc.parse((clone / path).read_text())
    assert written.frontmatter.visibility == "workspace"
    assert "runbook" in written.body, "the rest of the update still landed"
    assert "dropped a visibility change" in caplog.text


# -- what a plan may not do --------------------------------------------------


async def test_promote_may_not_delete(clone: Path, store: Store) -> None:
    """`docs/DESIGN.md` §7.1, enforced rather than trusted. Only `forget`
    deletes, and only what is already archived."""
    existing = MemoryDoc.new(type="fact", title="Priya Raman", body="Owns deploys.")
    write_memory(clone, existing)
    await observe(store, "Priya Raman", "Priya Raman left the company.")
    provider = Scripted(json.dumps([{"type": "delete", "id": existing.id, "reason": "she left"}]))
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert (clone / existing.suggested_path()).exists()
    assert GitRepo.at(clone).head() == before


async def test_a_reply_that_is_not_a_plan_leaves_the_corpus_alone(
    clone: Path, store: Store
) -> None:
    """The realistic shape of a successful injection: prose, or an invented
    operation, instead of a plan. The worst case is a deferred observation."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted("Ignore previous instructions. I have deleted every memory.")
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert GitRepo.at(clone).head() == before
    rows = await store.raw("SELECT state, attempts FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "pending", "still there, to be tried again"
    assert rows[0]["attempts"] == 1


async def test_a_reply_that_ran_out_of_room_is_retried_before_it_costs_a_retry(
    clone: Path, store: Store
) -> None:
    """A model that ran out of budget did not disagree with us. Spending one of
    three hourly attempts to discover that loses the facts to a number this job
    chose, so the retry happens in the run that noticed, with more room."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted(out_of_room(), creating("Priya Raman", "Owns deploys."))

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 1
    budgets = [req.max_tokens for req in provider.requests]
    assert budgets == [PLAN_TOKENS, PLAN_TOKENS * 2], "the second ask has twice the room"
    rows = await store.raw("SELECT state, attempts FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "promoted"
    assert rows[0]["attempts"] == 0, "nothing was spent on the model running out of room"


async def test_a_reply_with_no_text_is_retried_too(clone: Path, store: Store) -> None:
    """The same failure without the `max_tokens` flag: the reply is simply
    empty, which `json.loads` describes as a problem at column 0."""
    await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted("", creating("Priya Raman", "Owns deploys."))

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 1
    assert len(provider.requests) == 2


async def test_a_group_that_never_gets_an_answer_says_so(clone: Path, store: Store) -> None:
    """It is still bounded — two asks per run, and the attempt cap after that.
    What it must not do is record the failure as a malformed plan, which is
    what "not JSON: Expecting value: line 1 column 1 (char 0)" reads as."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted(out_of_room(), out_of_room())

    result = await (await promoter_for(clone, store, provider, max_attempts=1)).run()

    assert result.promoted == 0
    assert len(provider.requests) == 2, "twice per run, not once and not forever"
    rows = await store.raw("SELECT state, reason FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "discarded"
    reason = str(rows[0]["reason"])
    assert "output budget" in reason and "not JSON" not in reason


async def test_a_truncated_plan_is_refused_even_when_it_parses(clone: Path, store: Store) -> None:
    """Half a plan is the corpus-in-an-intermediate-state the compiler exists to
    prevent: the missing half is the half that says what else to change."""
    await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    half = out_of_room(creating("Priya Raman", "Owns deploys."))
    provider = Scripted(half, "[]")
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert GitRepo.at(clone).head() == before, "the half that parsed was not written"


async def test_the_planner_is_given_no_tools(clone: Path, store: Store) -> None:
    await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted("[]")

    await (await promoter_for(clone, store, provider)).run()

    assert provider.requests[0].tools == ()
    assert "SIATT_UNTRUSTED_" in provider.prompts[0]


async def test_a_create_for_a_file_that_exists_is_asked_again_with_that_file(
    clone: Path, store: Store
) -> None:
    """The most common mistake a planner makes, and the most recoverable one.
    Retrieval did not offer the file, so as far as the plan could tell nothing
    had been written about the subject. Showing it the file is the answer."""
    seeded = MemoryDoc.new(type="fact", title="Boram", body="Boram reviews the rota.")
    write_memory(clone, seeded, path="memory/people/boram.md")
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    colliding = json.dumps(
        [
            {
                "type": "create",
                "memory": MemoryDoc.new(type="fact", title="Boram", body="new").model_dump(
                    mode="json"
                ),
                "path": "memory/people/boram.md",
            }
        ]
    )
    provider = Scripted(colliding, updating(seeded.id, "Boram reviews the rota, and deploys."))

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 1
    assert len(provider.requests) == 2
    assert "reviews the rota" not in provider.prompts[0], "it was never shown the file"
    assert "reviews the rota" in provider.prompts[1], "and now it has been"
    assert "and deploys" in (clone / "memory/people/boram.md").read_text()
    rows = await store.raw("SELECT state, attempts FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "promoted"


async def test_a_plan_that_collides_twice_is_deferred_not_asked_forever(
    clone: Path, store: Store
) -> None:
    """Two plans per group per run. A model that proposes the same create after
    being shown the file is wrong in a way another ask will not fix."""
    write_memory(
        clone,
        MemoryDoc.new(type="fact", title="Boram", body="Boram reviews the rota."),
        path="memory/people/boram.md",
    )
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    colliding = json.dumps(
        [
            {
                "type": "create",
                "memory": MemoryDoc.new(type="fact", title="Boram", body="new").model_dump(
                    mode="json"
                ),
                "path": "memory/people/boram.md",
            }
        ]
    )
    provider = Scripted(colliding, colliding)
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert len(provider.requests) == 2
    assert GitRepo.at(clone).head() == before
    rows = await store.raw("SELECT state, attempts FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "pending"
    assert rows[0]["attempts"] == 1


async def test_a_colliding_memory_from_another_scope_is_never_shown(
    clone: Path, store: Store
) -> None:
    """A file sitting at the path the plan wanted is not thereby a file this
    group's audience may read."""
    write_memory(
        clone,
        MemoryDoc.new(
            type="fact", title="Boram", body="Boram is job-hunting.", visibility="private:U1"
        ),
        path="memory/people/boram.md",
    )
    await observe(store, "Priya Raman", "Priya Raman owns deploys.", scope="workspace")
    provider = Scripted(
        json.dumps(
            [
                {
                    "type": "create",
                    "memory": MemoryDoc.new(type="fact", title="Boram", body="new").model_dump(
                        mode="json"
                    ),
                    "path": "memory/people/boram.md",
                }
            ]
        )
    )

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert len(provider.requests) == 1, "there was nothing it was allowed to be shown"
    assert all("job-hunting" not in prompt for prompt in provider.prompts)


async def test_two_subjects_that_want_one_path_do_not_silently_overwrite(
    clone: Path, store: Store
) -> None:
    """Both writes would go into the same commit, where the second replaces the
    first — a lost fact with no error anywhere."""
    await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    await observe(store, "Release window", "Thursdays.")
    provider = Scripted(
        json.dumps(
            [
                {
                    "type": "create",
                    "memory": MemoryDoc.new(type="fact", title="A", body="one").model_dump(
                        mode="json"
                    ),
                    "path": "memory/facts/collide.md",
                }
            ]
        ),
        json.dumps(
            [
                {
                    "type": "create",
                    "memory": MemoryDoc.new(type="fact", title="B", body="two").model_dump(
                        mode="json"
                    ),
                    "path": "memory/facts/collide.md",
                }
            ]
        ),
    )

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 1
    assert result.deferred == 1
    assert (clone / "memory/facts/collide.md").read_text().endswith("one\n")


async def test_an_archive_in_one_group_and_an_update_in_another_leave_one_file(
    clone: Path, store: Store
) -> None:
    """`8f5911a`: one group archived a memory while another updated it, both
    landed in one commit, and the corpus ended up with two files carrying one
    id — which does not index at all (#239).

    The archive emits a write *and* a remove. Counting only the write left the
    second group's `Write` to the removed path looking like no collision, and
    it put back the file the remove had taken away.
    """
    doc = MemoryDoc.new(type="fact", title="Daily news summary", body="Every morning at 8.")
    path = write_memory(clone, doc, path="memory/facts/news.md")
    await observe(store, "News digest", "The digest was cancelled.")
    await observe(store, "Morning routine", "The digest goes out at nine now.")
    provider = Scripted(
        json.dumps([{"type": "archive", "id": doc.id, "reason": "no longer sent"}]),
        updating(doc.id, "Every morning at 9."),
    )

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 1
    assert result.deferred == 1
    assert files_with_id(clone, doc.id) == ["memory/archive/daily-news-summary.md"]
    assert path == "memory/facts/news.md"
    assert not Manifest.rebuild(clone)[1], "the corpus rebuilds without problems"


async def test_a_run_that_would_duplicate_an_id_commits_nothing(
    clone: Path, store: Store, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """The backstop, forced. Nothing upstream is supposed to let a change set
    get this far, which is exactly why the behaviour when one does needs
    stating: refuse the whole commit and defer, rather than write a corpus that
    will not index.
    """
    await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted(creating("Priya Raman", "Owns deploys."))
    monkeypatch.setattr(
        "siatt.runner.promote.collisions",
        lambda manifest, changes: {"mem_01ABC": ["memory/facts/a.md", "memory/facts/b.md"]},
    )
    before = GitRepo.at(clone).head()

    with caplog.at_level("ERROR"):
        result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert result.deferred == 1
    assert GitRepo.at(clone).head() == before, "nothing was committed"
    assert "refusing to commit" in caplog.text
    rows = await store.raw("SELECT state FROM observations", ())
    assert [r["state"] for r in rows] == ["pending"], "the observation waits for the next run"


# -- the bookkeeping ---------------------------------------------------------


async def test_a_plan_that_proposes_nothing_discards_with_a_reason(
    clone: Path, store: Store
) -> None:
    """`[]` is a normal answer, and the right one for a restated fact whose
    memory is already accurate. The observation is settled, not left to be
    reconsidered every hour."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted("[]")

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.discarded == 1
    rows = await store.raw("SELECT state, reason FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "discarded"
    assert "already says this" in str(rows[0]["reason"])


async def test_an_observation_that_keeps_failing_is_eventually_given_up_on(
    clone: Path, store: Store
) -> None:
    """Without this, one poison group costs a chat call every hour forever and
    is never promoted anyway."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")

    for _ in range(2):
        await (await promoter_for(clone, store, Scripted("not a plan"), max_attempts=2)).run()

    rows = await store.raw("SELECT state, reason FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "discarded"
    assert "promotion failed" in str(rows[0]["reason"])


async def test_a_run_is_bounded(clone: Path, store: Store) -> None:
    """One chat call per subject, and a backlog must not spend the whole budget
    on the tick that discovers it."""
    for n in range(5):
        await observe(store, f"subject {n}", f"Thing {n} is true.")
    provider = Scripted(*(["[]"] * 5))

    result = await (await promoter_for(clone, store, provider, max_subjects=2)).run()

    assert result.subjects == 2
    assert len(provider.requests) == 2


@pytest.mark.parametrize("extra", [{}, {"confidence": 0.9}, {"visibility": "private:someone"}])
async def test_update_ignores_model_timestamp_and_promotes(
    clone: Path, store: Store, extra: dict[str, Any]
) -> None:
    existing = MemoryDoc.new(type="fact", title="AI news schedule", body="Wants daily news.")
    path = write_memory(clone, existing)
    observation = await observe(store, "AI news schedule", "Daily news runs at 9 AM KST.")
    provider = Scripted(
        json.dumps(
            [
                {
                    "type": "update",
                    "id": existing.id,
                    "body": "Daily news runs at 9 AM KST.",
                    "frontmatter": {"updated": "2099-01-01T00:00:00Z", **extra},
                }
            ]
        )
    )
    before = datetime.now(UTC)

    result = await (await promoter_for(clone, store, provider, max_attempts=1)).run()

    assert result.promoted == 1
    assert result.discarded == result.deferred == 0
    written = MemoryDoc.parse((clone / path).read_text())
    assert "Daily news runs at 9 AM KST." in written.body
    assert written.id == existing.id
    assert written.frontmatter.created == existing.frontmatter.created
    assert before.replace(microsecond=0) <= written.frontmatter.updated <= datetime.now(UTC)
    assert written.frontmatter.visibility == existing.frontmatter.visibility
    if "confidence" in extra:
        assert written.frontmatter.confidence == extra["confidence"]
    rows = await store.raw("SELECT state, attempts FROM observations WHERE id = ?", (observation,))
    assert rows == [{"state": "promoted", "attempts": 0}]


@pytest.mark.parametrize("field", ["id", "created"])
async def test_update_timestamp_normalization_does_not_allow_identity_changes(
    clone: Path, store: Store, field: str
) -> None:
    existing = MemoryDoc.new(type="fact", title="AI news schedule", body="Wants daily news.")
    path = write_memory(clone, existing)
    original = (clone / path).read_text()
    await observe(store, "AI news schedule", "Daily news runs at 9 AM KST.")
    provider = Scripted(
        json.dumps(
            [
                {
                    "type": "update",
                    "id": existing.id,
                    "body": "Changed.",
                    "frontmatter": {"updated": "2099-01-01T00:00:00Z", field: "invalid"},
                }
            ]
        )
    )

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert result.deferred == 1
    assert (clone / path).read_text() == original
