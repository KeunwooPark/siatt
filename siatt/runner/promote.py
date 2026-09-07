"""`promote`: move distilled knowledge from SQLite into the git repo.

The job that makes the product exist. Everything before it accumulates
candidate facts in a database nobody reads; this is what turns them into files
a person can open, disagree with, and revert.

The shape of one run:

1. Read the pending observations and group them by `(subject, scope)`. The pair
   rather than the subject, because two visibility scopes must never meet
   inside one prompt — a group is reconciled as a unit, and the unit inherits
   one audience.
2. For each group, retrieve the memories already in the corpus that compete
   with it. This is the step that makes a restated fact an *update* instead of
   a second file saying the same thing.
3. Ask the chat model for a typed patch plan over that group.
4. Compile every accepted plan and apply them as **one commit**.
5. Mark the observations promoted or discarded, with the reason recorded.

Three rules it is built around.

**The model proposes; deterministic code disposes.** The reply is a JSON array
of patches and nothing else — it cannot mark an observation, choose a path, or
touch git. `promote` may not emit `Delete` at all, which `PatchCompiler`
enforces rather than trusts (#13); only `forget` deletes, and only what is
already archived.

**Visibility is inherited.** Every group carries one scope, retrieval is
filtered to it, and a plan that sets anything else on a document it creates is
corrected to the group's scope before compiling — loudly, because a model
getting that wrong is worth knowing about, and rejecting the plan over it would
lose the fact instead.

**Re-running is a no-op.** Idempotence comes from the observations table: a run
that commits marks its inputs `promoted`, and the next run finds nothing
pending. A crash between the commit and the marking re-promotes, and the model
sees its own memory in the competition and updates it rather than duplicating —
which is why the competition step is not only about quality.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from siatt.config import MemorySettings, PromoteSettings
from siatt.llm.registry import ModelRole, ProviderRegistry
from siatt.memory import blobref
from siatt.memory.changeset import collisions
from siatt.memory.consolidate import (
    PLAN_TOKENS,
    ConsolidationInput,
    build_request,
    decode_plan,
    unanswered,
)
from siatt.memory.document import (
    Frontmatter,
    MemoryDoc,
    MemoryError_,
    is_visibility,
    new_memory_id,
)
from siatt.memory.ltm import ApplyResult, Change, CommitMeta, MemoryStore, MemoryStoreError
from siatt.memory.manifest import Manifest
from siatt.memory.observation import Cited
from siatt.memory.patch import (
    Create,
    MemoryPatch,
    Merge,
    PatchCompiler,
    PatchError,
    Rejection,
    Supersede,
    Update,
)
from siatt.memory.retrieve import Retriever
from siatt.memory.schema import render_schema_md
from siatt.store import Store

log = logging.getLogger(__name__)

JOB = "promote"

#: Fresh ids offered to the model per group, on top of one per observation. A
#: `Create` needs an id that passes `Frontmatter._valid_id`, and a model asked
#: to invent a ULID produces something that fails validation often enough to
#: matter. Handing it valid ones costs nothing and removes the failure mode.
SPARE_IDS = 2

#: What the output budget is multiplied by when a reply comes back with no plan
#: in it. A model that ran out of room is not a model that disagreed, and
#: `max_attempts` is there to stop a group whose *content* cannot be planned —
#: spending three runs, an hour apart, discovering that the budget was the
#: problem loses the facts to a number this job chose. One retry at twice the
#: room settles it inside the run that noticed, and still terminates (#225).
RETRY_FACTOR = 2

#: Plans one group may cost in a single run. The second ask is for the case
#: where the first plan collided with a memory retrieval did not offer as
#: competition: the model is shown the file and asked again, rather than the
#: whole plan — including the patches that were fine — being thrown away and
#: the observations behind it discarded a few runs later (#226).
_ASKS_PER_GROUP = 2

TASK = """Reconcile these candidate observations about one subject against what
long-term memory already says.

`channel_messages` holds the observations, one per line, numbered. Each is a
claim distilled from a conversation. `memory_files` holds the memories already
in the corpus that compete with them, keyed by path — these are what you are
reconciling against, and updating one of them is almost always better than
writing another file about the same thing.

Return a JSON array of patch objects. The allowed operations are:

- `{{"type": "create", "memory": {{"frontmatter": {{"id": "<memory id>",
  "type": "fact", "title": "<title>", "tags": [], "visibility": "<scope>",
  "created": "<timestamp>", "updated": "<timestamp>"}}, "body": "<prose>"}},
  "path": "memory/<dir>/<slug>.md"}}` — a subject the corpus says nothing about
  yet. The frontmatter fields must be nested under `memory.frontmatter`; they
  are not fields of `memory` itself. `path` is optional; omit it and the
  conventional path for the document's `type` is used.
- `{{"type": "update", "id": "<memory id>", "body": "<full new body>",
  "frontmatter": {{...}}}}` — the corpus already covers this subject. `body`
  replaces the old body entirely, so write the whole thing, not the change.
  `frontmatter` carries only the fields you are changing. Never include `id`,
  `created`, or `updated` in an update's frontmatter: identity and creation time
  are immutable, and Siatt stamps `updated` automatically. Omit `frontmatter`
  when only the body changes.
- `{{"type": "merge", "into": "<memory id>", "from_ids": [...], "body": "..."}}`
  — two existing memories say the same thing. The sources are archived, not
  deleted, and their ids keep resolving.
- `{{"type": "supersede", "old_id": "<memory id>", "new": <document>}}` — the
  new claim contradicts an existing memory rather than extending it. The old
  one is archived and the new one records that it replaced it.

There is no delete. Nothing you return can remove a memory.

Return raw JSON only, with no Markdown or code fences. Return `[]` when the
corpus already says everything these observations say.
That is a normal answer, and it is the right one for a restated fact whose
memory is already accurate.

Write for a person reading the file in a year. One claim per memory; split
rather than append when a file starts covering two subjects.

{attachments}Use these ids for any memory you create, each at most once:
{ids}

Set `created` and `updated` to {now} on anything you create.
Set `visibility` to exactly `{scope}` on anything you create. Every observation
here came from a conversation with that audience, and a memory may not be
written to a wider one.

{schema}"""


#: Added to the task only for a group that has files to cite.
#:
#: Conditional because most groups have none, and a paragraph about attachments
#: on every prompt forever is a paragraph the model reads instead of the ones
#: that apply to it. It is also the only place the link syntax is stated, so a
#: group with nothing to cite is never told how — which is one fewer shape to
#: compose an invented digest into.
CITING = """`attachments` lists files that came with these observations, one per line, as
`[label] name — siatt://blob/<digest>`. A claim that is about one says so, and
where a memory records such a claim, link the file in its body as an ordinary
Markdown link: `[name](siatt://blob/<digest>)`, with the digest copied exactly
from that line. Write no other `siatt://blob/` link. A digest that is not on one
of those lines points at nothing, and is unwrapped out of your plan before it is
applied — so the memory keeps your sentence and loses the evidence it claimed.

"""


@dataclass(frozen=True, slots=True)
class Group:
    """The observations about one subject, from one audience."""

    subject: str
    scope: str
    rows: list[dict[str, Any]]

    @property
    def ids(self) -> list[str]:
        return [str(row["id"]) for row in self.rows]

    def rendered(self, known: Container[str]) -> tuple[list[str], list[str]]:
        """The claim lines and the attachment lines, which are labelled together.

        One method because the two halves have to agree. A claim says `a1` and
        the attachment list says what `a1` is; computing either alone would let
        them drift, and a plan that cites the wrong picture is worse than one
        that cites none.

        Labels rather than filenames as the link between them, because two
        people can send `IMG_3604.jpg` in one week and a group is a week of
        observations about one subject.

        `known` is the attachments that still exist. One collected since the
        observation was written is dropped here rather than offered and
        unwrapped later: a model told about a file it may cite will cite it,
        and the sentence it writes around a link that then disappears is a
        sentence about evidence.
        """
        labels: dict[str, str] = {}
        files: list[str] = []
        claims: list[str] = []
        for n, row in enumerate(self.rows, start=1):
            here = []
            for cited in _cited(row):
                if cited.sha256 not in known:
                    continue
                if cited.sha256 not in labels:
                    labels[cited.sha256] = f"a{len(labels) + 1}"
                    files.append(
                        f"[{labels[cited.sha256]}] {cited.name} — {blobref.uri(cited.sha256)}"
                    )
                here.append(labels[cited.sha256])
            attached = f" (attachments: {', '.join(here)})" if here else ""
            claims.append(
                f"[{n}] ({row['kind']}, confidence {row['confidence']}) {row['claim']}{attached}"
            )
        return claims, files


@dataclass(slots=True)
class Promotion:
    """What one run did, in the terms a person would ask about it."""

    subjects: int = 0
    promoted: int = 0
    discarded: int = 0
    deferred: int = 0
    changed: list[str] = field(default_factory=list)
    sha: str | None = None
    pull_request_url: str | None = None

    def summary(self) -> str:
        if not self.subjects:
            return "nothing pending"
        parts = [f"{self.subjects} subject(s)", f"{self.promoted} observation(s) promoted"]
        if self.discarded:
            parts.append(f"{self.discarded} discarded")
        if self.deferred:
            parts.append(f"{self.deferred} left pending")
        if self.changed:
            parts.append(f"{len(self.changed)} file(s) in {self.sha or 'no commit'}")
        return ", ".join(parts)


class Promoter:
    """One `promote` run, from the pending queue to a commit."""

    def __init__(
        self,
        store: Store,
        memory: MemoryStore,
        retriever: Retriever,
        registry: ProviderRegistry,
        *,
        policy: MemorySettings | None = None,
        settings: PromoteSettings | None = None,
        job_id: str | None = None,
    ) -> None:
        self._store = store
        self._blobs: frozenset[str] | None = None
        self._memory = memory
        self._retriever = retriever
        self._registry = registry
        self._policy = policy or MemorySettings()
        self._settings = settings or PromoteSettings()
        self._job_id = job_id

    async def run(self) -> Promotion:
        rows = await self._store.pending_observations(self._settings.max_observations)
        groups = _group(rows)[: self._settings.max_subjects]
        if not groups:
            return Promotion()

        manifest = self._memory.manifest()
        changes: list[Change] = []
        claimed: set[str] = set()
        promoted: list[Group] = []
        discarded: list[tuple[Group, str]] = []
        deferred: list[tuple[Group, str]] = []
        touched_ids: list[str] = []

        for group in groups:
            plan, compiled, problem = await self._propose(group, manifest)
            if problem is not None:
                deferred.append((group, problem))
                continue
            if not plan:
                discarded.append((group, "the corpus already says this; no change was proposed"))
                continue

            # Each group compiles against the corpus as it stands, not against
            # what the groups before it proposed, so two of them can land on
            # one path — and both writes would go into the same commit, where
            # the second silently replaces the first. Rare, and a lost fact
            # with no error is not a thing to leave to chance. The loser waits
            # for the next run, by which point the winner is on disk and shows
            # up as competition.
            #
            # Every path the group touches, `Remove` included. An archive is a
            # write *and* a remove of the file it moved out of, and counting
            # only the write left a second group's `Write` to that same path
            # looking like no collision at all: it put back the file the remove
            # had taken away, and the corpus ended up with two files under one
            # id, which does not index (#239).
            paths = {c.path for c in compiled}
            if overlap := paths & claimed:
                deferred.append((group, f"another subject in this run already writes {overlap}"))
                continue

            claimed |= paths
            changes.extend(compiled)
            promoted.append(group)
            touched_ids.extend(_memory_ids(plan))

        if broken := collisions(manifest, changes):
            # Not expected to fire: `claimed` above is what stops this, and the
            # groups are compiled one at a time, so no single plan reaches here
            # holding two files for one memory either. It is a backstop, at the
            # last moment the invariant is cheap to check and the commit has
            # not happened yet, because the cost of being wrong is not one bad
            # memory: a corpus with two files under one id does not index at
            # all, so retrieval goes on quietly answering from a stale index
            # until somebody runs `siatt doctor` (#239, #240).
            log.error(
                "promote: refusing to commit; %s would each live at two paths",
                "; ".join(f"{k} at {', '.join(v)}" for k, v in broken.items()),
            )
            at_two = ", ".join(broken)
            deferred.extend(
                (group, f"the run would have left {at_two} at two paths") for group in promoted
            )
            changes, promoted, touched_ids = [], [], []

        result = await self._commit(changes, promoted, touched_ids)
        return await self._record(result, promoted, discarded, deferred, len(groups))

    # -- one group -----------------------------------------------------------

    async def _propose(
        self, group: Group, manifest: Manifest
    ) -> tuple[list[MemoryPatch], list[Change], str | None]:
        """One group's plan and the writes it means, or why there are neither.

        Planned twice when the first plan collides with a memory the model was
        never shown. A `create` for a file the corpus already has is usually
        not the model being wrong about the corpus — it is the model answering
        the question it was asked, because retrieval did not offer that file as
        competition and so, as far as the plan could tell, nothing had been
        written about the subject.

        Handing it the file and asking again is the fix. Compiling the `create`
        as an update is not: the body it wrote does not know what is in the
        file, so writing it over the top would delete whatever the model was
        not shown — including, under `people/`, the block `identity` owns.
        """
        extra: dict[str, str] = {}
        problem: str | None = None
        for _ in range(_ASKS_PER_GROUP):
            plan, problem = await self._plan(group, extra=extra)
            if problem is not None or not plan:
                return [], [], problem
            compiler = PatchCompiler(
                self._memory.path,
                manifest,
                policy=self._policy,
                # A model writes this prose, and a model that has read one
                # attachment reference will compose another.
                blobs=await self._known_blobs(),
            )
            try:
                return plan, compiler.compile(plan, job=JOB), None
            except PatchError as exc:
                problem = str(exc)
                unshown = self._collided(exc.rejections, group, shown=extra)
                if not unshown:
                    break
                log.info(
                    "promote: re-planning %r with %s, which it was not shown",
                    group.subject,
                    ", ".join(sorted(unshown)),
                )
                extra.update(unshown)
        return [], [], problem

    def _collided(
        self, rejections: Sequence[Rejection], group: Group, *, shown: Mapping[str, str]
    ) -> dict[str, str]:
        """The memories a plan collided with, as competition for the next ask.

        Scoped to the group exactly as `_competing` is. A file sitting at the
        path the plan wanted is not thereby a file this group's audience may
        read, and "the model needs to see it" is not a reason to put a private
        memory in front of a workspace prompt. It is read from disk rather than
        trusted to the manifest, because visibility is a property of the
        document and a manifest can be stale.
        """
        files: dict[str, str] = {}
        for path in sorted({r.conflict for r in rejections if r.conflict} - set(shown)):
            content = self._offerable(path)
            if content is None:
                continue
            try:
                doc = MemoryDoc.parse(content, source=path)
            except MemoryError_ as exc:
                log.warning("promote: %s is in the way and does not parse: %s", path, exc)
                continue
            if doc.frontmatter.visibility != group.scope:
                log.warning(
                    "promote: %s is in the way of a %r group and is %r, so it was not shown",
                    path,
                    group.scope,
                    doc.frontmatter.visibility,
                )
                continue
            files[path] = content
        return files

    async def _known_blobs(self) -> frozenset[str]:
        """Attachments that exist, read once per run.

        Cached because the compiler is rebuilt per patch in some of these jobs,
        and re-reading the table for each one would be a query per memory to
        answer a question whose answer does not change inside a run.
        """
        if self._blobs is None:
            self._blobs = await self._store.attachment_hashes()
        return self._blobs

    async def _plan(
        self, group: Group, *, extra: Mapping[str, str] | None = None
    ) -> tuple[list[MemoryPatch], str | None]:
        """The model's plan for one group, or why there is not one.

        Asked at most twice. A reply with no plan in it — truncated, or empty
        because a reasoning model spent the budget thinking — is not an answer
        to argue with, and the second ask has twice the room. A reply that *is*
        a plan is decoded once; being wrong is what `max_attempts` is for.
        """
        if not is_visibility(group.scope):
            # The scope came off a session row, so this is a bug upstream
            # rather than anything the model did. Writing the memory anyway
            # would put an unparseable `visibility` in the corpus, and every
            # later read of that file fails.
            return [], f"{group.scope!r} is not a visibility scope a memory may carry"
        competing = await self._competing(group) | dict(extra or {})
        claims, files = group.rendered(await self._known_blobs())
        task = TASK.format(
            ids="\n".join(f"  {i}" for i in _fresh_ids(len(group.rows) + SPARE_IDS)),
            now=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            scope=group.scope,
            schema=render_schema_md(),
            attachments=CITING if files else "",
        )
        content = ConsolidationInput(
            channel_messages=claims, memory_files=competing, attachments=files
        )

        silence = ""
        for budget in (PLAN_TOKENS, PLAN_TOKENS * RETRY_FACTOR):
            request = build_request(job=JOB, task=task, content=content, max_tokens=budget)
            response = await self._registry.complete(ModelRole.CHAT, request, tag="promote.plan")
            silence = unanswered(response)
            if not silence:
                try:
                    plan = decode_plan(response.text, job=JOB)
                except PatchError as exc:
                    return [], str(exc)
                return _normalize_plan(plan, group.scope), None
            log.warning("promote: %s, planning %r", silence, group.subject)
        return [], silence

    async def _competing(self, group: Group) -> dict[str, str]:
        """The memories already in the corpus that this group is about.

        Scoped to the group, so a private observation is never reconciled
        against — or into — a memory it is not allowed to see. Read as whole
        files rather than as the retriever's snippets, because an `Update`
        rewrites a body and a model shown half of one would write half of one
        back.
        """
        query = " ".join([group.subject, *(str(row["claim"]) for row in group.rows)])
        retrieval = await self._retriever.retrieve(
            query,
            scope=group.scope,
            include_pinned=False,
            limit=self._settings.competing_memories,
        )
        manifest = self._memory.manifest()
        files: dict[str, str] = {}
        for memory_id in dict.fromkeys(retrieval.memory_ids):
            entry = manifest.resolve(memory_id)
            if entry is None or entry.path in files:
                continue
            content = self._offerable(entry.path)
            if content is None:
                continue
            files[entry.path] = content
            if len(files) >= self._settings.competing_memories:
                break
        return files

    def _offerable(self, path: str) -> str | None:
        """A memory's text, or `None` if it may not go in a plan prompt.

        One rule for every route a file takes into one, so that a document too
        large to reconcile is too large whether retrieval found it or a
        collision did.
        """
        try:
            content = self._memory.read(path)
        except MemoryStoreError as exc:
            log.warning("promote: could not read competing memory %s: %s", path, exc)
            return None
        if len(content) > self._settings.max_memory_chars:
            # Skipped, not truncated. See `PromoteSettings.max_memory_chars`.
            log.warning(
                "promote: %s is %d chars and was not offered as competition; "
                "reorganize should be splitting it",
                path,
                len(content),
            )
            return None
        return content

    # -- the commit, and the bookkeeping -------------------------------------

    async def _commit(
        self, changes: Sequence[Change], promoted: Sequence[Group], memory_ids: Sequence[str]
    ) -> ApplyResult:
        if not changes:
            return ApplyResult()
        observations = sum(len(group.rows) for group in promoted)
        subjects = ", ".join(group.subject for group in promoted[:3])
        if len(promoted) > 3:
            subjects += f" and {len(promoted) - 3} more"
        return await self._memory.apply(
            changes,
            CommitMeta(
                summary=f"promote {observations} observation(s) about {subjects}",
                job=JOB,
                job_id=self._job_id,
                memory_ids=list(dict.fromkeys(memory_ids)),
            ),
        )

    async def _record(
        self,
        result: ApplyResult,
        promoted: Sequence[Group],
        discarded: Sequence[tuple[Group, str]],
        deferred: Sequence[tuple[Group, str]],
        subjects: int,
    ) -> Promotion:
        """Move every observation this run decided about out of `pending`.

        After the commit, deliberately. A row marked `promoted` before the
        write lands is a fact that was never written and will never be
        proposed again; the other order re-proposes at worst, and the model
        sees the memory it already wrote and updates it.
        """
        landed = result.sha is not None or result.pull_request_url is not None
        where = result.pull_request_url or result.sha or "no commit"
        outcome = Promotion(
            subjects=subjects,
            changed=list(result.changed),
            sha=result.sha,
            pull_request_url=result.pull_request_url,
        )

        waiting = list(deferred)
        for group in promoted:
            if landed:
                outcome.promoted += await self._store.resolve_observations(
                    group.ids, state="promoted", reason=f"written to long-term memory in {where}"
                )
            else:
                # The plan compiled and the write did not land, so nothing was
                # promoted. Deferred like a rejected plan, attempt cap
                # included: a group that compiles to a commit git decides is
                # empty would otherwise be re-planned every hour forever.
                waiting.append((group, "the plan compiled but nothing was committed"))

        for group, reason in discarded:
            outcome.discarded += await self._store.resolve_observations(
                group.ids, state="discarded", reason=reason
            )

        for group, reason in waiting:
            await self._store.note_observation_attempt(group.ids)
            exhausted = [
                row for row in group.rows if int(row["attempts"]) + 1 >= self._settings.max_attempts
            ]
            if exhausted:
                log.warning(
                    "promote: giving up on %d observation(s) about %r after %d attempt(s): %s",
                    len(exhausted),
                    group.subject,
                    self._settings.max_attempts,
                    reason,
                )
                outcome.discarded += await self._store.resolve_observations(
                    [str(row["id"]) for row in exhausted],
                    state="discarded",
                    reason=f"promotion failed {self._settings.max_attempts} time(s): {reason}",
                )
            else:
                log.info("promote: deferring %r to the next run: %s", group.subject, reason)
            outcome.deferred += len(group.rows) - len(exhausted)

        return outcome


# -- helpers -----------------------------------------------------------------


def _group(rows: Sequence[dict[str, Any]]) -> list[Group]:
    """Pending observations, gathered by the unit `promote` reconciles.

    `(subject, scope)` and not `subject`: a group becomes one prompt and one
    memory's audience, and mixing two scopes in it is how something said in a
    DM ends up in a workspace file.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["subject"]), str(row["scope"])), []).append(row)
    return [
        Group(subject=subject, scope=scope, rows=members)
        for (subject, scope), members in grouped.items()
    ]


def _fresh_ids(count: int) -> list[str]:
    return [new_memory_id() for _ in range(count)]


def _cited(row: Mapping[str, Any]) -> list[Cited]:
    """What one observation was looking at, off the row.

    Tolerant of a row that predates the column and of one whose JSON is not
    what this expects. An observation is a fact somebody may have been waiting
    hours for, and dropping it because its attachment list will not decode
    would lose the claim over the part of it that is decoration.
    """
    try:
        payload = json.loads(str(row.get("attachments") or "[]"))
    except json.JSONDecodeError:
        log.warning("observation %s has an attachment list that is not JSON", row.get("id"))
        return []
    if not isinstance(payload, list):
        return []
    return [
        Cited(sha256=str(item["sha256"]), name=str(item.get("name") or "an attachment"))
        for item in payload
        if isinstance(item, dict) and item.get("sha256")
    ]


def _memory_ids(plan: Sequence[MemoryPatch]) -> list[str]:
    ids = []
    for patch in plan:
        match patch:
            case Create():
                ids.append(patch.memory.id)
            case Update():
                ids.append(patch.id)
            case Merge():
                ids.append(patch.into)
            case Supersede():
                ids.append(patch.new.id)
            case _:
                pass
    return ids


def _normalize_plan(plan: Sequence[MemoryPatch], scope: str) -> list[MemoryPatch]:
    """Enforce scope and drop model-supplied update timestamps.

    Corrected rather than rejected. The scope is not the model's to choose, so
    a plan that got it wrong is not a plan to argue with — and rejecting it
    would throw away the fact to punish the formatting. `PatchCompiler` still
    refuses to *widen* an existing memory, which is the case this cannot reach.
    """
    corrected: list[MemoryPatch] = []
    for patch in plan:
        match patch:
            case Create():
                corrected.append(patch.model_copy(update={"memory": _scoped(patch.memory, scope)}))
            case Supersede():
                corrected.append(patch.model_copy(update={"new": _scoped(patch.new, scope)}))
            case Update() if {"visibility", "updated"} & patch.frontmatter.keys():
                # An update may not change visibility at all: the memory's
                # audience was set when it was written, and this plan is about
                # one group's claims, not about who may read it.
                if "visibility" in patch.frontmatter:
                    log.warning(
                        "promote: dropped a visibility change from an update to %s", patch.id
                    )
                # Siatt owns this timestamp. Ignore the model's value so an
                # otherwise valid observation does not exhaust its retries.
                # Identity fields still reach the compiler and are rejected.
                if "updated" in patch.frontmatter:
                    log.warning(
                        "promote: dropped an updated timestamp from an update to %s", patch.id
                    )
                corrected.append(
                    patch.model_copy(
                        update={
                            "frontmatter": {
                                k: v
                                for k, v in patch.frontmatter.items()
                                if k not in {"visibility", "updated"}
                            }
                        }
                    )
                )
            case _:
                corrected.append(patch)
    return corrected


def _scoped(doc: MemoryDoc, scope: str) -> MemoryDoc:
    if doc.frontmatter.visibility == scope:
        return doc
    log.warning(
        "promote: a plan set visibility %r on a new memory; the observations came from %r",
        doc.frontmatter.visibility,
        scope,
    )
    # Re-validated rather than `model_copy`d in: `model_copy` does not run the
    # validators, and an unparseable `visibility` written to a file is one
    # every later read of that file fails on. `_plan` has already checked the
    # scope, so this cannot raise; it is here because "cannot" is a property of
    # today's callers.
    fields = doc.frontmatter.model_dump() | {"visibility": scope}
    return doc.model_copy(update={"frontmatter": Frontmatter.model_validate(fields)})
