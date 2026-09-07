"""Configuration loading and object construction.

Secrets are never stored in the config file — only the *name* of the
environment variable holding them. `siatt init` writes this file; with no file
present a usable config is synthesized from the environment, so a first run
needs nothing but an API key exported.

That name is resolved by `siatt.vault.resolve`: the environment first, then the
local vault. So the same config file works whether the key is exported or
stored by `siatt vault set`, and nothing here has to know which.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from siatt.core.agent import AgentConfig
from siatt.core.context import ContextBudget
from siatt.errors import ConfigError
from siatt.fetch.browser import BrowserRenderer
from siatt.fetch.client import WebFetcher
from siatt.llm.anthropic_compat import AnthropicCompatProvider
from siatt.llm.base import LLMProvider
from siatt.llm.cost import CostMeter, Price, PriceBook
from siatt.llm.openai_compat import OpenAICompatProvider
from siatt.llm.registry import ModelRole, ProviderRegistry, RetryPolicy
from siatt.memory.salience import Decay
from siatt.search.base import SearchProvider
from siatt.search.brave import BraveSearch
from siatt.store import Store
from siatt.store.blobs import (
    DEFAULT_ALLOWED_MIME,
    DEFAULT_MAX_BYTES,
    Attachments,
    BlobStore,
)
from siatt.vault import resolve

ProviderKind = Literal["openai", "anthropic"]
SearchKind = Literal["brave"]

NO_CHAT_PROVIDER = (
    "no model configured for the 'chat' role.\n"
    "Export ANTHROPIC_API_KEY or OPENAI_API_KEY, or write a config file "
    "(see `siatt config` for where it is looked for)."
)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CLONE_PATH = "~/.siatt/ltm"

#: The one destination that is not a channel somebody configured: the channel
#: the task was created in, posted at top level so every firing starts its own
#: thread. Reserved, so `[tasks.destinations]` cannot define a name that means
#: something else — and it is the only destination a tool may set, because it
#: reaches exactly the conversation the task already belongs to (§11.1).
HERE = "here"

#: What `siatt task add --destination` takes, and therefore what a key in
#: `[tasks.destinations]` may be. A short word a person types on a terminal.
_DESTINATION_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")

#: A Slack channel. `C` is what every channel is today, `G` a private group
#: from before they were; `D` is a DM and `U` is a person, and neither is
#: somewhere a schedule may be pointed.
_CHANNEL_ID = re.compile(r"[CG][A-Z0-9]{1,40}")


def config_path() -> Path:
    if override := os.environ.get("SIATT_CONFIG"):
        return Path(override).expanduser()
    return Path(user_config_dir("siatt")) / "config.toml"


def default_db_path() -> Path:
    if override := os.environ.get("SIATT_DB"):
        return Path(override).expanduser()
    return Path(user_data_dir("siatt")) / "siatt.db"


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ProviderKind
    model: str
    base_url: str | None = None
    key_env: str | None = None
    embedding_dimensions: int | None = None
    timeout_seconds: float | None = None
    fallbacks: list[ProviderConfig] = Field(default_factory=list)

    def api_key(self) -> str:
        env = self.key_env or default_key_env(self.kind)
        key = resolve(env)
        if not key:
            raise ConfigError(
                f"{env} is not set and is not in the vault "
                f"(needed for the {self.kind} provider {self.model!r}).\n"
                f"Export it, or run `siatt vault set {env}`."
            )
        return key

    def build(self) -> LLMProvider:
        if self.kind == "anthropic":
            return AnthropicCompatProvider(
                model=self.model,
                api_key=self.api_key(),
                base_url=self.base_url or "https://api.anthropic.com/v1",
            )
        return OpenAICompatProvider(
            model=self.model,
            api_key=self.api_key(),
            base_url=self.base_url or "https://api.openai.com/v1",
            embedding_dimensions=self.embedding_dimensions,
            timeout=self.timeout_seconds,
        )

    def chain(self) -> list[LLMProvider]:
        return [self.build(), *(f.build() for f in self.fallbacks)]


class LTMSettings(BaseModel):
    """The private GitHub repository that holds long-term memory."""

    model_config = ConfigDict(extra="forbid")

    #: `owner/name`, or any URL git can clone. None until `siatt init` has run.
    repo: str | None = None
    clone_path: str | None = None
    branch: str = "main"
    token_env: str = "SIATT_GITHUB_TOKEN"
    #: Jobs that open a pull request instead of pushing to the branch. Defaults
    #: to the destructive one: `forget` is the job you want to read before it
    #: lands, and `promote` is the one you would get tired of approving.
    supervised: list[str] = Field(default_factory=lambda: ["forget"])

    @property
    def configured(self) -> bool:
        return bool(self.repo)

    def resolved_clone_path(self) -> Path:
        return Path(self.clone_path or DEFAULT_CLONE_PATH).expanduser()

    def token(self) -> str | None:
        return resolve(self.token_env)


class SlackSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_token_env: str | None = None  # xapp-, Socket Mode
    bot_token_env: str | None = None  # xoxb-
    #: Override Slack's Web API endpoint. Useful for private proxies and for
    #: exercising the real Socket Mode client against a local QA server.
    api_url: str = "https://slack.com/api/"
    allowed_channels: list[str] = Field(default_factory=list)
    #: Post a placeholder and rewrite it as the answer arrives, rather than
    #: saying nothing until the turn is over. On by default: a long turn that
    #: shows no sign of life gets asked again, and that is a second model call
    #: and two answers. Off is a single message, and costs one API call a turn.
    stream: bool = True
    #: Emoji name to verdict, for reactions on Siatt's own answers (#36). The
    #: names are Slack's own, without colons and without a skin tone. Replacing
    #: this replaces the whole map rather than adding to it: a workspace where
    #: ✅ means "I have actioned this" should be able to say so, and a default
    #: that could only be extended would keep boosting memory on the strength
    #: of a checkbox.
    reactions: dict[str, str] = Field(
        default_factory=lambda: {
            "+1": "up",
            "thumbsup": "up",
            "white_check_mark": "up",
            "x": "down",
            "-1": "down",
            "thumbsdown": "down",
        }
    )

    @field_validator("reactions")
    @classmethod
    def _known_verdicts(cls, value: dict[str, str]) -> dict[str, str]:
        if unknown := {v for v in value.values() if v not in ("up", "down")}:
            raise ValueError(f"a reaction maps to 'up' or 'down', not {', '.join(sorted(unknown))}")
        # Slack sends the name without colons; accepting them written either
        # way costs one line and saves a config that silently never matches.
        return {name.strip().strip(":"): verdict for name, verdict in value.items()}

    @property
    def configured(self) -> bool:
        return bool(self.app_token_env and self.bot_token_env)


class MemorySettings(BaseModel):
    """Limits on what a consolidation job is allowed to do in one commit.

    These are the numbers the patch validator enforces. They exist because the
    plan on the other side of them was written by a model reading text that
    someone else typed.
    """

    model_config = ConfigDict(extra="forbid")

    #: A memory is a paragraph or two. Anything near this is a transcript that
    #: escaped, or a prompt-injection payload.
    max_file_bytes: int = 65_536
    #: One job touching more than this is a runaway, not a consolidation.
    max_files_per_commit: int = 25
    #: How long an archived memory must sit before it can be deleted at all.
    retention_floor_days: int = 30


class EpisodeSettings(BaseModel):
    """When a stretch of conversation is over, and how much of it to read.

    The boundaries are what `docs/DESIGN.md` §6 draws: a thread that has gone
    quiet, or one that has run long enough to be worth consolidating before it
    goes quiet at all.
    """

    model_config = ConfigDict(extra="forbid")

    #: Quiet for this long and the segment is over. Twenty minutes is the
    #: design's number: long enough that a slow reply does not split a
    #: conversation in half, short enough that today's thread is consolidated
    #: today.
    idle_minutes: int = 20
    #: A long thread is closed on length rather than left to grow. An episode
    #: that never ends is one whose transcript eventually stops fitting in a
    #: context window, and consolidating the first half of a conversation is
    #: better than consolidating none of it.
    max_messages: int = 40
    #: Episodes consolidated per sweep. A bound rather than a target: it is
    #: what stops a backlog — a daemon down for a day — from spending its whole
    #: token budget in one tick.
    max_per_run: int = 20
    #: How much of an episode the extractor reads. Above `max_messages` on
    #: purpose: a session end closes an episode of any length.
    transcript_messages: int = 200
    #: More candidate facts than this out of one conversation is a model
    #: narrating the transcript rather than distilling it.
    max_observations: int = 12
    #: The cost gate (`docs/DESIGN.md` §6.1). An episode scored below this
    #: closes with its summary and is never extracted from, so it never
    #: reaches `promote`. Low, because the thing being traded is tokens
    #: against knowledge, and only one of those can be got back later. `0.0`
    #: turns the gate off — every score clears it.
    signal_threshold: float = 0.3


class PromoteSettings(BaseModel):
    """How much of the pending queue one `promote` run works through.

    Every number here bounds a cost. `promote` calls the *chat* model once per
    subject and writes one commit per run, so an unbounded run is both the
    largest bill in the system and the largest diff a person has to review.
    """

    model_config = ConfigDict(extra="forbid")

    #: Pending observations read per run.
    max_observations: int = 100
    #: Subjects reconciled per run — one chat-model call each.
    max_subjects: int = 10
    #: Existing memories offered per subject as competition. This is what makes
    #: a restated fact an update instead of a duplicate, so it is the number to
    #: raise if duplicates start appearing.
    competing_memories: int = 5
    #: A memory larger than this is not shown as competition. Not truncated:
    #: an `Update` replaces the body wholesale, so a model shown half a memory
    #: would rewrite it as half a memory. One this size is a file `reorganize`
    #: should be splitting.
    max_memory_chars: int = 8_000
    #: Rejected plans a subject may produce before its observations are
    #: discarded. Without it, one poison group costs a chat call every hour
    #: forever and is never promoted anyway.
    max_attempts: int = 3


class ForgetSettings(BaseModel):
    """When a memory stops being worth keeping, and how slowly.

    Read alongside `MemorySettings.retention_floor_days`, which is the hard
    floor the patch validator enforces on any deletion. Nothing here can lower
    it: these numbers only decide how much *more* cautious than that a
    particular installation wants to be.
    """

    model_config = ConfigDict(extra="forbid")

    #: Salience at which a live memory moves to `archive/`. At the default
    #: decay curve this is roughly two months with no recall at all — and
    #: `reflect` boosts on every recall, so a memory anybody has needed since
    #: the summer is nowhere near it.
    archive_below: float = 0.12
    #: How long an archived memory sits before it is collected. Comfortably
    #: past the retention floor on purpose: the floor is the line the validator
    #: refuses to cross, and a policy that sat exactly on it would be relying
    #: on the last check in the system rather than on its own judgement.
    archive_grace_days: int = 60
    #: Files this may touch in one week, across both transitions.
    max_per_run: int = 20


class ReorganizeSettings(BaseModel):
    """The weekly librarian pass: what it looks at, and how much it may do.

    Every number is a bound on a job that rewrites files nobody asked it to
    touch. The value of the corpus being in git is that a person can read what
    changed, and a pass that rewrote a hundred files a week would take that
    away while technically remaining reversible.
    """

    model_config = ConfigDict(extra="forbid")

    #: A memory past this is a candidate to split. Well under
    #: `memory.max_file_bytes`, which is the point at which the validator
    #: refuses a file outright: this is "has grown into two subjects", not
    #: "is a transcript that escaped".
    split_above_bytes: int = 6_000
    #: Token overlap at which two memories are worth *asking* about. Not a
    #: judgement that they are duplicates — that costs a model call over both
    #: whole documents — only a filter that never suggests a pair sharing no
    #: vocabulary, so a tidy corpus costs nothing to check.
    #:
    #: Low, and deliberately so. Two memories written months apart about one
    #: fact share the names and the verbs and little else, so a strict
    #: threshold misses exactly the duplicates worth finding. The two errors
    #: are not symmetric: a false positive costs one bounded model call that
    #: answers `[]`, and a false negative is a duplicate that stays forever.
    duplicate_overlap: float = 0.45
    #: Model calls per run, across merges and splits together.
    max_operations: int = 6
    #: Memories that may end up in one merge. Three files about one person are
    #: one question; a cap is what stops a corpus of near-identical notes from
    #: becoming a single unreadable file.
    max_cluster: int = 4
    #: Memories read per run when looking for candidates.
    max_candidates: int = 500


class ReflectSettings(BaseModel):
    """The nightly pass: the journal, the salience curve, and the digest."""

    model_config = ConfigDict(extra="forbid")

    #: The salience curve. See `siatt/memory/salience.py` — salience is
    #: recomputed from age and recall rather than adjusted, which is what lets
    #: a bounded nightly pass converge instead of double-counting.
    base_salience: float = 0.5
    half_life_days: float = 30.0
    hit_boost: float = 0.08
    max_hit_boost: float = 0.3
    salience_floor: float = 0.05
    #: How far back recalls count. The same order as the half-life: a memory
    #: that was busy last season and quiet since should be fading.
    hit_window_days: int = 30
    #: Below this, a memory's salience is left alone. The commit is a diff a
    #: person reads, and a file rewritten for a change in the third decimal
    #: place is noise in it.
    min_salience_move: float = 0.02
    #: Salience rewrites per run, under the per-commit file cap. On a corpus
    #: larger than one commit, the memories furthest from their true score go
    #: first and the rest converge over the following nights.
    max_salience_updates: int = 20
    #: How many recently-touched memories are read together when looking for
    #: contradictions. One prompt, so this is a token budget as much as a
    #: coverage decision.
    max_conflict_candidates: int = 12
    #: Contradictions surfaced per night. More than a handful is a corpus with
    #: a structural problem, and a list nobody reads to the end.
    max_conflicts: int = 5
    journal_tokens: int = 800
    #: Slack channel id for the nightly digest. None posts nothing.
    digest_channel: str | None = None
    #: What a 👍 on an answer adds to the salience of the memories behind it,
    #: and the ceiling on that (#36).
    endorsement_boost: float = 0.2
    max_endorsement_boost: float = 0.4
    #: What an ❌ multiplies a memory's confidence by, once per reaction. A
    #: person disagreeing with one answer is a reason to trust a memory less,
    #: not a reason to stop believing it.
    suspect_factor: float = 0.7
    #: Confidence rewrites per run, under the same per-commit cap the salience
    #: updates share.
    max_suspect_updates: int = 10

    def decay(self) -> Decay:
        return Decay(
            base=self.base_salience,
            half_life_days=self.half_life_days,
            per_hit=self.hit_boost,
            max_boost=self.max_hit_boost,
            per_endorsement=self.endorsement_boost,
            max_endorsement_boost=self.max_endorsement_boost,
            floor=self.salience_floor,
        )


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str | None = None
    #: Rounds of tool calls one turn may spend, and the wall clock it may spend
    #: them over. Both bound the same thing — how much work a turn does before
    #: it has to answer — and a turn that hits either writes up what it has
    #: rather than returning nothing (#200).
    max_tool_iterations: int = 40
    max_turn_seconds: float = 600.0
    max_tokens: int = 4096
    temperature: float | None = None
    history_limit: int = 200


class ContextSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = 128_000
    system: float = 0.05
    pinned: float = 0.10
    retrieved: float = 0.30
    episode: float = 0.10
    recent: float = 0.35
    headroom: float = 0.10

    @model_validator(mode="after")
    def _budget_is_constructible(self) -> ContextSettings:
        """Fail at load time, not at the first turn.

        `ContextBudget` validates the shares in `__post_init__`, and nothing
        constructed one until a command built a packer — so a config whose
        shares did not sum to 1.0 passed `siatt config` and `siatt doctor` and
        then stopped `siatt run` from starting (#76). A health check that is
        green on a config the program refuses is worse than no health check.
        """
        self.to_budget()
        return self

    def to_budget(self) -> ContextBudget:
        return ContextBudget(**self.model_dump())

    def tokens_for_retrieval(self) -> int:
        """The share retrieval may fill. Retrieval truncates to this itself, so
        that what `siatt why` reports is what the packer would have received."""
        return self.to_budget().tokens_for(self.retrieved)


class StoreSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None

    def resolved(self) -> Path:
        return Path(self.path).expanduser() if self.path else default_db_path()


class PriceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class SearchSettings(BaseModel):
    """Web search, off unless a `kind` is set.

    Absent configuration means the tool is never registered, rather than
    registered and failing on use. A model told it can search will spend a turn
    finding out that it cannot, and then apologize for it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: SearchKind | None = None
    key_env: str | None = None
    base_url: str | None = None
    #: Well under `DEFAULT_TOOL_TIMEOUT`. A search is a detour on a turn
    #: somebody is waiting through, and a slow one should be abandoned rather
    #: than spend the whole budget for the reply.
    timeout_seconds: float = Field(default=10.0, gt=0)
    max_results: int = Field(default=5, ge=1, le=10)
    #: USD per call, from the vendor's price list. Zero — the default — still
    #: counts calls; it just cannot contribute to the daily ceiling. Same
    #: bargain as `[pricing]`: a stale built-in number is worse than none.
    cost_per_call_usd: float = Field(default=0.0, ge=0)

    @property
    def configured(self) -> bool:
        return self.kind is not None

    def api_key(self) -> str:
        if self.kind is None:
            raise ConfigError("web search is not configured; set `kind` under [search]")
        env = self.key_env or default_search_key_env(self.kind)
        key = resolve(env)
        if not key:
            raise ConfigError(
                f"{env} is not set and is not in the vault "
                f"(needed for {self.kind} web search).\n"
                f"Export it, or run `siatt vault set {env}`."
            )
        return key

    def build(self) -> SearchProvider:
        return BraveSearch(
            api_key=self.api_key(),
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )


class FetchSettings(BaseModel):
    """Reading a page, on unless an install says otherwise.

    The other way round from `[search]`, and for a reason that is not
    inconsistency: search cannot work without a key somebody went and got, so
    its absence is the honest default. Fetching needs nothing, and a capability
    that has to be discovered and enabled is a capability that is missing on
    the day it was needed — the model searched, found the page that had the
    answer, and could not open it.

    What makes fetching safe is `siatt/fetch/guard.py`, not this flag. The flag
    is for an install that wants the whole outbound surface gone: set
    `enabled = false` and the tool is not registered, the same as an unnamed
    search `kind`.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: The whole request, redirects included. A page is a detour on a turn
    #: somebody is waiting through.
    timeout_seconds: float = Field(default=15.0, gt=0)
    #: Off the wire, before any text is extracted from it.
    max_bytes: int = Field(default=2_000_000, ge=1_024)
    #: Handed to the model. What actually costs context.
    max_chars: int = Field(default=20_000, ge=500)
    max_redirects: int = Field(default=4, ge=0, le=10)
    #: Zero by default and still counted, on the same terms as search.
    cost_per_call_usd: float = Field(default=0.0, ge=0)

    def build(self, renderer: BrowserRenderer | None = None) -> WebFetcher:
        return WebFetcher(
            timeout=self.timeout_seconds,
            max_bytes=self.max_bytes,
            max_chars=self.max_chars,
            max_redirects=self.max_redirects,
            renderer=renderer,
        )


class AttachmentSettings(BaseModel):
    """Files people send, kept only if an install asks for them.

    Off by default, on the same terms as `[search]` rather than `[fetch]`. The
    argument for fetching being on is that it needs nothing and its absence is
    only discovered on the day it was needed; this needs a directory that grows
    without bound until the collector exists, and a Slack scope somebody has to
    go and add. A capability that writes to disk should be asked for.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: Per file, enforced while the bytes go past rather than after they are all
    #: in hand -- a cap applied to something already in memory is not a cap.
    max_bytes: int = Field(default=DEFAULT_MAX_BYTES, ge=1_024)
    #: Mime prefixes. Images and video by default: those are the two kinds a
    #: person sends into a conversation expecting them to be looked at.
    allowed_mime: tuple[str, ...] = DEFAULT_ALLOWED_MIME

    def blobs(self, db_path: Path) -> BlobStore:
        return BlobStore.beside(db_path, max_bytes=self.max_bytes)

    def build(self, store: Store, db_path: Path) -> Attachments:
        """The scoped API, or nothing when the install has not asked for it.

        Returning `Attachments` only when enabled is the same bargain the search
        tool makes: nothing downstream has to re-check a flag, because there is
        nothing to call.
        """
        return Attachments(store, self.blobs(db_path), allowed_mime=tuple(self.allowed_mime))


class BrowserSettings(BaseModel):
    """Running a page rather than reading it, off unless an install asks.

    The opposite default from `[fetch]`, and for a reason `[fetch]`'s docstring
    does not apply to: this one is not free. It needs an extra whose browser is
    about 650MB on disk, and a render costs several seconds and a few hundred
    megabytes of RSS while it runs. A capability with that price attached is one
    an install should choose, and until it does the tool does not mention it —
    the `render` parameter is absent from the schema, not present and refused.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    #: The whole render: launch, navigate, and the settle below.
    timeout_seconds: float = Field(default=30.0, gt=0)
    #: How long script is allowed to keep working after the document is ready.
    #: Fixed rather than waiting for network idle, which a page that polls never
    #: reaches.
    settle_ms: int = Field(default=3_000, ge=0, le=30_000)
    #: A page wanting more than this is not a page.
    max_requests: int = Field(default=600, ge=1)
    max_bytes: int = Field(default=20_000_000, ge=1_024)

    def build(self) -> BrowserRenderer:
        return BrowserRenderer(
            timeout=self.timeout_seconds,
            settle_ms=self.settle_ms,
            max_requests=self.max_requests,
            max_bytes=self.max_bytes,
        )


class BudgetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    daily_usd_ceiling: float | None = Field(default=None, ge=0)


class TaskSettings(BaseModel):
    """Bounds on the schedules a person may create.

    Every firing of a standing task is a full turn — retrieval, a frontier
    model, however many tools it reaches for — so an unbounded table is a bill
    rather than a feature. These are the three numbers that keep it one.
    """

    model_config = ConfigDict(extra="forbid")

    #: Per person, counting active and paused. Generous enough that nobody
    #: bumps into it doing the obvious thing, low enough that a runaway loop of
    #: "make me a schedule" cannot fill the table.
    max_per_owner: int = Field(default=20, ge=1)

    #: The floor between two fires, checked on the gap the expression actually
    #: produces rather than on how it is written. `*/15` and `0,15,30,45` are
    #: the same schedule and are treated as one.
    min_interval_minutes: int = Field(default=15, ge=1)

    #: Consecutive failed runs before a task is paused and its owner told. A
    #: task failing quietly forever is worse than one that stops.
    disable_after_failures: int = Field(default=5, ge=1)

    #: Named Slack channels a standing task may be pointed at, as
    #: `name = "C0123ABCD"`. Empty by default, which is what keeps a fresh
    #: install unable to post anywhere nobody asked it to.
    #:
    #: This is the only place a channel id can enter the feature. A task row
    #: holds one of these *names*; the id it stands for is read from here at
    #: fire time, so pointing a schedule somewhere new is an operator editing
    #: this file, and text arriving in a conversation cannot name a channel
    #: that is not already in it (§7.1). Only `siatt task add --destination`
    #: writes one — the `schedule_create` tool has no argument for it.
    destinations: dict[str, str] = Field(default_factory=dict)

    @field_validator("destinations")
    @classmethod
    def _real_channels(cls, value: dict[str, str]) -> dict[str, str]:
        """Refuse at load time what would otherwise fail at nine in the morning.

        A destination is a *channel*: `C` for one Slack converted to that id,
        `G` for a legacy private group. A user id or a DM (`D`) is refused
        outright rather than posted into — "every weekday, message this person"
        is somebody else's conversation, and it is not what this feature is.
        """
        cleaned = {}
        for name, channel in value.items():
            key = name.strip()
            if key == HERE:
                raise ValueError(
                    f"{HERE!r} is reserved: it means the channel a task was created in, "
                    "so a configured destination cannot be called that"
                )
            if not _DESTINATION_NAME.fullmatch(key):
                raise ValueError(
                    f"{name!r} is not a destination name; use lowercase letters, digits, "
                    "'-' and '_' (it is what `siatt task add --destination` takes)"
                )
            if not _CHANNEL_ID.fullmatch(channel.strip()):
                raise ValueError(
                    f"destination {key!r} is {channel!r}, which is not a Slack channel id. "
                    "Channels start with C (or G); a D is a DM and a U is a person."
                )
            cleaned[key] = channel.strip()
        return cleaned


class RetrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 20.0
    jitter: float = 0.25

    def to_policy(self) -> RetryPolicy:
        return RetryPolicy(**self.model_dump())


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ltm: LTMSettings = Field(default_factory=LTMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    episodes: EpisodeSettings = Field(default_factory=EpisodeSettings)
    promote: PromoteSettings = Field(default_factory=PromoteSettings)
    reflect: ReflectSettings = Field(default_factory=ReflectSettings)
    reorganize: ReorganizeSettings = Field(default_factory=ReorganizeSettings)
    forget: ForgetSettings = Field(default_factory=ForgetSettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)
    llm: dict[str, ProviderConfig] = Field(default_factory=dict)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    fetch: FetchSettings = Field(default_factory=FetchSettings)
    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    attachments: AttachmentSettings = Field(default_factory=AttachmentSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    tasks: TaskSettings = Field(default_factory=TaskSettings)
    #: USD per million tokens, keyed by model-name prefix. Empty by default:
    #: a stale built-in price table is worse than an absent one.
    pricing: dict[str, PriceSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _destinations_are_reachable(self) -> Config:
        """Refuse a destination Siatt would post into and then ignore.

        `slack.allowed_channels` narrows what ingress reads. A destination
        outside it would still be posted into — egress does not consult the
        allowlist — and every reply under that post would be dropped as
        chatter, which is a schedule that talks to a channel it cannot hear.
        Two settings that only disagree at nine in the morning are worth one
        check at load time.
        """
        allowed = set(self.slack.allowed_channels)
        if not allowed:
            return self
        stranded = sorted(
            f"{name} ({channel})"
            for name, channel in self.tasks.destinations.items()
            if channel not in allowed
        )
        if stranded:
            raise ValueError(
                f"task destination(s) {', '.join(stranded)} are not in slack.allowed_channels, "
                "so a reply under anything posted there would be ignored. Add them to the "
                "allowlist, or point the destination somewhere Siatt reads."
            )
        return self

    def agent_config(self) -> AgentConfig:
        settings = self.agent
        base = AgentConfig()
        return AgentConfig(
            system_prompt=settings.system_prompt or base.system_prompt,
            max_tool_iterations=settings.max_tool_iterations,
            max_turn_seconds=settings.max_turn_seconds,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            history_limit=settings.history_limit,
        )

    def price_book(self) -> PriceBook:
        return PriceBook({name: Price(**p.model_dump()) for name, p in self.pricing.items()})

    def chains(self) -> dict[ModelRole, list[LLMProvider]]:
        chains: dict[ModelRole, list[LLMProvider]] = {}
        for role in ModelRole:
            if entry := self.llm.get(role.value):
                chains[role] = entry.chain()
        if ModelRole.CHAT not in chains:
            raise ConfigError(NO_CHAT_PROVIDER)
        # Utility falls back to the chat model rather than failing: an expensive
        # summary beats no summary, and it keeps first-run setup to one key.
        if ModelRole.UTILITY not in chains:
            chains[ModelRole.UTILITY] = chains[ModelRole.CHAT]
        return chains

    def build_registry(self, *, store: Store | None = None) -> ProviderRegistry:
        meter = CostMeter(
            self.price_book(),
            sink=store.record_call if store is not None else _null_sink,
            daily_usd_ceiling=self.budget.daily_usd_ceiling,
            spent_since=store.spend_since if store is not None else None,
        )
        return ProviderRegistry(self.chains(), meter=meter, retry=self.retry.to_policy())

    def redacted(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


async def _null_sink(record: Any) -> None:
    return None


def default_key_env(kind: ProviderKind) -> str:
    return "ANTHROPIC_API_KEY" if kind == "anthropic" else "OPENAI_API_KEY"


def default_search_key_env(kind: SearchKind) -> str:
    return {"brave": "BRAVE_SEARCH_API_KEY"}[kind]


def anchored(value: str, *, base: Path) -> Path:
    """Resolve a configured path, reading a relative one as relative to `base`.

    `base` is the directory holding config.toml. Resolved against the process's
    working directory instead — which is what `expanduser()` alone leaves —
    a relative path in a config file means a different memory repo and a
    different database depending on where Siatt was started, and neither
    Siatt nor the operator is told: the second directory gets a freshly
    bootstrapped, empty world reported as healthy (#88).

    Relative to the config file is the reading that gives the value a stable
    meaning, and it is what makes a config directory portable.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_config(path: Path | None = None) -> Config:
    target = path or config_path()
    if not target.exists():
        return config_from_env()
    try:
        raw = tomllib.loads(target.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{target} is not valid TOML: {exc}") from exc
    try:
        cfg = Config.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"{target} is not a valid Siatt config: {exc}") from exc
    return anchor_paths(cfg, base=target.expanduser().resolve().parent)


def anchor_paths(cfg: Config, *, base: Path) -> Config:
    """Rewrite relative configured paths to be absolute. See `anchored`.

    Done once here rather than inside each `resolved()`, so that everything
    downstream — including `siatt config`, which prints the config back — sees
    the path that will actually be used.

    A value that is already absolute is left exactly as written, `~` included:
    it is unambiguous as it stands, and rewriting it would mean `siatt init`
    could not round-trip the file it just wrote.
    """
    cfg.ltm.clone_path = _anchor(cfg.ltm.clone_path, base)
    cfg.store.path = _anchor(cfg.store.path, base)
    return cfg


def _anchor(value: str | None, base: Path) -> str | None:
    if not value:
        return value
    path = Path(value).expanduser()
    return value if path.is_absolute() else str(base / path)


def config_from_env() -> Config:
    """Synthesize a config from whatever API key can be resolved.

    Keeps first run to `export ANTHROPIC_API_KEY=... && siatt run`, or to
    `siatt vault set ANTHROPIC_API_KEY` and nothing exported at all. `siatt init`
    replaces this with a real config file.

    Only the *keys* fall back to the vault. `SIATT_CHAT_MODEL` and
    `OPENAI_BASE_URL` are settings rather than secrets, so they stay
    environment-only: the vault is for credentials, and a config knob that
    could hide in it would be a config knob nobody can find.

    Returns a provider-less config rather than raising when no key is present:
    `siatt db migrate`, `siatt cost` and `siatt config` are all useful before any
    model is configured, and only `chains()` actually needs a provider.
    """
    if resolve("ANTHROPIC_API_KEY"):
        chat = ProviderConfig(
            kind="anthropic",
            model=os.environ.get("SIATT_CHAT_MODEL") or DEFAULT_ANTHROPIC_MODEL,
        )
    elif resolve("OPENAI_API_KEY"):
        chat = ProviderConfig(
            kind="openai",
            model=os.environ.get("SIATT_CHAT_MODEL") or DEFAULT_OPENAI_MODEL,
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
    else:
        return Config()
    return Config(llm={"chat": chat})


# -- writing -----------------------------------------------------------------

_HEADER = """# Siatt configuration — written by `siatt init`.
#
# This file holds no secrets. Credentials are referenced by the *name* of the
# environment variable that carries them, so this file is safe to read, diff,
# and back up. See docs/DESIGN.md Appendix A.
"""

_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    text = "".join(_ESCAPES.get(ch, ch) for ch in str(value))
    return f'"{text}"'


def _map(name: str, values: dict[str, str], *, comment: str = "") -> list[str]:
    """Render a table of names to strings as its own section.

    `[tasks.destinations]` is a table, not a field: `_table` renders scalars
    and lists, and a dict handed to `_toml_value` comes out as a quoted Python
    repr that the loader then refuses. Its own section is also what somebody
    editing this file by hand would write.
    """
    if not values:
        return []
    lines = [f"[{name}]"]
    if comment:
        lines.insert(0, comment)
    width = max(len(key) for key in values)
    lines += [f"{key.ljust(width)} = {_toml_value(value)}" for key, value in values.items()]
    return [*lines, ""]


def _table(
    name: str,
    model: BaseModel,
    *,
    comment: str = "",
    full: bool = False,
    exclude: set[str] | None = None,
) -> list[str]:
    """Render one table, omitting fields left at their default unless `full`."""
    fields = model.model_dump(
        mode="json", exclude_defaults=not full, exclude_none=True, exclude=exclude
    )
    if not fields:
        return []
    lines = [f"[{name}]"]
    if comment:
        lines.insert(0, comment)
    width = max(len(key) for key in fields)
    lines += [f"{key.ljust(width)} = {_toml_value(value)}" for key, value in fields.items()]
    return [*lines, ""]


def render_toml(cfg: Config) -> str:
    """Serialize a config back to TOML.

    Hand-rolled rather than via a TOML writer so the generated file can carry
    the comments that make it editable by hand — which is the only reason to
    write a config file instead of a pickle.
    """
    lines = [_HEADER]

    if cfg.ltm.configured:
        lines += _table(
            "ltm",
            cfg.ltm,
            comment="# The private GitHub repo holding long-term memory.",
            full=True,
        )

    for role, provider in cfg.llm.items():
        # `fallbacks` is a TOML array of tables, so it is emitted as its own
        # `[[...]]` sections rather than inline in the parent.
        lines += _table(f"llm.{role}", provider, full=True, exclude={"fallbacks"})
        for fallback in provider.fallbacks:
            lines += _table(f"[llm.{role}.fallbacks]", fallback, full=True, exclude={"fallbacks"})

    if cfg.slack.configured:
        lines += _table(
            "slack",
            cfg.slack,
            comment="# Socket Mode: the connection is outbound, so there is no ingress.",
        )

    if cfg.search.configured:
        lines += _table(
            "search",
            cfg.search,
            comment="# Web search. Results are untrusted text; see docs/DESIGN.md.",
            full=True,
        )

    for name, section in (
        # Only what an install changed. Fetching is on with every bound at its
        # default, so a fresh config says nothing about it — and a config that
        # does mention it is a config where somebody moved a limit.
        ("fetch", cfg.fetch),
        ("browser", cfg.browser),
        ("memory", cfg.memory),
        ("episodes", cfg.episodes),
        ("promote", cfg.promote),
        ("reflect", cfg.reflect),
        ("reorganize", cfg.reorganize),
        ("forget", cfg.forget),
        ("agent", cfg.agent),
        ("context", cfg.context),
        ("store", cfg.store),
        ("retry", cfg.retry),
        ("budget", cfg.budget),
    ):
        lines += _table(name, section)
    lines += _table("tasks", cfg.tasks, exclude={"destinations"})
    lines += _map(
        "tasks.destinations",
        cfg.tasks.destinations,
        comment="# Channels a standing task may be pointed at, by name (§7.1).",
    )
    for model, price in cfg.pricing.items():
        lines += _table(f'pricing."{model}"', price, full=True)

    return "\n".join(lines).rstrip() + "\n"


def write_config(cfg: Config, path: Path) -> None:
    """Write the config file with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_toml(cfg))
    path.chmod(0o600)
