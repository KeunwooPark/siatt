"""Socket Mode: Slack ingress and egress for a daemon with no public address.

Ingress does exactly one thing — normalize, enqueue, return. Slack gives a
listener three seconds before it treats the event as undelivered and re-sends
it, and bolt acknowledges once the listener returns, so anything slow on this
path becomes a retry storm on top of whatever was already slow. The agent runs
behind the queue, where taking a minute costs nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Self

from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from siatt.adapters.slack.events import (
    Accepted,
    Changed,
    Ignored,
    Reacted,
    SlackContext,
    message_id,
    normalize,
    reaction,
)
from siatt.adapters.slack.files import SlackFiles
from siatt.adapters.slack.identity import Directory
from siatt.adapters.slack.limits import split
from siatt.adapters.slack.stream import (
    DEFAULT_INTERVAL,
    LiveMessage,
    SlackRateLimited,
    SlackRefused,
)
from siatt.adapters.slack.upload import SlackUploads, UploadRefused
from siatt.config import SlackSettings
from siatt.core.agent import Agent, AgentResult
from siatt.core.events import InboundEvent
from siatt.core.feedback import Feedback
from siatt.core.revise import Reviser
from siatt.core.runtime import DEFAULT_CONCURRENCY, Reply, Runtime, one_message
from siatt.errors import ConfigError
from siatt.llm.types import Delta
from siatt.store.blobs import Attachments
from siatt.vault import resolve

log = logging.getLogger(__name__)

#: Socket Mode carries no signed HTTP requests, so bolt has nothing to verify
#: and no secret to verify it with. Saying so is clearer than leaving bolt to
#: warn about a secret that does not apply here.
NO_HTTP_VERIFICATION = "unused-in-socket-mode"

#: How many unfinished live messages are remembered, so a redelivered event
#: rewrites the message its last attempt posted instead of adding another
#: "thinking…" to the thread. Only turns that failed outright stay here — a
#: finished one is forgotten — and the inbox dead-letters after five attempts,
#: so this is a cap on a leak rather than a working set.
MAX_UNFINISHED = 256


class SlackAdapter:
    """One Slack workspace, connected over a socket Siatt opens outbound."""

    def __init__(
        self,
        agent: Agent,
        *,
        app: AsyncApp,
        context: SlackContext,
        app_token: str,
        concurrency: int = DEFAULT_CONCURRENCY,
        stream: bool = True,
        interval: float = DEFAULT_INTERVAL,
        reactions: Mapping[str, str] | None = None,
        scrub: Callable[[str], str] | None = None,
        files: SlackFiles | None = None,
        owns_files: bool = False,
        attachments: Attachments | None = None,
    ) -> None:
        self._app = app
        self._context = context
        self._app_token = app_token
        self._interval = interval
        self._reactions = dict(reactions if reactions is not None else SlackSettings().reactions)
        self._unfinished: dict[str, str] = {}
        #: Every write Slack takes from this adapter goes through here, which
        #: is what makes `_translate` unavoidable: a `chat.postMessage` called
        #: directly would raise `SlackApiError`, and a refusal nobody
        #: classified is a refusal the inbox retries five times (#258).
        self._poster = _ClientPoster(self.client)
        self.directory = Directory(agent.store, self._users_info, team_id=context.team_id)
        self.files = files
        self._owns_files = owns_files
        self.reviser = Reviser(agent.store)
        self.feedback = Feedback(agent.store)
        self.uploads = (
            SlackUploads(
                uploader=_ClientUploader(self.client),
                poster=self._poster,
                store=agent.store,
                attachments=attachments,
            )
            if attachments is not None
            else None
        )
        self.runtime = Runtime(
            agent,
            self.open_reply if stream else one_message(self.reply),
            concurrency=concurrency,
            prepare=self._prepare,
            scrub=scrub,
            # An answer here can carry a file, so `send_file` may resolve one
            # (#247). Only with a store behind it: without `[attachments]`
            # there is nothing on disk to send, and the honest answer to the
            # model is the same one a terminal gives.
            sends_files=attachments is not None,
            # And an answer here is a thread of messages rather than one
            # stream of text, so a turn may end one and begin another (#259).
            sends_messages=True,
        )
        self._handler: AsyncSocketModeHandler | None = None
        self._register()

    @classmethod
    async def connect(
        cls,
        agent: Agent,
        settings: SlackSettings,
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
        scrub: Callable[[str], str] | None = None,
        attachments: Attachments | None = None,
    ) -> Self:
        """Build an adapter, having asked Slack who it is.

        The identity is not decoration: the bot's own user id is what tells a
        mention from chatter and Siatt's own messages from everyone else's, and
        the team id is half of every session key.
        """
        client = AsyncWebClient(
            token=_token(settings.bot_token_env, "bot"), base_url=settings.api_url
        )
        app = AsyncApp(
            client=client,
            # Socket Mode sends its acknowledgement after dispatch returns.
            # The listener only normalizes and commits one inbox row, and it
            # must finish before that ack or a crash in between loses an event
            # Slack has already been told was durable.
            process_before_response=True,
            signing_secret=NO_HTTP_VERIFICATION,
            request_verification_enabled=False,
        )
        identity = await app.client.auth_test()
        return cls(
            agent,
            app=app,
            context=SlackContext(
                bot_user_id=str(identity["user_id"]),
                team_id=str(identity["team_id"]),
                allowed_channels=frozenset(settings.allowed_channels),
            ),
            app_token=_token(settings.app_token_env, "app"),
            concurrency=concurrency,
            scrub=scrub,
            stream=settings.stream,
            reactions=settings.reactions,
            # The same bot token the client holds. It is the only credential
            # that can read a private file, which is why `files.py` is so
            # careful about where it is willing to send it.
            files=SlackFiles(
                attachments,
                token=_token(settings.bot_token_env, "bot"),
                hosts=tuple(settings.file_hosts),
            ),
            owns_files=True,
            # The same store the fetch writes into and the agent reads back
            # through. Egress reads it a third way -- under the session's scope,
            # which is the whole of what stops a DM's photograph from being
            # posted into a channel.
            attachments=attachments,
        )

    async def _prepare(self, event: InboundEvent) -> InboundEvent:
        """What happens between the queue and the turn.

        Names first, then files. Both are network calls that must not sit in
        front of Slack's three-second ack, and both are written never to raise:
        a directory that is down or a file that will not come is a reason to
        answer with less, not a reason to fail the turn and have the message
        redelivered until its retry budget runs out.

        Files last so that the note the model reads is appended to text whose
        mentions are already resolved, rather than being walked over by the
        hydration that follows it.
        """
        event = await self.directory.hydrate(event)
        return event if self.files is None else await self.files.collect(event)

    @property
    def context(self) -> SlackContext:
        return self._context

    @property
    def client(self) -> AsyncWebClient:
        return self._app.client

    # -- ingress -------------------------------------------------------------

    async def on_event(self, event: dict[str, Any]) -> None:
        """The whole three-second path: one decision and one INSERT.

        Nothing here awaits a model, which is the entire reason `inbox` exists.
        """
        try:
            decision = await normalize(
                event,
                context=self._context,
                known_session=self._known_session,
                thread_session=self._thread_session,
            )
        except Exception:
            # Belt to the braces in `events.py`. Everything up to the INSERT is
            # a judgement about a payload, and a judgement that raises loses
            # the message: bolt hands the failure back to Slack, Slack retries
            # into the same exception three times, and then the message is gone
            # with nothing recording that it arrived. Dropping it loudly is
            # worse than answering it and better than dropping it in silence.
            #
            # Only around `normalize`. A failure out of `submit` is the store,
            # and there the row genuinely was not written — letting that reach
            # bolt is what gets the message re-sent, which is the outcome we
            # want.
            log.exception(
                "could not read a slack event (type=%r, ts=%r); ignoring it",
                event.get("type"),
                event.get("ts"),
            )
            return
        if isinstance(decision, Ignored):
            log.debug("ignoring a slack event: %s", decision.reason)
            return
        if isinstance(decision, Changed):
            await self.revise(decision)
            return
        if not isinstance(decision, Accepted):
            # `normalize` reads messages and never returns a reaction, but the
            # decision type covers every surface event; saying so here is what
            # keeps a future one from falling through into `submit` and being
            # answered.
            log.debug("a message decision this path does not handle: %s", decision)
            return
        enqueued = await self.runtime.submit(decision.event)
        if enqueued.duplicate:
            # Either Slack re-sent the event, or the same message reached us as
            # both `app_mention` and `message`. Both are normal; both are the
            # unique constraint doing its job.
            log.debug("slack sent %s again; already queued", decision.event.external_id)

    async def on_reaction(self, event: dict[str, Any]) -> None:
        """An emoji on one of Siatt's answers, which is the whole feedback loop.

        On the ack path with the revisions, and for the same reasons: a lookup
        and an INSERT, no model and no network, and it must not reach the agent
        — a 👍 is not a question.
        """
        try:
            decision = reaction(event, context=self._context, verdicts=self._reactions)
        except Exception:
            log.exception("could not read a slack reaction (%r)", event.get("reaction"))
            return
        if not isinstance(decision, Reacted):
            log.debug("ignoring a slack reaction: %s", getattr(decision, "reason", decision))
            return

        act = self.feedback.withdraw if decision.removed else self.feedback.record
        outcome = await act(
            source="slack",
            external_id=decision.external_id,
            verdict=decision.verdict,
            author=decision.author,
        )
        log.debug("reaction on %s: %s", decision.external_id, outcome.summary())

    async def remember_answer(self, event: InboundEvent, result: AgentResult, ts: str) -> None:
        """Note which memories produced the message just posted at `ts`.

        Recorded whatever it used, including nothing: a reaction on an answer
        that recalled no memory is still a fact about the answer, and a row
        that exists with an empty list is how a later 👍 tells "nothing to
        boost" from "not one of ours" (#36).
        """
        if not event.channel or not ts:
            return
        await self.runtime.store.record_answer(
            source="slack",
            external_id=message_id(self._context.team_id, event.channel, ts),
            memory_ids=result.memory_ids,
            session_id=event.session_id,
            scope=event.scope,
        )

    async def opened_thread(self, event: InboundEvent, ts: str) -> None:
        """Remember a thread this answer started, so replies under it are read.

        Only a message posted with no `thread_ts`, which on this surface means
        a standing task firing into a channel: everything a person says arrives
        in a thread already. Without the row, ingress derives a session id from
        the new thread that nothing has ever used, finds no mention in the
        reply, and ignores it — a briefing nobody can answer (#215).
        """
        if event.reply_to is not None or not event.channel or not ts:
            return
        await self.runtime.store.open_slack_thread(
            team_id=self._context.team_id,
            channel=event.channel,
            thread_ts=ts,
            session_id=event.session_id,
        )

    async def revise(self, decision: Changed) -> None:
        """Apply an edit or a deletion, here on the ack path rather than behind
        the queue.

        Deliberately, and it is the one thing besides the INSERT that runs
        here. The work is a handful of indexed statements against SQLite with
        no model and no network in it, so it fits the three seconds many times
        over — and it must not go through the inbox, because everything that
        comes out of the inbox is delivered to the agent as something to answer,
        and "Jane fixed a typo" is not a question.

        Durability comes from Slack instead: raising means bolt does not ack,
        Slack re-sends, and every step here is idempotent — a rewrite to the
        same words, a state already set, a review deduped on its key.
        """
        revision = decision.revision
        if revision.text is not None:
            # Cache-only, so the ack path stays free of `users.info`. The
            # people in an edited message were almost always resolved when the
            # original arrived.
            revision = replace(revision, text=await self.directory.rename_known(revision.text))
        outcome = await self.reviser.apply(revision)
        log.debug("revision %s: %s", revision.external_id, outcome.summary())

    async def _known_session(self, session_id: str) -> bool:
        return await self.runtime.store.get_session(session_id) is not None

    async def _thread_session(self, channel: str, thread_ts: str) -> str | None:
        return await self.runtime.store.slack_thread_session(
            team_id=self._context.team_id, channel=channel, thread_ts=thread_ts
        )

    async def _users_info(self, user_id: str) -> Mapping[str, Any]:
        """One profile, unwrapped from the envelope Slack puts it in.

        `Directory` is given this rather than the client so that it needs no
        `slack_sdk` — and so the thing under test is the caching, not a mock of
        somebody else's response object.
        """
        response = await self.client.users_info(user=user_id)
        user = response.get("user")
        return user if isinstance(user, Mapping) else {}

    # -- egress --------------------------------------------------------------

    async def open_reply(self, event: InboundEvent) -> Reply:
        """Put a message up now, and rewrite it as the turn produces an answer.

        In the thread the question was asked in, always. A turn that takes
        thirty seconds and says nothing for twenty-nine of them is
        indistinguishable from one that broke, and somebody who thinks Siatt
        broke asks again — which is a second turn, a second model call, and two
        answers to one question.

        Except when there is no thread yet, which is a standing task posting
        into a channel (#215). Nobody is waiting on that one — its whole
        premise is that nobody asked just now — and a `thinking…` sitting at
        top level in a channel is not reassurance, it is a message people reply
        to. That turn says nothing until it has something to say.
        """
        if not event.channel or event.reply_to is None:
            return _Posted(self, event)
        message = LiveMessage(
            self._poster,
            channel=event.channel,
            thread_ts=event.reply_to,
            interval=self._interval,
            ts=self._unfinished.get(event.external_id),
        )
        await message.open()
        if message.ts is not None:
            self._remember(event.external_id, message.ts)
        return _Live(self, event, message)

    async def reply(self, event: InboundEvent, result: AgentResult) -> None:
        """Post the answer, having shown nothing before it.

        What a build with `stream: false` uses, and what a standing task gets:
        nobody is watching either, so the first anyone sees is the answer.

        In as many messages as the answer needs — the ones the turn asked for
        (#259), each cut again if it is still too long for one (#258). They go
        *under* the first: `event.reply_to` when there is a thread already, and
        otherwise the message this turn has just made into one, which is the
        same thread `opened_thread` is about to record. A standing task that
        posts three sections must not post three top-level messages.
        """
        parts = [part for message in messages(result) for part in split(message)]
        if not parts or not event.channel:
            log.warning("nothing to post for %s", event.external_id)
            return
        ts = await self._poster.post(channel=event.channel, thread_ts=event.reply_to, text=parts[0])
        for part in parts[1:]:
            await self._poster.post(
                channel=event.channel, thread_ts=event.reply_to or ts, text=part
            )
        await self.opened_thread(event, ts)
        await self.remember_answer(event, result, ts)
        await self.send_files(event, result)

    async def send_files(self, event: InboundEvent, result: AgentResult) -> None:
        """Put this turn's files in the thread, after the answer is in it.

        Both egress paths end here, and both call it last. The answer is what
        somebody is waiting for; a file is what the answer is about, and an
        upload that fails must not be able to take the reply down with it —
        which is why `SlackUploads.send` never raises.

        `event.reply_to` and not the answer's own `ts`: the file goes where the
        answer went. In a thread that is the thread; in a DM with no thread it
        is the next message, which is where somebody would look for it.
        """
        if self.uploads is None:
            return
        await self.uploads.send(
            result,
            external_id=event.external_id,
            channel=event.channel,
            thread_ts=event.reply_to,
        )

    def _remember(self, external_id: str, ts: str) -> None:
        self._unfinished[external_id] = ts
        while len(self._unfinished) > MAX_UNFINISHED:
            # Oldest first, which for a dict is insertion order. Losing one
            # costs a duplicate placeholder on a redelivery that was already
            # four attempts deep.
            self._unfinished.pop(next(iter(self._unfinished)))

    def _forget(self, external_id: str) -> None:
        self._unfinished.pop(external_id, None)

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Open the socket and return. Reconnects are the client's own affair."""
        self._handler = AsyncSocketModeHandler(self._app, self._app_token)
        # Unannotated in bolt, so `strict` will not take its word for it.
        await self._handler.connect_async()  # type: ignore[no-untyped-call]

    async def aclose(self) -> None:
        if self._handler is not None:
            await self._handler.close_async()  # type: ignore[no-untyped-call]
            self._handler = None
        if self._owns_files and self.files is not None:
            # Only a pool this adapter built. One passed in belongs to whoever
            # passed it, and closing somebody else's client is how a test
            # discovers its transport has gone.
            await self.files.aclose()

    # -- internals -----------------------------------------------------------

    def _register(self) -> None:
        async def listener(event: dict[str, Any]) -> None:
            await self.on_event(event)

        # Both, on purpose. `app_mention` is all a minimally-scoped install
        # gets; `message` is what carries DMs and thread replies. Where an
        # install has both, the same message arrives twice and the dedupe key
        # in `events.message_id` is what makes that harmless — which is only
        # true because `normalize` reads the two payloads identically. The
        # duplicate is discarded unexamined, so anything the two disagree
        # about is decided by whichever arrived first.
        for name in ("app_mention", "message"):
            self._app.event(name)(listener)

        async def reacted(event: dict[str, Any]) -> None:
            await self.on_reaction(event)

        # Their own listener, because a reaction is not a message: it has no
        # text and nothing to answer, and everything `normalize` decides about
        # a message is a question it cannot be asked.
        for name in ("reaction_added", "reaction_removed"):
            self._app.event(name)(reacted)


def answer(result: AgentResult) -> str:
    """The finished message: what the model said, and why if it stopped early.

    One function because both egress paths render it, and a live reply that
    formatted the answer differently from the plain one would make "did the
    placeholder go up?" visible to the person reading the thread.
    """
    text = result.text.strip()
    if note := result.note:
        return f"{text}\n\n_{note}_" if text else f"_{note}_"
    return text


def messages(result: AgentResult) -> list[str]:
    """Every message this turn asked for, in the order it asked (#259).

    The parts a turn queued, then the answer it ended with. The note goes on
    the last of them and nowhere else: it explains how the turn stopped, and a
    turn stops once however many messages it took to get there.

    An empty one is dropped rather than sent. A turn that queued its sections
    and had nothing left to add ends with no text, and Slack will not take an
    empty message anyway.
    """
    return [message for message in (*result.parts, answer(result)) if message.strip()]


class _Live:
    """A turn whose answer is a message already in the thread."""

    def __init__(self, adapter: SlackAdapter, event: InboundEvent, message: LiveMessage) -> None:
        self._adapter = adapter
        self._event = event
        self._message = message

    async def delta(self, delta: Delta) -> None:
        await self._message.delta(delta)

    async def finish(self, result: AgentResult) -> None:
        await self._message.finish(*messages(result))
        # Only now: until the answer is in the thread, a redelivery of this
        # event should rewrite this message rather than post another.
        self._adapter._forget(self._event.external_id)
        if self._message.ts is not None:
            await self._adapter.remember_answer(self._event, result, self._message.ts)
        # Last, and after the repaint has stopped: a file uploaded while the
        # message was still being rewritten would land above an answer that had
        # not finished arriving.
        await self._adapter.send_files(self._event, result)

    async def aclose(self) -> None:
        await self._message.aclose()


class _Posted:
    """A turn with nowhere to put a placeholder. Answers the old way."""

    def __init__(self, adapter: SlackAdapter, event: InboundEvent) -> None:
        self._adapter = adapter
        self._event = event

    async def delta(self, delta: Delta) -> None:
        pass

    async def finish(self, result: AgentResult) -> None:
        await self._adapter.reply(self._event, result)

    async def aclose(self) -> None:
        pass


class _ClientPoster:
    """`Poster`, over the Slack web client.

    The translation lives here rather than in `stream.py` for the reason the
    directory's lookup does: what streaming needs is two writes and a way to
    hear "not so fast", and keeping `slack_sdk` on this side is what lets the
    throttling be tested without it.
    """

    def __init__(self, client: AsyncWebClient) -> None:
        self._client = client

    async def post(self, *, channel: str, thread_ts: str | None, text: str) -> str:
        try:
            response = await self._client.chat_postMessage(
                channel=channel, thread_ts=thread_ts, text=text
            )
        except SlackApiError as exc:
            raise _translate(exc) from exc
        return str(response["ts"])

    async def update(self, *, channel: str, ts: str, text: str) -> None:
        try:
            await self._client.chat_update(channel=channel, ts=ts, text=text)
        except SlackApiError as exc:
            raise _translate(exc) from exc


class _ClientUploader:
    """`Uploader`, over the Slack web client.

    `files.upload` is retired; the flow is `files.getUploadURLExternal`, a POST
    of the bytes to the address it answers with, and `files.completeUploadExternal`.
    `files_upload_v2` is all three, and taking it from `slack_sdk` rather than
    writing it out is also what keeps the bot token off the upload hop: that
    address is pre-signed and carries its own authorization, and a hand-rolled
    version would be one edit away from attaching the header anyway.
    """

    def __init__(self, client: AsyncWebClient) -> None:
        self._client = client

    async def upload(
        self, *, channel: str, thread_ts: str | None, filename: str, title: str, data: bytes
    ) -> None:
        try:
            await self._client.files_upload_v2(
                channel=channel,
                thread_ts=thread_ts,
                filename=filename,
                title=title,
                file=data,
                # The file info is a round trip nothing here reads.
                request_file_info=False,
            )
        except SlackApiError as exc:
            raise UploadRefused(_refusal(exc)) from exc


#: Slack's way of saying the token cannot upload. All three mean the app is
#: missing `files:write`, and none of them says so — which is the difference
#: between a thread that reads "the app needs the files:write scope" and one
#: that reads "not_allowed_token_type".
_NO_SCOPE = ("not_allowed_token_type", "missing_scope", "no_permission", "access_denied")


def _code(exc: SlackApiError) -> str:
    """Slack's own name for what was wrong, or empty if it did not say."""
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    return str(response.get("error") or "")


def _refusal(exc: SlackApiError) -> str:
    """Why the file did not go, for somebody reading the thread."""
    code = _code(exc)
    if code in _NO_SCOPE:
        return "Siatt's Slack app lacks the files:write scope"
    if code == "ratelimited":
        return "Slack is rate-limiting uploads just now"
    if code:
        return f"Slack refused it ({code})"
    return "Slack refused it"


#: Refusals that another identical attempt will never satisfy. Three kinds,
#: and the reason they are one list is that the caller's move is the same for
#: all three — stop, rather than spend another model call finding out again:
#:
#: the message is not sendable as it stands (`msg_too_long`); the message being
#: rewritten is gone or beyond editing, which `_write_final` recovers from by
#: posting instead; and the app is not allowed to write here at all, which is
#: an operator's problem and wants a dead letter naming it, not five turns.
#:
#: Anything not listed stays an ordinary exception and keeps the old behaviour,
#: a failed turn that is delivered again. An unrecognised code is more likely
#: to be something transient we have not seen yet than something permanent, and
#: the cost of guessing wrong that way is a retry rather than a lost answer.
_PERMANENT = frozenset(
    {
        # the message as it stands
        "as_user_not_supported",
        "invalid_blocks",
        "invalid_blocks_format",
        "msg_too_long",
        "no_text",
        # the message being written, rather than the workspace
        "cant_update_message",
        "edit_window_closed",
        "message_not_found",
        # the door: an operator's problem, and a dead letter names it
        "account_inactive",
        "channel_not_found",
        "ekm_access_denied",
        "invalid_auth",
        "is_archived",
        "missing_scope",
        "not_allowed_token_type",
        "not_in_channel",
        "restricted_action",
        "restricted_action_non_threadable_channel",
        "restricted_action_read_only_channel",
        "restricted_action_thread_locked",
        "team_access_not_granted",
        "thread_not_found",
        "token_expired",
        "token_revoked",
    }
)


def _translate(exc: SlackApiError) -> Exception:
    """Slack's "too fast", told from Slack's "no", told from the rest.

    A 429 becomes `SlackRateLimited`, because it is the one answer where
    waiting is the response. A code on `_PERMANENT` becomes `SlackRefused`,
    because it is an answer no repetition changes and the queue's default —
    deliver it again, model call and all — is exactly wrong for it (#258).

    Everything else is passed along as it was: a live frame swallows it, and
    the final write fails the turn and earns a retry.
    """
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        headers = getattr(response, "headers", {}) or {}
        try:
            retry_after = float(headers.get("Retry-After", 1))
        except (TypeError, ValueError):
            retry_after = 1.0
        return SlackRateLimited(retry_after)
    if (code := _code(exc)) in _PERMANENT:
        return SlackRefused(code)
    return exc


def _token(env: str | None, kind: str) -> str:
    if not env:
        raise ConfigError(f"no {kind} token configured for Slack; run `siatt init`")
    value = resolve(env)
    if not value:
        raise ConfigError(
            f"{env} is not set and is not in the vault (the Slack {kind} token).\n"
            f"Export it, or run `siatt vault set {env}`."
        )
    return value
