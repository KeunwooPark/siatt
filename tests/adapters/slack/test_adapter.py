"""The Slack adapter: ack fast, dedupe hard, answer in the thread."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("slack_bolt", reason="the `slack` extra")

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from siatt.adapters import slack as package
from siatt.adapters.slack.app import NO_HTTP_VERIFICATION, SlackAdapter, messages
from siatt.adapters.slack.events import Accepted, SlackContext, normalize
from siatt.adapters.slack.limits import MAX_TEXT
from siatt.config import AttachmentSettings
from siatt.core.agent import Agent, AgentResult
from siatt.core.context import ContextPacker
from siatt.core.events import InboundEvent
from siatt.core.file_tools import file_tools
from siatt.core.message_tools import message_tools
from siatt.core.revise import TOMBSTONE
from siatt.core.tools import Tool, ToolRegistry
from siatt.llm.registry import ModelRole, ProviderRegistry
from siatt.llm.tokens import Tokenizer
from siatt.llm.types import ChatRequest, ChatResponse, Delta, Message, ToolUseBlock, Usage
from siatt.store import Store
from siatt.store.blobs import Attachments
from tests.conftest import until
from tests.core.test_agent import ScriptedProvider, says
from tests.core.test_agent_images import png
from tests.core.test_file_tools import _sends as sends

BOT = "U0SIATT"
TEAM = "T0TEAM"
HUMAN = "U0HUMAN"


class RecordingClient(AsyncWebClient):
    """A Slack client that keeps what it was asked to post instead of posting.

    It has to model a *thread* rather than a list of calls, because a reply is
    now one message written several times (#22): the assertion worth making is
    what the thread says when the turn is over, and how many messages it took
    to say it.
    """

    def __init__(self) -> None:
        super().__init__(token="xoxb-test")
        self.posted: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.uploaded: list[dict[str, Any]] = []
        #: What happened, in order. A file must arrive after the answer it
        #: belongs to, and "both happened" is not the assertion worth making.
        self.order: list[str] = []
        self.profiles: dict[str, dict[str, Any]] = {
            HUMAN: {"name": "jane", "profile": {"display_name": "jane"}}
        }
        self._ts = 0

    async def chat_postMessage(self, **kwargs: Any) -> Any:
        self._ts += 1
        ts = f"1700009999.{self._ts:06d}"
        self.posted.append(kwargs | {"ts": ts})
        self.order.append("post")
        return {"ok": True, "ts": ts}

    async def chat_update(self, **kwargs: Any) -> Any:
        self.updates.append(kwargs)
        self.order.append("update")
        return {"ok": True, "ts": kwargs["ts"]}

    async def files_upload_v2(self, **kwargs: Any) -> Any:
        self.uploaded.append(kwargs)
        self.order.append("upload")
        return {"ok": True, "files": []}

    async def users_info(self, *, user: str, **kwargs: Any) -> Any:
        # Answered rather than left to the real client: every delivered event
        # resolves its author now (#23), and a test suite that reached
        # slack.com to find that out would be a test suite that needs a network.
        return {"ok": True, "user": self.profiles.get(user, {})}

    @property
    def messages(self) -> list[str]:
        """What each message in the thread says now, in the order posted."""
        latest = {post["ts"]: post["text"] for post in self.posted}
        for update in self.updates:
            latest[update["ts"]] = update["text"]
        return [latest[post["ts"]] for post in self.posted]


async def answered(client: RecordingClient, count: int = 1) -> None:
    """Wait for `count` turns to have delivered an answer.

    The final `chat.update` is the one that carries it. Waiting on the *post*
    would be waiting on the placeholder, which goes up before the model is
    called and therefore before anything a test wants to assert on exists.
    """
    await until(lambda: len(client.updates) >= count)


class SlowProvider(ScriptedProvider):
    """A model that takes its time, which is the case ingress has to survive."""

    def __init__(self, script: list[Any], *, delay: float) -> None:
        super().__init__(script)
        self._delay = delay

    async def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:
        await asyncio.sleep(self._delay)
        async for delta in super().stream(req):
            yield delta


def make_agent(
    store: Store,
    tokenizer: Tokenizer,
    provider: ScriptedProvider,
    tools: list[Tool] | None = None,
) -> Agent:
    return Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=ToolRegistry(tools or []),
        packer=ContextPacker(tokenizer=tokenizer),
    )


def make_adapter(
    store: Store,
    tokenizer: Tokenizer,
    *,
    provider: ScriptedProvider | None = None,
    concurrency: int = 8,
    stream: bool = True,
    attachments: Attachments | None = None,
    tools: list[Tool] | None = None,
) -> tuple[SlackAdapter, RecordingClient]:
    client = RecordingClient()
    app = AsyncApp(
        client=client,
        signing_secret=NO_HTTP_VERIFICATION,
        request_verification_enabled=False,
    )
    adapter = SlackAdapter(
        make_agent(store, tokenizer, provider or ScriptedProvider([says("noted")] * 200), tools),
        app=app,
        context=SlackContext(bot_user_id=BOT, team_id=TEAM),
        app_token="xapp-test",
        concurrency=concurrency,
        stream=stream,
        attachments=attachments,
    )
    return adapter, client


def mention(
    ts: str = "1700000000.000100", text: str = f"<@{BOT}> what did we decide?"
) -> dict[str, Any]:
    return {
        "type": "message",
        "channel_type": "channel",
        "channel": "C0DEPLOY",
        "user": HUMAN,
        "text": text,
        "ts": ts,
    }


# -- ingress ------------------------------------------------------------------


async def test_the_ack_path_never_waits_for_a_turn(store: Store, tokenizer: Tokenizer) -> None:
    """The acceptance criterion. Slack gives a listener three seconds and bolt
    acks once it returns, so the measurement has to be taken with every turn
    slot occupied — which is the state a busy channel is normally in."""
    provider = SlowProvider([says("noted")] * 200, delay=1.0)
    adapter, _ = make_adapter(store, tokenizer, provider=provider, concurrency=8)
    running = asyncio.create_task(adapter.runtime.run())

    slowest = 0.0
    try:
        for n in range(8):
            await adapter.on_event(mention(ts=f"1700000000.{n:06d}"))
        await until(lambda: adapter.runtime.dispatcher.in_flight == 8)

        for n in range(50):
            started = time.perf_counter()
            await adapter.on_event(mention(ts=f"1700000001.{n:06d}"))
            slowest = max(slowest, time.perf_counter() - started)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=30.0)

    assert len(provider.requests) == 8, "the slots really were full"
    assert slowest < 0.5, f"slowest ack was {slowest:.3f}s against a 3s budget"


async def test_an_ignored_event_is_not_queued(store: Store, tokenizer: Tokenizer) -> None:
    adapter, _ = make_adapter(store, tokenizer)

    await adapter.on_event(mention(text="nothing to do with the bot"))

    assert await adapter.runtime.inbox.counts() == {}


async def test_a_forced_retry_produces_exactly_one_reply(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The other acceptance criterion. Slack re-sends aggressively, and the
    same message also arrives as both `message` and `app_mention`."""
    adapter, client = make_adapter(store, tokenizer)
    body = mention()
    as_mention = {k: v for k, v in body.items() if k != "channel_type"} | {"type": "app_mention"}

    running = asyncio.create_task(adapter.runtime.run())
    try:
        for delivery in (body, body, as_mention, body):
            await adapter.on_event(delivery)
        await answered(client)
        await asyncio.sleep(0.2)  # a second answer would land in this window
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert len(client.posted) == 1, client.posted
    assert await adapter.runtime.inbox.counts() == {"done": 1}


async def test_a_file_entry_that_is_not_an_object_still_reaches_the_inbox(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The repro from #123, measured where it hurts: the row, not the decision.

    `_with_attachments` read the payload's shape and trusted it, so an entry
    that was not an object raised out of `normalize` — before the INSERT, and
    therefore before anything recorded that the message had arrived at all.
    """
    adapter, client = make_adapter(store, tokenizer)
    body = mention(text=f"<@{BOT}> what's in this?") | {
        "subtype": "file_share",
        "files": ["F0123456"],
    }

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(body)
        await answered(client)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert await adapter.runtime.inbox.counts() == {"done": 1}


async def test_an_event_that_cannot_be_read_is_ignored_rather_than_raised(
    store: Store, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Ingress must reach a decision for every payload Slack sends.

    An exception out of `normalize` lands before the inbox row is written,
    which is the one thing this path exists to guarantee: Slack retries three
    times, each raises, and the message is gone with no record that it ever
    arrived. `normalize` is patched rather than fed a payload that breaks it
    today, because the guard is for the shape nobody has thought of yet.
    """

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("a shape nobody anticipated")

    monkeypatch.setattr("siatt.adapters.slack.app.normalize", boom)
    adapter, _ = make_adapter(store, tokenizer)

    with caplog.at_level(logging.ERROR, logger="siatt.adapters.slack.app"):
        await adapter.on_event(mention())

    assert "could not read a slack event" in caplog.text
    assert "a shape nobody anticipated" in caplog.text, "the traceback is the whole point"


async def test_ingress_still_works_after_an_event_it_could_not_read(
    store: Store, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surviving one bad event is only worth having if the socket survives too."""
    adapter, client = make_adapter(store, tokenizer)
    real = normalize
    calls = 0

    async def once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("a shape nobody anticipated")
        return await real(*args, **kwargs)

    monkeypatch.setattr("siatt.adapters.slack.app.normalize", once)
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention(ts="1700000000.000100"))
        await adapter.on_event(mention(ts="1700000000.000200"))
        await answered(client)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert await adapter.runtime.inbox.counts() == {"done": 1}


# -- egress -------------------------------------------------------------------


async def test_the_answer_goes_back_into_the_thread(store: Store, tokenizer: Tokenizer) -> None:
    adapter, client = make_adapter(store, tokenizer)
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await answered(client)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.posted[0]["channel"] == "C0DEPLOY"
    assert client.posted[0]["thread_ts"] == "1700000000.000100"
    assert client.messages == ["noted"], "one message in the thread, and it is the answer"


def a_firing(session_id: str = "slack:task:01J@2026-09-08T09:00") -> InboundEvent:
    """A standing task with a destination, as `task_handler` queues one: a
    channel, no thread, and nobody waiting."""
    return InboundEvent(
        source="slack",
        external_id=f"task:{session_id}",
        session_id=session_id,
        text="what happened in AI overnight",
        scope="channel:C0DEPLOY",
        author=HUMAN,
        channel="C0DEPLOY",
        reply_to=None,
        origin="scheduled",
    )


async def test_a_firing_with_no_thread_posts_once_and_says_nothing_first(
    store: Store, tokenizer: Tokenizer
) -> None:
    """A top-level `thinking…` is not reassurance — nobody asked just now — it
    is a message people reply to, in a channel, that is about to be rewritten
    into something else."""
    adapter, client = make_adapter(store, tokenizer)
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.runtime.submit(a_firing())
        await until(lambda: len(client.posted) >= 1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert len(client.posted) == 1
    assert client.posted[0].get("thread_ts") is None, "no thread is what starts one"
    assert client.messages == ["noted"]
    assert not client.updates, "nothing was posted early enough to need rewriting"


async def test_a_reply_under_a_firing_is_answered_without_a_mention(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The whole point of posting a briefing: somebody can answer it. The
    thread is one nobody mentioned Siatt in, under a timestamp no session has
    ever used, so this only works because the post was recorded as ours."""
    adapter, client = make_adapter(store, tokenizer)
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.runtime.submit(a_firing())
        await until(lambda: len(client.posted) >= 1)
        root = client.posted[0]["ts"]

        await adapter.on_event(
            {
                "type": "message",
                "channel_type": "channel",
                "channel": "C0DEPLOY",
                "user": HUMAN,
                "text": "which of those is worth reading?",
                "ts": "1700000000.000200",
                "thread_ts": root,
            }
        )
        await until(lambda: len(client.posted) >= 2)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.posted[1]["thread_ts"] == root, "answered in the thread it started"
    # And in the conversation that posted the briefing, so "those" refers to
    # something.
    assert (
        await store.slack_thread_session(team_id=TEAM, channel="C0DEPLOY", thread_ts=root)
        == "slack:task:01J@2026-09-08T09:00"
    )
    history = await store.raw(
        "SELECT session_id FROM messages WHERE session_id = ? ORDER BY seq",
        ("slack:task:01J@2026-09-08T09:00",),
    )
    assert len(history) == 4, "two turns, question and answer each, in one conversation"


async def test_a_turn_with_nothing_to_say_says_why(store: Store, tokenizer: Tokenizer) -> None:
    """A silent turn on Slack is no more debuggable than a silent one on a tty."""
    adapter, client = make_adapter(store, tokenizer, provider=ScriptedProvider([says("")] * 4))
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await answered(client)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.messages == ["_the model returned nothing._"]


async def test_a_channel_and_a_dm_share_one_pool(store: Store, tokenizer: Tokenizer) -> None:
    """One person talking in two places, and one memory of both (#265). The
    session rows record where each conversation happened; neither is fenced off
    from what the other learned."""
    adapter, client = make_adapter(store, tokenizer)
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await answered(client)
        await adapter.on_event(
            {
                "type": "message",
                "channel_type": "im",
                "channel": "D0PRIVATE",
                "user": HUMAN,
                "text": "remember that I hate standups",
                "ts": "1700000000.000200",
            }
        )
        await answered(client, 2)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    keys = (
        f"slack:{TEAM}:C0DEPLOY:1700000000.000100",
        f"slack:{TEAM}:D0PRIVATE:1700000000.000200",
    )
    for key in keys:
        session = await store.get_session(key)
        assert session is not None
        assert session["scope"] == "workspace"


# -- streaming ----------------------------------------------------------------


async def test_the_thread_shows_a_reply_before_the_turn_is_over(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The point of #22. A turn that says nothing for thirty seconds is
    indistinguishable from one that broke, and somebody who thinks Siatt broke
    asks again — a second turn, a second model call, two answers."""
    provider = SlowProvider([says("noted")] * 4, delay=0.3)
    adapter, client = make_adapter(store, tokenizer, provider=provider)

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await until(lambda: len(client.posted) == 1)
        assert client.messages == ["_thinking…_"], "up before the model answered"
        await answered(client)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.messages == ["noted"], "and the same message carries the answer"


async def test_streaming_off_posts_the_answer_and_nothing_else(
    store: Store, tokenizer: Tokenizer
) -> None:
    """One API call a turn, for a workspace that would rather not watch a
    message rewrite itself."""
    adapter, client = make_adapter(store, tokenizer, stream=False)

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await until(lambda: len(client.posted) == 1)
        await asyncio.sleep(0.1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.messages == ["noted"]
    assert client.updates == []


async def test_an_answer_too_long_for_one_message_arrives_in_several(
    store: Store, tokenizer: Tokenizer
) -> None:
    """#258. It used to arrive in none of them: Slack refused the write, the
    turn failed, and the retry — reading the answer in its own history — told
    the person it had already been posted."""
    long_answer = "\n\n".join(f"item {n}: " + "detail " * 30 for n in range(20))
    assert len(long_answer) > MAX_TEXT, "the case under test"
    adapter, client = make_adapter(
        store, tokenizer, provider=ScriptedProvider([says(long_answer)] * 4)
    )

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await answered(client)
        await until(lambda: len(client.posted) > 1)
        await asyncio.sleep(0.1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    thread = client.messages
    assert len(thread) > 1, "more than one message carried it"
    assert all(len(message) <= MAX_TEXT for message in thread)
    assert "item 0" in thread[0] and "item 19" in thread[-1], "in order, all of it"
    assert all(post.get("thread_ts") == "1700000000.000100" for post in client.posted), (
        "every part in the thread that asked"
    )


async def test_a_standing_task_s_parts_go_under_its_first_message(
    store: Store, tokenizer: Tokenizer
) -> None:
    """A task posts into a channel rather than a thread (#215), so there is no
    `thread_ts` to inherit. Three sections must not become three top-level
    messages in the channel."""
    adapter, client = make_adapter(store, tokenizer)
    event = InboundEvent(
        source="slack",
        external_id="task:01TEST@2026-09-08T00:00+00:00",
        session_id="slack:task:01TEST@2026-09-08T00:00+00:00",
        text="the morning digest",
        scope="channel:C0DEPLOY",
        author=HUMAN,
        channel="C0DEPLOY",
        reply_to=None,
        origin="scheduled",
    )

    sections = [f"section {name}\n" + "detail " * 300 for name in ("one", "two", "three")]
    await adapter.reply(event, AgentResult(text="\n\n".join(sections)))

    first, *rest = client.posted
    assert rest, "it did not fit in one"
    assert first["thread_ts"] is None, "the first opens the thread"
    assert [post["thread_ts"] for post in rest] == [first["ts"]] * len(rest)


async def test_a_retried_turn_rewrites_its_own_placeholder(
    store: Store, tokenizer: Tokenizer
) -> None:
    """Delivery is at-least-once, so a turn that fails is run again. A fresh
    placeholder per attempt would leave the thread full of "thinking…"."""
    provider = ScriptedProvider([says("noted")] * 4)
    adapter, client = make_adapter(store, tokenizer, provider=provider)
    attempts = 0

    # Failing the *final* write is the case that matters: the placeholder is
    # already up, and the event goes back on the queue with it still there.
    async def finish_or_fail(**kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("slack went away mid-answer")
        return await RecordingClient.chat_update(client, **kwargs)

    client.chat_update = finish_or_fail  # type: ignore[method-assign]
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await until(lambda: attempts >= 2)
        await until(lambda: client.messages == ["noted"])
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=30.0)

    assert len(client.posted) == 1, client.posted


# -- files on the way out -----------------------------------------------------


async def opening(event: dict[str, Any]) -> Any:
    """What ingress makes of this message, so a test can attach a file to the
    same session and scope the turn will run under."""

    async def never(session_id: str) -> bool:
        return False

    return await normalize(
        event, context=SlackContext(bot_user_id=BOT, team_id=TEAM), known_session=never
    )


def keeping_files(store: Store, tmp_path: Path) -> Attachments:
    return AttachmentSettings(enabled=True).build(store, tmp_path / "siatt.db")


async def attached(attachments: Attachments, event: InboundEvent) -> str:
    """A photograph, arrived in the conversation the mention opens."""
    return await attachments.put(
        png(),
        mime="image/png",
        source_name="slack",
        scope=event.scope,
        session_id=event.session_id,
        name="shot.png",
    )


def asking_for_it(store: Store, tokenizer: Tokenizer, attachments: Attachments, sha: str) -> Agent:
    return Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [ScriptedProvider([sends(sha), says("here")])]}),
        store=store,
        tools=ToolRegistry(file_tools(store=store, attachments=attachments)),
        packer=ContextPacker(tokenizer=tokenizer),
        attachments=attachments,
    )


async def test_a_turn_that_asks_to_send_a_file_uploads_it_after_the_answer(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """End to end: the model names a file it was sent, and the file lands in the
    thread — after the answer, because `LiveMessage` is still repainting until
    the answer is final."""
    attachments = keeping_files(store, tmp_path)
    incoming = await opening(mention())
    assert isinstance(incoming, Accepted)
    sha = await attached(attachments, incoming.event)
    client = RecordingClient()
    app = AsyncApp(
        client=client, signing_secret=NO_HTTP_VERIFICATION, request_verification_enabled=False
    )
    adapter = SlackAdapter(
        asking_for_it(store, tokenizer, attachments, sha),
        app=app,
        context=SlackContext(bot_user_id=BOT, team_id=TEAM),
        app_token="xapp-test",
        attachments=attachments,
    )

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await until(lambda: bool(client.uploaded))
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.order[-1] == "upload", "the answer first, then the file"
    assert client.uploaded[0]["filename"] == "shot.png"
    assert client.uploaded[0]["file"] == png()
    # Where the answer went: the thread the mention was in.
    assert client.uploaded[0]["thread_ts"] == incoming.event.reply_to
    assert client.uploaded[0]["channel"] == "C0DEPLOY"


async def test_the_plain_path_uploads_too(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """`stream: false` is one post and no repaint, and the file still follows it."""
    attachments = keeping_files(store, tmp_path)
    incoming = await opening(mention())
    assert isinstance(incoming, Accepted)
    sha = await attached(attachments, incoming.event)
    client = RecordingClient()
    app = AsyncApp(
        client=client, signing_secret=NO_HTTP_VERIFICATION, request_verification_enabled=False
    )
    adapter = SlackAdapter(
        asking_for_it(store, tokenizer, attachments, sha),
        app=app,
        context=SlackContext(bot_user_id=BOT, team_id=TEAM),
        app_token="xapp-test",
        stream=False,
        attachments=attachments,
    )

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await until(lambda: bool(client.uploaded))
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.order == ["post", "upload"]
    assert client.updates == []


async def test_an_install_that_keeps_no_files_says_the_surface_cannot_send(
    store: Store, tokenizer: Tokenizer
) -> None:
    """Nothing on disk to send, so the tool refuses rather than the upload
    failing — and there is no uploader at all."""
    adapter, _ = make_adapter(store, tokenizer)

    assert adapter.uploads is None
    assert adapter.runtime._sends_files is False


async def test_keeping_files_makes_the_surface_one_that_can_send(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    adapter, _ = make_adapter(store, tokenizer, attachments=keeping_files(store, tmp_path))

    assert adapter.uploads is not None
    assert adapter.runtime._sends_files is True


# -- revisions ----------------------------------------------------------------


async def answer_once(adapter: SlackAdapter, client: RecordingClient) -> None:
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await answered(client)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)


async def test_editing_a_message_rewrites_what_siatt_stored(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The whole chain, which no unit test covers: the turn records the Slack
    key on the row it writes, and the edit finds the row by that key."""
    adapter, client = make_adapter(store, tokenizer)
    await answer_once(adapter, client)

    await adapter.on_event(
        {
            "type": "message",
            "subtype": "message_changed",
            "channel": "C0DEPLOY",
            "ts": "1700000099.000000",
            "message": {"ts": "1700000000.000100", "user": HUMAN, "text": "what did we ship?"},
        }
    )

    stored = await store.recent_messages(f"slack:{TEAM}:C0DEPLOY:1700000000.000100")
    assert stored[0].text == "what did we ship?"


async def test_deleting_a_message_tombstones_it(store: Store, tokenizer: Tokenizer) -> None:
    adapter, client = make_adapter(store, tokenizer)
    await answer_once(adapter, client)

    await adapter.on_event(
        {
            "type": "message",
            "subtype": "message_deleted",
            "channel": "C0DEPLOY",
            "ts": "1700000099.000000",
            "deleted_ts": "1700000000.000100",
        }
    )

    stored = await store.recent_messages(f"slack:{TEAM}:C0DEPLOY:1700000000.000100")
    assert stored[0].text == TOMBSTONE
    assert "what did we decide" not in str(
        await store.raw("SELECT content FROM messages WHERE seq = 1")
    )


async def test_a_revision_never_reaches_the_agent(store: Store, tokenizer: Tokenizer) -> None:
    """Everything that comes out of the inbox is delivered as something to
    answer, and "Jane fixed a typo" is not a question."""
    provider = ScriptedProvider([says("noted")] * 4)
    adapter, client = make_adapter(store, tokenizer, provider=provider)
    await answer_once(adapter, client)

    await adapter.on_event(
        {
            "type": "message",
            "subtype": "message_deleted",
            "channel": "C0DEPLOY",
            "ts": "1700000099.000000",
            "deleted_ts": "1700000000.000100",
        }
    )

    assert await adapter.runtime.inbox.counts() == {"done": 1}, "the revision was not queued"
    assert len(provider.requests) == 1, "and no second turn ran"


async def test_siatts_own_streamed_updates_are_not_revisions(
    store: Store, tokenizer: Tokenizer
) -> None:
    """A streamed reply is one `chat.update` per second, and Slack echoes every
    one of them back as `message_changed`."""
    adapter, client = make_adapter(store, tokenizer)
    await answer_once(adapter, client)

    await adapter.on_event(
        {
            "type": "message",
            "subtype": "message_changed",
            "channel": "C0DEPLOY",
            "ts": "1700000099.000000",
            "message": {"ts": client.posted[0]["ts"], "user": BOT, "text": "noted"},
        }
    )

    stored = await store.recent_messages(f"slack:{TEAM}:C0DEPLOY:1700000000.000100")
    assert stored[0].text == "what did we decide?", "the person's message is untouched"


# -- identity -----------------------------------------------------------------


async def test_the_model_is_shown_names_rather_than_user_ids(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The end-to-end half of #23: the resolution happens after the queue and
    before the session, so what the model reads and what is stored as the
    user's message are the same text."""
    provider = ScriptedProvider([says("noted")] * 4)
    adapter, client = make_adapter(store, tokenizer, provider=provider)
    client.profiles["U0RAJ"] = {"name": "raj", "profile": {"display_name": "raj"}}

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention(text=f"<@{BOT}> did <@U0RAJ> ship it?"))
        await answered(client)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    asked = [m for m in provider.requests[0].messages if m.role == "user"][-1]
    assert asked.text == "did @raj ship it?"
    stored = await store.recent_messages(f"slack:{TEAM}:C0DEPLOY:1700000000.000100", 10)
    assert stored[0].text == "did @raj ship it?"


async def test_everybody_a_message_saw_is_recorded_for_mapping(
    store: Store, tokenizer: Tokenizer
) -> None:
    adapter, client = make_adapter(store, tokenizer)
    client.profiles["U0RAJ"] = {"name": "raj", "profile": {"display_name": "raj"}}

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention(text=f"<@{BOT}> ask <@U0RAJ>"))
        await answered(client)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    for uid, name in ((HUMAN, "jane"), ("U0RAJ", "raj")):
        row = await store.get_slack_user(TEAM, uid)
        assert row is not None and row["display_name"] == name


# -- the package --------------------------------------------------------------


def test_the_adapter_resolves_through_the_lazy_import() -> None:
    """#119 put `SlackAdapter` behind a module `__getattr__` so that
    `siatt.adapters.slack.events` imports on an install that never asked for the
    `slack` extra. The two tests that cover *that* half run in subprocesses,
    because making `slack_bolt` unimportable inside an environment that has it
    is the only way to reproduce a missing extra — and a subprocess is
    something coverage cannot see, so the lazy path read as dead code.

    This is the half that runs here: with the extra installed, the name
    resolves to the class. It is also the assertion that fails if the name is
    dropped from `__all__` or misspelled inside `__getattr__`, neither of which
    the subprocess pair would notice — both assert on the failure path.
    """
    assert "SlackAdapter" in package.__all__
    assert package.SlackAdapter is SlackAdapter


def test_every_name_the_package_exports_can_be_reached() -> None:
    """`__all__` is what `from ... import *` and a reader both go by, so a name
    on it that resolves to nothing is a promise the package does not keep."""
    for name in package.__all__:
        assert getattr(package, name) is not None, name


def test_the_package_says_no_to_a_name_it_does_not_have() -> None:
    """A `__getattr__` that falls off the end returns `None` instead of
    raising, which makes `hasattr` true for everything and turns a typo into a
    silent `None` at the call site rather than an import error."""
    missing = "Nonexistent"

    with pytest.raises(AttributeError, match="has no attribute 'Nonexistent'"):
        getattr(package, missing)


# -- feedback -----------------------------------------------------------------


def reaction_event(
    emoji: str = "+1", *, on: str, user: str = HUMAN, removed: bool = False
) -> dict[str, Any]:
    return {
        "type": "reaction_removed" if removed else "reaction_added",
        "user": user,
        "reaction": emoji,
        "item": {"type": "message", "channel": "C0DEPLOY", "ts": on},
        "item_user": BOT,
        "event_ts": "1700000009.000000",
    }


async def test_an_answer_records_the_memories_behind_it(store: Store, tokenizer: Tokenizer) -> None:
    """The chain #36 rests on: the reaction arrives days later, long after the
    process that produced the answer has forgotten everything."""
    adapter, client = make_adapter(store, tokenizer)
    await answer_once(adapter, client)

    answer = await store.answer_at("slack", f"slack:{TEAM}:C0DEPLOY:{client.posted[0]['ts']}")
    assert answer is not None
    assert answer["session_id"] == f"slack:{TEAM}:C0DEPLOY:1700000000.000100"


async def test_a_thumbs_up_reaches_the_memories_that_produced_the_answer(
    store: Store, tokenizer: Tokenizer
) -> None:
    adapter, client = make_adapter(store, tokenizer)
    await answer_once(adapter, client)
    external = f"slack:{TEAM}:C0DEPLOY:{client.posted[0]['ts']}"
    answer = await store.answer_at("slack", external)
    assert answer is not None
    # The scripted provider has no retriever behind it, so the memory is put on
    # the answer directly: what is under test here is the wiring from a Slack
    # reaction to a feedback row, not what the ranker packed.
    await store.write(
        "UPDATE answers SET memory_ids = ? WHERE id = ?",
        ('["mem_01K8XQ0000000000000000001"]', answer["id"]),
    )

    await adapter.on_reaction(reaction_event(on=str(client.posted[0]["ts"])))

    assert await store.endorsements_since("2000-01-01") == {"mem_01K8XQ0000000000000000001": 1}


async def test_a_reaction_never_reaches_the_agent(store: Store, tokenizer: Tokenizer) -> None:
    """A 👍 is not a question."""
    provider = ScriptedProvider([says("noted")] * 4)
    adapter, client = make_adapter(store, tokenizer, provider=provider)
    await answer_once(adapter, client)

    await adapter.on_reaction(reaction_event(on=str(client.posted[0]["ts"])))

    assert await adapter.runtime.inbox.counts() == {"done": 1}
    assert len(provider.requests) == 1


async def test_a_reaction_on_a_message_siatt_never_posted_does_nothing(
    store: Store, tokenizer: Tokenizer
) -> None:
    adapter, _ = make_adapter(store, tokenizer)

    await adapter.on_reaction(reaction_event(on="1700009999.999999"))

    assert await store.endorsements_since("2000-01-01") == {}


async def test_a_turn_that_ends_a_message_gets_another_one(
    store: Store, tokenizer: Tokenizer
) -> None:
    """#259. The turn used to end there: it wrote section ①, ended the message
    meaning to begin another, and sections ② and ③ were never written."""
    provider = ScriptedProvider(
        [
            ChatResponse(
                message=Message(
                    role="assistant",
                    content=(
                        ToolUseBlock(id="t0", name="send_message", input={"text": "section ①"}),
                    ),
                ),
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5),
                model="m",
            ),
            says("section ②"),
        ]
        * 4
    )
    adapter, client = make_adapter(store, tokenizer, provider=provider, tools=message_tools())

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await answered(client)
        await until(lambda: len(client.posted) > 1)
        await asyncio.sleep(0.1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.messages == ["section ①", "section ②"], "both of them, in order"


async def test_the_note_goes_on_the_last_message_and_nowhere_else(
    store: Store, tokenizer: Tokenizer
) -> None:
    """It explains how the turn stopped, and a turn stops once however many
    messages it took to get there."""
    rendered = messages(
        AgentResult(text="section ②", parts=("section ①",), stop_reason="max_tokens")
    )

    assert rendered[0] == "section ①"
    assert rendered[1].startswith("section ②") and "output limit" in rendered[1]


async def test_a_turn_with_nothing_left_to_add_sends_no_empty_message(
    store: Store, tokenizer: Tokenizer
) -> None:
    assert messages(AgentResult(text="  ", parts=("all of it",))) == ["all of it"]
