"""What Slack sends, and what Siatt decides it means.

No `slack_bolt` here on purpose: every judgement about which messages are for
Siatt is in `events.py`, and none of it needs a socket to test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any

import pytest

from siatt.adapters.slack.events import (
    Accepted,
    Changed,
    Decision,
    Ignored,
    Reacted,
    SlackContext,
    normalize,
    reaction,
)
from siatt.config import SlackSettings

BOT = "U0SIATT"
TEAM = "T0TEAM"
HUMAN = "U0HUMAN"


def context(*, allowed: frozenset[str] = frozenset()) -> SlackContext:
    return SlackContext(bot_user_id=BOT, team_id=TEAM, allowed_channels=allowed)


async def never(session_id: str) -> bool:
    return False


async def always(session_id: str) -> bool:
    return True


def dm(text: str = "what did we decide?", ts: str = "1700000000.000100") -> dict[str, Any]:
    return {
        "type": "message",
        "channel_type": "im",
        "channel": "D1",
        "user": HUMAN,
        "text": text,
        "ts": ts,
    }


def in_channel(
    text: str = f"<@{BOT}> what did we decide?",
    ts: str = "1700000000.000100",
    *,
    channel: str = "C1",
    thread: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": "message",
        "channel_type": "channel",
        "channel": channel,
        "user": HUMAN,
        "text": text,
        "ts": ts,
    }
    if thread is not None:
        body["thread_ts"] = thread
    return body


def as_mention(body: dict[str, Any]) -> dict[str, Any]:
    """The same message as Slack's other delivery of it.

    An `app_mention` payload carries no `channel_type` — the field is on the
    conversation, and this event is about the mention.
    """
    return {key: value for key, value in body.items() if key != "channel_type"} | {
        "type": "app_mention"
    }


def accepted(decision: Decision) -> Accepted:
    assert isinstance(decision, Accepted), decision
    return decision


# -- what gets through -------------------------------------------------------


async def test_a_dm_is_answered() -> None:
    decision = await normalize(dm(), context=context(), known_session=never)

    event = accepted(decision).event
    assert event.session_id == f"slack:{TEAM}:D1:1700000000.000100"
    assert event.author == HUMAN
    assert event.text == "what did we decide?"


async def test_a_mention_in_a_channel_is_answered() -> None:
    decision = await normalize(in_channel(), context=context(), known_session=never)

    event = accepted(decision).event
    assert event.text == "what did we decide?", "Siatt's own mention is addressing, not content"


async def test_an_app_mention_carries_no_channel_type_and_is_still_a_channel() -> None:
    """A minimally-scoped install gets `app_mention` and nothing else, and that
    event has no `channel_type` to read."""
    body = {
        "type": "app_mention",
        "channel": "C1",
        "user": HUMAN,
        "text": f"<@{BOT}> hello",
        "ts": "1700000000.000100",
    }

    accepted(await normalize(body, context=context(), known_session=never))


async def test_a_thread_reply_needs_no_second_mention() -> None:
    """Being spoken to once starts a conversation; making people re-address
    every message in a thread is how a bot becomes tedious."""
    body = in_channel("and what about staging?", "1700000000.000200", thread="1700000000.000100")

    ignored = await normalize(body, context=context(), known_session=never)
    accepted_again = await normalize(body, context=context(), known_session=always)

    assert isinstance(ignored, Ignored)
    assert accepted(accepted_again).event.session_id == f"slack:{TEAM}:C1:1700000000.000100"


async def test_a_reply_goes_to_the_thread_the_question_was_asked_in() -> None:
    top = accepted(await normalize(in_channel(), context=context(), known_session=never)).event
    reply = accepted(
        await normalize(
            in_channel(f"<@{BOT}> more", "1700000000.000200", thread="1700000000.000100"),
            context=context(),
            known_session=never,
        )
    ).event

    assert top.reply_to == "1700000000.000100", "a top-level message starts a thread"
    assert reply.reply_to == "1700000000.000100"
    assert reply.session_id == top.session_id


# -- threads Siatt started itself --------------------------------------------


def opened(by: dict[str, str]) -> Any:
    """A `ThreadSession` over a map of thread timestamp to conversation."""

    async def lookup(channel: str, thread_ts: str) -> str | None:
        return by.get(thread_ts)

    return lookup


async def test_a_reply_under_a_post_siatt_started_needs_no_mention() -> None:
    """A standing task's morning briefing is a thread nobody mentioned Siatt
    in, under a timestamp no session has ever used. Both tests that make a
    reply worth answering say no, so without the thread map the one thing
    people will actually do with a briefing — reply to it — is ignored."""
    body = in_channel(
        "which of those is worth reading?", "1700000000.000200", thread="1700000000.000100"
    )

    decision = await normalize(
        body,
        context=context(),
        known_session=never,
        thread_session=opened({"1700000000.000100": "slack:task:01J@2026-09-08T09:00"}),
    )

    assert accepted(decision).event.session_id == "slack:task:01J@2026-09-08T09:00"


async def test_the_reply_continues_the_turn_that_posted_rather_than_a_new_one() -> None:
    """The session is the one that produced the post, not the one this thread's
    timestamp would derive — which is the difference between "what about the
    third one?" being answerable and it arriving with nothing to refer to."""
    body = in_channel("and the third?", "1700000000.000200", thread="1700000000.000100")

    event = accepted(
        await normalize(
            body,
            context=context(),
            known_session=never,
            thread_session=opened({"1700000000.000100": "slack:task:01J@2026-09-08T09:00"}),
        )
    ).event

    assert event.session_id != f"slack:{TEAM}:C1:1700000000.000100"
    assert event.reply_to == "1700000000.000100", "and the answer goes back into that thread"


async def test_a_thread_siatt_did_not_start_is_still_ignored() -> None:
    body = in_channel("morning all", "1700000000.000200", thread="1700000000.000999")

    decision = await normalize(
        body,
        context=context(),
        known_session=never,
        thread_session=opened({"1700000000.000100": "slack:task:01J@2026-09-08T09:00"}),
    )

    assert isinstance(decision, Ignored)


async def test_a_channel_off_the_allowlist_is_not_even_looked_up() -> None:
    """The allowlist is what an install uses to say "do not read this channel",
    and asking a question about a message from one is already reading it."""
    asked: list[str] = []

    async def lookup(channel: str, thread_ts: str) -> str | None:
        asked.append(channel)
        return "slack:task:01J@2026-09-08T09:00"

    decision = await normalize(
        in_channel(channel="C0PRIVATE"),
        context=context(allowed=frozenset({"C1"})),
        known_session=always,
        thread_session=lookup,
    )

    assert isinstance(decision, Ignored)
    assert not asked


async def test_somebody_elses_mention_stays_in_the_text() -> None:
    body = in_channel(f"<@{BOT}> ask <@U0OTHER|jane> about deploys")

    assert (
        accepted(await normalize(body, context=context(), known_session=never)).event.text
        == "ask <@U0OTHER|jane> about deploys"
    )


async def test_the_shape_of_a_message_survives() -> None:
    """Stripping the mention used to cost every line break in the message —
    `str.split()` with no argument splits on newlines too. Pasted code, stack
    traces, numbered lists and multi-paragraph questions all arrived as one
    run-on line."""
    sent = f"<@{BOT}> please look at this:\n\n```\ndef f():\n    return 1\n```\n\nthanks"

    event = accepted(
        await normalize(in_channel(sent), context=context(), known_session=never)
    ).event

    assert event.text == "please look at this:\n\n```\ndef f():\n    return 1\n```\n\nthanks"


async def test_the_gap_the_mention_leaves_is_still_tidied() -> None:
    """One space where a mention was between two words, none where it was at
    an edge — and nothing else touched."""
    cases = {
        f"<@{BOT}> hello": "hello",
        f"hello <@{BOT}>": "hello",
        f"hey <@{BOT}> what's up": "hey what's up",
        f"<@{BOT}>\nhello": "hello",
        f"a  b <@{BOT}> c": "a  b c",
    }

    for sent, expected in cases.items():
        event = accepted(
            await normalize(in_channel(sent), context=context(), known_session=never)
        ).event
        assert event.text == expected, sent


async def test_a_question_with_a_file_attached_is_answered() -> None:
    """`file_share` is not an edit. It is somebody asking a question with a
    document attached, and it used to be dropped in silence — the failure the
    design doc is otherwise careful about ("a chat assistant that silently
    ignores you is worse than one that answers you twice")."""
    body = in_channel(f"<@{BOT}> what's in this?") | {
        "subtype": "file_share",
        "files": [
            {
                "name": "q3.pdf",
                "mimetype": "application/pdf",
                "size": 4096,
                "url_private_download": "https://files.slack.com/files-pri/T1-F1/download/q3.pdf",
            }
        ],
    }

    event = accepted(await normalize(body, context=context(), known_session=never)).event

    # The text is the person's words alone. What arrived with them is a
    # descriptor, because saying anything true about a file needs a fetch, and a
    # fetch does not belong inside the three-second ack.
    assert event.text == "what's in this?"
    assert [(a.name, a.mime, a.size) for a in event.attachments] == [
        ("q3.pdf", "application/pdf", 4096)
    ]
    assert event.attachments[0].url == "https://files.slack.com/files-pri/T1-F1/download/q3.pdf"


async def test_a_file_with_no_address_is_still_an_attachment() -> None:
    """A tombstoned upload, or one this install may not read. Dropping it here
    would answer "what's in this?" as though nothing had been sent."""
    body = dm("") | {"subtype": "file_share", "files": [{"name": "q3.pdf"}, {}]}

    event = accepted(await normalize(body, context=context(), known_session=never)).event

    assert event.text == ""
    assert [(a.name, a.url) for a in event.attachments] == [("q3.pdf", None), (None, None)]


async def test_a_file_entry_that_is_not_an_object_is_still_an_attachment() -> None:
    """Slack sends file objects, and #121 believed the payload's shape. The
    judgement has to reach a decision for anything JSON can hold, because the
    alternative is an exception on the path that has not written the inbox row
    yet — and an entry we cannot read the name of is one we cannot open, which
    is what the note downstream says about it."""
    body = dm("what's in this?") | {
        "subtype": "file_share",
        "files": ["F0123456", {"name": "q3.pdf"}, None],
    }

    event = accepted(await normalize(body, context=context(), known_session=never)).event

    assert event.text == "what's in this?"
    assert [a.name for a in event.attachments] == [None, "q3.pdf", None]
    assert all(a.url is None for a in event.attachments)


@pytest.mark.parametrize("files", ["F0123456", {"id": "F0123456"}, 7])
async def test_a_files_field_that_is_not_a_list_is_not_iterated(files: Any) -> None:
    """A string would be walked character by character and a mapping key by
    key, both of which invent attachments that were never sent. Nothing is
    said about a field nothing can be read from; the message still gets
    through, which is the part that matters."""
    body = dm("what's in this?") | {"subtype": "file_share", "files": files}

    event = accepted(await normalize(body, context=context(), known_session=never)).event

    assert event.text == "what's in this?"
    assert event.attachments == ()


@pytest.mark.parametrize("subtype", ["file_share", "me_message", "thread_broadcast"])
async def test_the_subtypes_that_are_somebody_talking_get_through(subtype: str) -> None:
    assert isinstance(
        await normalize(dm() | {"subtype": subtype}, context=context(), known_session=never),
        Accepted,
    )


# -- what does not ------------------------------------------------------------


async def test_a_channel_message_that_is_not_addressed_to_siatt_is_ignored() -> None:
    decision = await normalize(
        in_channel("what did we decide?"), context=context(), known_session=never
    )

    assert isinstance(decision, Ignored)
    assert decision.reason == "not addressed to Siatt"


@pytest.mark.parametrize("subtype", ["channel_join", "channel_topic", "invented"])
async def test_a_subtype_that_is_not_somebody_talking_is_ignored(subtype: str) -> None:
    """An unknown subtype is ignored rather than answered, because Siatt replies
    in any thread it is already part of and "Bob joined the channel" is not a
    question."""
    decision = await normalize(dm() | {"subtype": subtype}, context=context(), known_session=never)

    assert isinstance(decision, Ignored)
    assert subtype in decision.reason


@pytest.mark.parametrize("subtype", ["message_changed", "message_deleted"])
async def test_a_revision_is_never_something_to_answer(subtype: str) -> None:
    """#25 made these mean something, and the thing they must not mean is a new
    message: an edit that reached the agent would be answered a second time."""
    decision = await normalize(dm() | {"subtype": subtype}, context=context(), known_session=always)

    assert not isinstance(decision, Accepted)


async def test_siatts_own_message_is_ignored() -> None:
    decision = await normalize(dm() | {"user": BOT}, context=context(), known_session=always)

    assert isinstance(decision, Ignored)
    assert decision.reason == "posted by Siatt"


async def test_another_bot_is_ignored() -> None:
    """Two bots in one channel is how a loop starts."""
    decision = await normalize(
        in_channel(f"<@{BOT}> status?") | {"bot_id": "B1"},
        context=context(),
        known_session=always,
    )

    assert isinstance(decision, Ignored)
    assert decision.reason == "posted by a bot"


async def test_a_channel_off_the_allowlist_is_ignored_even_when_addressed() -> None:
    decision = await normalize(
        in_channel(channel="CSECRET"),
        context=context(allowed=frozenset({"C1"})),
        known_session=always,
    )

    assert isinstance(decision, Ignored)
    assert decision.reason == "channel CSECRET is not on the allowlist"


async def test_an_empty_allowlist_allows_the_channels_siatt_was_invited_to() -> None:
    """Inviting a bot to a channel is already a deliberate act by a person."""
    assert isinstance(
        await normalize(in_channel(), context=context(), known_session=never), Accepted
    )


async def test_the_channel_allowlist_does_not_govern_dms() -> None:
    decision = await normalize(
        dm(), context=context(allowed=frozenset({"C1"})), known_session=never
    )

    assert isinstance(decision, Accepted)


async def test_both_deliveries_of_one_message_normalize_identically() -> None:
    """A mention in a readable channel arrives twice, as `message` and as
    `app_mention`, under one `ts` and therefore one dedupe key — so the second
    is discarded unexamined and whichever landed first decided everything. The
    two payloads must not be able to disagree about anything."""
    for body in (dm(f"<@{BOT}> what did we decide?"), in_channel()):
        as_message = accepted(await normalize(body, context=context(), known_session=never)).event
        as_app_mention = accepted(
            await normalize(as_mention(body), context=context(), known_session=never)
        ).event

        assert as_message == as_app_mention, body


async def test_the_allowlist_does_not_govern_a_dm_however_slack_delivered_it() -> None:
    """`app_mention` carries no `channel_type`, so reading `is_dm` off it filed
    a DM as a channel — and the `app_mention` copy was then dropped for being
    off-allowlist. The channel id says what kind of conversation it is."""
    body = as_mention(dm(f"<@{BOT}> what did we decide?"))

    decision = await normalize(
        body, context=context(allowed=frozenset({"C1"})), known_session=never
    )

    assert accepted(decision).event.author == HUMAN


# -- revisions ----------------------------------------------------------------


def edited(
    text: str = "what did we decide, again?",
    *,
    ts: str = "1700000000.000100",
    channel: str = "D1",
    user: str = HUMAN,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "type": "message",
        "subtype": "message_changed",
        "channel": channel,
        # The change's own timestamp, which is *not* the message's. Written
        # differently on purpose: reading this one looks up a message that
        # never existed and silently does nothing.
        "ts": "1700000099.000000",
        "message": {"ts": ts, "user": user, "text": text} | extra,
    }


def deleted(ts: str = "1700000000.000100", *, channel: str = "D1") -> dict[str, Any]:
    return {
        "type": "message",
        "subtype": "message_deleted",
        "channel": channel,
        "ts": "1700000099.000000",
        "deleted_ts": ts,
    }


async def test_an_edit_names_the_message_it_edited() -> None:
    decision = await normalize(edited(), context=context(), known_session=never)

    assert isinstance(decision, Changed)
    assert decision.revision.external_id == f"slack:{TEAM}:D1:1700000000.000100"
    assert decision.revision.text == "what did we decide, again?"


async def test_a_deletion_names_the_message_it_deleted() -> None:
    decision = await normalize(deleted(), context=context(), known_session=never)

    assert isinstance(decision, Changed)
    assert decision.revision.external_id == f"slack:{TEAM}:D1:1700000000.000100"
    assert decision.revision.deleted


async def test_an_edited_message_is_keyed_the_same_way_the_original_was() -> None:
    """The revision has to find the row ingress wrote, and the only thing
    connecting them is that both sides spell the key the same way."""
    original = await normalize(dm(), context=context(), known_session=never)
    revision = await normalize(edited(channel="D1"), context=context(), known_session=never)

    assert isinstance(original, Accepted) and isinstance(revision, Changed)
    assert revision.revision.external_id == original.event.external_id


async def test_siatts_own_message_changing_is_not_a_revision() -> None:
    """A streamed reply is one `chat.update` per second (#22), and every one of
    them comes back as a `message_changed`. Without this an answer would revise
    itself thirty times."""
    decision = await normalize(edited(user=BOT), context=context(), known_session=always)

    assert isinstance(decision, Ignored)


async def test_a_bot_message_changing_is_not_a_revision() -> None:
    decision = await normalize(edited(bot_id="B0SIATT"), context=context(), known_session=always)

    assert isinstance(decision, Ignored)


async def test_siatts_own_mention_is_stripped_from_an_edit_as_well() -> None:
    """The stored text had it stripped, so an edit that put it back would
    rewrite the message into something ingress would never have written."""
    decision = await normalize(
        edited(f"<@{BOT}> what did we decide, again?"), context=context(), known_session=never
    )

    assert isinstance(decision, Changed)
    assert decision.revision.text == "what did we decide, again?"


async def test_a_revision_outside_the_allowlist_is_ignored() -> None:
    """It would find no stored message anyway — ingress never accepted one. Said
    out loud rather than left to fall through, because "it happens to be
    harmless" is not the same as "it is refused"."""
    decision = await normalize(
        deleted(channel="C0OTHER"), context=context(allowed=frozenset({"C1"})), known_session=never
    )

    assert isinstance(decision, Ignored)
    assert "allowlist" in decision.reason


@pytest.mark.parametrize(
    "event",
    [
        {"type": "message", "subtype": "message_deleted", "channel": "D1"},
        {"type": "message", "subtype": "message_changed", "channel": "D1"},
        {"type": "message", "subtype": "message_changed", "channel": "D1", "message": "nope"},
        {"type": "message", "subtype": "message_deleted", "deleted_ts": "1.0"},
    ],
    ids=["no deleted_ts", "no message", "message is not an object", "no channel"],
)
async def test_a_revision_that_names_nothing_is_ignored(event: dict[str, Any]) -> None:
    """These reach the ack path, where an exception is a message lost with
    nothing recording that it arrived."""
    decision = await normalize(event, context=context(), known_session=never)

    assert isinstance(decision, Ignored)


# -- dedupe -------------------------------------------------------------------


async def test_the_same_message_has_one_id_however_it_arrives() -> None:
    """An install with both subscriptions gets a mention twice, under two
    different `event_id`s. Keying on the message is what makes that harmless."""
    as_message = accepted(
        await normalize(in_channel(), context=context(), known_session=never)
    ).event
    as_mention = accepted(
        await normalize(
            {
                "type": "app_mention",
                "channel": "C1",
                "user": HUMAN,
                "text": f"<@{BOT}> what did we decide?",
                "ts": "1700000000.000100",
            },
            context=context(),
            known_session=never,
        )
    ).event

    assert as_message.external_id == as_mention.external_id == f"slack:{TEAM}:C1:1700000000.000100"


# -- one pool (#265) ----------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        dm(),
        in_channel(),
        in_channel() | {"channel_type": "group"},
        in_channel() | {"channel_type": "mpim"},
    ],
)
async def test_everything_from_slack_is_one_pool(body: dict[str, Any]) -> None:
    """Siatt serves one person. A DM and a channel are that person talking in
    two places, and a fact learned in either answers a question asked in the
    other — which the per-channel scoping made impossible (#265)."""
    event = accepted(await normalize(body, context=context(), known_session=always)).event

    assert event.scope == "workspace"


# -- and the import costs nothing either --------------------------------------


def test_the_judgements_import_without_the_slack_extra() -> None:
    """The docstring above claims no `slack_bolt`, and that was true of every
    line in this file except the first: importing `events` runs the package
    `__init__`, which imported the adapter, which imports `slack_bolt`. On a
    `uv sync --dev` that is a collection *error*, not a skip, and pytest aborts
    the whole run — zero of 600-odd tests, which reads a lot like a pass.
    """
    program = textwrap.dedent(
        """
        import sys

        class Absent:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "slack_bolt":
                    raise ModuleNotFoundError("No module named 'slack_bolt'")
                return None

        sys.meta_path.insert(0, Absent())

        import siatt.adapters.slack as package
        from siatt.adapters.slack.events import normalize

        assert "slack_bolt" not in sys.modules, "something imported it anyway"
        assert package.normalize is normalize
        """
    )

    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


def test_asking_for_the_adapter_is_what_needs_the_extra() -> None:
    """Lazy, not gone. The name still resolves where it always did."""
    program = textwrap.dedent(
        """
        import sys

        class Absent:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "slack_bolt":
                    raise ModuleNotFoundError("No module named 'slack_bolt'")
                return None

        sys.meta_path.insert(0, Absent())

        import siatt.adapters.slack as package

        try:
            package.SlackAdapter
        except ModuleNotFoundError as exc:
            assert "slack_bolt" in str(exc)
        else:
            raise AssertionError("the adapter resolved without its extra")

        try:
            package.NotAThing
        except AttributeError:
            pass
        else:
            raise AssertionError("an unknown attribute resolved")
        """
    )

    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


# -- reactions ----------------------------------------------------------------

#: The default map, which is what an install with no `[slack] reactions` gets.
VERDICTS = SlackSettings().reactions


def reacted(
    emoji: str = "+1",
    *,
    on: str = "1700000001.000000",
    channel: str = "D1",
    user: str = HUMAN,
    item_user: str | None = BOT,
    removed: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": "reaction_removed" if removed else "reaction_added",
        "user": user,
        "reaction": emoji,
        "item": {"type": "message", "channel": channel, "ts": on},
        "event_ts": "1700000009.000000",
    }
    if item_user is not None:
        body["item_user"] = item_user
    return body | extra


def approved(decision: Decision) -> Reacted:
    assert isinstance(decision, Reacted), decision
    return decision


async def test_a_thumbs_up_names_the_answer_it_is_on() -> None:
    """The answer, not the question: a reaction names the message it sits on,
    and the message it sits on is Siatt's reply."""
    decision = approved(reaction(reacted(), context=context(), verdicts=VERDICTS))

    assert decision.external_id == f"slack:{TEAM}:D1:1700000001.000000"
    assert decision.verdict == "up"
    assert decision.author == HUMAN
    assert not decision.removed


async def test_a_cross_is_a_down_vote() -> None:
    assert approved(reaction(reacted("x"), context=context(), verdicts=VERDICTS)).verdict == "down"


async def test_removing_a_reaction_is_a_retraction_not_a_vote_the_other_way() -> None:
    decision = approved(reaction(reacted(removed=True), context=context(), verdicts=VERDICTS))

    assert decision.removed and decision.verdict == "up"


async def test_a_skin_tone_is_still_the_same_thumb() -> None:
    """Slack appends `::skin-tone-3` to the name, so a configured `+1` would
    match the default-toned thumb and silently ignore everybody else's."""
    decision = approved(reaction(reacted("+1::skin-tone-5"), context=context(), verdicts=VERDICTS))

    assert decision.verdict == "up"


async def test_an_emoji_nobody_mapped_means_nothing() -> None:
    """A 🎉 is not a verdict, and the whole value of the signal is that what
    counts as one was chosen."""
    decision = reaction(reacted("tada"), context=context(), verdicts=VERDICTS)

    assert isinstance(decision, Ignored)
    assert "not mapped" in decision.reason


async def test_the_mapping_is_configurable() -> None:
    """A workspace where ✅ means "I have actioned this" should be able to say
    so, rather than have memory boosted on the strength of a checkbox."""
    theirs = SlackSettings(reactions={"eyes": "down"}).reactions

    assert approved(reaction(reacted("eyes"), context=context(), verdicts=theirs)).verdict == "down"
    assert isinstance(reaction(reacted("+1"), context=context(), verdicts=theirs), Ignored)


async def test_a_reaction_on_somebody_elses_message_is_not_feedback() -> None:
    """People react to each other all day. Only a reaction on one of Siatt's own
    answers says anything about memory."""
    decision = reaction(reacted(item_user=HUMAN), context=context(), verdicts=VERDICTS)

    assert isinstance(decision, Ignored)


async def test_siatts_own_reaction_is_not_feedback() -> None:
    decision = reaction(reacted(user=BOT), context=context(), verdicts=VERDICTS)

    assert isinstance(decision, Ignored)


async def test_a_reaction_on_a_file_is_not_feedback() -> None:
    body = reacted() | {"item": {"type": "file", "file": "F0123"}}

    assert isinstance(reaction(body, context=context(), verdicts=VERDICTS), Ignored)


async def test_a_reaction_outside_the_allowlist_is_ignored() -> None:
    decision = reaction(
        reacted(channel="C0OTHER"), context=context(allowed=frozenset({"C1"})), verdicts=VERDICTS
    )

    assert isinstance(decision, Ignored)
    assert "allowlist" in decision.reason


async def test_a_message_is_not_a_reaction() -> None:
    assert isinstance(reaction(dm(), context=context(), verdicts=VERDICTS), Ignored)


@pytest.mark.parametrize(
    "body",
    [
        {"type": "reaction_added", "reaction": "+1", "item": {"type": "message", "ts": "1.0"}},
        {"type": "reaction_added", "reaction": "+1", "user": HUMAN, "item": "nope"},
        {"type": "reaction_added", "user": HUMAN, "item": {"type": "message", "channel": "D1"}},
    ],
    ids=["no user or channel", "item is not an object", "no ts"],
)
async def test_a_reaction_that_names_nothing_is_ignored(body: dict[str, Any]) -> None:
    """This runs on the ack path, where an exception is an event lost with
    nothing recording that it arrived."""
    assert isinstance(reaction(body, context=context(), verdicts=VERDICTS), Ignored)
