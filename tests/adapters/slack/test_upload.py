"""Sending a file out: the scope check, and what happens when Slack says no.

The inbound half's hazard is a bearer token following a URL somewhere it should
not go. This half's is quieter — bytes leaving the conversation they belong to —
and it has exactly one guard, which is that everything is read through
`Attachments.read` under the scope the turn ran under.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from siatt.adapters.slack.upload import SOURCE, SlackUploads, UploadRefused
from siatt.core.agent import AgentResult
from siatt.store import Store
from siatt.store.blobs import Attachment, Attachments
from tests.core.test_agent_images import png
from tests.core.test_file_tools import files


class FakeUploader:
    def __init__(self, refuse: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._refuse = refuse

    async def upload(
        self, *, channel: str, thread_ts: str | None, filename: str, title: str, data: bytes
    ) -> None:
        if self._refuse is not None:
            raise self._refuse
        self.calls.append(
            {
                "channel": channel,
                "thread_ts": thread_ts,
                "filename": filename,
                "title": title,
                "data": data,
            }
        )


class FakePoster:
    def __init__(self, refuse: Exception | None = None) -> None:
        self.posted: list[dict[str, Any]] = []
        self._refuse = refuse

    async def post(self, *, channel: str, thread_ts: str | None, text: str) -> str:
        if self._refuse is not None:
            raise self._refuse
        self.posted.append({"channel": channel, "thread_ts": thread_ts, "text": text})
        return "1700000000.000200"


async def conversation(store: Store, *, scope: str = "channel:C1") -> None:
    await store.ensure_session("s1", surface="slack", scope=scope)


async def kept(
    attachments: Attachments, *, scope: str = "channel:C1", name: str | None = "shot.png"
) -> Attachment:
    sha = await attachments.put(
        png(), mime="image/png", source_name="slack", scope=scope, session_id="s1", name=name
    )
    held = await attachments.get(sha, scope=scope)
    assert held is not None
    return held


def uploads(
    store: Store,
    attachments: Attachments | None,
    *,
    uploader: FakeUploader | None = None,
    poster: FakePoster | None = None,
) -> tuple[SlackUploads, FakeUploader, FakePoster]:
    up = uploader or FakeUploader()
    post = poster or FakePoster()
    return (
        SlackUploads(uploader=up, poster=post, store=store, attachments=attachments),
        up,
        post,
    )


async def send(
    sending: SlackUploads, *held: Attachment, external_id: str = "1700000000.000100"
) -> None:
    await sending.send(
        AgentResult(text="here it is", attachments=tuple(held)),
        session_id="s1",
        external_id=external_id,
        channel="C1",
        thread_ts="1700000000.000100",
    )


# -- the ordinary case -------------------------------------------------------


async def test_the_file_goes_into_the_thread_the_answer_went_to(
    store: Store, tmp_path: Path
) -> None:
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    sending, uploader, poster = uploads(store, attachments)

    await send(sending, held)

    assert uploader.calls == [
        {
            "channel": "C1",
            "thread_ts": "1700000000.000100",
            "filename": "shot.png",
            "title": "shot.png",
            "data": png(),
        }
    ]
    assert poster.posted == [], "nothing went wrong, so nothing is said about it"


async def test_a_turn_with_nothing_to_send_does_not_call_slack(
    store: Store, tmp_path: Path
) -> None:
    await conversation(store)
    sending, uploader, poster = uploads(store, files(store, tmp_path))

    await send(sending)

    assert uploader.calls == []
    assert poster.posted == []


async def test_an_install_that_keeps_no_files_sends_none(store: Store, tmp_path: Path) -> None:
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    sending, uploader, _ = uploads(store, None)

    await send(sending, held)

    assert uploader.calls == []


async def test_a_file_with_no_name_is_called_after_its_kind(store: Store, tmp_path: Path) -> None:
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments, name=None)
    sending, uploader, _ = uploads(store, attachments)

    await send(sending, held)

    assert uploader.calls[0]["filename"] == "attachment.png"


async def test_a_filename_is_flattened_before_it_is_sent(store: Store, tmp_path: Path) -> None:
    """Somebody else chose it. It is shown in the thread, and it is never a path."""
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments, name="../../etc/one\ntwo.png")
    sending, uploader, _ = uploads(store, attachments)

    await send(sending, held)

    name = uploader.calls[0]["filename"]
    assert "\n" not in name
    assert "/" not in name


# -- the scope ---------------------------------------------------------------


async def test_a_photograph_from_a_dm_is_not_uploaded_into_a_channel(
    store: Store, tmp_path: Path
) -> None:
    """The one check. The blob is real, the digest is right, and the session is
    somewhere else — so there are no bytes to send."""
    await conversation(store, scope="channel:C1")
    attachments = files(store, tmp_path)
    private = await kept(attachments, scope="private:U123", name="dm.png")
    sending, uploader, poster = uploads(store, attachments)

    await send(sending, private)

    assert uploader.calls == []
    assert "dm.png" in poster.posted[0]["text"]


async def test_the_session_scope_is_what_is_read_under(store: Store, tmp_path: Path) -> None:
    await conversation(store, scope="private:U123")
    attachments = files(store, tmp_path)
    held = await kept(attachments, scope="private:U123", name="dm.png")
    sending, uploader, _ = uploads(store, attachments)

    await send(sending, held)

    assert [call["filename"] for call in uploader.calls] == ["dm.png"]


# -- when it fails -----------------------------------------------------------


async def test_a_refused_upload_says_so_in_the_thread(store: Store, tmp_path: Path) -> None:
    """The answer already said "here it is". Silence would read as a file that
    was sent and lost."""
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    refused = UploadRefused("Siatt's Slack app lacks the files:write scope")
    sending, _, poster = uploads(store, attachments, uploader=FakeUploader(refuse=refused))

    await send(sending, held)

    said = poster.posted[0]["text"]
    assert "shot.png" in said
    assert "files:write" in said
    assert poster.posted[0]["thread_ts"] == "1700000000.000100"


async def test_an_unexpected_failure_is_still_only_a_line_in_the_thread(
    store: Store, tmp_path: Path
) -> None:
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    sending, _, poster = uploads(store, attachments, uploader=FakeUploader(refuse=RuntimeError()))

    await send(sending, held)

    assert "shot.png" in poster.posted[0]["text"]


async def test_slack_being_down_entirely_does_not_raise(store: Store, tmp_path: Path) -> None:
    """Two failures in a row is Slack being down, not something a third call fixes."""
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    sending, _, _ = uploads(
        store,
        attachments,
        uploader=FakeUploader(refuse=UploadRefused("no")),
        poster=FakePoster(refuse=RuntimeError()),
    )

    await send(sending, held)


async def test_one_file_failing_does_not_stop_the_other(store: Store, tmp_path: Path) -> None:
    await conversation(store)
    attachments = files(store, tmp_path)
    here = await kept(attachments)
    gone = Attachment(sha256="a" * 64, mime="image/png", size=45, name="missing.png")
    sending, uploader, poster = uploads(store, attachments)

    await send(sending, gone, here)

    assert [call["filename"] for call in uploader.calls] == ["shot.png"]
    assert "missing.png" in poster.posted[0]["text"]


# -- once, across a restart --------------------------------------------------


async def test_a_redelivered_turn_does_not_send_the_file_twice(
    store: Store, tmp_path: Path
) -> None:
    """The inbox delivers at least once and the whole turn re-runs. The answer
    is rewritten; the photograph is not posted again."""
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    sending, uploader, _ = uploads(store, attachments)

    await send(sending, held)
    await send(sending, held)

    assert len(uploader.calls) == 1


async def test_the_second_run_is_stopped_by_the_database_not_by_memory(
    store: Store, tmp_path: Path
) -> None:
    """A restart between the two is the case a dictionary in the process misses."""
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    first, uploader, _ = uploads(store, attachments)
    await send(first, held)

    second, again, _ = uploads(store, attachments)
    await send(second, held)

    assert len(uploader.calls) == 1
    assert again.calls == []


async def test_the_same_file_in_a_later_turn_is_sent_again(store: Store, tmp_path: Path) -> None:
    """Asked for twice is sent twice. The arrival key is the request, not the file."""
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    sending, uploader, _ = uploads(store, attachments)

    await send(sending, held, external_id="1700000000.000100")
    await send(sending, held, external_id="1700000000.000900")

    assert len(uploader.calls) == 2


async def test_what_was_sent_is_recorded_as_an_arrival(store: Store, tmp_path: Path) -> None:
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    sending, _, _ = uploads(store, attachments)

    await send(sending, held)

    rows = await store.raw(
        "SELECT source, scope, name FROM attachment_refs WHERE source = ?", (SOURCE,)
    )
    assert [dict(row) for row in rows] == [
        {"source": SOURCE, "scope": "channel:C1", "name": "shot.png"}
    ]


async def test_a_file_that_could_not_be_sent_is_not_recorded_as_sent(
    store: Store, tmp_path: Path
) -> None:
    await conversation(store)
    attachments = files(store, tmp_path)
    held = await kept(attachments)
    sending, _, _ = uploads(store, attachments, uploader=FakeUploader(refuse=UploadRefused("no")))

    await send(sending, held)

    rows = await store.raw("SELECT id FROM attachment_refs WHERE source = ?", (SOURCE,))
    assert rows == []
