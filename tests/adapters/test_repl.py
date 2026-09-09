"""The terminal adapter's user-facing strings, and how it hands over a file."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from siatt import __version__
from siatt.adapters.cli.repl import SCOPE, Repl, banner
from siatt.core.agent import Agent
from siatt.core.file_tools import file_tools
from siatt.llm.tokens import Tokenizer
from siatt.store import Store
from siatt.store.blobs import Attachments
from tests.core.test_agent import build, says
from tests.core.test_agent_images import png
from tests.core.test_file_tools import _sends as sends
from tests.core.test_file_tools import files


def test_the_banner_names_the_running_version() -> None:
    """#48: a hand-maintained label told users of a memory build there was none.

    The point is not the words but where they come from — a version written
    into a string is one that goes stale without anything failing.
    """
    assert __version__ in banner()
    assert "v0" not in banner()
    assert "No memory yet" not in banner()


# -- handing over a file ------------------------------------------------------


def repl(agent: Agent) -> tuple[Repl, io.StringIO]:
    out = io.StringIO()
    # Wide, so an assertion is about what was printed rather than about where
    # rich decided to wrap it.
    return Repl(agent=agent, console=Console(file=out, width=200), session_id="cli:1"), out


async def kept(
    attachments: Attachments, *, name: str | None = "shot.png", session_id: str = "cli:1"
) -> str:
    return await attachments.put(
        png(),
        mime="image/png",
        source_name="cli",
        scope=SCOPE,
        session_id=session_id,
        name=name,
    )


def asking_for_it(store: Store, tokenizer: Tokenizer, attachments: Attachments, sha: str) -> Agent:
    agent, _ = build(
        store,
        tokenizer,
        [sends(sha), says("here it is")],
        tools=file_tools(store=store, attachments=attachments),
        attachments=attachments,
    )
    return agent


async def test_a_sent_file_is_reported_with_the_path_to_it(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """The terminal's whole delivery: the person and the process share a disk."""
    attachments = files(store, tmp_path)
    sha = await kept(attachments)
    session, out = repl(asking_for_it(store, tokenizer, attachments, sha))

    await session._turn("send me that shot")

    printed = out.getvalue()
    assert "shot.png" in printed
    assert "image/png" in printed
    assert str(await attachments.path(sha)) in printed


async def test_a_filename_is_one_line_whatever_it_contains(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """It came off an upload, so somebody else wrote it."""
    attachments = files(store, tmp_path)
    sha = await kept(attachments, name="one\ntwo [dim]three[/dim].png")
    session, out = repl(asking_for_it(store, tokenizer, attachments, sha))

    await session._turn("send it")

    sent = [line for line in out.getvalue().splitlines() if "one" in line]
    assert len(sent) == 1
    assert "three" in sent[0], "and the markup in it was shown, not obeyed"


async def test_a_file_collected_since_the_turn_says_so_instead(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    sha = await kept(attachments)
    agent = asking_for_it(store, tokenizer, attachments, sha)
    session, out = repl(agent)
    (await attachments.path(sha)).unlink()

    await session._turn("send it")

    printed = out.getvalue()
    assert "no longer on disk" in printed
    assert "blobs" not in printed, "and no path to a file that is not there"


async def test_a_turn_with_nothing_to_send_prints_what_it_always_did(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    await kept(attachments)
    agent, _ = build(store, tokenizer, [says("nothing to send")], attachments=attachments)
    session, out = repl(agent)

    await session._turn("hello")

    assert "nothing to send" in out.getvalue()
    assert "sent" not in out.getvalue().replace("nothing to send", "")


async def test_the_terminal_says_it_can_send_files(
    store: Store, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """Otherwise `send_file` refuses here, which was the state before #249."""
    attachments = files(store, tmp_path)
    sha = await kept(attachments)
    agent = asking_for_it(store, tokenizer, attachments, sha)
    session, _ = repl(agent)

    await session._turn("send it")

    assert session.last_result is not None
    assert [held.sha256 for held in session.last_result.attachments] == [sha]
