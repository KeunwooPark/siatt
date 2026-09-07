"""Whether the way out can carry a file, from the surface to the tool.

`send_file` refuses before it resolves anything on a surface that only sends
strings, which is only as good as the fact reaching it. This is that wire: the
surface declares it when it builds its runtime, and the turn carries it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from siatt.core.agent import AgentResult
from siatt.core.events import InboundEvent
from siatt.core.runtime import Runtime, one_message
from siatt.core.session import SessionState, Turn
from siatt.store import Store


class RecordingAgent:
    """Answers nothing, and remembers what it was told about the surface."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.asked: list[bool] = []

    async def respond(self, session_id: str, text: str, **kwargs: Any) -> AgentResult:
        self.asked.append(bool(kwargs["can_send_files"]))
        return AgentResult(text="ok")


async def _one_turn(store: Store, **runtime: Any) -> RecordingAgent:
    agent = RecordingAgent(store)

    async def sink(event: InboundEvent, result: AgentResult) -> None:
        pass

    engine = Runtime(agent, one_message(sink), **runtime)  # type: ignore[arg-type]
    event = InboundEvent(source="slack", external_id="E1", session_id="s1", text="send it")
    session = SessionState(
        id="s1", surface="slack", scope="workspace", message_count=0, episode_id=None
    )
    await engine._turn(Turn(event=event, session=session))
    return agent


async def test_a_surface_that_sends_files_says_so_on_the_turn(tmp_path: Path) -> None:
    async with await Store.open(tmp_path / "runtime.db") as store:
        agent = await _one_turn(store, sends_files=True)

    assert agent.asked == [True]


async def test_a_surface_is_mute_about_files_until_it_is_wired_up(tmp_path: Path) -> None:
    """The default is the safe one: a new surface promises nothing."""
    async with await Store.open(tmp_path / "runtime.db") as store:
        agent = await _one_turn(store)

    assert agent.asked == [False]
