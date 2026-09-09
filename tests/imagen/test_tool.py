"""`image_generate`: what it settles before it draws, and what it leaves behind.

Three of these tests are about things that happen *before* a request goes out,
and they are the point of the module. Generation is the first tool that spends
real money on one call, so a refusal that arrives after the money is gone is not
a refusal. The fourth thing it leaves behind is a label: bytes Siatt invented,
marked as such, so nothing can later file them as evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from siatt.config import AttachmentSettings
from siatt.core.file_tools import CANNOT_SEND, MAX_SENT, file_tools
from siatt.core.tools import ToolContext, ToolRegistry
from siatt.errors import AuthError
from siatt.imagen.base import GeneratedImage
from siatt.imagen.tool import FULL, OVER_BUDGET, image_generate_tool
from siatt.llm.cost import CallRecord, CostMeter, Price, PriceBook
from siatt.llm.types import ToolUseBlock, Usage
from siatt.memory.blobref import handle
from siatt.memory.observation import GENERATED, citable, named
from siatt.store import Store
from siatt.store.blobs import Attachments
from tests.core.test_agent_images import png

USAGE = Usage(input_tokens=12, output_tokens=272)


class FakeDrawer:
    """An endpoint that answers from a script and records what it was asked."""

    name = "fake"
    model = "openai/gpt-image-2"

    def __init__(self, *replies: GeneratedImage | Exception) -> None:
        self.replies: list[GeneratedImage | Exception] = list(replies) or [
            GeneratedImage(data=png(), mime="image/png", model=self.model, usage=USAGE)
        ]
        self.prompts: list[str] = []
        self.closed = False

    async def generate(self, prompt: str) -> GeneratedImage:
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def aclose(self) -> None:
        self.closed = True


class Collector:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    async def __call__(self, record: CallRecord) -> None:
        self.records.append(record)


def files(store: Store, tmp_path: Path, **kwargs: Any) -> Attachments:
    return AttachmentSettings(enabled=True, **kwargs).build(store, tmp_path / "siatt.db")


def asking(**kwargs: Any) -> ToolContext:
    """A turn on a surface that can send files, unless a test says otherwise."""
    return ToolContext(session_id="s1", can_send_files=True, **kwargs)


async def draw(
    attachments: Attachments,
    drawer: FakeDrawer,
    context: ToolContext,
    *,
    prompt: str = "a red circle on white",
    meter: CostMeter | None = None,
) -> str:
    registry = ToolRegistry(
        [image_generate_tool(provider=drawer, attachments=attachments, meter=meter)]
    )
    result = await registry.dispatch(
        ToolUseBlock(id="t1", name="image_generate", input={"prompt": prompt}), context
    )
    return result.content


# -- before anything is spent ------------------------------------------------


async def test_a_surface_that_cannot_carry_a_file_is_refused_before_drawing(
    store: Store, tmp_path: Path
) -> None:
    """An answer that says "here it is" where nothing can be sent is the failure
    this exists to prevent, and it costs a drawing to discover it late."""
    drawer = FakeDrawer()

    said = await draw(files(store, tmp_path), drawer, ToolContext(session_id="s1"))

    assert said == CANNOT_SEND
    assert drawer.prompts == [], "the endpoint was called on a turn that cannot deliver"


async def test_an_answer_that_is_already_full_is_refused_before_drawing(
    store: Store, tmp_path: Path
) -> None:
    drawer = FakeDrawer()
    context = asking(outgoing=["a" * 64] * MAX_SENT)

    said = await draw(files(store, tmp_path), drawer, context)

    assert said == FULL
    assert drawer.prompts == []


async def test_the_daily_ceiling_is_checked_before_drawing_not_after(
    store: Store, tmp_path: Path
) -> None:
    """A request that has been billed cannot be unspent."""
    drawer = FakeDrawer()
    meter = CostMeter(
        PriceBook(),
        sink=Collector(),
        daily_usd_ceiling=1.0,
        spent_since=_spent(2.0),
    )

    said = await draw(files(store, tmp_path), drawer, asking(), meter=meter)

    assert said == OVER_BUDGET
    assert drawer.prompts == []


async def test_a_prompt_with_nothing_in_it_is_refused(store: Store, tmp_path: Path) -> None:
    drawer = FakeDrawer()

    said = await draw(files(store, tmp_path), drawer, asking(), prompt="   ")

    assert "needs a description" in said
    assert drawer.prompts == []


# -- what it makes -----------------------------------------------------------


async def test_the_picture_is_kept_and_goes_out_with_this_answer(
    store: Store, tmp_path: Path
) -> None:
    attachments = files(store, tmp_path)
    context = asking()

    said = await draw(attachments, FakeDrawer(), context)

    assert len(context.outgoing) == 1
    held = await attachments.get(context.outgoing[0])
    assert held is not None
    assert held.mime == "image/png"
    assert await attachments.read(held.sha256) == png()
    assert "going out with this answer" in said
    assert "Nothing has been sent yet" in said


async def test_the_file_is_named_after_the_prompt(store: Store, tmp_path: Path) -> None:
    """Four drawings in a thread all called `attachment.png` are four files
    nobody can tell apart."""
    attachments = files(store, tmp_path)
    context = asking()

    said = await draw(attachments, FakeDrawer(), context, prompt="A red circle on white paper")

    held = await attachments.get(context.outgoing[0])
    assert held is not None
    assert held.name == "a-red-circle-on-white-paper.png"
    assert "a-red-circle-on-white-paper.png" in said


async def test_the_name_is_rebuilt_from_the_prompt_rather_than_sliced_out_of_it(
    store: Store, tmp_path: Path
) -> None:
    """The prompt is the model's own text, and a filename is somewhere text is
    shown. A separator, a newline or a leading dot must not survive into one."""
    attachments = files(store, tmp_path)
    context = asking()

    await draw(attachments, FakeDrawer(), context, prompt="../../etc/passwd\nand a dot .")

    held = await attachments.get(context.outgoing[0])
    assert held is not None
    assert held.name == "etc-passwd-and-a-dot.png"


async def test_the_picture_is_recorded_as_arriving_in_this_conversation(
    store: Store, tmp_path: Path
) -> None:
    """A drawing is an arrival like any other, so it is citable exactly where it
    was made and nowhere else."""
    attachments = files(store, tmp_path)
    context = asking()

    await draw(attachments, FakeDrawer(), context)

    sha = context.outgoing[0]
    assert [r["sha256"] for r in await store.attachments_for_session("s1")] == [sha]
    assert await store.attachments_for_session("s2") == []


async def test_it_is_not_also_put_in_front_of_the_model(store: Store, tmp_path: Path) -> None:
    """It costs tokens by area, and the model asked for it — it knows."""
    context = asking()

    await draw(files(store, tmp_path), FakeDrawer(), context)

    assert context.surfaced == []


# -- what it is not: evidence ------------------------------------------------


async def test_what_it_draws_is_stored_as_generated(store: Store, tmp_path: Path) -> None:
    attachments = files(store, tmp_path)
    context = asking()

    await draw(attachments, FakeDrawer(), context)

    rows = await store.attachments_for_session("s1")
    assert [row["source"] for row in rows] == [GENERATED]


async def test_a_generated_picture_cannot_be_cited_but_can_still_be_sent(
    store: Store, tmp_path: Path
) -> None:
    """The whole of the rule, in the two functions that decide it. A memory
    citing an image Siatt invented is a fabricated exhibit; sending the same
    file again is just sending a file."""
    attachments = files(store, tmp_path)
    context = asking()
    await draw(attachments, FakeDrawer(), context)
    sha = context.outgoing[0]

    rows = await store.attachments_for_session("s1")

    assert citable(rows) == {}
    assert sha in named(rows)

    # And `send_file` resolves it, on a fresh turn that has not sent anything.
    later = asking()
    sending = ToolRegistry(file_tools(store=store, attachments=attachments))
    said = await sending.dispatch(
        ToolUseBlock(id="t2", name="send_file", input={"ids": [handle(sha)]}), later
    )
    assert later.outgoing == [sha]
    assert "going out with this answer" in said.content


# -- accounting --------------------------------------------------------------


async def test_the_call_is_metered_with_the_tokens_the_endpoint_reported(
    store: Store, tmp_path: Path
) -> None:
    collector = Collector()
    meter = CostMeter(PriceBook({"openai/gpt-image": Price(output=40.0)}), sink=collector)

    await draw(files(store, tmp_path), FakeDrawer(), asking(), meter=meter)

    [record] = collector.records
    assert record.role == "image"
    assert record.tag == "image_generate"
    assert record.usage == USAGE
    assert record.ok
    assert record.cost_usd == pytest.approx(272 * 40.0 / 1_000_000)


async def test_a_failed_drawing_is_recorded_and_reaches_the_model_as_an_error(
    store: Store, tmp_path: Path
) -> None:
    collector = Collector()
    meter = CostMeter(PriceBook(), sink=collector)
    drawer = FakeDrawer(AuthError("the key was rejected", provider="fake", status=401))
    context = asking()

    registry = ToolRegistry(
        [image_generate_tool(provider=drawer, attachments=files(store, tmp_path), meter=meter)]
    )
    result = await registry.dispatch(
        ToolUseBlock(id="t1", name="image_generate", input={"prompt": "x"}), context
    )

    assert result.is_error
    assert "AuthError" in result.content
    assert context.outgoing == []
    [record] = collector.records
    assert not record.ok
    assert record.error is not None


# -- an install that cannot keep what it drew --------------------------------


async def test_a_picture_that_cannot_be_kept_is_not_claimed_to_be_sent(
    store: Store, tmp_path: Path
) -> None:
    """`allowed_mime` narrowed to video, or a cap smaller than the drawing. The
    money is gone either way; the answer must not also promise a file."""
    attachments = files(store, tmp_path, allowed_mime=("video/",))
    context = asking()

    said = await draw(attachments, FakeDrawer(), context)

    assert "could not be kept" in said
    assert "nothing was sent" in said
    assert context.outgoing == []


def _spent(usd: float) -> Any:
    async def spend(_since: str) -> float:
        return usd

    return spend
