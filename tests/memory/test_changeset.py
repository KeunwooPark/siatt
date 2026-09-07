"""What a change list leaves behind, and whether it is still a corpus."""

from __future__ import annotations

import pytest

from siatt.memory.changeset import collisions, project
from siatt.memory.document import MemoryDoc
from siatt.memory.ltm import Remove, Write
from siatt.memory.manifest import Manifest


@pytest.fixture
def jane() -> MemoryDoc:
    return MemoryDoc.new(type="person", title="Jane", body="Owns deploys.")


@pytest.fixture
def deploy() -> MemoryDoc:
    return MemoryDoc.new(type="project", title="Deploy pipeline", body="Ships on Thursdays.")


def manifest_of(*entries: tuple[str, MemoryDoc]) -> Manifest:
    manifest = Manifest()
    for path, doc in entries:
        manifest.record(path, doc, checksum="seeded")
    return manifest


# -- projection --------------------------------------------------------------


def test_a_write_lands_in_the_projection(jane: MemoryDoc) -> None:
    after = project(Manifest(), [Write("memory/people/jane.md", jane.render())])

    assert after.path_of(jane.id) == "memory/people/jane.md"


def test_a_remove_takes_the_memory_out(jane: MemoryDoc) -> None:
    before = manifest_of(("memory/people/jane.md", jane))

    after = project(before, [Remove("memory/people/jane.md")])

    assert jane.id not in after
    assert jane.id in before, "the manifest handed in is not modified"


def test_an_archive_moves_the_memory(jane: MemoryDoc) -> None:
    """The pair `_archive` emits, in the order it emits them."""
    before = manifest_of(("memory/people/jane.md", jane))

    after = project(
        before,
        [Write("memory/archive/jane.md", jane.render()), Remove("memory/people/jane.md")],
    )

    assert after.path_of(jane.id) == "memory/archive/jane.md"


def test_a_write_that_does_not_parse_is_left_alone(jane: MemoryDoc) -> None:
    before = manifest_of(("memory/people/jane.md", jane))

    after = project(before, [Write("memory/people/jane.md", "not a memory at all")])

    assert after.path_of(jane.id) == "memory/people/jane.md"


# -- the invariant -----------------------------------------------------------


def test_an_ordinary_run_collides_with_nothing(jane: MemoryDoc, deploy: MemoryDoc) -> None:
    before = manifest_of(("memory/people/jane.md", jane))

    assert not collisions(
        before,
        [
            Write("memory/people/jane.md", jane.render()),
            Write("memory/projects/deploy.md", deploy.render()),
        ],
    )


def test_an_archive_alone_collides_with_nothing(jane: MemoryDoc) -> None:
    """The write comes first and the file it replaces is still there when it
    does. Order is the whole question, so this is worth stating."""
    before = manifest_of(("memory/people/jane.md", jane))

    assert not collisions(
        before,
        [Write("memory/archive/jane.md", jane.render()), Remove("memory/people/jane.md")],
    )


def test_an_archive_and_an_update_in_one_commit_are_caught(jane: MemoryDoc) -> None:
    """`8f5911a`, reduced: one group archived the memory, another rewrote the
    file it had just been moved out of, and the corpus ended up unindexable."""
    before = manifest_of(("memory/facts/news.md", jane))

    broken = collisions(
        before,
        [
            Write("memory/archive/news.md", jane.render()),
            Remove("memory/facts/news.md"),
            Write("memory/facts/news.md", jane.render()),
        ],
    )

    assert broken == {jane.id: ["memory/archive/news.md", "memory/facts/news.md"]}


def test_a_write_beside_a_memory_the_run_never_touches_is_caught(jane: MemoryDoc) -> None:
    """Nothing in the change list is wrong on its own; the second path only
    duplicates an id that was already in the corpus."""
    before = manifest_of(("memory/people/jane.md", jane))

    broken = collisions(before, [Write("memory/facts/jane-again.md", jane.render())])

    assert broken == {jane.id: ["memory/facts/jane-again.md", "memory/people/jane.md"]}


def test_two_new_files_sharing_one_id_are_caught(jane: MemoryDoc) -> None:
    broken = collisions(
        Manifest(),
        [Write("memory/facts/a.md", jane.render()), Write("memory/facts/b.md", jane.render())],
    )

    assert broken == {jane.id: ["memory/facts/a.md", "memory/facts/b.md"]}


def test_rewriting_a_path_with_another_memory_moves_the_path(
    jane: MemoryDoc, deploy: MemoryDoc
) -> None:
    """The old id no longer lives there, so it does not count as a second home
    for it. A file is owned by whatever was written to it last."""
    before = manifest_of(("memory/facts/x.md", jane))

    assert not collisions(before, [Write("memory/facts/x.md", deploy.render())])


def test_an_unparseable_write_is_not_guessed_at(jane: MemoryDoc) -> None:
    """Nothing can be said about an id it does not carry. The file is somebody
    else's problem — `Manifest.rebuild` reports it — not a false collision."""
    before = manifest_of(("memory/people/jane.md", jane))

    assert not collisions(before, [Write("memory/facts/broken.md", "no frontmatter here")])


def test_a_manifest_cannot_answer_this_itself(jane: MemoryDoc) -> None:
    """Why `collisions` exists rather than a check over `project`: a manifest
    maps an id to *one* path, so projecting the same change list keeps the last
    write and the duplicate is gone from the structure you would ask."""
    before = manifest_of(("memory/facts/news.md", jane))
    changes = [
        Write("memory/archive/news.md", jane.render()),
        Remove("memory/facts/news.md"),
        Write("memory/facts/news.md", jane.render()),
    ]

    assert project(before, changes).path_of(jane.id) == "memory/facts/news.md"
    assert collisions(before, changes)
