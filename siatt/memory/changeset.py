"""What a list of file changes means for the corpus it will land in.

Two questions come up wherever a job assembles several independently compiled
plans into one commit: what the manifest looks like once the changes are on
disk, and whether what they leave behind is still a corpus.

Both are answered here rather than inside a runner because both runners need
them, and because the invariant the second one checks — *one memory id lives at
one path* — is the one the whole corpus rests on. `Manifest` is a mapping from
id to a single path, retrieval resolves links through it, and the indexer keys
its chunks on `<memory id>:<ordinal>`. Nothing downstream has anywhere to put a
second file with the same id, so each of them fails in its own way when one
appears: the manifest reports it as a broken file, the indexer dies on a
`UNIQUE constraint`. The cheapest place to notice is before the commit.
"""

from __future__ import annotations

from collections.abc import Sequence

from siatt.memory.document import MemoryDoc, MemoryError_
from siatt.memory.ltm import Change, Write
from siatt.memory.manifest import Manifest


def project(manifest: Manifest, changes: Sequence[Change]) -> Manifest:
    """The manifest as it will be once `changes` are on disk.

    Used two ways. `reorganize` regenerates its index pages from this rather
    than from what is currently there, so a run that merges two memories does
    not publish a listing naming both of them until the following week.
    `promote` compiles each group against it, so the second group plans against
    the corpus the first one leaves behind instead of the one they both started
    from.
    """
    after = manifest.model_copy(deep=True)
    for change in changes:
        if isinstance(change, Write):
            try:
                doc = MemoryDoc.parse(change.content, source=change.path)
            except MemoryError_:
                continue
            after.record(change.path, doc, checksum="pending")
        elif (memory_id := after.id_at(change.path)) is not None:
            after.forget(memory_id)
    return after


def collisions(manifest: Manifest, changes: Sequence[Change]) -> dict[str, list[str]]:
    """Memory ids that `changes` would leave living at more than one path.

    Replayed in order over the paths the manifest already knows, because order
    is what decides it: an archive emits `[Write(archive/x.md), Remove(y.md)]`,
    and a later `Write(y.md)` for the same memory puts back the file the remove
    took away. Both land in one commit and the corpus ends up with two files
    carrying one id (#239).

    A `Manifest` cannot represent the answer — it maps an id to *one* path, so
    `project` above silently keeps the last write and the duplicate disappears
    from the very structure you would ask. Hence the separate walk, and hence
    ids to paths rather than the other way round.

    A `Write` whose content does not parse is skipped rather than guessed at.
    Nothing here rejects a plan on its own; it reports, and the caller decides.
    """
    paths: dict[str, set[str]] = {
        memory_id: {entry.path} for memory_id, entry in manifest.memories.items()
    }
    for change in changes:
        for owned in paths.values():
            owned.discard(change.path)
        if not isinstance(change, Write):
            continue
        try:
            doc = MemoryDoc.parse(change.content, source=change.path)
        except MemoryError_:
            continue
        paths.setdefault(doc.id, set()).add(change.path)
    return {
        memory_id: sorted(owned) for memory_id, owned in sorted(paths.items()) if len(owned) > 1
    }
