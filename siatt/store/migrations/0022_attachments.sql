-- Somewhere to put the bytes people send.
--
-- Until now a file arriving in Slack was named and thrown away: ingress
-- appended "[attached, which Siatt cannot open: ...]" so the agent could say
-- honestly what it was missing, and nothing was kept. `messages.content` is a
-- JSON array of content blocks and long-term memory is a git repository of
-- Markdown; neither is a home for a photograph.
--
-- Long-term memory is the wrong home for a reason stronger than size. It is a
-- corpus a person reads on GitHub, rewritten wholesale by `reorganize`,
-- `forget` and `consolidate`. A binary there would live in git history forever
-- after `forget` deleted it -- which quietly defeats the forgetting -- and
-- would force every curation job to reason about files it cannot read.
--
-- The bytes are not in here either. A 40MB video as a row makes backups, WAL
-- churn and every incautious SELECT expensive, and the filesystem wins well
-- before that size. So the split is: the filesystem holds the bytes, keyed by
-- their own hash, and this database is the index -- and the authority on who
-- may see them, which is the half that leaks when it is wrong.

-- One row per distinct set of bytes. Keyed by the hash, so the same picture
-- pasted twice is one blob, and an inbox row that replays after a restart
-- re-writes nothing.
--
-- `bytes` is the size on disk, not the content; the content is at
-- <db parent>/blobs/<sha256[:2]>/<sha256>.
CREATE TABLE attachments (
    sha256     TEXT PRIMARY KEY,
    mime       TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

-- One row per arrival: where a blob came from, and who may see it.
--
-- Scope lives here rather than on the blob, and that is the whole point of
-- there being two tables. The same picture sent in a DM and posted in a public
-- channel is one set of bytes with two visibilities, and the wider arrival must
-- not widen the narrower one -- deduplicating on content would otherwise make
-- "someone already shared this in #general" into a way to read a DM. Every read
-- path filters on this column before it returns anything, the way retrieval
-- already does for chunks.
--
-- `message_id` cascades: a ref exists because of the message it arrived on, and
-- `clear_session` should not leave it behind. It is nullable because the fetch
-- can land before the turn appends its message, and a ref with no message is
-- one the collector will deal with on age.
--
-- `session_id` is deliberately a plain column with no foreign key. It is here to
-- explain a blob, not to constrain it, and a ref must never fail to be written
-- because a session row has not been created yet.
CREATE TABLE attachment_refs (
    id          TEXT PRIMARY KEY,   -- ULID
    sha256      TEXT NOT NULL REFERENCES attachments(sha256),
    source      TEXT NOT NULL,      -- 'slack' | 'cli' | 'http'
    -- The surface's own id for the message it came on. Part of the arrival key
    -- below, which is what makes a redelivery one ref rather than two.
    external_id TEXT,
    session_id  TEXT,
    message_id  TEXT REFERENCES messages(id) ON DELETE CASCADE,
    author      TEXT,
    scope       TEXT NOT NULL DEFAULT 'workspace',
    -- What the surface called it. Shown to a person and to the model; never
    -- used as a path, because it is somebody else's text.
    name        TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX attachment_refs_sha ON attachment_refs (sha256);
CREATE INDEX attachment_refs_message ON attachment_refs (message_id);

-- At-most-one ref per file per delivery. The same trick as UNIQUE (source,
-- external_id) on the inbox: an at-least-once queue plus this is at-most-once
-- in effect, without anything having to remember whether it already ran.
-- Partial, because a ref written outside a surface's delivery -- the CLI, a
-- test -- has no external id to be unique on.
CREATE UNIQUE INDEX attachment_refs_arrival
    ON attachment_refs (source, external_id, sha256)
    WHERE external_id IS NOT NULL;
