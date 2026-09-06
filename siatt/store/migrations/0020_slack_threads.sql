-- Threads Siatt started, and the conversation each one belongs to.
--
-- Ingress derives a session id from the thread a message is in, and answers in
-- a channel only when it is mentioned or when that session already exists. A
-- thread Siatt opened itself satisfies neither: nobody mentioned anything, and
-- the session id under the *new* thread's timestamp has never been seen — so
-- without this table, replies under a standing task's morning post are read as
-- chatter and ignored, which is the one thing that would make it useless.
--
-- One row per message Siatt posted at top level, written once the timestamp
-- comes back from Slack. It maps that thread to the session that produced it,
-- so a reply continues the conversation the post came out of rather than
-- opening an empty one beside it.
--
-- Keyed by the thread rather than by the task: what ingress has in its hand is
-- a channel and a thread timestamp, and the lookup is on the three-second ack
-- path. A row is small and permanent — a thread stays answerable for as long
-- as people reply in it, and there is no point at which "this is one of ours"
-- stops being true.

CREATE TABLE slack_threads (
    team_id    TEXT NOT NULL,
    channel    TEXT NOT NULL,
    thread_ts  TEXT NOT NULL,   -- the root message: the thread's own id
    session_id TEXT NOT NULL,   -- the conversation that posted it
    created_at TEXT NOT NULL,
    PRIMARY KEY (team_id, channel, thread_ts)
);
