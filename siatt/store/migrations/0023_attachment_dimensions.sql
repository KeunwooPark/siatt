-- How big a picture is, so the packer can budget for it.
--
-- An image costs tokens by area -- both provider families charge roughly
-- width x height / 750 -- and an `ImageBlock` costed by the length of its JSON
-- is costed at nearly nothing. The packer would then assemble a context that
-- looks like it fits and the provider would answer 400, which is the one
-- failure `siatt/core/context.py` exists to prevent.
--
-- Read once, from the header, at the moment the bytes are stored: the file is
-- in hand exactly then, and re-reading it on every turn to count tokens would
-- put a filesystem read inside the packer's loop.
--
-- Null for anything whose header we do not parse -- HEIC, a format that does
-- not exist yet, a video -- and null is not "unknown, assume small": the
-- packer charges the ceiling for it. Guessing small is the answer that costs a
-- turn.

ALTER TABLE attachments ADD COLUMN width INTEGER;
ALTER TABLE attachments ADD COLUMN height INTEGER;
