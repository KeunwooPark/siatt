-- One person, one pool of memory (#265).
--
-- Visibility scoping was built for the failure mode of a *shared* memory bot:
-- repeating something from a DM in a public channel (`docs/DESIGN.md` §11.1).
-- Siatt serves one person, so there was never a second audience for it to
-- protect, and what it did instead was hide the user's own facts from the user
-- -- a preference stated in one channel was invisible from the next, and the
-- turn that could not see it answered "nothing is stored".
--
-- Every gate is gone from the code. This is the corresponding backfill: rows
-- written under a narrower scope become `workspace`, so nothing already in the
-- store stays partitioned by a boundary nothing enforces any more.
--
-- The columns stay. They record what a row was written under, which is a fact
-- about history rather than a permission, and re-threading a field is a much
-- smaller job than re-deriving one from a corpus that never recorded it. The
-- `visibility` field on the memory documents themselves is rewritten
-- separately, through the patch path, because those live in git and a commit
-- is how they are meant to change.
UPDATE sessions        SET scope = 'workspace' WHERE scope != 'workspace';
UPDATE observations    SET scope = 'workspace' WHERE scope != 'workspace';
UPDATE tasks           SET scope = 'workspace' WHERE scope != 'workspace';
UPDATE answers         SET scope = 'workspace' WHERE scope != 'workspace';
UPDATE reviews         SET scope = 'workspace' WHERE scope != 'workspace';
UPDATE attachment_refs SET scope = 'workspace' WHERE scope != 'workspace';

-- `chunks` is derived from the corpus and rebuilt by the indexer, but it is
-- what retrieval actually reads: leaving it stale would keep the old scopes on
-- every row until the next full reindex.
UPDATE chunks          SET scope = 'workspace' WHERE scope != 'workspace';
