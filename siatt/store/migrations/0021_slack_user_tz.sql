-- #223: the timezone a relative day is relative to.
--
-- `어제` means the speaker's yesterday. Resolving it in UTC is wrong for the
-- first nine hours of every day at UTC+9, which is when a good deal of the
-- conversation happens.
--
-- `users.info` has been returning this all along — an IANA name like
-- `Asia/Seoul` — and `read_user` was dropping it. So it rides the cache and
-- the TTL that are already here, and costs no extra call: a workspace where
-- somebody travels picks the new zone up within `DEFAULT_TTL`.
--
-- Empty rather than NULL for "Slack did not say", so the column reads the same
-- as `real_name` beside it and every existing row is immediately valid.
-- `today_in` treats empty and unknown alike, and answers UTC for both.

ALTER TABLE slack_users ADD COLUMN tz TEXT NOT NULL DEFAULT '';
