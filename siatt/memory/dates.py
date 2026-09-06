"""Relative days in a question, resolved to the dates memories are written with.

A memory records the day it is about; a person asks about it with a word. The
memory says `2026-09-05` and the question says `어제`, and because every term
`build_match` produces is a literal, those two never share a token. The memory
that answers the question is not ranked low — it is absent from the pool.

That is #221, and it is the failure mode a diary is most exposed to. "What did
I do yesterday?" names no other noun, so there is nothing else for the lexical
search to catch the memory by, and the one question with no fallback is the one
people actually ask a diary.

Neither the rewriter nor an embedder rescues this. `is_self_contained` is true
for "내가 어제 어디갔는지 기억해?" — four words, no anaphor — so the rewriter is
never called, and a model asked to rewrite text has no reason to know today's
date anyway. The resolution has to happen where the date is known, which is
here.

**The output is phrases, not terms.** `build_match` ORs what it is given, so
`9월` on its own would pull in every memory written in September and rank it on
a token the question never asked about. `"9월 5일"` matches the day and nothing
else. This is the one place the module departs from `retrieve`'s "OR it and let
bm25 discriminate" rule, and it departs because these terms are not the
person's words — they are inferred, and inferred terms should not be able to
outvote the ones actually typed.

**Which day is "today" is the caller's to say.** `today_in` turns a zone name
into the date it is there now, and everything above works from that. Resolving
in UTC was #223: at UTC+9 every conversation before 09:00 local resolved `어제`
to the day before the one meant, so #221 worked only after lunch. The zone
comes from the speaker's Slack profile, because `어제` means *their* yesterday
and nobody else's.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: Relative day words and the offset each names, longest first so that "day
#: before yesterday" is consumed before the "yesterday" inside it.
#:
#: Korean entries are matched as substrings because Korean glues its particles
#: on: 어제, 어젠, 어제는 and 어제부터 are all the same word to a reader and four
#: different tokens to FTS. English entries are matched on word boundaries,
#: where that gluing does not happen and a substring search would find "today"
#: inside a longer word.
_RELATIVE_DAYS: tuple[tuple[str, int], ...] = (
    ("그끄저께", -3),
    ("그저께", -2),
    ("그제", -2),
    ("어저께", -1),
    ("어제", -1),
    ("오늘", 0),
    ("내일", 1),
    ("모레", 2),
    ("글피", 3),
    ("day before yesterday", -2),
    ("yesterday", -1),
    ("today", 0),
    ("tomorrow", 1),
)

#: Counted days: "3일 전", "이틀 전", "5 days ago". The native-Korean numerals
#: are listed rather than parsed because only the first handful are ever used
#: this way — past 나흘 people switch to Sino-Korean digits, which the numeric
#: pattern already reads.
_COUNTED: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"하루\s*전"), 1),
    (re.compile(r"이틀\s*전"), 2),
    (re.compile(r"사흘\s*전"), 3),
    (re.compile(r"나흘\s*전"), 4),
)

_NUMBERED: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d{1,3})\s*일\s*전"),
    re.compile(r"(\d{1,3})\s+days?\s+ago"),
)

#: Past this, a number followed by "days ago" is not a day somebody remembers,
#: and expanding it only adds phrases nothing will match.
_MAX_DAYS_AGO = 366

#: How many distinct days one question may expand to. A question naming more
#: than a few is not asking about a day, and the bound keeps a pathological
#: message from building a MATCH expression out of hundreds of phrases.
_MAX_DAYS = 4

#: Month names for the English form, spelled out rather than taken from
#: `strftime("%B")`, which answers in whatever locale the process happens to
#: run under. A memory written in English says "September"; a MATCH built on a
#: German host must not go looking for "September" spelled "Septembers".
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

#: Left alone by the scrubber below, which blanks each match so a longer
#: expression cannot be re-read as the shorter one inside it. A space keeps the
#: word boundaries either side intact.
_BLANK = " "

_ENGLISH = re.compile(r"[a-z]")

#: Hangul syllable arithmetic. A syllable is `0xAC00 + (initial*21 + medial)*28
#: + final`, so a syllable written without a final consonant sits at a multiple
#: of 28 and the 27 code points above it are that same syllable carrying one.
_HANGUL_BASE = 0xAC00
_HANGUL_COUNT = 11172
_FINALS = 28


def _pattern(word: str) -> str:
    """A regex matching `word` as it is actually written in a sentence.

    English gets word boundaries. Korean does not, because it glues its
    particles on: 어제, 어제는 and 어제부터 are one word to a reader and three
    tokens to FTS, and a substring search finds all three.

    A substring search is not quite enough, though. When the particle begins
    with a consonant that can be a final, Korean absorbs it into the preceding
    syllable rather than adding one: 어제 + 는 is written 어젠, which does not
    contain 어제 at all. So a word ending in a final-less syllable is matched
    with that syllable widened to the 28 that share its opening — which also
    picks up 어젯밤, "last night", for free.
    """
    if _ENGLISH.search(word):
        return rf"\b{re.escape(word)}\b"
    last = ord(word[-1]) - _HANGUL_BASE
    if 0 <= last < _HANGUL_COUNT and last % _FINALS == 0:
        first = _HANGUL_BASE + last
        return re.escape(word[:-1]) + f"[{chr(first)}-{chr(first + _FINALS - 1)}]"
    return re.escape(word)


def date_phrases(query: str, today: date) -> list[str]:
    """Absolute date phrases for every relative day `query` names.

    Returns phrases in the several forms a memory might have been written in —
    `2026-09-05`, `2026년 9월 5일`, `9월 5일`, `September 5` — because the
    assistant writes a memory in the language of the conversation that produced
    it, and the conversation asking about it later may be in the other one.

    Empty for a question that names no day, which is nearly all of them.
    """
    days = _offsets(query.lower())
    seen: list[str] = []
    for offset in days[:_MAX_DAYS]:
        for phrase in _forms(today + timedelta(days=offset)):
            if phrase not in seen:
                seen.append(phrase)
    return seen


def _offsets(query: str) -> list[int]:
    """Day offsets named in `query`, in the order they are named.

    Each match is blanked out as it is taken, so "the day before yesterday"
    yields -2 alone and not -2 and -1 both.
    """
    found: list[int] = []
    for word, offset in _RELATIVE_DAYS:
        query, hits = re.subn(_pattern(word), _BLANK, query)
        if hits:
            found.append(offset)
    for counted, days in _COUNTED:
        query, hits = counted.subn(_BLANK, query)
        if hits:
            found.append(-days)
    for numbered in _NUMBERED:
        for match in numbered.finditer(query):
            days = int(match.group(1))
            if 0 < days <= _MAX_DAYS_AGO:
                found.append(-days)
    return list(dict.fromkeys(found))


def _forms(day: date) -> tuple[str, ...]:
    """The ways this day might be spelled in a memory.

    `9월 5일` is here without its year because that is how a sentence written
    inside a conversation refers to a day in the current one, and it is still a
    phrase of two tokens rather than a loose `9월` — precise enough to be worth
    the small risk of catching the same day a year out.
    """
    return (
        day.isoformat(),
        f"{day.year}년 {day.month}월 {day.day}일",
        f"{day.month}월 {day.day}일",
        f"{_MONTHS[day.month - 1]} {day.day}",
    )


def today_in(zone: str | tzinfo | None, *, now: datetime | None = None) -> date:
    """What day it is in `zone` — UTC if it does not name one.

    Never raises. An IANA name reaches this from a Slack profile, on the turn
    path, and a workspace where somebody has a zone this build's tzdata has
    never heard of is a workspace that answers questions slightly wrong — not
    one that fails the turn. An unknown zone is the same answer as no zone.

    A `tzinfo` is accepted as well as a name because the local zone cannot
    survive the trip through one: `datetime.now().astimezone().tzname()` says
    "KST", and `ZoneInfo("KST")` does not exist. `local_zone` hands the object
    over instead of a string nothing can look up.
    """
    moment = now or datetime.now(UTC)
    return moment.astimezone(_zone(zone)).date()


def _zone(zone: str | tzinfo | None) -> tzinfo:
    if isinstance(zone, tzinfo):
        return zone
    if not zone:
        return UTC
    try:
        return ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError):
        return UTC


def local_zone() -> tzinfo | None:
    """This machine's zone, for the surfaces with no profile to ask.

    `siatt why` and the CLI have no Slack user behind them, and the person at
    the terminal means their own yesterday exactly as much as somebody typing
    in a channel does. `None` when the platform will not say, which `today_in`
    reads as UTC.
    """
    return datetime.now().astimezone().tzinfo
