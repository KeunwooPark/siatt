"""Where a message too long for Slack is cut.

A string in and a list of strings out, so the rule is checkable without a
socket, a `LiveMessage`, or an opinion about which Slack endpoint refuses what.
"""

from __future__ import annotations

import pytest

from siatt.adapters.slack.limits import MAX_TEXT, split


def test_something_that_fits_is_one_message() -> None:
    assert split("It was Tuesday.", 100) == ["It was Tuesday."]


def test_nothing_to_say_is_nothing_to_post() -> None:
    """Slack will not take an empty message, so the caller must not be handed
    one to try."""
    assert split("", 100) == []
    assert split("   \n\n  \t ", 100) == []


def test_every_character_survives_the_cut() -> None:
    """Cutting is not truncation. What changes is which message carries a
    character, never whether one goes at all."""
    text = "\n\n".join(f"item {n} " + "x" * 40 for n in range(20))
    parts = split(text, 100)

    assert len(parts) > 1
    assert "".join(part.replace("\n", "") for part in parts).replace(" ", "") == text.replace(
        "\n", ""
    ).replace(" ", "")


def test_no_part_is_over_the_limit() -> None:
    text = " ".join(f"word{n}" for n in range(500))
    assert all(len(part) <= 40 for part in split(text, 40))


def test_the_break_goes_between_paragraphs_when_one_is_in_reach() -> None:
    """A digest of five items should split between items. That is the whole
    reason the separators are ranked rather than "wherever it stops fitting"."""
    text = "one one one\n\ntwo two two\n\nthree three three"

    assert split(text, 26) == ["one one one\n\ntwo two two", "three three three"]


def test_a_line_break_will_do_when_a_paragraph_will_not() -> None:
    text = "alpha line\nbravo line\ncharlie line"

    assert split(text, 22) == ["alpha line\nbravo line", "charlie line"]


def test_and_a_word_break_when_nothing_else_will() -> None:
    assert split("alpha bravo charlie delta", 12) == ["alpha bravo", "charlie", "delta"]


def test_one_unbroken_run_is_cut_where_it_stops_fitting() -> None:
    """A URL, or a language that does not put spaces between words. There is
    no good break, and refusing to make a bad one means sending nothing."""
    assert split("a" * 25, 10) == ["a" * 10, "a" * 10, "a" * 5]


def test_a_break_at_the_very_start_is_not_taken() -> None:
    """It would produce an empty message and leave the text no shorter, which
    is how a loop like this one runs forever."""
    assert split("x" + " " + "y" * 30, 12) == ["x", "y" * 12, "y" * 12, "y" * 6]


def test_a_break_sitting_exactly_on_the_limit_is_found() -> None:
    """The character after a full message is where the next one starts, so it
    has to be looked at — an off-by-one here costs a clean break."""
    assert split("123456789 rest", 9) == ["123456789", "rest"]


def test_a_limit_of_nothing_is_a_mistake_rather_than_a_hang() -> None:
    with pytest.raises(ValueError):
        split("It was Tuesday.", 0)


def test_the_default_is_well_under_what_slack_publishes() -> None:
    """40,000 is the documented ceiling on `text` and not the only limit that
    applies. See the module docstring for why aiming low costs nothing."""
    assert MAX_TEXT <= 4000
