"""How a memory points at a file, and what happens to a pointer nobody stored."""

from __future__ import annotations

from siatt.memory.blobref import HANDLE_CHARS, handle, link, matching, prune, referenced, uri

REAL = "a" * 64
INVENTED = "b" * 64


def test_a_reference_is_an_ordinary_markdown_link() -> None:
    """The corpus is Markdown a person reads on GitHub. A reference that needed
    its own syntax would be a reference that broke every reader."""
    assert link(REAL, "shot.png") == f"[shot.png]({uri(REAL)})"


def test_the_name_is_what_a_person_reads() -> None:
    """A bare URI is sixty-four characters of noise in the middle of a
    sentence, which is why the text is half of the reference."""
    assert "shot.png" in link(REAL, "shot.png")


def test_referenced_finds_every_pointer() -> None:
    body = f"See {link(REAL, 'a.png')} and also {link(INVENTED, 'b.png')}."

    assert referenced(body) == {REAL, INVENTED}


def test_a_hash_that_is_not_sixty_four_hex_is_not_a_reference() -> None:
    """It must never become a path, and the digest is the only part that could."""
    assert referenced("[x](siatt://blob/../../etc/passwd)") == set()
    assert referenced(f"[x](siatt://blob/{'a' * 63})") == set()
    assert referenced(f"[x](siatt://blob/{'a' * 65})") == set()
    assert referenced("[x](siatt://blob/NOTHEX" + "a" * 58 + ")") == set()


# -- pruning -----------------------------------------------------------------


def test_an_invented_reference_is_unwrapped_not_deleted() -> None:
    """A model that has read one of these will compose another. What it claimed
    to see may well be true; what it may not do is leave a pointer to evidence
    that is not there."""
    body = f"The board says {link(INVENTED, 'a photo of the whiteboard')} to ship on Friday."

    pruned, dropped = prune(body, {REAL})

    assert pruned == "The board says a photo of the whiteboard to ship on Friday."
    assert dropped == {INVENTED}


def test_a_real_reference_survives() -> None:
    body = f"See {link(REAL, 'shot.png')}."

    pruned, dropped = prune(body, {REAL})

    assert pruned == body
    assert dropped == set()


def test_one_invented_reference_does_not_take_the_others_with_it() -> None:
    body = f"{link(REAL, 'real.png')} and {link(INVENTED, 'fake.png')}"

    pruned, _ = prune(body, {REAL})

    assert link(REAL, "real.png") in pruned
    assert INVENTED not in pruned
    assert "fake.png" in pruned


def test_nothing_known_prunes_everything() -> None:
    """An install that keeps no attachments cannot have blobs, so every
    reference in a memory it writes is one nothing can resolve."""
    body = f"See {link(REAL, 'shot.png')}."

    pruned, dropped = prune(body, set())

    assert pruned == "See shot.png."
    assert dropped == {REAL}


def test_text_that_would_close_the_link_early_is_neutralized() -> None:
    """The name came off an upload, so somebody else chose it."""
    written = link(REAL, "not[really]a name")

    assert referenced(written) == {REAL}
    assert "[" not in written[1 : -len(uri(REAL)) - 2]


def test_a_newline_in_the_name_cannot_split_the_link() -> None:
    written = link(REAL, "shot.png\n- and another thing")

    assert written.count("\n") == 0
    assert referenced(written) == {REAL}


def test_an_empty_name_still_reads_as_something() -> None:
    assert "an attachment" in link(REAL, "")


# -- naming one in front of a model ------------------------------------------


def test_a_handle_is_short_enough_to_copy() -> None:
    """The whole reason it exists. Sixty-four characters is a citation that
    fails on one typo, and fails silently."""
    assert handle(REAL) == "a" * 12
    assert len(handle(REAL)) == HANDLE_CHARS


def test_a_handle_resolves_back_to_the_one_blob_it_names() -> None:
    assert matching(handle(REAL), {REAL, INVENTED}) == [REAL]


def test_a_handle_naming_two_blobs_resolves_to_both() -> None:
    """Not to the first one. "Which of these?" and "no such file" are different
    answers, and only the caller knows what to do about either."""
    twin = "a" * 20 + "c" * 44

    assert matching("a" * 8, {REAL, twin}) == sorted([REAL, twin])


def test_the_ellipsis_off_an_attachment_note_still_resolves() -> None:
    """A model copies what it was shown, and what it was shown ends in one."""
    assert matching(f"{handle(REAL)}…", {REAL}) == [REAL]


def test_a_whole_uri_still_resolves() -> None:
    """A model that has read a memory has seen the reference in this shape."""
    assert matching(uri(REAL), {REAL}) == [REAL]


def test_a_prefix_too_short_to_mean_anything_names_nothing() -> None:
    assert matching("aa", {REAL}) == []


def test_a_handle_that_is_not_hex_names_nothing() -> None:
    """A filename typed where an id belongs must not be scanned against every
    digest in the conversation on the chance that it prefixes one."""
    assert matching("shot.png", {REAL}) == []
    assert matching("", {REAL}) == []
