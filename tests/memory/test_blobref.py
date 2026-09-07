"""How a memory points at a file, and what happens to a pointer nobody stored."""

from __future__ import annotations

from siatt.memory.blobref import link, prune, referenced, uri

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
