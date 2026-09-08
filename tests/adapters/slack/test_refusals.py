"""Slack's "too fast", told from Slack's "no", told from the rest.

The classification is the whole of #258 on the ingress-of-errors side: a
refusal that is passed along as an ordinary exception is a refusal the inbox
delivers again, four more times, model call included.
"""

from __future__ import annotations

from typing import Any

import pytest
from slack_sdk.errors import SlackApiError

from siatt.adapters.slack.app import _translate
from siatt.adapters.slack.stream import SlackRateLimited, SlackRefused
from siatt.errors import DeliveryRefused


class FakeResponse:
    """As much of `SlackResponse` as `_translate` reads."""

    def __init__(self, error: str = "", status_code: int = 200, retry_after: str | None = None):
        self._body = {"ok": False, "error": error} if error else {"ok": False}
        self.status_code = status_code
        self.headers = {"Retry-After": retry_after} if retry_after is not None else {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._body.get(key, default)


def refused(error: str = "", status_code: int = 200, retry_after: str | None = None):
    response = FakeResponse(error, status_code, retry_after)
    return SlackApiError(f"slack said {error or status_code}", response)  # type: ignore[arg-type]


def test_a_429_is_something_to_wait_out() -> None:
    translated = _translate(refused("ratelimited", status_code=429, retry_after="7"))

    assert isinstance(translated, SlackRateLimited)
    assert translated.retry_after == 7.0


def test_a_429_with_no_usable_header_still_waits() -> None:
    assert isinstance(_translate(refused("ratelimited", 429, "soon")), SlackRateLimited)
    assert isinstance(_translate(refused("ratelimited", 429)), SlackRateLimited)


@pytest.mark.parametrize(
    "code",
    [
        "msg_too_long",
        "edit_window_closed",
        "cant_update_message",
        "channel_not_found",
        "not_in_channel",
        "is_archived",
        "token_revoked",
        "missing_scope",
        "invalid_blocks",
    ],
)
def test_a_refusal_no_repetition_would_satisfy_says_so(code: str) -> None:
    translated = _translate(refused(code))

    assert isinstance(translated, SlackRefused), code
    assert translated.code == code
    assert isinstance(translated, DeliveryRefused), "which is what the inbox reads"


def test_a_code_nobody_has_classified_keeps_the_old_behaviour() -> None:
    """A failed turn, delivered again. An unrecognised code is likelier to be
    something transient we have not seen than something permanent, and the cost
    of guessing that way round is a retry rather than a lost answer."""
    original = refused("fatal_error")

    assert _translate(original) is original


def test_an_error_slack_attached_no_response_to_is_passed_along() -> None:
    original = SlackApiError("the socket went away", None)  # type: ignore[arg-type]

    assert _translate(original) is original


def test_a_rate_limit_is_not_permanent_however_it_is_labelled() -> None:
    """`ratelimited` is on no permanent list, and the 429 is what decides."""
    assert not isinstance(_translate(refused("ratelimited", 429)), SlackRefused)
