"""Slack, over Socket Mode.

`SlackAdapter` needs `slack_bolt`, which is the `slack` extra: `uv sync --extra
slack`. Nothing else here does — every judgement that can leak a private
conversation lives in `events.py`, the workspace directory lives in
`identity.py`, and neither needs a socket; `files.py` needs only `httpx`.
`identity.py` is why `siatt.runner` can import from this package at all:
the `identity` job writes what the directory saw, on a build that may have no Slack extra installed.

So the adapter is resolved lazily and the judgements are not. Importing this
package on an install that never asked for Slack therefore costs nothing until
something actually reaches for the adapter, which is what `siatt run` with no
flags relies on (`cli._serve_slack` defers its own import for the same reason)
and what makes `tests/adapters/slack/test_events.py` importable without the
extra rather than an error that aborts collection of the whole suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from siatt.adapters.slack.events import (
    Accepted,
    Decision,
    Ignored,
    SlackContext,
    message_id,
    normalize,
    session_id,
)
from siatt.adapters.slack.files import DEFAULT_FILE_HOSTS, SlackFiles
from siatt.adapters.slack.identity import Directory, SlackUser, user_ref

if TYPE_CHECKING:
    from siatt.adapters.slack.app import SlackAdapter

__all__ = [
    "DEFAULT_FILE_HOSTS",
    "Accepted",
    "Decision",
    "Directory",
    "Ignored",
    "SlackAdapter",
    "SlackContext",
    "SlackFiles",
    "SlackUser",
    "message_id",
    "normalize",
    "session_id",
    "user_ref",
]


def __getattr__(name: str) -> Any:
    """Resolve `SlackAdapter` on first use, so `slack_bolt` is imported then."""
    if name == "SlackAdapter":
        from siatt.adapters.slack.app import SlackAdapter

        return SlackAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
