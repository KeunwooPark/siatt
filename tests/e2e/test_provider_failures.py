from __future__ import annotations

import sqlite3

import pytest

from tests.e2e.conftest import SiattRig


@pytest.mark.parametrize(
    ("prompt", "diagnostic", "attempts"),
    [
        ("Fail with 401.", "AuthError", 1),
        ("Fail with 429.", "RateLimitError", 3),
        ("Fail with 503.", "TransientError", 3),
        ("Return malformed SSE.", "ProviderProtocolError", 1),
        ("Time out.", "TransientError", 3),
    ],
)
def test_provider_failures_have_deterministic_retry_and_diagnostics(
    siatt_rig: SiattRig, prompt: str, diagnostic: str, attempts: int
) -> None:
    result = siatt_rig.run(f"{prompt}\n/quit\n")

    # A failed turn does not crash the interactive process; it reports the
    # terminal cause and remains able to consume /quit.
    assert result.returncode == 0, result.stderr
    assert diagnostic in result.stdout
    assert result.stderr == ""
    assert len(siatt_rig.server.requests) == attempts

    connection = sqlite3.connect(siatt_rig.database)
    try:
        calls = connection.execute("SELECT ok, error FROM llm_calls ORDER BY id").fetchall()
    finally:
        connection.close()
    assert calls == [(0, diagnostic)] * attempts

    combined = result.stdout + result.stderr + repr(calls)
    assert "not-a-real-secret" not in combined
    if prompt.startswith("Fail with"):
        assert "planned failure echoed" in result.stdout
