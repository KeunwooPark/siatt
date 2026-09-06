from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
import time

import pytest

from tests.e2e.conftest import SiattRig


@pytest.mark.external
def test_live_slack_socket_mode_connects(siatt_rig: SiattRig) -> None:
    """A protected smoke test for the dedicated Slack QA workspace.

    Local CI exercises ingress and egress against the protocol-level fake. This
    optional check catches token rotation, app installation, and Socket Mode
    configuration drift without putting third-party credentials in PR jobs.
    """
    app_token = os.environ.get("SIATT_SLACK_SMOKE_APP_TOKEN")
    bot_token = os.environ.get("SIATT_SLACK_SMOKE_BOT_TOKEN")
    if not app_token or not bot_token:
        pytest.skip("dedicated Slack smoke credentials are not configured")

    with siatt_rig.config.open("a") as config:
        config.write(
            '\n[slack]\napp_token_env = "SIATT_SLACK_SMOKE_APP_TOKEN"\n'
            'bot_token_env = "SIATT_SLACK_SMOKE_BOT_TOKEN"\nstream = false\n'
        )
    process = subprocess.Popen(
        [sys.executable, "-m", "siatt.cli", "run", "--slack", "--config", str(siatt_rig.config)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=siatt_rig.env
        | {
            "SIATT_SLACK_SMOKE_APP_TOKEN": app_token,
            "SIATT_SLACK_SMOKE_BOT_TOKEN": bot_token,
        },
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + 20
    output = ""
    try:
        while "Connected to Slack" not in output and time.monotonic() < deadline:
            if selector.select(timeout=0.2):
                output += process.stdout.readline()
            if process.poll() is not None:
                break
        assert "Connected to Slack" in output
    finally:
        selector.close()
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, (output + stdout, stderr)
