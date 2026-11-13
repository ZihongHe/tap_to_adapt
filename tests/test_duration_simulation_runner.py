"""Regression tests for duration simulation runner.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import websocket

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_duration_simulation_public import (  # noqa: E402
    wait_while_servicing_timing_socket,
)


class TimeoutSocket:
    def __init__(self) -> None:
        self.timeout = 120.0
        self.recv_calls = 0

    def gettimeout(self) -> float:
        return self.timeout

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def recv(self) -> str:
        self.recv_calls += 1
        time.sleep(min(self.timeout, 0.005))
        raise websocket.WebSocketTimeoutException()


def test_idle_wait_services_socket_and_restores_timeout() -> None:
    socket = TimeoutSocket()

    wait_while_servicing_timing_socket(socket, 0.02)

    assert socket.recv_calls >= 2
    assert socket.timeout == 120.0
