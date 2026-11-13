"""Regression tests for transport probe static.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "static"


def test_transport_probe_page_is_public_automation_surface() -> None:
    page = (STATIC / "transport-probe.html").read_text(encoding="utf-8")

    assert '<html lang="zh-CN" data-app-ready="false">' in page
    assert "不登录、不创建实验 session" in page
    assert 'id="probe-start"' in page
    assert "static/transport-probe.js?v=20260719-transport-probe-v1" in page


def test_transport_probe_matches_v1_wire_contract() -> None:
    script = (STATIC / "transport-probe.js").read_text(encoding="utf-8")

    assert 'const SCHEMA = "tap_transport_probe_result_v1"' in script
    assert 'const PROTOCOL_VERSION = "transport-probe-v1"' in script
    assert '["timing", "study_pair", "companion_pair"]' in script
    assert 'wsUrl("/ws/transport-probe")' in script
    assert 'type: "hello"' in script
    assert 'message.type === "hello_ack"' in script
    assert 'message.type === "probe_ack"' in script
    assert "view.setUint32(0, seq, false)" in script
    assert "view.setFloat64(4, clientSentMs, false)" in script
    assert "const FRAME_BYTES = 12 + PCM_BYTES" in script
    assert 'type: "heartbeat", seq, client_sent_ms: sentMs' in script


def test_transport_probe_exposes_nonblocking_pollable_api_and_raw_evidence() -> None:
    script = (STATIC / "transport-probe.js").read_text(encoding="utf-8")

    assert "window.__transportProbe = Object.freeze(api)" in script
    assert "Promise.resolve().then(() => executeRun(currentRun))" in script
    assert "function getStatus()" in script
    assert "function getResult()" in script
    assert "timing_samples: []" in script
    assert "pair_samples: []" in script
    assert "socket_events: []" in script
    assert "scheduler_lag_samples: []" in script
    assert "buffered_amount_samples: []" in script
    assert "function buildWindows(run)" in script
    assert "client_received_ms" in script
    assert "server_received_ms" in script
    assert "server_sent_ms" in script


def test_transport_probe_hard_gate_is_not_shortened_or_silently_relaxed() -> None:
    script = (STATIC / "transport-probe.js").read_text(encoding="utf-8")

    expected = [
        "minimum_duration_ms: 60000",
        "minimum_timing_ack_ratio: 0.99",
        "maximum_rtt_median_ms: 200",
        "maximum_rtt_p95_ms: 500",
        "maximum_rtt_p99_ms: 1000",
        "maximum_deadline_miss_ratio: 0.01",
        "maximum_adjacent_rtt_change_p95_ms: 150",
        "maximum_ack_gap_ms_exclusive: 2000",
        "maximum_buffered_amount_bytes: 32768",
        "maximum_scheduler_lag_p95_ms: 50",
        "minimum_pair_ack_ratio: 0.95",
        "maximum_unexpected_closes: 0",
        "maximum_protocol_errors: 0",
    ]
    for marker in expected:
        assert marker in script

    assert "Object.values(run.result.checks).every(Boolean)" in script
    assert "sample.rtt_ms > ACK_DEADLINE_MS" in script
    assert "clearAllTimers(run)" in script
    assert 'context.socket.close(1000, "probe complete")' in script
