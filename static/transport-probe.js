/**
 * @file Diagnostic transport client for WebSocket ordering, timing, and reconnect behavior. Probe measurements are separate from participant outcome records.
 */

"use strict";

(() => {
  const SCHEMA = "tap_transport_probe_result_v1";
  const PROTOCOL_VERSION = "transport-probe-v1";
  const CHANNELS = Object.freeze(["timing", "study_pair", "companion_pair"]);
  const TIMING_RATE_HZ = 10;
  const PAIR_RATE_HZ = 1;
  const PCM_BYTES = 3200;
  const FRAME_BYTES = 12 + PCM_BYTES;
  const WINDOW_MS = 10000;
  const ACK_DEADLINE_MS = 2000;
  const HELLO_TIMEOUT_MS = 4500;
  const ACK_GRACE_MS = 2000;
  const THRESHOLDS = Object.freeze({
    minimum_duration_ms: 60000,
    minimum_timing_ack_ratio: 0.99,
    maximum_rtt_median_ms: 200,
    maximum_rtt_p95_ms: 500,
    maximum_rtt_p99_ms: 1000,
    maximum_deadline_miss_ratio: 0.01,
    maximum_adjacent_rtt_change_p95_ms: 150,
    maximum_ack_gap_ms_exclusive: 2000,
    maximum_buffered_amount_bytes: 32768,
    maximum_scheduler_lag_p95_ms: 50,
    minimum_pair_ack_ratio: 0.95,
    maximum_unexpected_closes: 0,
    maximum_protocol_errors: 0,
  });

  const script = document.querySelector('script[src*="static/transport-probe.js"]');
  const APP_BASE = script
    ? new URL(script.src).pathname.replace(/\/static\/transport-probe\.js$/, "")
    : "";

  let currentRun = null;
  let lastResult = idleResult();

  function idleResult() {
    return {
      schema: SCHEMA,
      protocol_version: PROTOCOL_VERSION,
      probe_id: null,
      status: "idle",
      done: false,
      passed: null,
      failure_reason: null,
      config: null,
      started_ms: null,
      traffic_started_ms: null,
      traffic_ended_ms: null,
      completed_ms: null,
      raw: {
        timing_samples: [],
        pair_samples: [],
        socket_events: [],
        scheduler_lag_samples: [],
        buffered_amount_samples: [],
      },
      summary: null,
      windows: [],
      checks: {},
      thresholds: { ...THRESHOLDS },
    };
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function epochNow() {
    return performance.timeOrigin + performance.now();
  }

  function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function normalizeConfig(input) {
    const config = input && typeof input === "object" ? input : {};
    const users = Math.trunc(finiteNumber(config.users ?? config.user_count, 2));
    const durationMs = Math.trunc(finiteNumber(config.duration_ms ?? config.durationMs, 60000));
    const channels = Array.isArray(config.channels) ? [...config.channels] : [...CHANNELS];
    const socketsPerUser = Math.trunc(finiteNumber(config.sockets_per_user, 3));
    const errors = [];
    if (![1, 2].includes(users)) errors.push("users_must_be_1_or_2");
    if (durationMs < 1000 || durationMs > 65000) errors.push("duration_ms_out_of_range");
    if (socketsPerUser !== 3) errors.push("sockets_per_user_must_equal_3");
    if (channels.length !== CHANNELS.length || CHANNELS.some((item, index) => channels[index] !== item)) {
      errors.push("channels_must_match_protocol_order");
    }
    return {
      errors,
      value: {
        duration_ms: durationMs,
        users,
        sockets_per_user: 3,
        channels: [...CHANNELS],
        timing_rate_hz: TIMING_RATE_HZ,
        pair_rate_hz: PAIR_RATE_HZ,
        pcm_bytes: PCM_BYTES,
        frame_bytes: FRAME_BYTES,
        window_ms: WINDOW_MS,
        ack_deadline_ms: ACK_DEADLINE_MS,
        ack_grace_ms: ACK_GRACE_MS,
      },
    };
  }

  function makeProbeId() {
    if (window.crypto?.randomUUID) return window.crypto.randomUUID();
    return `${Date.now()}-${Math.random().toString(16).slice(2)}-${Math.random().toString(16).slice(2)}`;
  }

  function wsUrl(path) {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    return `${scheme}//${location.host}${APP_BASE}${path}`;
  }

  function order(run) {
    run.order += 1;
    return run.order;
  }

  function addSocketEvent(run, context, event, detail = {}) {
    run.result.raw.socket_events.push({
      order: order(run),
      event,
      user_index: context.userIndex,
      channel: context.channel,
      at_ms: epochNow(),
      ...detail,
    });
  }

  function addTimer(run, callback, delayMs) {
    const id = window.setTimeout(() => {
      run.timers.delete(id);
      callback();
    }, Math.max(0, delayMs));
    run.timers.add(id);
    return id;
  }

  function clearTimer(run, id) {
    if (id === null || id === undefined) return;
    window.clearTimeout(id);
    run.timers.delete(id);
  }

  function clearAllTimers(run) {
    for (const id of run.timers) window.clearTimeout(id);
    run.timers.clear();
  }

  function pcmPayload(userIndex) {
    const bytes = new Uint8Array(PCM_BYTES);
    const samples = new Int16Array(PCM_BYTES / 2);
    const frequency = userIndex === 0 ? 220 : 277;
    for (let index = 0; index < samples.length; index += 1) {
      samples[index] = Math.round(Math.sin((2 * Math.PI * frequency * index) / 16000) * 9000);
    }
    bytes.set(new Uint8Array(samples.buffer));
    return bytes;
  }

  function timingFrame(seq, clientSentMs, payload) {
    const frame = new ArrayBuffer(FRAME_BYTES);
    const view = new DataView(frame);
    view.setUint32(0, seq, false);
    view.setFloat64(4, clientSentMs, false);
    new Uint8Array(frame, 12, PCM_BYTES).set(payload);
    return frame;
  }

  function percentile(values, fraction) {
    if (!values.length) return null;
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!sorted.length) return null;
    const index = Math.max(0, Math.ceil(fraction * sorted.length) - 1);
    return round(sorted[Math.min(index, sorted.length - 1)]);
  }

  function round(value) {
    return Number.isFinite(value) ? Math.round(value * 1000) / 1000 : null;
  }

  function ratio(numerator, denominator) {
    return denominator > 0 ? round(numerator / denominator) : 0;
  }

  function createRun(config) {
    const result = idleResult();
    result.probe_id = makeProbeId();
    result.status = "starting";
    result.config = config;
    result.started_ms = epochNow();
    return {
      result,
      order: 0,
      timers: new Set(),
      sockets: [],
      finishing: false,
      trafficActive: false,
      trafficStartPerf: null,
      trafficEndPerf: null,
      expectedTimingPerUser: Math.floor(config.duration_ms / (1000 / TIMING_RATE_HZ)),
      expectedPairPerSocket: Math.floor(config.duration_ms / (1000 / PAIR_RATE_HZ)),
      protocolErrors: 0,
      unexpectedCloses: 0,
      payloads: Array.from({ length: config.users }, (_, index) => pcmPayload(index)),
    };
  }

  function invalidResult(config, reasons) {
    const result = idleResult();
    result.probe_id = makeProbeId();
    result.status = "failed";
    result.done = true;
    result.passed = false;
    result.failure_reason = `invalid_config:${reasons.join(",")}`;
    result.config = config;
    result.started_ms = epochNow();
    result.completed_ms = result.started_ms;
    result.checks = { valid_config: false };
    return result;
  }

  function connectSocket(run, userIndex, channel) {
    return new Promise((resolve, reject) => {
      if (run.finishing) {
        reject(new Error("probe_finishing"));
        return;
      }
      const socket = new WebSocket(wsUrl("/ws/transport-probe"));
      socket.binaryType = "arraybuffer";
      const context = {
        userIndex,
        channel,
        socket,
        seq: 0,
        pending: new Map(),
        ready: false,
        createdPerf: performance.now(),
        createdMs: epochNow(),
        openedPerf: null,
        openedMs: null,
        intentionalClose: false,
        closeSettled: false,
        closeResolve: null,
        closePromise: null,
        helloTimer: null,
      };
      context.closePromise = new Promise((closeResolve) => { context.closeResolve = closeResolve; });
      run.sockets.push(context);

      context.helloTimer = addTimer(run, () => {
        if (context.ready || run.finishing) return;
        reject(new Error(`hello_timeout:${userIndex}:${channel}`));
        failRun(run, `hello_timeout:${userIndex}:${channel}`);
      }, HELLO_TIMEOUT_MS);

      socket.addEventListener("open", () => {
        if (run.finishing) {
          context.intentionalClose = true;
          try { socket.close(1000, "probe already finished"); } catch (_) { /* Best effort. */ }
          return;
        }
        context.openedPerf = performance.now();
        context.openedMs = epochNow();
        addSocketEvent(run, context, "open", {
          created_ms: round(context.createdMs),
          opened_ms: round(context.openedMs),
          tcp_websocket_open_ms: round(context.openedPerf - context.createdPerf),
          ready_state: socket.readyState,
        });
        socket.send(JSON.stringify({
          type: "hello",
          probe_id: run.result.probe_id,
          user_index: userIndex,
          channel,
        }));
      });

      socket.addEventListener("message", (event) => {
        let message;
        try {
          if (typeof event.data !== "string") throw new Error("non_json_server_frame");
          message = JSON.parse(event.data);
        } catch (error) {
          recordProtocolError(run, context, error.message || "invalid_server_frame");
          return;
        }
        if (message.type === "hello_ack") {
          if (context.ready) {
            recordProtocolError(run, context, "duplicate_hello_ack");
            return;
          }
          if (message.protocol_version !== PROTOCOL_VERSION) {
            recordProtocolError(run, context, "protocol_version_mismatch");
            return;
          }
          context.ready = true;
          clearTimer(run, context.helloTimer);
          addSocketEvent(run, context, "hello_ack", {
            protocol_version: message.protocol_version,
            opened_ms: round(context.openedMs),
            open_latency_ms: round(performance.now() - context.createdPerf),
            hello_latency_ms: round(performance.now() - context.openedPerf),
            server_received_ms: finiteNumber(message.server_received_ms, null),
            server_sent_ms: finiteNumber(message.server_sent_ms, null),
          });
          resolve(context);
          return;
        }
        if (message.type === "probe_ack") {
          handleAck(run, context, message);
          return;
        }
        if (message.type === "probe_error" || message.type === "error") {
          recordProtocolError(run, context, String(message.code || message.message || "server_probe_error"));
          return;
        }
        recordProtocolError(run, context, `unexpected_message_type:${String(message.type)}`);
      });

      socket.addEventListener("error", () => {
        addSocketEvent(run, context, "error");
      });

      socket.addEventListener("close", (event) => {
        clearTimer(run, context.helloTimer);
        if (!context.closeSettled) {
          context.closeSettled = true;
          context.closeResolve();
        }
        const unexpected = !context.intentionalClose && !run.finishing;
        if (unexpected) run.unexpectedCloses += 1;
        addSocketEvent(run, context, "close", {
          code: event.code,
          reason: event.reason,
          was_clean: event.wasClean,
          unexpected,
        });
        if (!context.ready) reject(new Error(`socket_closed_before_ready:${userIndex}:${channel}:${event.code}`));
        if (unexpected) failRun(run, `unexpected_socket_close:${userIndex}:${channel}:${event.code}`);
      });
    });
  }

  function recordProtocolError(run, context, code) {
    run.protocolErrors += 1;
    addSocketEvent(run, context, "protocol_error", { code });
    failRun(run, `protocol_error:${code}`);
  }

  function handleAck(run, context, message) {
    const seq = Number(message.seq);
    const sample = Number.isInteger(seq) ? context.pending.get(seq) : null;
    if (!sample || sample.acked) {
      recordProtocolError(run, context, `unknown_or_duplicate_ack:${String(message.seq)}`);
      return;
    }
    if (message.channel !== context.channel) {
      recordProtocolError(run, context, "ack_channel_mismatch");
      return;
    }
    const echoedClientSent = Number(message.client_sent_ms);
    if (!Number.isFinite(echoedClientSent) || Math.abs(echoedClientSent - sample.sent_ms) > 0.01) {
      recordProtocolError(run, context, "ack_client_timestamp_mismatch");
      return;
    }
    const serverReceivedMs = Number(message.server_received_ms);
    const serverSentMs = Number(message.server_sent_ms);
    if (!Number.isFinite(serverReceivedMs) || !Number.isFinite(serverSentMs)
      || serverSentMs < serverReceivedMs) {
      recordProtocolError(run, context, "ack_server_timestamp_invalid");
      return;
    }
    const receivedMs = epochNow();
    sample.acked = true;
    sample.client_received_ms = round(receivedMs);
    sample.server_received_ms = round(serverReceivedMs);
    sample.server_sent_ms = round(serverSentMs);
    sample.rtt_ms = round(receivedMs - sample.sent_ms);
    sample.deadline_miss = sample.rtt_ms > ACK_DEADLINE_MS;
    sample.ack_order = order(run);
    context.pending.delete(seq);
  }

  function recordScheduler(run, context, seq, scheduledMs, lagMs, skipped) {
    run.result.raw.scheduler_lag_samples.push({
      order: order(run),
      user_index: context.userIndex,
      channel: context.channel,
      seq,
      scheduled_ms: round(scheduledMs),
      observed_ms: round(epochNow()),
      lag_ms: round(Math.max(0, lagMs)),
      skipped,
    });
  }

  function sendTiming(run, context, seq, scheduledEpochMs) {
    const socket = context.socket;
    if (socket.readyState !== WebSocket.OPEN) {
      failRun(run, `timing_socket_not_open:${context.userIndex}`);
      return;
    }
    const sentMs = epochNow();
    const schedulerLagMs = Math.max(0, sentMs - scheduledEpochMs);
    const before = socket.bufferedAmount;
    socket.send(timingFrame(seq, sentMs, run.payloads[context.userIndex]));
    const buffered = socket.bufferedAmount;
    const sample = {
      order: order(run),
      user_index: context.userIndex,
      seq,
      window_index: Math.min(
        Math.floor((scheduledEpochMs - run.result.traffic_started_ms) / WINDOW_MS),
        Math.ceil(run.result.config.duration_ms / WINDOW_MS) - 1,
      ),
      scheduled_ms: round(scheduledEpochMs),
      sent_ms: round(sentMs),
      acked: false,
      client_received_ms: null,
      server_received_ms: null,
      server_sent_ms: null,
      rtt_ms: null,
      scheduler_lag_ms: round(schedulerLagMs),
      buffered_amount: buffered,
      deadline_miss: null,
      ack_order: null,
    };
    run.result.raw.timing_samples.push(sample);
    context.pending.set(seq, sample);
    run.result.raw.buffered_amount_samples.push({
      order: order(run),
      user_index: context.userIndex,
      channel: context.channel,
      seq,
      at_ms: round(sentMs),
      before_bytes: before,
      buffered_amount: buffered,
    });
    return schedulerLagMs;
  }

  function sendPair(run, context, seq, scheduledEpochMs) {
    const socket = context.socket;
    if (socket.readyState !== WebSocket.OPEN) {
      failRun(run, `pair_socket_not_open:${context.userIndex}:${context.channel}`);
      return;
    }
    const sentMs = epochNow();
    const before = socket.bufferedAmount;
    socket.send(JSON.stringify({ type: "heartbeat", seq, client_sent_ms: sentMs }));
    const buffered = socket.bufferedAmount;
    const sample = {
      order: order(run),
      user_index: context.userIndex,
      channel: context.channel,
      seq,
      window_index: Math.min(
        Math.floor((scheduledEpochMs - run.result.traffic_started_ms) / WINDOW_MS),
        Math.ceil(run.result.config.duration_ms / WINDOW_MS) - 1,
      ),
      scheduled_ms: round(scheduledEpochMs),
      sent_ms: round(sentMs),
      acked: false,
      client_received_ms: null,
      server_received_ms: null,
      server_sent_ms: null,
      rtt_ms: null,
      buffered_amount: buffered,
      deadline_miss: null,
      ack_order: null,
    };
    run.result.raw.pair_samples.push(sample);
    context.pending.set(seq, sample);
    run.result.raw.buffered_amount_samples.push({
      order: order(run),
      user_index: context.userIndex,
      channel: context.channel,
      seq,
      at_ms: round(sentMs),
      before_bytes: before,
      buffered_amount: buffered,
    });
  }

  function startScheduledStream(run, context, rateHz, expectedCount) {
    const periodMs = 1000 / rateHz;
    let slot = 0;
    const tick = () => {
      if (!run.trafficActive || run.finishing) return;
      const nowPerf = performance.now();
      const elapsed = nowPerf - run.trafficStartPerf;
      const dueSlot = Math.max(0, Math.floor(elapsed / periodMs));
      while (slot < dueSlot && slot < expectedCount) {
        const scheduledPerf = run.trafficStartPerf + slot * periodMs;
        if (context.channel === "timing") {
          recordScheduler(run, context, slot, run.result.traffic_started_ms + slot * periodMs, nowPerf - scheduledPerf, true);
        }
        slot += 1;
      }
      if (slot >= expectedCount) return;
      const targetPerf = run.trafficStartPerf + slot * periodMs;
      if (nowPerf + 0.5 < targetPerf) {
        addTimer(run, tick, targetPerf - nowPerf);
        return;
      }
      const scheduledEpochMs = run.result.traffic_started_ms + slot * periodMs;
      const lagMs = Math.max(0, nowPerf - targetPerf);
      if (context.channel === "timing") {
        const actualLagMs = sendTiming(run, context, slot, scheduledEpochMs);
        recordScheduler(run, context, slot, scheduledEpochMs, actualLagMs, false);
      } else {
        sendPair(run, context, slot, scheduledEpochMs);
      }
      slot += 1;
      if (slot < expectedCount) {
        const nextTarget = run.trafficStartPerf + slot * periodMs;
        addTimer(run, tick, nextTarget - performance.now());
      }
    };
    addTimer(run, tick, 0);
  }

  async function executeRun(run) {
    try {
      run.result.status = "connecting";
      const connections = [];
      for (let userIndex = 0; userIndex < run.result.config.users; userIndex += 1) {
        for (const channel of CHANNELS) connections.push(connectSocket(run, userIndex, channel));
      }
      await Promise.all(connections);
      if (run.finishing) return;
      run.result.status = "running";
      run.trafficActive = true;
      run.trafficStartPerf = performance.now();
      run.trafficEndPerf = run.trafficStartPerf + run.result.config.duration_ms;
      run.result.traffic_started_ms = epochNow();
      run.result.traffic_ended_ms = round(run.result.traffic_started_ms + run.result.config.duration_ms);

      for (const context of run.sockets) {
        if (context.channel === "timing") {
          startScheduledStream(run, context, TIMING_RATE_HZ, run.expectedTimingPerUser);
        } else {
          startScheduledStream(run, context, PAIR_RATE_HZ, run.expectedPairPerSocket);
        }
      }
      addTimer(run, () => {
        run.trafficActive = false;
        run.result.status = "draining";
        addTimer(run, () => finalizeRun(run), ACK_GRACE_MS);
      }, run.result.config.duration_ms);
    } catch (error) {
      failRun(run, error?.message || "connection_setup_failed");
    }
  }

  function failRun(run, reason) {
    if (run.finishing) return;
    run.result.failure_reason = run.result.failure_reason || String(reason);
    run.trafficActive = false;
    finalizeRun(run);
  }

  function adjacentRttChanges(samples) {
    const changes = [];
    const byUser = new Map();
    for (const sample of samples) {
      if (!sample.acked || !Number.isFinite(sample.rtt_ms)) continue;
      if (!byUser.has(sample.user_index)) byUser.set(sample.user_index, []);
      byUser.get(sample.user_index).push(sample);
    }
    for (const entries of byUser.values()) {
      entries.sort((left, right) => left.seq - right.seq);
      for (let index = 1; index < entries.length; index += 1) {
        changes.push(Math.abs(entries[index].rtt_ms - entries[index - 1].rtt_ms));
      }
    }
    return changes;
  }

  function maximumAckGap(samples, users, measurementStartMs, measurementEndMs) {
    let maximum = 0;
    for (let userIndex = 0; userIndex < users; userIndex += 1) {
      const received = samples
        .filter((sample) => sample.user_index === userIndex && sample.acked && Number.isFinite(sample.client_received_ms))
        .map((sample) => sample.client_received_ms)
        .sort((a, b) => a - b);
      if (!received.length) {
        maximum = Math.max(maximum, measurementEndMs - measurementStartMs);
        continue;
      }
      maximum = Math.max(maximum, received[0] - measurementStartMs);
      for (let index = 1; index < received.length; index += 1) {
        maximum = Math.max(maximum, received[index] - received[index - 1]);
      }
      maximum = Math.max(maximum, measurementEndMs - received[received.length - 1]);
    }
    return round(maximum);
  }

  function timingMetrics(run, samples, expected, schedulerSamples, measurementStartMs, measurementEndMs) {
    const acked = samples.filter((sample) => sample.acked && Number.isFinite(sample.rtt_ms));
    const rtts = acked.map((sample) => sample.rtt_ms);
    const schedulerMissed = Math.max(0, expected - samples.length);
    const networkMissed = samples.filter((sample) => !sample.acked).length;
    const late = acked.filter((sample) => sample.rtt_ms > ACK_DEADLINE_MS).length;
    return {
      timing_expected: expected,
      timing_sent: samples.length,
      timing_acked: acked.length,
      timing_ack_ratio: ratio(acked.length, expected),
      rtt_median_ms: percentile(rtts, 0.5),
      rtt_p95_ms: percentile(rtts, 0.95),
      rtt_p99_ms: percentile(rtts, 0.99),
      adjacent_rtt_change_p95_ms: percentile(adjacentRttChanges(samples), 0.95),
      deadline_miss_count: schedulerMissed + networkMissed + late,
      deadline_miss_ratio: ratio(schedulerMissed + networkMissed + late, expected),
      max_ack_gap_ms: maximumAckGap(
        samples,
        run.result.config.users,
        measurementStartMs,
        measurementEndMs,
      ),
      scheduler_lag_p95_ms: percentile(
        schedulerSamples.filter((entry) => !entry.skipped).map((entry) => entry.lag_ms),
        0.95,
      ),
    };
  }

  function buildSummary(run) {
    const result = run.result;
    for (const sample of [...result.raw.timing_samples, ...result.raw.pair_samples]) {
      if (sample.deadline_miss === null) sample.deadline_miss = !sample.acked || sample.rtt_ms > ACK_DEADLINE_MS;
    }
    const expectedTiming = run.expectedTimingPerUser * result.config.users;
    const timing = timingMetrics(
      run,
      result.raw.timing_samples,
      expectedTiming,
      result.raw.scheduler_lag_samples,
      result.traffic_started_ms || result.started_ms,
      result.traffic_ended_ms || (result.started_ms + result.config.duration_ms),
    );
    const expectedPair = run.expectedPairPerSocket * result.config.users * 2;
    const pairAcked = result.raw.pair_samples.filter((sample) => sample.acked).length;
    return {
      ...timing,
      max_buffered_amount_bytes: result.raw.buffered_amount_samples.reduce(
        (maximum, entry) => Math.max(maximum, entry.buffered_amount),
        0,
      ),
      pair_expected: expectedPair,
      pair_sent: result.raw.pair_samples.length,
      pair_acked: pairAcked,
      pair_ack_ratio: ratio(pairAcked, expectedPair),
      unexpected_close_count: run.unexpectedCloses,
      protocol_error_count: run.protocolErrors,
    };
  }

  function buildWindows(run) {
    const windows = [];
    const result = run.result;
    const count = Math.ceil(result.config.duration_ms / WINDOW_MS);
    const timingPerUser = Math.floor(WINDOW_MS / (1000 / TIMING_RATE_HZ));
    const pairPerSocket = Math.floor(WINDOW_MS / (1000 / PAIR_RATE_HZ));
    for (let windowIndex = 0; windowIndex < count; windowIndex += 1) {
      const startOffset = windowIndex * WINDOW_MS;
      const endOffset = Math.min(result.config.duration_ms, startOffset + WINDOW_MS);
      const fraction = (endOffset - startOffset) / WINDOW_MS;
      const timingSamples = result.raw.timing_samples.filter((sample) => sample.window_index === windowIndex);
      const pairSamples = result.raw.pair_samples.filter((sample) => sample.window_index === windowIndex);
      const schedulerSamples = result.raw.scheduler_lag_samples.filter((sample) => {
        const offset = sample.scheduled_ms - result.traffic_started_ms;
        return offset >= startOffset && offset < endOffset;
      });
      const expectedTiming = Math.floor(timingPerUser * fraction) * result.config.users;
      const expectedPair = Math.floor(pairPerSocket * fraction) * result.config.users * 2;
      const metrics = timingMetrics(
        run,
        timingSamples,
        expectedTiming,
        schedulerSamples,
        result.traffic_started_ms + startOffset,
        result.traffic_started_ms + endOffset,
      );
      const pairAcked = pairSamples.filter((sample) => sample.acked).length;
      windows.push({
        window_index: windowIndex,
        start_offset_ms: startOffset,
        end_offset_ms: endOffset,
        ...metrics,
        pair_expected: expectedPair,
        pair_sent: pairSamples.length,
        pair_acked: pairAcked,
        pair_ack_ratio: ratio(pairAcked, expectedPair),
        max_buffered_amount_bytes: result.raw.buffered_amount_samples
          .filter((entry) => entry.at_ms >= result.traffic_started_ms + startOffset
            && entry.at_ms < result.traffic_started_ms + endOffset)
          .reduce((maximum, entry) => Math.max(maximum, entry.buffered_amount), 0),
      });
    }
    return windows;
  }

  function buildChecks(result) {
    const summary = result.summary;
    const config = result.config;
    return {
      duration: config.duration_ms >= THRESHOLDS.minimum_duration_ms,
      users: [1, 2].includes(config.users),
      channels: config.channels.length === CHANNELS.length
        && CHANNELS.every((channel, index) => config.channels[index] === channel),
      timing_ack_ratio: summary.timing_ack_ratio >= THRESHOLDS.minimum_timing_ack_ratio,
      rtt_median: summary.rtt_median_ms !== null
        && summary.rtt_median_ms <= THRESHOLDS.maximum_rtt_median_ms,
      rtt_p95: summary.rtt_p95_ms !== null && summary.rtt_p95_ms <= THRESHOLDS.maximum_rtt_p95_ms,
      rtt_p99: summary.rtt_p99_ms !== null && summary.rtt_p99_ms <= THRESHOLDS.maximum_rtt_p99_ms,
      deadline_miss: summary.deadline_miss_ratio <= THRESHOLDS.maximum_deadline_miss_ratio,
      adjacent_rtt_change: summary.adjacent_rtt_change_p95_ms !== null
        && summary.adjacent_rtt_change_p95_ms <= THRESHOLDS.maximum_adjacent_rtt_change_p95_ms,
      ack_gap: summary.max_ack_gap_ms !== null
        && summary.max_ack_gap_ms < THRESHOLDS.maximum_ack_gap_ms_exclusive,
      buffered_amount: summary.max_buffered_amount_bytes <= THRESHOLDS.maximum_buffered_amount_bytes,
      scheduler_lag: summary.scheduler_lag_p95_ms !== null
        && summary.scheduler_lag_p95_ms <= THRESHOLDS.maximum_scheduler_lag_p95_ms,
      pair_ack_ratio: summary.pair_ack_ratio >= THRESHOLDS.minimum_pair_ack_ratio,
      unexpected_closes: summary.unexpected_close_count === THRESHOLDS.maximum_unexpected_closes,
      protocol_errors: summary.protocol_error_count === THRESHOLDS.maximum_protocol_errors,
      no_runtime_failure: !result.failure_reason,
    };
  }

  async function finalizeRun(run, explicitReason = null) {
    if (run.finishing) return;
    run.finishing = true;
    run.trafficActive = false;
    clearAllTimers(run);
    if (explicitReason && !run.result.failure_reason) run.result.failure_reason = explicitReason;
    for (const context of run.sockets) {
      context.intentionalClose = true;
      if (context.socket.readyState === WebSocket.OPEN || context.socket.readyState === WebSocket.CONNECTING) {
        try { context.socket.close(1000, "probe complete"); } catch (_) { /* Best effort. */ }
      } else if (!context.closeSettled) {
        context.closeSettled = true;
        context.closeResolve();
      }
    }
    await Promise.race([
      Promise.all(run.sockets.map((context) => context.closePromise)),
      new Promise((resolve) => window.setTimeout(resolve, 1000)),
    ]);
    run.result.summary = buildSummary(run);
    run.result.windows = buildWindows(run);
    run.result.checks = buildChecks(run.result);
    run.result.passed = Object.values(run.result.checks).every(Boolean);
    run.result.done = true;
    run.result.status = run.result.passed ? "completed" : "failed";
    run.result.completed_ms = epochNow();
    lastResult = run.result;
  }

  async function start(config = {}) {
    if (currentRun && !currentRun.result.done) {
      return clone(currentRun.result);
    }
    const normalized = normalizeConfig(config);
    if (normalized.errors.length) {
      lastResult = invalidResult(normalized.value, normalized.errors);
      currentRun = null;
      return clone(lastResult);
    }
    currentRun = createRun(normalized.value);
    lastResult = currentRun.result;
    Promise.resolve().then(() => executeRun(currentRun));
    return clone(currentRun.result);
  }

  function getResult() {
    return clone(currentRun?.result || lastResult);
  }

  function getStatus() {
    const result = currentRun?.result || lastResult;
    return {
      schema: result.schema,
      protocol_version: result.protocol_version,
      probe_id: result.probe_id,
      status: result.status,
      done: result.done,
      passed: result.passed,
      failure_reason: result.failure_reason,
      started_ms: result.started_ms,
      traffic_started_ms: result.traffic_started_ms,
      traffic_ended_ms: result.traffic_ended_ms,
      counts: {
        timing_samples: result.raw.timing_samples.length,
        timing_acked: result.raw.timing_samples.filter((sample) => sample.acked).length,
        pair_samples: result.raw.pair_samples.length,
        pair_acked: result.raw.pair_samples.filter((sample) => sample.acked).length,
        socket_events: result.raw.socket_events.length,
      },
    };
  }

  async function stop(reason = "stopped_by_caller") {
    if (currentRun && !currentRun.result.done) await finalizeRun(currentRun, String(reason));
    return getResult();
  }

  const api = { start, getStatus, getResult, stop };
  Object.defineProperty(api, "result", { enumerable: true, get: getResult });
  window.__transportProbe = Object.freeze(api);

  const status = document.querySelector("#probe-status");
  const startButton = document.querySelector("#probe-start");
  const stopButton = document.querySelector("#probe-stop");
  const durationInput = document.querySelector("#probe-duration");
  const output = document.querySelector("#probe-result");
  const metricElements = {
    ack: document.querySelector("#probe-ack-ratio"),
    p95: document.querySelector("#probe-p95"),
    gap: document.querySelector("#probe-ack-gap"),
    jitter: document.querySelector("#probe-jitter"),
    scheduler: document.querySelector("#probe-scheduler"),
    buffered: document.querySelector("#probe-buffered"),
  };

  function valueOrDash(value, suffix = "") {
    return value === null || value === undefined ? "—" : `${value}${suffix}`;
  }

  let terminalDisplay = null;
  function render() {
    if (!status || !output) return;
    const currentStatus = getStatus();
    if (!currentStatus.done) terminalDisplay = null;
    if (currentStatus.done && (!terminalDisplay || terminalDisplay.probe_id !== currentStatus.probe_id)) {
      terminalDisplay = getResult();
    }
    const result = terminalDisplay || currentStatus;
    status.dataset.state = result.status;
    status.textContent = result.done
      ? (result.passed ? "通过" : `未通过${result.failure_reason ? `：${result.failure_reason}` : ""}`)
      : ({ idle: "待运行", starting: "正在启动", connecting: "正在连接", running: "负载运行中", draining: "等待最后 ACK" }[result.status] || result.status);
    const summary = result.summary;
    metricElements.ack.textContent = summary ? `${(summary.timing_ack_ratio * 100).toFixed(2)}%` : "—";
    metricElements.p95.textContent = valueOrDash(summary?.rtt_p95_ms, " ms");
    metricElements.gap.textContent = valueOrDash(summary?.max_ack_gap_ms, " ms");
    metricElements.jitter.textContent = valueOrDash(summary?.adjacent_rtt_change_p95_ms, " ms");
    metricElements.scheduler.textContent = valueOrDash(summary?.scheduler_lag_p95_ms, " ms");
    metricElements.buffered.textContent = valueOrDash(summary?.max_buffered_amount_bytes, " B");
    const display = {
      schema: result.schema,
      protocol_version: result.protocol_version,
      probe_id: result.probe_id,
      status: result.status,
      done: result.done,
      passed: result.passed,
      failure_reason: result.failure_reason,
      config: result.config,
      summary: result.summary,
      windows: result.windows,
      checks: result.checks,
    };
    output.textContent = JSON.stringify(display, null, 2);
    if (startButton) startButton.disabled = !result.done && result.status !== "idle";
    if (stopButton) stopButton.disabled = result.done || result.status === "idle";
  }

  startButton?.addEventListener("click", async () => {
    const durationMs = Math.round(finiteNumber(durationInput?.value, 60) * 1000);
    await start({ duration_ms: durationMs, users: 2, sockets_per_user: 3, channels: [...CHANNELS] });
    render();
  });
  stopButton?.addEventListener("click", async () => {
    await stop();
    render();
  });

  const renderTimer = window.setInterval(render, 500);
  window.addEventListener("pagehide", () => {
    window.clearInterval(renderTimer);
    if (currentRun && !currentRun.result.done) {
      currentRun.finishing = true;
      currentRun.trafficActive = false;
      clearAllTimers(currentRun);
      for (const context of currentRun.sockets) {
        context.intentionalClose = true;
        try { context.socket.close(1000, "page hidden"); } catch (_) { /* Best effort. */ }
      }
    }
  }, { once: true });

  document.documentElement.dataset.appReady = "true";
  render();
})();
