/**
 * @file Operator dashboard for authenticated inspection of enrollment, session locks, study progress, and behavioral summaries. Administrative requests require an explicit local access token.
 */

"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const APP_BASE = new URL(document.querySelector('script[src*="static/admin.js"]').src).pathname.replace(/\/static\/admin\.js$/, "");
const state = {
  token: "",
  stages: [],
  overview: null,
  core: null,
  liveSubjects: null,
  shareFitTrend: null,
  allRuns: [],
  filterInitialized: false,
  activeView: "core",
  createdAccounts: [],
  createdBatch: null,
  selectedParticipantId: "",
};

async function adminApi(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Admin-Token", state.token);
  if (options.body && typeof options.body !== "string") {
    headers.set("Content-Type", "application/json");
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch(`${APP_BASE}${path}`, { ...options, headers, cache: "no-store" });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (payload.detail?.message) {
        detail = payload.detail.message;
      } else {
        detail = JSON.stringify(payload.detail);
      }
    } catch (_) { /* Keep HTTP status text. */ }
    throw new Error(detail);
  }
  return (response.headers.get("content-type") || "").includes("application/json") ? response.json() : response;
}

function formatTime(value) {
  if (!value) return "—";
  return new Date(Number(value)).toLocaleString("zh-CN", { hour12: false });
}

function todayCstDate() {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

function cstSlotStartMs(dateStr, hour, minute) {
  const [year, month, day] = String(dateStr).split("-").map(Number);
  return Date.UTC(year, month - 1, day, hour - 8, minute);
}

function blockedSlotStartsForDate(dateStr) {
  const starts = [];
  for (let hour = 8; hour < 22; hour += 1) {
    starts.push(cstSlotStartMs(dateStr, hour, 0));
    starts.push(cstSlotStartMs(dateStr, hour, 30));
  }
  return starts;
}

function renderBlockedSlotGrid(share) {
  const grid = $("#orch-block-slot-grid");
  const dateInput = $("#orch-block-date");
  if (!grid || !dateInput) return;
  if (!dateInput.value) dateInput.value = todayCstDate();
  const dateStr = dateInput.value;
  const blocked = new Set(
    (share.operator_blocked_slots || [])
      .filter((row) => row.date_cst === dateStr)
      .map((row) => Number(row.scheduled_start_ms)),
  );
  const booked = new Set(
    ((share.serial_schedule || {}).appointments || [])
      .filter((row) => ["booked", "claimed", "active"].includes(String(row.status || "")))
      .map((row) => Number(row.scheduled_start_ms)),
  );
  grid.replaceChildren();
  for (const startMs of blockedSlotStartsForDate(dateStr)) {
    const label = document.createElement("label");
    const checked = blocked.has(startMs);
    label.className = `orch-block-slot${checked ? " is-blocked" : ""}${booked.has(startMs) ? " is-booked" : ""}`;
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = String(startMs);
    box.checked = checked;
    box.addEventListener("change", () => {
      label.classList.toggle("is-blocked", box.checked);
    });
    const text = document.createElement("span");
    const clock = new Date(startMs).toLocaleTimeString("zh-CN", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "Asia/Shanghai",
    });
    text.textContent = booked.has(startMs) ? `${clock} · 已有预约` : clock;
    label.append(box, text);
    grid.appendChild(label);
  }
}

function cstDateTimeToMs(value) {
  const match = String(value || "").match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (!match) return 0;
  const [, year, month, day, hour, minute] = match.map(Number);
  return Date.UTC(year, month - 1, day, hour - 8, minute);
}

function msToCstDateTime(ms) {
  if (!ms) return "";
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date(Number(ms)));
  const get = (type) => parts.find((part) => part.type === type)?.value || "";
  return `${get("year")}-${get("month")}-${get("day")}T${get("hour")}:${get("minute")}`;
}

function defaultSessionLockRange() {
  const now = Date.now();
  const slotMs = 30 * 60 * 1000;
  const start = Math.ceil((now + 60_000) / slotMs) * slotMs;
  return { start, end: start + (2 * 60 * 60 * 1000) };
}

function renderSessionLock(lock) {
  const status = $("#session-lock-status");
  const passwordEl = $("#session-lock-password");
  const copyBtn = $("#session-lock-copy");
  const startInput = $("#session-lock-start");
  const endInput = $("#session-lock-end");
  if (!status) return;
  const row = lock || {};
  if (startInput && !startInput.value) {
    const range = defaultSessionLockRange();
    startInput.value = msToCstDateTime(range.start);
    if (endInput && !endInput.value) endInput.value = msToCstDateTime(range.end);
  }
  if (row.configured && row.start_ms && row.end_ms) {
    if (startInput) startInput.value = msToCstDateTime(row.start_ms);
    if (endInput) endInput.value = msToCstDateTime(row.end_ms);
  }
  const display = row.password_display || row.password || "";
  state.sessionLockPassword = display;
  if (copyBtn) copyBtn.disabled = !display;
  if (passwordEl) {
    passwordEl.hidden = !display;
    passwordEl.classList.toggle("hidden", !display);
    passwordEl.textContent = display ? `召集密码 ${display}` : "";
  }
  if (!row.configured && !row.active) {
    status.textContent = row.expired
      ? "上次锁定已过期，外部报名已恢复。"
      : "当前未锁定。";
    return;
  }
  const windows = Array.isArray(row.windows) && row.windows.length
    ? row.windows
    : (row.start_ms && row.end_ms ? [{ start_ms: row.start_ms, end_ms: row.end_ms }] : []);
  const ranges = windows
    .map((item) => `${formatTime(item.start_ms)} – ${formatTime(item.end_ms)}`)
    .join("；");
  if (row.active) {
    status.textContent = `锁定中：${ranges}。外部不能开始；页底凭密码可开启。`;
    return;
  }
  if (row.upcoming) {
    status.textContent = `已预约锁定：${ranges}。到点后外部不能开始，预约格已灰掉。`;
    return;
  }
  status.textContent = `锁定记录：${ranges}`;
}

async function saveSessionLock() {
  const error = $("#session-lock-error");
  if (error) error.textContent = "";
  const startMs = cstDateTimeToMs($("#session-lock-start")?.value);
  const endMs = cstDateTimeToMs($("#session-lock-end")?.value);
  if (!startMs || !endMs) {
    if (error) error.textContent = "请填写开始和结束时间（北京时间）。";
    return;
  }
  try {
    const payload = await adminApi("/api/admin/session-lock", {
      method: "POST",
      body: { start_ms: startMs, end_ms: endMs },
    });
    state.sessionLock = payload;
    renderSessionLock(payload);
  } catch (err) {
    if (error) error.textContent = err.message;
  }
}

async function clearSessionLock() {
  const error = $("#session-lock-error");
  if (error) error.textContent = "";
  try {
    const payload = await adminApi("/api/admin/session-lock", { method: "DELETE" });
    state.sessionLock = payload;
    renderSessionLock(payload);
  } catch (err) {
    if (error) error.textContent = err.message;
  }
}

async function copySessionLockPassword() {
  const value = String(state.sessionLockPassword || "").trim();
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
    $("#session-lock-status").textContent = `已复制召集密码 ${value}`;
  } catch (_) {
    $("#session-lock-error").textContent = "复制失败，请手动抄写密码。";
  }
}

async function saveBlockedSlots() {
  const dateInput = $("#orch-block-date");
  if (!dateInput?.value) return;
  const scheduledStartMs = $$("#orch-block-slot-grid input[type='checkbox']")
    .filter((box) => box.checked)
    .map((box) => Number(box.value));
  await orchestrationAction("/api/admin/orchestration/blocked-slots", {
    date_cst: dateInput.value,
    scheduled_start_ms: scheduledStartMs,
  });
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function yesNo(value) {
  return value ? "是" : "否";
}

function contactChannelLabel(value) {
  return ({ email: "邮箱", phone: "电话", wechat: "微信" })[value] || value;
}

function contactChannels(value) {
  const channels = Array.isArray(value) ? value : [];
  return channels.length ? channels.map(contactChannelLabel).join(" / ") : "—";
}

function displayName(value) {
  const name = String(value || "").trim();
  return name || "—";
}

function contactField(value) {
  const text = String(value || "").trim();
  return text || "—";
}

function contactLine(contact) {
  const row = contact || {};
  const parts = [
    row.email ? `邮箱 ${row.email}` : "",
    row.phone ? `电话 ${row.phone}` : "",
    row.wechat ? `微信 ${row.wechat}` : "",
  ].filter(Boolean);
  return parts.length ? parts.join(" · ") : "—";
}

function windowTapCell(windowTaps, windowIndex) {
  const slot = (windowTaps?.windows || []).find((row) => Number(row.window) === windowIndex);
  return `${Number(slot?.taps || 0)} / ${Number(slot?.interrupts || 0)}`;
}

function buildTable(rows, columns, labels, renderers = {}) {
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = labels[column] || column;
    headRow.appendChild(th);
  }
  head.appendChild(headRow);
  const body = document.createElement("tbody");
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = columns.length;
    td.className = "admin-empty";
    td.textContent = "暂无记录";
    tr.appendChild(td);
    body.appendChild(tr);
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      if (renderers[column]) {
        const rendered = renderers[column](row[column], row);
        if (rendered instanceof Node) td.appendChild(rendered);
        else td.textContent = rendered ?? "";
      } else {
        td.textContent = row[column] ?? "";
      }
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  table.append(head, body);
  return table;
}

function codeCell(value) {
  const code = document.createElement("code");
  code.className = "admin-code";
  code.textContent = value ?? "—";
  return code;
}

function experimentArmLabel(value) {
  return {
    share_fit: "Share 拟合（链式）",
    cold_learn: "① 冷启动在线学",
    shared_learn: "② share 启动在线学",
    shared_frozen: "③ share 启动冻结",
    unassigned: "未编入当前阶段",
  }[value] || value || "—";
}

function longTextCell(value) {
  const text = document.createElement("div");
  text.className = "admin-long-text";
  text.textContent = value ?? "—";
  return text;
}

function eventTextCell(value) {
  const container = document.createElement("div");
  container.className = "admin-text-fields";
  for (const [field, text] of Object.entries(value || {})) {
    const row = document.createElement("div");
    const label = document.createElement("strong");
    label.textContent = `${field}：`;
    const content = document.createElement("span");
    content.textContent = text;
    row.append(label, content);
    container.appendChild(row);
  }
  return container;
}

function conversationSourceLabel(value) {
  if (value === "primary") return "电脑实验系统";
  if (value === "companion") return "手机陪伴系统";
  return value || "—";
}

$("#admin-login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  state.token = $("#admin-token").value;
  $("#admin-login-error").textContent = "";
  try {
    await refreshAdmin();
    $("#admin-login").classList.add("hidden");
    $("#admin-console").classList.remove("hidden");
    $("#admin-dot").className = "status-dot online";
    $("#admin-status").textContent = "已认证";
  } catch (err) {
    state.token = "";
    $("#admin-login-error").textContent = `认证失败：${err.message}`;
  }
});

async function refreshAdmin() {
  const selectedRun = $("#run-filter")?.value || "";
  const overviewPath = selectedRun
    ? `/api/admin/data-overview?run_slug=${encodeURIComponent(selectedRun)}`
    : "/api/admin/data-overview";
  const corePath = selectedRun
    ? `/api/admin/research-core?run_slug=${encodeURIComponent(selectedRun)}`
    : "/api/admin/research-core";
  const [stageData, readiness, overview, orchestration, core, liveSubjects, shareFitTrend, sessionLock] = await Promise.all([
    adminApi("/api/admin/stages"),
    adminApi("/api/admin/readiness"),
    adminApi(overviewPath),
    adminApi("/api/admin/orchestration"),
    adminApi(corePath),
    adminApi("/api/admin/live-subjects"),
    adminApi("/api/admin/share-fit-trend-board").catch(() => null),
    adminApi("/api/admin/session-lock").catch(() => null),
  ]);
  state.stages = stageData.stages;
  if (!selectedRun) state.allRuns = overview.runs;
  state.overview = overview;
  state.orchestration = orchestration;
  state.core = core;
  state.liveSubjects = liveSubjects;
  state.shareFitTrend = shareFitTrend;
  state.sessionLock = sessionLock;
  renderStageTemplates(stageData);
  renderReadiness(readiness);
  renderOverview(overview);
  renderOrchestration(orchestration);
  renderResearchCore(core, overview);
  renderLiveSubjects(liveSubjects);
  renderShareFitTrendBoard(shareFitTrend);
  if (!state.filterInitialized) {
    state.filterInitialized = true;
    const preferredRun = core.filter?.run_slug || "";
    const select = $("#run-filter");
    if (preferredRun && [...select.options].some((option) => option.value === preferredRun)) {
      select.value = preferredRun;
      await refreshAdmin();
    }
  }
}

function liveStatusChip(label, status) {
  const chip = document.createElement("span");
  chip.className = `admin-live-status is-${status || "setup"}`;
  chip.textContent = label || "—";
  return chip;
}

function yesNoOnline(value) {
  return value ? "在线" : "—";
}

function renderLiveSubjects(data) {
  const counts = data?.counts || {};
  const subjects = (data?.subjects || []).filter((row) =>
    ["running", "setup", "companion_only", "claimed"].includes(String(row.board_status || ""))
  );
  const countEl = $("#live-subjects-count");
  const updatedEl = $("#live-subjects-updated");
  const tableEl = $("#live-subjects-table");
  if (countEl) {
    const inSession = subjects.length || Number(counts.running || 0) + Number(counts.setup || 0) + Number(counts.claimed || 0);
    countEl.textContent = `只读 · 正在试验 ${inSession} 人`;
  }
  if (updatedEl) {
    updatedEl.textContent = data?.server_ms
      ? `最近同步：${formatTime(data.server_ms)}（刷新看板只读取最新状态，不改写试验）`
      : "";
  }
  if (!tableEl) return;
  const rows = subjects.map((row) => ({
    ...row,
    contact_line: contactLine(row.contact),
    official_start_label: formatTime(row.official_start_ms),
    computer_label: yesNoOnline(row.computer_online),
    phone_label: yesNoOnline(row.phone_online),
    realtime_label: row.realtime_connected ? "已连接" : "—",
    appointment_label: row.appointment?.status
      ? `${row.appointment.status}${row.holder_mode ? ` · ${row.holder_mode}` : ""}`
      : (row.holder_mode || "—"),
  }));
  tableEl.replaceChildren(buildTable(
    rows,
    [
      "display_name",
      "contact_line",
      "code_hint",
      "experiment_arm_label",
      "board_status_label",
      "official_start_label",
      "elapsed_label",
      "progress_label",
      "computer_label",
      "phone_label",
      "realtime_label",
      "appointment_label",
    ],
    {
      display_name: "姓名",
      contact_line: "联系方式",
      code_hint: "受试者号",
      experiment_arm_label: "实验阶段",
      board_status_label: "当前状态",
      official_start_label: "正式开始",
      elapsed_label: "已进行",
      progress_label: "估计进度",
      computer_label: "电脑端",
      phone_label: "手机端",
      realtime_label: "实时连接",
      appointment_label: "预约/占位",
    },
    {
      display_name: displayName,
      board_status_label: (value, row) => liveStatusChip(value, row.board_status),
    },
  ));
}

async function refreshLiveSubjects() {
  const data = await adminApi("/api/admin/live-subjects");
  state.liveSubjects = data;
  renderLiveSubjects(data);
}

function formatPValue(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  if (number < 0.001) return "<0.001";
  return number.toFixed(3);
}

function formatConfidence(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toFixed(2);
}

function formatTrendNumber(value, digits = 2) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function trendReading(trend) {
  if (!trend) return "—";
  const dropped = (trend.dropped_codes || []).filter(Boolean);
  if (!dropped.length) return trend.reading || "—";
  return `${trend.reading || "—"}（剔除 ${dropped.join("、")}）`;
}

function renderFitRemainingCard(remaining) {
  const remainingNode = $("#orch-fit-remaining");
  if (remainingNode) {
    remainingNode.textContent = remaining.locked
      ? `已锁定 N*=${remaining.locked_n ?? "—"}`
      : remaining.estimated_additional_donors == null
        ? "—"
        : `${remaining.estimated_additional_donors} 人`;
  }
  const confidenceNode = $("#orch-fit-confidence");
  if (confidenceNode) {
    confidenceNode.textContent = remaining.confidence == null
      ? "自信 —"
      : `自信 ${formatConfidence(remaining.confidence)}`;
  }
  const remainingCopy = $("#orch-fit-remaining-copy");
  if (remainingCopy) {
    const behaviorLabels = {
      plateau: "平台",
      improving: "改善中",
      rebound: "回弹",
      baseline: "基线",
      insufficient: "样本不足",
    };
    remainingCopy.textContent = [
      remaining.summary || "拟合点预计人数为只读计算，不改写模型或锁仓状态。",
      `已完成 donor=${remaining.completed_n ?? 0}`,
      `参数候选=${remaining.parameter_candidate_n ?? "—"}`,
      `开场候选=${remaining.behavior_candidate_n ?? "—"}`,
      `拟选点=${remaining.candidate_n ?? "—"}`,
      `开场状态=${behaviorLabels[remaining.current_behavior_state] || remaining.current_behavior_state || "—"}`,
      `平台确认=${remaining.persist_ready ? "已满足" : `还需到 n=${remaining.persist_need_n ?? "—"}`}`,
      `回弹确认=${remaining.rebound_confirmed ? "已满足" : `需连续 ${remaining.rebound_confirm_windows ?? 2} 窗`}`,
      `确认帽=${remaining.confirm_horizon_donors ?? "—"}`,
      `参数连续稳定=${remaining.parameter_trailing_stable ?? 0}/${remaining.stable_required ?? 3}`,
      `自信程度=${formatConfidence(remaining.confidence)}`,
      `上限剩余=${remaining.upper_bound_additional_donors ?? "—"}`,
      remaining.pending_questionnaire_count
        ? `待最终问卷=${remaining.pending_questionnaire_count}`
        : "",
    ].filter(Boolean).join(" · ");
  }
  const historyNode = $("#orch-fit-remaining-history");
  if (historyNode) {
    historyNode.replaceChildren(buildTable(
      remaining.history || [],
      ["n", "estimated_additional_donors", "confidence", "current_behavior_state", "parameter_trailing_stable"],
      {
        n: "人数点 n",
        estimated_additional_donors: "当时预计还需",
        confidence: "自信程度",
        current_behavior_state: "开场闸门",
        parameter_trailing_stable: "参数连续稳定",
      },
      {
        confidence: (value) => formatConfidence(value),
        current_behavior_state: (value) => ({
          plateau: "平台",
          improving: "改善中",
          rebound: "回弹",
          baseline: "基线",
          insufficient: "样本不足",
        }[value] || value || "—"),
      },
    ));
  }
}

function renderShareFitTrendBoard(data) {
  state.shareFitTrend = data || null;
  const countEl = $("#share-fit-trend-count");
  const updatedEl = $("#share-fit-trend-updated");
  if (countEl) {
    countEl.textContent = `只读 · share fit ${data?.completed_n ?? 0} 人`;
  }
  if (updatedEl) {
    updatedEl.textContent = data?.generated_ms
      ? `最近同步：${formatTime(data.generated_ms)}（IQR 1.5 后 Spearman，α=${data?.definitions?.significance_alpha ?? 0.05}；不改写试验）`
      : "";
  }
  if (data?.fit_remaining_estimate) {
    renderFitRemainingCard(data.fit_remaining_estimate);
  }
  const blockNode = $("#share-fit-block-trend-table");
  if (blockNode) {
    const session = data?.session_trends || {};
    const rows = [
      {
        label: "全场",
        taps_rho: session.taps_per_minute?.rho,
        taps_p: session.taps_per_minute?.p_value,
        taps_sig: session.taps_per_minute?.significant_label,
        taps_read: trendReading(session.taps_per_minute),
        q_rho: session.q?.rho,
        q_p: session.q?.p_value,
        q_sig: session.q?.significant_label,
        q_read: trendReading(session.q),
      },
      ...((data?.block_trends || []).map((row) => ({
        label: `块 ${row.block}`,
        taps_rho: row.taps_per_minute?.rho,
        taps_p: row.taps_per_minute?.p_value,
        taps_sig: row.taps_per_minute?.significant_label,
        taps_read: trendReading(row.taps_per_minute),
        q_rho: row.q?.rho,
        q_p: row.q?.p_value,
        q_sig: row.q?.significant_label,
        q_read: trendReading(row.q),
      }))),
    ];
    blockNode.replaceChildren(buildTable(
      rows,
      ["label", "taps_rho", "taps_p", "taps_sig", "taps_read", "q_rho", "q_p", "q_sig", "q_read"],
      {
        label: "范围",
        taps_rho: "taps/min ρ",
        taps_p: "taps/min p",
        taps_sig: "taps 是否显著",
        taps_read: "taps 读法",
        q_rho: "开场/块内 q ρ",
        q_p: "q p",
        q_sig: "q 是否显著",
        q_read: "q 读法",
      },
      {
        taps_rho: (value) => formatTrendNumber(value, 2),
        taps_p: formatPValue,
        q_rho: (value) => formatTrendNumber(value, 2),
        q_p: formatPValue,
      },
    ));
  }
  const qNode = $("#share-fit-questionnaire-trend-table");
  if (qNode) {
    qNode.replaceChildren(buildTable(
      data?.questionnaire_trends || [],
      ["label", "participants", "rho", "p_value", "significant_label", "reading", "inlier_mean"],
      {
        label: "问卷维度",
        participants: "作答人数",
        rho: "ρ",
        p_value: "变化趋势显著性 p",
        significant_label: "是否显著",
        reading: "读法",
        inlier_mean: "剔除后均值",
      },
      {
        rho: (value) => formatTrendNumber(value, 2),
        p_value: formatPValue,
        inlier_mean: (value) => formatTrendNumber(value, 2),
      },
    ));
  }
  const donorNode = $("#share-fit-donor-trend-table");
  if (donorNode) {
    const rows = (data?.donors || []).map((row) => {
      const next = {
        n: row.n,
        code_hint: row.code_hint || "—",
        session_taps: formatTrendNumber(row.session_taps_per_minute, 2),
        session_q: formatTrendNumber(row.session_q, 2),
        opening_q: formatTrendNumber(row.opening_q, 2),
      };
      for (let block = 1; block <= 5; block += 1) {
        const metric = row.blocks?.[String(block)] || {};
        next[`b${block}_taps`] = formatTrendNumber(metric.taps_per_minute, 2);
        next[`b${block}_q`] = formatTrendNumber(metric.q, 2);
      }
      return next;
    });
    donorNode.replaceChildren(buildTable(
      rows,
      [
        "n", "code_hint", "opening_q",
        "b1_taps", "b2_taps", "b3_taps", "b4_taps", "b5_taps",
        "b1_q", "b2_q", "b3_q", "b4_q", "b5_q",
        "session_taps", "session_q",
      ],
      {
        n: "n",
        code_hint: "受试者号",
        opening_q: "开场 q",
        b1_taps: "块1 taps/min",
        b2_taps: "块2 taps/min",
        b3_taps: "块3 taps/min",
        b4_taps: "块4 taps/min",
        b5_taps: "块5 taps/min",
        b1_q: "块1 q",
        b2_q: "块2 q",
        b3_q: "块3 q",
        b4_q: "块4 q",
        b5_q: "块5 q",
        session_taps: "全场 taps/min",
        session_q: "全场 q",
      },
    ));
  }
}

async function refreshShareFitTrendBoard() {
  const data = await adminApi("/api/admin/share-fit-trend-board");
  renderShareFitTrendBoard(data);
}

function formatRate(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatLatency(value) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return `${Math.round(Number(value))} ms`;
}

function renderResearchCore(data, overview) {
  const overall = data.overall || {};
  const complete = data.data_completeness || {};
  const scope = data.filter?.run_slug
    ? `当前 Run：${data.filter.run_slug}`
    : `全部 ${data.filter?.run_count || 0} 个 Run`;
  $("#core-scope").textContent =
    `${scope} · 仅端到端完成者进入主要结局 · 参与者为统计单位 · 5 × 6 分钟活跃块 · Rest 排除 · 更新于 ${formatTime(data.generated_ms)}`;
  $("#core-instance-badge").textContent =
    `${data.instance_id === "formal" ? "正式" : "测试"}实例 · 数据独立隔离`;
  $("#core-participant-rate").textContent = formatRate(overall.participant_mean_correction_rate);
  $("#core-opportunities").textContent = overall.opportunities ?? 0;
  $("#core-corrections").textContent = overall.eligible_corrections ?? 0;
  $("#core-tap-split").textContent =
    `${overall.eligible_manual_activations ?? 0} / ${overall.eligible_interruptions ?? 0}`;
  $("#core-completed").textContent =
    `${complete.completed ?? 0} / ${complete.participants ?? 0}`;
  $("#core-completeness").textContent =
    `${complete.five_blocks ?? 0} / ${complete.final_questionnaire ?? 0}`;

  const metricsByArm = new Map((data.by_arm || []).map((row) => [row.arm, row]));
  const armRows = (data.sample_progress || []).map((sample) => ({
    ...sample,
    ...(metricsByArm.get(sample.arm) || {}),
  }));
  $("#core-arm-table").replaceChildren(buildTable(
    armRows,
    [
      "label", "enrolled", "completed", "five_blocks", "final_questionnaire",
      "opportunities", "eligible_manual_activations", "eligible_interruptions",
      "participant_mean_correction_rate", "state_transition_mistouches", "raw_taps",
      "interrupt_latency_median_ms",
    ],
    {
      label: "实验臂", enrolled: "报名", completed: "完成", five_blocks: "5 blocks",
      final_questionnaire: "结束问卷", opportunities: "Eligible 机会",
      eligible_manual_activations: "Activate", eligible_interruptions: "Interrupt",
      participant_mean_correction_rate: "参与者均值纠错率", raw_taps: "原始 tap",
      state_transition_mistouches: "误触（不训练）",
      interrupt_latency_median_ms: "Interrupt 延迟中位数",
    },
    {
      participant_mean_correction_rate: formatRate,
      interrupt_latency_median_ms: formatLatency,
    },
  ));

  const subjectRows = (overview?.participants || []).map((row) => ({
    ...row,
    contact_line: contactLine(row.contact),
    identity_label: row.baseline_demographics?.identity_label || "—",
    spoken_accent_label: row.baseline_demographics?.spoken_accent_label || "—",
    w1: windowTapCell(row.window_taps, 1),
    w2: windowTapCell(row.window_taps, 2),
    w3: windowTapCell(row.window_taps, 3),
    w4: windowTapCell(row.window_taps, 4),
    w5: windowTapCell(row.window_taps, 5),
    tap_total: row.window_taps?.tap_total ?? 0,
    interrupt_total: row.window_taps?.interrupt_total ?? 0,
  }));
  $("#core-subject-window-table").replaceChildren(buildTable(
    subjectRows,
    [
      "display_name", "code_hint", "experiment_arm", "identity_label",
      "spoken_accent_label", "contact_line",
      "w1", "w2", "w3", "w4", "w5", "tap_total", "interrupt_total",
    ],
    {
      display_name: "姓名",
      code_hint: "编号尾号",
      experiment_arm: "实验阶段 / 臂",
      identity_label: "身份",
      spoken_accent_label: "口音或方言习惯",
      contact_line: "联系方式",
      w1: "窗口1 点击/打断",
      w2: "窗口2 点击/打断",
      w3: "窗口3 点击/打断",
      w4: "窗口4 点击/打断",
      w5: "窗口5 点击/打断",
      tap_total: "点击合计",
      interrupt_total: "打断合计",
    },
    {
      display_name: displayName,
      experiment_arm: experimentArmLabel,
    },
  ));

  $("#core-block-table").replaceChildren(buildTable(
    data.by_block || [],
    [
      "label", "block", "participants", "opportunities", "auto_activations",
      "eligible_manual_activations", "eligible_interruptions",
      "participant_mean_correction_rate", "state_transition_mistouches",
    ],
    {
      label: "实验臂", block: "Block", participants: "参与者", opportunities: "Eligible 机会",
      auto_activations: "自动响应暴露", eligible_manual_activations: "Activate",
      eligible_interruptions: "Interrupt", participant_mean_correction_rate: "参与者均值纠错率",
      state_transition_mistouches: "误触（不训练）",
    },
    { participant_mean_correction_rate: formatRate },
  ));

  const armOrder = ["share_fit", "unassigned", "cold_learn", "shared_learn", "shared_frozen"];
  const armLabels = new Map((data.by_arm || []).map((row) => [row.arm, row.label]));
  const dimensionLabels = {
    predictability: "Predictability 可预测性",
    perceived_control: "Perceived control 掌控感",
    correctability: "Correctability 可纠错性",
    legibility: "Legibility 变化可理解性",
    behavioral_adaptation: "Behavioral adaptation 低行为迁就",
    response_accuracy: "Response accuracy 时机符合度",
    cognitive_demand: "Cognitive demand 低认知负担",
    annoyance: "Annoyance 低烦扰",
    habitability: "Habitability 可习得性",
    speed: "Speed 交互效率",
    likeability: "Likeability 好感",
  };
  const facetDimensionOrder = {
    trend: [
      "response_accuracy", "likeability", "cognitive_demand", "annoyance",
      "habitability", "perceived_control", "speed",
    ],
    overall: [
      "response_accuracy", "likeability", "cognitive_demand", "annoyance",
      "habitability", "speed",
    ],
    construct: [
      "predictability", "perceived_control", "correctability",
      "legibility", "behavioral_adaptation",
    ],
  };
  const coreQuestionnaireLabels = {
    dimension: "问卷维度",
    ...Object.fromEntries(armOrder.map((arm) => [arm, armLabels.get(arm) || arm])),
  };
  for (const [facet, selector] of [
    ["trend", "#core-questionnaire-trend-table"],
    ["overall", "#core-questionnaire-overall-table"],
    ["construct", "#core-questionnaire-construct-table"],
  ]) {
    const values = new Map(
      (data.questionnaire_dimensions || [])
        .filter((row) => (row.facet || "construct") === facet)
        .map((row) => [`${row.arm}:${row.dimension}`, row]),
    );
    const questionnaireRows = facetDimensionOrder[facet].map((dimension) => {
      const row = { dimension: dimensionLabels[dimension] };
      for (const arm of armOrder) {
        const value = values.get(`${arm}:${dimension}`);
        row[arm] = value
          ? `${Number(value.mean_1_to_7).toFixed(2)} (n=${value.participants})`
          : "—";
      }
      return row;
    });
    $(selector)?.replaceChildren(buildTable(
      questionnaireRows,
      ["dimension", ...armOrder],
      coreQuestionnaireLabels,
    ));
  }

  const diffs = data.three_arm_differences || {};
  const formatPairwise = (pairs = []) => pairs.map((row) => {
    const mark = row.sig && row.sig !== "—" ? row.sig : (row.note || "—");
    return `${row.label} ${row.direction || ""} ${mark}`.trim();
  }).join("；") || "—";
  const formatOmnibus = (omnibus = {}) => {
    if (omnibus.note && omnibus.p == null) return omnibus.note;
    const p = omnibus.p == null ? "—" : Number(omnibus.p).toFixed(3);
    return `${omnibus.sig || "—"} (p=${p})`;
  };
  const formatArmCell = (cell, { counts = false } = {}) => {
    if (!cell || cell.n === 0) return "—";
    if (!counts) return `${cell.mean_label} (n=${cell.n})`;
    const n = Number(cell.n) || 0;
    const avg = (value) => (
      n > 0 && Number.isFinite(Number(value))
        ? (Number(value) / n).toFixed(1)
        : "—"
    );
    return `纠错 ${avg(cell.behavior_corrections)} · 实际tap ${avg(cell.raw_taps)} (n=${n})`;
  };
  const contrastArms = ["cold_learn", "shared_learn", "shared_frozen"];
  const blockRateRows = (diffs.blocks || []).map((row) => ({
    label: `Block ${row.block} 平均纠错率`,
    cold_learn: formatArmCell(row.groups?.cold_learn),
    shared_learn: formatArmCell(row.groups?.shared_learn),
    shared_frozen: formatArmCell(row.groups?.shared_frozen),
    omnibus: formatOmnibus(row.omnibus),
    pairwise: formatPairwise(row.pairwise),
  }));
  const blockCountRows = (diffs.blocks || []).map((row) => ({
    label: `Block ${row.block} 人均激活 / 打断`,
    cold_learn: formatArmCell(row.groups?.cold_learn, { counts: true }),
    shared_learn: formatArmCell(row.groups?.shared_learn, { counts: true }),
    shared_frozen: formatArmCell(row.groups?.shared_frozen, { counts: true }),
    omnibus: formatOmnibus(row.omnibus),
    pairwise: formatPairwise(row.pairwise),
  }));
  const mapQuestionRows = (rows) => (rows || []).map((row) => ({
    label: row.label,
    cold_learn: formatArmCell(row.groups?.cold_learn),
    shared_learn: formatArmCell(row.groups?.shared_learn),
    shared_frozen: formatArmCell(row.groups?.shared_frozen),
    omnibus: formatOmnibus(row.omnibus),
    pairwise: formatPairwise(row.pairwise),
  }));
  const questionFacetRows = (key, facet) => {
    const dedicated = diffs[key];
    if (Array.isArray(dedicated) && dedicated.length) return dedicated;
    return (diffs.questionnaire || []).filter((row) => row.facet === facet);
  };
  const diffColumns = [
    "label", ...contrastArms, "omnibus", "pairwise",
  ];
  const diffLabels = {
    label: "指标",
    cold_learn: "① 冷启动在线学",
    shared_learn: "② Share 启动在线学",
    shared_frozen: "③ Share 启动冻结",
    omnibus: "三组差异",
    pairwise: "成对（Holm）",
  };
  $("#three-arm-block-diff-table")?.replaceChildren(buildTable(
    blockRateRows, diffColumns, diffLabels,
  ));
  $("#three-arm-block-count-table")?.replaceChildren(buildTable(
    blockCountRows, diffColumns, { ...diffLabels, label: "次数" },
  ));
  $("#three-arm-questionnaire-trend-table")?.replaceChildren(buildTable(
    mapQuestionRows(questionFacetRows("questionnaire_trend", "trend")),
    diffColumns,
    { ...diffLabels, label: "向好性" },
  ));
  $("#three-arm-questionnaire-overall-table")?.replaceChildren(buildTable(
    mapQuestionRows(questionFacetRows("questionnaire_overall", "overall")),
    diffColumns,
    { ...diffLabels, label: "整体性" },
  ));
  $("#three-arm-questionnaire-construct-table")?.replaceChildren(buildTable(
    mapQuestionRows(questionFacetRows("questionnaire_construct", "construct")),
    diffColumns,
    { ...diffLabels, label: "构念题" },
  ));

  $("#core-final-recovery-table").replaceChildren(buildTable(
    data.final_questionnaire_recoveries || [],
    [
      "code_hint", "credential", "entered_ms", "model_finalized_ms",
      "chain_committed_ms", "final_questionnaire_submitted", "final_submitted_ms",
      "end_to_end_complete", "last_error",
    ],
    {
      code_hint: "编号尾号", credential: "受试者凭据", entered_ms: "进入最终问卷",
      model_finalized_ms: "模型最终化", chain_committed_ms: "模型链可供下一位",
      final_questionnaire_submitted: "最终问卷已提交", final_submitted_ms: "问卷提交时间",
      end_to_end_complete: "端到端完成", last_error: "最近错误",
    },
    {
      entered_ms: formatTime,
      model_finalized_ms: formatTime,
      chain_committed_ms: formatTime,
      final_submitted_ms: formatTime,
      final_questionnaire_submitted: (value, row) => (
        value
          ? "是"
          : row.end_to_end_complete
            ? "否（计时已完成，可凭据补填）"
            : "否"
      ),
      end_to_end_complete: (value) => value ? "是" : "否",
      last_error: (value) => value || "—",
    },
  ));
  $("#interrupt-continue-table")?.replaceChildren(buildTable(
    data.interrupt_continues || [],
    [
      "code_hint", "interrupt_count", "continue_count", "restart_count",
      "continue_timestamps_ms", "last_ms",
    ],
    {
      code_hint: "编号尾号",
      interrupt_count: "中断次数",
      continue_count: "从中断处继续",
      restart_count: "选择重新开始",
      continue_timestamps_ms: "继续时间戳",
      last_ms: "最近一次",
    },
    {
      continue_timestamps_ms: (value) => (
        Array.isArray(value) && value.length
          ? value.map((item) => formatTime(item)).join("；")
          : "—"
      ),
      last_ms: formatTime,
    },
  ));
}

function setAdminView(view) {
  state.activeView = view;
  const coreActive = view === "core";
  $("#admin-core-view").classList.toggle("hidden", !coreActive);
  $("#admin-other-view").classList.toggle("hidden", coreActive);
  $("#show-core-view").classList.toggle("active", coreActive);
  $("#show-other-view").classList.toggle("active", !coreActive);
  $("#show-core-view").setAttribute("aria-selected", String(coreActive));
  $("#show-other-view").setAttribute("aria-selected", String(!coreActive));
}

$("#show-core-view").addEventListener("click", () => setAdminView("core"));
$("#show-other-view").addEventListener("click", () => setAdminView("other"));

function phaseLabel(phase, share) {
  if (phase === "share_fit") {
    const n = Number(share?.completed || 0);
    if (share?.locked_n != null) {
      return `Share 拟合（已锁定 N*=${share.locked_n}）`;
    }
    if (n > 0) {
      return `Share 拟合（从 M(${n}) 继续）`;
    }
    return "Share 拟合";
  }
  if (phase === "contrast") return "三臂对比";
  if (phase === "done") return "已结束";
  return phase || "—";
}

function statusLabel(status) {
  if (status === "running") return "报名中";
  if (status === "paused") return "已停止报名";
  if (status === "stopped") return "已结束";
  return status || "—";
}

function renderOrchestration(data) {
  if (!data) return;
  $("#orch-auto").textContent = data.auto_enabled ? "开启" : "关闭";
  $("#orch-status").textContent = statusLabel(data.status);
  const share = data.share_fit || {};
  $("#orch-phase").textContent = phaseLabel(data.phase, share);
  const serial = share.serial_schedule || {};
  const remaining = share.fit_remaining_estimate || {};
  const lockReasonLabels = {
    manual: "管理员锁定",
    parameter_and_behavior_plateau: "参数+开场平台",
    behavior_rebound: "行为回弹",
    max_donors: "达到50人上限",
    confirm_horizon: "震荡确认上限",
  };
  const reason = lockReasonLabels[share.lock_reason] || share.lock_reason || "";
  const locked = share.locked_n != null
    ? `N*=${share.locked_n}${reason ? `（${reason}）` : ""}`
    : (share.candidate_n != null ? `当前近优点 ${share.candidate_n}` : "未锁定");
  const fitPoint = $("#orch-fit-point");
  if (fitPoint) {
    if (data.phase === "share_fit" && share.locked_n == null) {
      fitPoint.textContent = `链尖 M(${share.completed ?? 0}) · 未锁定 · 拟选 ${remaining.candidate_n ?? share.candidate_n ?? "—"}`;
    } else if (share.locked_n != null) {
      fitPoint.textContent = `已锁定 M(${share.locked_n})`;
    } else {
      fitPoint.textContent = "—";
    }
  }
  $("#orch-share").textContent =
    `${share.completed ?? 0} 链 / ${share.end_to_end_completed_donors ?? 0} 端到端 / ${share.max_donors_for_auto ?? 50} 上限 / ${locked}`;
  const pendingChain = share.protocol_complete_pending_ids || [];
  $("#orch-chain-pending").textContent = `${pendingChain.length} 人`;
  $("#orch-model-finalized").textContent =
    `${share.model_finalization_count ?? 0} / ${share.completed ?? 0}`;
  const counts = data.contrast?.completed_counts || {};
  const assignedCounts = data.contrast?.counts || {};
  const caps = data.contrast?.caps || {};
  if (data.phase === "share_fit") {
    $("#orch-arm1").textContent = "未开始（已退回 Share 拟合）";
    $("#orch-arm2").textContent = "未开始（已退回 Share 拟合）";
    $("#orch-arm3").textContent = "未开始（已退回 Share 拟合）";
    $("#orch-next").textContent = "仍在 Share 拟合，尚未进入三臂";
  } else {
    $("#orch-arm1").textContent = `${counts.cold_learn ?? 0} / ${caps.cold_learn ?? 32}（已分配 ${assignedCounts.cold_learn ?? 0}）`;
    $("#orch-arm2").textContent = `${counts.shared_learn ?? 0} / ${caps.shared_learn ?? 32}（已分配 ${assignedCounts.shared_learn ?? 0}）`;
    $("#orch-arm3").textContent = `${counts.shared_frozen ?? 0} / ${caps.shared_frozen ?? 32}（已分配 ${assignedCounts.shared_frozen ?? 0}）`;
    const labels = data.contrast?.labels || {};
    $("#orch-next").textContent = labels[data.contrast?.next_arm] || data.contrast?.next_arm || "—";
  }
  if (!state.shareFitTrend?.fit_remaining_estimate) {
    renderFitRemainingCard(share.fit_remaining_estimate || {});
  }
  renderBlockedSlotGrid(share);
  renderSessionLock(state.sessionLock);
  const toggle = $("#orch-auto-toggle");
  if (toggle && document.activeElement !== toggle) {
    toggle.checked = Boolean(data.auto_enabled);
  }
  const curveBtn = $("#orch-download-curve");
  const weightsBtn = $("#orch-download-weights");
  const canDownload = Boolean(share.curve_download_available) || Number(share.completed || 0) > 0 || share.locked_n != null;
  if (curveBtn) {
    curveBtn.disabled = !canDownload;
  }
  if (weightsBtn) {
    weightsBtn.disabled = !canDownload;
  }
  $("#orchestration-meta").textContent =
    `实例 ${data.instance_id || "—"} · 更新于 ${formatTime(data.updated_ms)} · ` +
    (data.enrollment_open ? "公开报名开放" : "公开报名关闭") +
    ` · Share 上限=${share.max_donors_for_auto ?? "—"}` +
    ` · 链提交版本=${share.chain_commit_version ?? "—"}` +
    (share.last_chain_error ? ` · 最近链错误=${share.last_chain_error}` : "") +
    ` · 串行位置=${serial.holder_present ? "占用中" : "空闲"} / 有效预约=${serial.active_appointments ?? 0}` +
    ` / 预约窗=${serial.horizon_hours ?? 12}h / 时间块=${serial.slot_minutes ?? 30}min / 缓冲=${serial.spacing_minutes ?? 60}min / 迟到释放=+${serial.late_release_minutes ?? 10}min` +
    ` / 准备栏目释放=${serial.setup_release_minutes ?? 30}min / 正式互动硬释放=${serial.unfinished_release_minutes ?? 90}min` +
    ` · 参数闸门 min=${share.min_donors_for_auto ?? "—"} / stable=${share.stable_required ?? "—"} / rel_delta<${share.rel_delta_threshold ?? "—"}` +
    ` · 开场闸门 median=${share.behavior_window ?? "—"} / 改善<${share.opening_improve_epsilon ?? "—"} / 回弹>${share.tap_delta_tolerance ?? "—"}` +
    ` / 平台确认=${share.plateau_persist_after_nonoverlap ?? remaining.plateau_persist_after_nonoverlap ?? "—"}` +
    ` / 回弹确认=${share.rebound_confirm_windows ?? remaining.rebound_confirm_windows ?? "—"}` +
    ` / 确认帽=${share.confirm_horizon_donors ?? remaining.confirm_horizon_donors ?? "—"}`;
  const appointmentStatusLabels = {
    booked: "已预约",
    claimed: "已验证/待开始",
    active: "实验中",
    completed: "已完成",
    released: "已释放",
  };
  const releaseReasonLabels = {
    late_over_10_minutes: "迟到超过10分钟",
    participant_rescheduled: "参与者改约",
    participant_logout: "参与者主动退出编号",
    session_aborted: "实验中止",
    session_hard_timeout: "未完成超时",
    setup_over_30_minutes: "准备栏目超过30分钟",
    serial_slot_timeout: "位置超时",
    active_predecessor_detected: "检测到前一位仍在实验",
  };
  const subjectById = new Map(
    (state.overview?.participants || []).map((row) => [String(row.id), row]),
  );
  $("#orch-appointment-table").replaceChildren(buildTable(
    (serial.appointments || []).map((row) => {
      const subject = subjectById.get(String(row.participant_id || "")) || {};
      return {
        ...row,
        display_name: subject.display_name || "",
        contact_line: contactLine(subject.contact),
      };
    }),
    [
      "scheduled_start_ms", "voucher", "display_name", "contact_line",
      "participant_id", "status", "arrived_ms", "credential_verified_ms",
      "release_reason",
    ],
    {
      scheduled_start_ms: "预约时间",
      voucher: "四位凭证",
      display_name: "姓名",
      contact_line: "联系方式",
      participant_id: "参与者 ID",
      status: "状态",
      arrived_ms: "到达时间",
      credential_verified_ms: "凭证验证时间",
      release_reason: "释放原因",
    },
    {
      scheduled_start_ms: formatTime,
      voucher: (value) => value || "—",
      display_name: displayName,
      participant_id: (value) => value ? `…${String(value).slice(-8)}` : "—",
      status: (value) => appointmentStatusLabels[value] || value || "—",
      arrived_ms: formatTime,
      credential_verified_ms: formatTime,
      release_reason: (value) => releaseReasonLabels[value] || value || "—",
    },
  ));
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function drawShareFitCurve(canvas, curve) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { top: 48, right: 36, bottom: 56, left: 64 };
  const points = curve.points || [];
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7f4ef";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#1f1a17";
  ctx.font = "600 20px 'Segoe UI', sans-serif";
  ctx.fillText("Share 拟合：tap 频次随完成人数变化", pad.left, 30);
  ctx.font = "13px 'Segoe UI', sans-serif";
  ctx.fillStyle = "#5c534c";
  ctx.fillText(
    `${curve.instance_id || ""} · 完成 ${curve.completed || 0}` +
      (curve.locked_n != null ? ` · N*=${curve.locked_n}` : "") +
      ` · 分母 ${curve.effective_interaction_minutes} min`,
    pad.left,
    48,
  );
  if (!points.length) {
    ctx.fillText("暂无数据点", pad.left, height / 2);
    return;
  }
  const xs = points.map((row) => Number(row.n));
  const ysIndividual = points.map((row) => Number(row.taps_per_minute));
  const ysMean = points.map((row) => Number(row.cumulative_mean_taps_per_minute));
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMax = Math.max(0.05, ...ysIndividual, ...ysMean) * 1.15;
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const xMap = (value) => pad.left + ((value - xMin) / Math.max(1, xMax - xMin)) * plotW;
  const yMap = (value) => pad.top + plotH - (value / yMax) * plotH;

  ctx.strokeStyle = "#d8cfc4";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#5c534c";
  ctx.font = "12px 'Segoe UI', sans-serif";
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = (yMax * tick) / 4;
    const y = yMap(value);
    ctx.strokeStyle = "#ece4da";
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = "#5c534c";
    ctx.fillText(value.toFixed(2), 12, y + 4);
  }
  xs.forEach((value) => {
    ctx.fillText(String(value), xMap(value) - 4, pad.top + plotH + 20);
  });
  ctx.fillText("完成人数 n", pad.left + plotW / 2 - 28, height - 16);
  ctx.save();
  ctx.translate(18, pad.top + plotH / 2 + 40);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("tap / 分钟", 0, 0);
  ctx.restore();

  function drawSeries(values, color, dashed) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash(dashed ? [8, 6] : []);
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = xMap(xs[index]);
      const y = yMap(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    values.forEach((value, index) => {
      ctx.beginPath();
      ctx.arc(xMap(xs[index]), yMap(value), 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  drawSeries(ysIndividual, "#b45309", true);
  drawSeries(ysMean, "#0f766e", false);

  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.fillStyle = "#b45309";
  ctx.fillRect(pad.left, height - 40, 14, 3);
  ctx.fillText("单人 taps/min", pad.left + 20, height - 36);
  ctx.fillStyle = "#0f766e";
  ctx.fillRect(pad.left + 140, height - 40, 14, 3);
  ctx.fillText("累计均值 taps/min（随人数）", pad.left + 160, height - 36);
}

async function downloadShareFitCurve() {
  $("#orchestration-error").textContent = "";
  try {
    const curve = await adminApi("/api/admin/orchestration/share-fit-curve");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const instance = curve.instance_id || "study";
    const csvHeader = [
      "n",
      "participant_id",
      "tap_count",
      "manual_activate_count",
      "interrupt_count",
      "mistouch_count",
      "eligible_opportunities",
      "eligible_correction_count",
      "correction_burden",
      "opening_correction_burden",
      "opening_eligible_opportunities",
      "opening_eligible_correction_count",
      "taps_per_minute",
      "cumulative_mean_taps_per_minute",
      "rel_delta",
      "rolling_3_correction_burden",
      "tap_delta",
      "behavior_state",
      "prior_id",
    ];
    const csvRows = (curve.points || []).map((row) => csvHeader.map((key) => {
      const value = row[key];
      if (value == null) return "";
      const text = String(value);
      return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    }).join(","));
    downloadBlob(
      `share-fit-tap-curve-${instance}-${stamp}.csv`,
      new Blob([`\ufeff${csvHeader.join(",")}\n${csvRows.join("\n")}\n`], {
        type: "text/csv;charset=utf-8",
      }),
    );
    const canvas = $("#orch-curve-canvas");
    drawShareFitCurve(canvas, curve);
    const png = await new Promise((resolve, reject) => {
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("无法生成折线图"))), "image/png");
    });
    downloadBlob(`share-fit-tap-curve-${instance}-${stamp}.png`, png);
  } catch (err) {
    $("#orchestration-error").textContent = err.message;
  }
}

async function orchestrationAction(path, body) {
  $("#orchestration-error").textContent = "";
  try {
    const payload = await adminApi(path, {
      method: "POST",
      body: body === undefined ? {} : body,
    });
    state.orchestration = payload;
    renderOrchestration(payload);
  } catch (err) {
    $("#orchestration-error").textContent = err.message;
  }
}

$("#orch-auto-toggle")?.addEventListener("change", async (event) => {
  await orchestrationAction("/api/admin/orchestration/auto", {
    enabled: Boolean(event.target.checked),
  });
});
$("#orch-pause")?.addEventListener("click", () => orchestrationAction("/api/admin/orchestration/pause"));
$("#orch-resume")?.addEventListener("click", () => orchestrationAction("/api/admin/orchestration/resume"));
$("#orch-refresh-share")?.addEventListener("click", () => orchestrationAction("/api/admin/orchestration/refresh-share-fit"));
$("#orch-lock-share")?.addEventListener("click", () => {
  if (!window.confirm("确认用当前已完成 donor 锁定 share 拟合，并开启 ①②③ 对比阶段？")) return;
  return orchestrationAction("/api/admin/orchestration/lock-share-fit", { donor_count: null });
});
$("#orch-early-stop")?.addEventListener("click", () => {
  if (!window.confirm("提前结束实验后将不再接受新报名。确定吗？")) return;
  return orchestrationAction("/api/admin/orchestration/early-stop");
});
$("#orch-download-curve")?.addEventListener("click", () => downloadShareFitCurve());
$("#orch-download-weights")?.addEventListener("click", () => downloadShareFitWeights());
$("#session-lock-save")?.addEventListener("click", () => saveSessionLock());
$("#session-lock-clear")?.addEventListener("click", () => clearSessionLock());
$("#session-lock-copy")?.addEventListener("click", () => copySessionLockPassword());
$("#orch-block-date")?.addEventListener("change", () => {
  renderBlockedSlotGrid(state.orchestration?.share_fit || {});
});
$("#orch-block-select-day")?.addEventListener("click", () => {
  $$("#orch-block-slot-grid input[type='checkbox']").forEach((box) => {
    box.checked = true;
    box.closest(".orch-block-slot")?.classList.add("is-blocked");
  });
});
$("#orch-block-clear-day")?.addEventListener("click", () => {
  $$("#orch-block-slot-grid input[type='checkbox']").forEach((box) => {
    box.checked = false;
    box.closest(".orch-block-slot")?.classList.remove("is-blocked");
  });
});
$("#orch-block-save")?.addEventListener("click", () => saveBlockedSlots());

async function downloadShareFitWeights() {
  $("#orchestration-error").textContent = "";
  const button = $("#orch-download-weights");
  if (button) button.disabled = true;
  try {
    const response = await fetch(`${APP_BASE}/api/admin/orchestration/share-fit-weights`, {
      headers: { "X-Admin-Token": state.token },
      cache: "no-store",
    });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        if (typeof payload.detail === "string") detail = payload.detail;
      } catch (_) { /* Keep HTTP status text. */ }
      throw new Error(detail);
    }
    const blob = await response.blob();
    const header = response.headers.get("content-disposition") || "";
    const match = /filename\*?=(?:UTF-8'')?["']?([^"';]+)/i.exec(header);
    const filename = match
      ? decodeURIComponent(match[1])
      : `share-fit-weights-${new Date().toISOString().replace(/[:.]/g, "-")}.zip`;
    downloadBlob(filename, blob);
  } catch (err) {
    $("#orchestration-error").textContent = err.message;
  } finally {
    if (button) button.disabled = false;
  }
}

function renderStageTemplates(data) {
  const select = $("#stage-template");
  const previous = select.value;
  select.replaceChildren();
  for (const stage of data.stages) {
    const option = document.createElement("option");
    option.value = stage.id;
    option.textContent = stage.initializable ? stage.label : `${stage.label}（流水线管理）`;
    option.disabled = !stage.initializable;
    select.appendChild(option);
  }
  if (previous && data.stages.some((stage) => stage.id === previous && stage.initializable)) {
    select.value = previous;
  }
  updateStageDescription();
}

function updateStageDescription() {
  const stage = state.stages.find((item) => item.id === $("#stage-template").value);
  if (!stage) return;
  $("#stage-personal-debug").checked = Boolean(stage.personal_debug_default);
  updatePersonalDebugControls(true);
}

function updatePersonalDebugControls(resetCount = false) {
  const stage = state.stages.find((item) => item.id === $("#stage-template").value);
  if (!stage) return;
  const personalDebug = $("#stage-personal-debug").checked;
  if (resetCount || personalDebug) {
    $("#stage-count").value = personalDebug ? 1 : (stage.recommended_count ?? "");
  }
  $("#stage-count").max = personalDebug ? 1 : (stage.maximum_count ?? "");
  $("#stage-description").textContent =
    `${stage.description} 配置：${stage.cohort || "—"} / ${stage.group_name || "—"} / ` +
    `${stage.protocol_mode || "—"}；` +
    (personalDebug
      ? "当前仅用于个人调试：限 1 个账号，数据与模型永久隔离。"
      : `单批上限 ${stage.maximum_count ?? "由正式流水线决定"}。`);
  $("#create-stage-accounts").disabled = !stage.initializable;
}

$("#stage-template").addEventListener("change", updateStageDescription);
$("#stage-personal-debug").addEventListener("change", () => updatePersonalDebugControls(true));

$("#stage-accounts-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#stage-error").textContent = "";
  $("#create-stage-accounts").disabled = true;
  try {
    const count = Number.parseInt($("#stage-count").value, 10);
    if (!Number.isInteger(count) || count < 1) throw new Error("指派人数必须是正整数");
    const result = await adminApi("/api/admin/stage-accounts", {
      method: "POST",
      body: {
        stage: $("#stage-template").value,
        count,
        run_slug: $("#stage-run-slug").value.trim() || null,
        label: $("#stage-label").value.trim(),
        personal_debug: $("#stage-personal-debug").checked,
      },
    });
    state.createdAccounts = result.accounts;
    state.createdBatch = result;
    $("#created-accounts-meta").textContent =
      `${result.stage_label} · ${result.run_slug} · 指派 ${result.assigned_count} 人 · ` +
      `${result.personal_debug ? "仅用于个人调试；使用独立调试目录，不影响本实例常规数据" :
        "实验数据按所选阶段管理"} · ` +
      `${result.personal_debug ? `隔离目录 ${result.archive.archive_path} · ` : ""}批次 ${result.batch_id}`;
    renderCreatedAccounts(result.accounts);
    $("#created-accounts").classList.remove("hidden");
    await refreshAdmin();
  } catch (err) {
    $("#stage-error").textContent = `生成失败：${err.message}`;
  } finally {
    const selected = state.stages.find((item) => item.id === $("#stage-template").value);
    $("#create-stage-accounts").disabled = !selected?.initializable;
  }
});

function renderCreatedAccounts(accounts) {
  $("#created-accounts-table").replaceChildren(buildTable(
    accounts,
    ["access_code", "participant_id", "run_slug", "cohort", "group_name", "protocol_mode"],
    {
      access_code: "一次性登录码",
      participant_id: "内部参与者 ID",
      run_slug: "Run",
      cohort: "Cohort",
      group_name: "组",
      protocol_mode: "协议",
    },
    {
      access_code: codeCell,
      participant_id: codeCell,
      run_slug: codeCell,
    },
  ));
}

function accountRows() {
  return state.createdAccounts.map((row) => ({
    access_code: row.access_code,
    participant_id: row.participant_id,
    run_slug: row.run_slug,
    cohort: row.cohort,
    group_name: row.group_name,
    protocol_mode: row.protocol_mode,
  }));
}

function rowsToDelimited(rows, delimiter) {
  const columns = ["access_code", "participant_id", "run_slug", "cohort", "group_name", "protocol_mode"];
  const escape = (value) => {
    const text = String(value ?? "");
    if (delimiter === "\t") return text.replaceAll("\t", " ").replaceAll("\n", " ");
    return `"${text.replaceAll('"', '""')}"`;
  };
  return [columns.join(delimiter), ...rows.map((row) => columns.map((column) => escape(row[column])).join(delimiter))].join("\n");
}

$("#copy-accounts").addEventListener("click", async () => {
  await navigator.clipboard.writeText(rowsToDelimited(accountRows(), "\t"));
  $("#copy-accounts").textContent = "已复制";
});

$("#download-accounts").addEventListener("click", () => {
  const csv = `\ufeff${rowsToDelimited(accountRows(), ",")}`;
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `tap-to-adapt-accounts-${Date.now()}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
});

function renderOverview(data) {
  const counts = data.counts;
  $("#metric-runs").textContent = counts.runs;
  $("#metric-participants").textContent = counts.participants;
  $("#metric-sessions").textContent = counts.sessions;
  $("#metric-events").textContent = counts.events;
  $("#metric-audio").textContent = `${counts.audio_files} / ${formatBytes(counts.audio_bytes)}`;
  $("#metric-releases").textContent = counts.dataset_releases;
  renderRunFilter();

  $("#paths-table").replaceChildren(buildTable(
    data.paths,
    ["label", "kind", "path", "exists"],
    { label: "用途", kind: "类型", path: "服务器绝对路径", exists: "当前存在" },
    { path: codeCell, exists: (value) => yesNo(value) },
  ));

  $("#runs-table").replaceChildren(buildTable(
    data.runs,
    ["run_slug", "mode", "participants", "sessions", "events", "audio_files", "audio_bytes", "created_ms"],
    {
      run_slug: "Run",
      mode: "协议模式",
      participants: "账号",
      sessions: "会话",
      events: "事件",
      audio_files: "音频",
      audio_bytes: "音频大小",
      created_ms: "创建时间",
    },
    { run_slug: codeCell, audio_bytes: formatBytes, created_ms: formatTime },
  ));

  const stageLabels = new Map(state.stages.map((stage) => [stage.id, stage.label]));
  $("#batches-table").replaceChildren(buildTable(
    data.account_batches,
    ["action", "label", "stage", "requested_count", "personal_debug", "archive_absolute_path", "created_ms"],
    {
      action: "",
      label: "批次备注",
      stage: "实验阶段",
      requested_count: "指派人数",
      personal_debug: "仅个人调试",
      archive_absolute_path: "统一归档目录",
      created_ms: "创建时间",
    },
    {
      action: (_value, row) => {
        if (!row.has_archive) return "—";
        const button = document.createElement("button");
        button.className = "btn btn-secondary admin-small-button";
        button.type = "button";
        button.textContent = "刷新归档";
        button.addEventListener("click", () => refreshBatchArchive(row.id, button));
        return button;
      },
      stage: (value) => stageLabels.get(value) || value,
      personal_debug: (value) => value ? "是（永久隔离）" : "否",
      archive_absolute_path: codeCell,
      created_ms: formatTime,
    },
  ));

  const participants = data.participants.map((row) => ({
    ...row,
    contact_email: row.contact?.email || "",
    contact_phone: row.contact?.phone || "",
    contact_wechat: row.contact?.wechat || "",
    w1: windowTapCell(row.window_taps, 1),
    w2: windowTapCell(row.window_taps, 2),
    w3: windowTapCell(row.window_taps, 3),
    w4: windowTapCell(row.window_taps, 4),
    w5: windowTapCell(row.window_taps, 5),
    identity_label: row.baseline_demographics?.identity_label || "—",
    spoken_accent_label: row.baseline_demographics?.spoken_accent_label || "—",
    consent_signed: row.consent?.signed,
    audio_consent: row.consent?.raw_audio,
  }));
  $("#participants-table").replaceChildren(buildTable(
    participants,
    [
      "action", "experiment_status_label", "display_name", "contact_email",
      "contact_phone", "contact_wechat",
      "w1", "w2", "w3", "w4", "w5", "code_hint", "experiment_arm",
      "identity_label", "spoken_accent_label",
      "days_started", "event_count", "consent_signed", "last_session_ms",
    ],
    {
      action: "",
      experiment_status_label: "实验状态",
      display_name: "姓名",
      contact_email: "邮箱",
      contact_phone: "电话",
      contact_wechat: "微信",
      w1: "窗口1 点击/打断",
      w2: "窗口2 点击/打断",
      w3: "窗口3 点击/打断",
      w4: "窗口4 点击/打断",
      w5: "窗口5 点击/打断",
      code_hint: "登录码尾号",
      experiment_arm: "实验阶段 / 臂",
      identity_label: "身份",
      spoken_accent_label: "口音或方言习惯",
      days_started: "已开始天数",
      event_count: "事件",
      consent_signed: "已签同意",
      last_session_ms: "最近会话",
    },
    {
      action: (_value, row) => {
        const button = document.createElement("button");
        button.className = "btn btn-secondary admin-small-button";
        button.type = "button";
        button.textContent = "查看";
        button.addEventListener("click", () => loadParticipant(row.id));
        return button;
      },
      experiment_status_label: (value, row) => liveStatusChip(
        value || "—",
        row.experiment_status || "not_started",
      ),
      display_name: displayName,
      contact_email: contactField,
      contact_phone: contactField,
      contact_wechat: contactField,
      experiment_arm: experimentArmLabel,
      consent_signed: yesNo,
      last_session_ms: formatTime,
    },
  ));

  $("#event-types-table").replaceChildren(buildTable(
    data.event_type_counts,
    ["event_type", "count"],
    { event_type: "事件类型", count: "数量" },
    { event_type: codeCell },
  ));
  $("#dataset-releases-table").replaceChildren(buildTable(
    data.dataset_releases,
    ["run_slug", "schema_version", "files", "path", "modified_ms"],
    { run_slug: "Run", schema_version: "Schema", files: "文件数", path: "发布目录", modified_ms: "更新时间" },
    { run_slug: codeCell, path: codeCell, modified_ms: formatTime },
  ));
  $("#audio-assets-table").replaceChildren(buildTable(
    data.audio_assets,
    ["participant_id", "day", "kind", "bytes", "absolute_path", "exists"],
    { participant_id: "参与者 ID", day: "Day", kind: "类型", bytes: "大小", absolute_path: "绝对路径", exists: "存在" },
    { participant_id: codeCell, bytes: formatBytes, absolute_path: codeCell, exists: yesNo },
  ));
  $("#model-assets-table").replaceChildren(buildTable(
    data.model_assets,
    ["kind", "participant_id", "encoder_path", "main_path", "created_ms"],
    { kind: "类型", participant_id: "参与者 ID", encoder_path: "Encoder", main_path: "Head/Main", created_ms: "创建时间" },
    { participant_id: codeCell, encoder_path: codeCell, main_path: codeCell, created_ms: formatTime },
  ));
}

async function refreshBatchArchive(batchId, button) {
  const status = $("#archive-status");
  status.textContent = "";
  button.disabled = true;
  try {
    const result = await adminApi(`/api/admin/account-batches/${encodeURIComponent(batchId)}/archive`, {
      method: "POST",
    });
    status.textContent =
      `归档已刷新：${result.archive_path}；参与者 ${result.assigned_count} 人，` +
      `事件 ${result.record_counts.events || 0} 条，音频 ${result.audio_file_count} 个。`;
    button.textContent = "已刷新";
  } catch (err) {
    status.textContent = `归档失败：${err.message}`;
  } finally {
    button.disabled = false;
  }
}

function renderRunFilter() {
  const select = $("#run-filter");
  const previous = select.value;
  select.replaceChildren();
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "全部 Run";
  select.appendChild(all);
  for (const run of state.allRuns) {
    const option = document.createElement("option");
    option.value = run.run_slug;
    option.textContent = `${run.run_slug}（${run.participants} 人）`;
    select.appendChild(option);
  }
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
}

$("#run-filter").addEventListener("change", () => {
  $("#participant-detail").classList.add("hidden");
  refreshAdmin().catch((err) => {
    $("#admin-status").textContent = `刷新失败：${err.message}`;
    $("#admin-dot").className = "status-dot error";
  });
});

async function loadParticipant(participantId) {
  const detail = await adminApi(`/api/admin/participants/${encodeURIComponent(participantId)}/records`);
  state.selectedParticipantId = participantId;
  $("#participant-detail-title").textContent =
    detail.participant?.display_name || participantId;
  const consent = detail.consent;
  $("#download-research-consent-copy").classList.toggle("hidden", !consent.signed);
  $("#download-research-consent-evidence").classList.toggle(
    "hidden",
    !consent.third_party_verifiable,
  );
  const cards = [
    ["姓名", displayName(detail.participant?.display_name)],
    ["邮箱", contactField(detail.contact?.email)],
    ["电话", contactField(detail.contact?.phone)],
    ["微信", contactField(detail.contact?.wechat)],
    ["点击合计", detail.window_taps?.tap_total ?? 0],
    ["打断合计", detail.window_taps?.interrupt_total ?? 0],
    ["Run", detail.participant.run_slug],
    ["实验阶段 / 臂", experimentArmLabel(detail.participant.experiment_arm)],
    ["身份", detail.baseline_demographics?.identity_label || "—"],
    ["口音或方言习惯", detail.baseline_demographics?.spoken_accent_label || "—"],
    ["年龄段", detail.baseline_demographics?.age_range_label || "—"],
    ["性别", detail.baseline_demographics?.gender_label || "—"],
    ["会话", detail.sessions.length],
    ["事件", detail.recent_events.length],
    ["已签同意", yesNo(consent.signed)],
    ["原始音频许可", yesNo(consent.raw_audio)],
  ];
  const summary = $("#participant-detail-summary");
  summary.replaceChildren();
  for (const [label, value] of cards) {
    const item = document.createElement("div");
    item.className = "admin-metric";
    const small = document.createElement("span");
    small.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    item.append(small, strong);
    summary.appendChild(item);
  }
  $("#participant-window-taps-table").replaceChildren(buildTable(
    detail.window_taps?.windows || [],
    ["window", "taps", "interrupts"],
    { window: "窗口", taps: "点击", interrupts: "打断" },
  ));
  $("#participant-events-table").replaceChildren(buildTable(
    detail.event_type_counts,
    ["event_type", "count"],
    { event_type: "事件类型", count: "数量" },
    { event_type: codeCell },
  ));
  $("#participant-sessions-table").replaceChildren(buildTable(
    detail.sessions,
    ["day", "state", "started_ms", "completed_ms", "ended_ms"],
    { day: "Day", state: "状态", started_ms: "开始", completed_ms: "完成", ended_ms: "结束" },
    { started_ms: formatTime, completed_ms: formatTime, ended_ms: formatTime },
  ));
  $("#participant-conversation-table").replaceChildren(buildTable(
    detail.conversation_turns || [],
    ["server_ms", "source", "transcript", "reply", "model"],
    {
      server_ms: "时间",
      source: "对话对象",
      transcript: "用户发言（ASR）",
      reply: "智能体回答",
      model: "模型",
    },
    {
      server_ms: formatTime,
      source: conversationSourceLabel,
      transcript: longTextCell,
      reply: longTextCell,
      model: codeCell,
    },
  ));
  const textEvents = (detail.recent_events || []).filter(
    (row) => row.text_fields && Object.keys(row.text_fields).length,
  );
  $("#participant-text-events-table").replaceChildren(buildTable(
    textEvents,
    ["server_ms", "event_type", "text_fields"],
    { server_ms: "时间", event_type: "事件类型", text_fields: "原文" },
    { server_ms: formatTime, event_type: codeCell, text_fields: eventTextCell },
  ));
  const questionnaireRows = (detail.responses || []).flatMap((response) => (
    (response.selections || []).map((selection) => ({
      instrument: response.instrument,
      position: selection.position,
      question_text: selection.question_text,
      answer_text: selection.answer_text,
      answer_value: selection.answer_value,
      created_ms: response.created_ms,
    }))
  ));
  $("#participant-responses-table").replaceChildren(buildTable(
    questionnaireRows,
    ["instrument", "position", "question_text", "answer_text", "answer_value", "created_ms"],
    {
      instrument: "问卷",
      position: "呈现顺序",
      question_text: "题目内容",
      answer_text: "所选选项内容",
      answer_value: "稳定编码",
      created_ms: "提交时间",
    },
    { created_ms: formatTime, answer_value: codeCell },
  ));
  const openResponseRows = (detail.responses || []).flatMap((response) => (
    (response.open_answers || []).map((answer) => ({
      instrument: response.instrument,
      question_text: answer.question_text,
      answer_text: answer.answer_text,
      created_ms: response.created_ms,
    }))
  ));
  $("#participant-open-responses-table").replaceChildren(buildTable(
    openResponseRows,
    ["instrument", "question_text", "answer_text", "created_ms"],
    {
      instrument: "问卷",
      question_text: "题目内容",
      answer_text: "受试者原文",
      created_ms: "提交时间",
    },
    { answer_text: longTextCell, created_ms: formatTime },
  ));
  $("#participant-audio-table").replaceChildren(buildTable(
    detail.audio_files,
    ["kind", "bytes", "content_type", "absolute_path", "exists"],
    { kind: "类型", bytes: "大小", content_type: "格式", absolute_path: "绝对路径", exists: "存在" },
    { bytes: formatBytes, absolute_path: codeCell, exists: yesNo },
  ));
  $("#participant-detail").classList.remove("hidden");
  $("#participant-detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("#close-participant-detail").addEventListener("click", () => $("#participant-detail").classList.add("hidden"));

$("#download-research-consent-copy").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  if (!state.selectedParticipantId) return;
  button.disabled = true;
  try {
    const response = await adminApi(
      `/api/admin/participants/${encodeURIComponent(state.selectedParticipantId)}/consent-copy`,
    );
    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") || "";
    const matchedName = disposition.match(/filename="([^"]+)"/i);
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = matchedName?.[1] || "tap-to-adapt-consent-research-archive.html";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
  } catch (err) {
    window.alert(`研究方归档副本下载失败：${err.message}`);
  } finally {
    button.disabled = false;
  }
});

$("#download-research-consent-evidence").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  if (!state.selectedParticipantId) return;
  button.disabled = true;
  try {
    const response = await adminApi(
      `/api/admin/participants/${encodeURIComponent(state.selectedParticipantId)}/consent-evidence`,
    );
    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") || "";
    const matchedName = disposition.match(/filename="([^"]+)"/i);
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = matchedName?.[1] || "tap-to-adapt-consent-evidence.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
  } catch (err) {
    window.alert(`同意证据包下载失败：${err.message}`);
  } finally {
    button.disabled = false;
  }
});

const READINESS_ADVISORY_KEYS = new Set([
  "consent_ethics_finalized",
  "compensation_ethics_finalized",
]);

function renderReadiness(data) {
  $("#readiness-summary").textContent = data.formal_ready
    ? "所有硬门禁已通过，可以按预注册流水线释放正式参与者。"
    : "正式 Study A/B 仍被锁定；Pilot 和 Simulation 阶段不受此状态影响。";
  const rows = Object.entries(data.checks).map(([name, passed]) => ({
    name: READINESS_ADVISORY_KEYS.has(name) ? `${name}（状态标记）` : name,
    passed: passed ? "通过" : "未通过",
  }));
  $("#readiness-table").replaceChildren(buildTable(rows, ["name", "passed"], { name: "检查项", passed: "状态" }));
}

$("#refresh-admin").addEventListener("click", () => {
  refreshAdmin().catch((err) => {
    $("#admin-status").textContent = `刷新失败：${err.message}`;
    $("#admin-dot").className = "status-dot error";
  });
});

$("#refresh-live-subjects")?.addEventListener("click", () => {
  refreshLiveSubjects().catch((err) => {
    $("#admin-status").textContent = `看板刷新失败：${err.message}`;
    $("#admin-dot").className = "status-dot error";
  });
});

$("#refresh-share-fit-trend")?.addEventListener("click", () => {
  refreshShareFitTrendBoard().catch((err) => {
    $("#admin-status").textContent = `看板刷新失败：${err.message}`;
    $("#admin-dot").className = "status-dot error";
  });
});

$("#export-data").addEventListener("click", async () => {
  const response = await adminApi("/api/admin/export");
  const blob = await response.blob();
  const disposition = response.headers.get("content-disposition") || "";
  const match = disposition.match(/filename="([^"]+)"/);
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = match ? match[1] : "tap-to-adapt-user-study-data.zip";
  anchor.click();
  URL.revokeObjectURL(anchor.href);
});

function pipelineRun() {
  const value = $("#pipeline-run").value.trim();
  if (!value) throw new Error("请先填写 Formal run");
  return value;
}

function renderPipeline(data) {
  const rows = Object.entries(data.waves).map(([wave, value]) => ({
    wave,
    count: value.count,
    released: value.released,
    complete: value.complete,
    priorReady: value.eligible_for_prior ?? "—",
    blockers: value.prior_blockers?.length ?? "—",
  }));
  const snapshots = Object.entries(data.snapshot_counts).map(([q, count]) => `q=${q}: ${count}`).join(" · ");
  const selected = data.shared_priors.find((prior) => prior.status === "selected");
  const probes = data.candidate_probes || {};
  $("#pipeline-meta").textContent =
    `Snapshots ${snapshots} · shared priors ${data.shared_priors.length} · ` +
    `candidate probes ${probes.participants_complete || 0}/${probes.participants_assigned || 0} · ` +
    `selected ${selected ? `n=${selected.n}, q=${selected.q}` : "尚未选择"}`;
  $("#pipeline-table").replaceChildren(buildTable(
    rows,
    ["wave", "count", "released", "complete", "priorReady", "blockers"],
    { wave: "Wave", count: "人数", released: "已释放", complete: "完整会话", priorReady: "Prior 可用", blockers: "Prior 阻塞" },
  ));
}

async function loadPipeline() {
  const data = await adminApi(`/api/admin/pipeline/${encodeURIComponent(pipelineRun())}`);
  renderPipeline(data);
  return data;
}

async function loadPipelineAnalysis() {
  const data = await adminApi(`/api/admin/pipeline/${encodeURIComponent(pipelineRun())}/analysis`);
  $("#pipeline-analysis").textContent = JSON.stringify(data, null, 2);
  return data;
}

async function pipelineAction(action) {
  const error = $("#pipeline-error");
  error.textContent = "";
  try {
    await action();
    await loadPipeline();
  } catch (err) {
    error.textContent = err.message;
  }
}

$("#pipeline-status-btn").addEventListener("click", () => pipelineAction(loadPipeline));
$("#pipeline-analysis-btn").addEventListener("click", () => pipelineAction(loadPipelineAnalysis));
$("#release-wave-btn").addEventListener("click", () => pipelineAction(() => adminApi("/api/admin/pipeline/release-study-a", {
  method: "POST",
  body: { run_slug: pipelineRun(), wave: Number($("#pipeline-wave").value) },
})));
$("#prepare-study-b-btn").addEventListener("click", () => pipelineAction(() => adminApi("/api/admin/pipeline/prepare-study-b", {
  method: "POST",
  body: {
    run_slug: pipelineRun(),
    selected_n: Number($("#selected-n").value),
    selected_q: Number($("#selected-q").value),
    decision_record: $("#decision-record").value.trim(),
  },
})));
$("#release-reserve-btn").addEventListener("click", () => pipelineAction(() => {
  const reason = $("#reserve-reason").value.trim();
  if (reason.length < 10) throw new Error("Reserve 释放原因至少填写 10 个字符");
  return adminApi("/api/admin/pipeline/release-reserve", {
    method: "POST",
    body: { run_slug: pipelineRun(), reason },
  });
}));
