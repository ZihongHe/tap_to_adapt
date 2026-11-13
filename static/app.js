/**
 * @file Desktop participant controller: enrollment, consent, device checks, pairing, timed interaction, corrective taps, and questionnaires. Server acknowledgements determine durable state; local rendering never substitutes for an acknowledged model update.
 */

"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));
const APP_BASE = new URL(document.querySelector('script[src*="static/app.js"]').src).pathname.replace(/\/static\/app\.js$/, "");
const I18N = window.TTA_I18N || {
  locale: "zh-CN",
  isEnglish: false,
  t: (value) => value,
  apiPath: (value) => value,
};
const STORAGE_PREFIX = `tta:${APP_BASE || "/"}`;
const storageKey = (name) => `${STORAGE_PREFIX}:${name}`;
const FULL_STUDY_CODE_PATTERN = /^TTA-[A-Z0-9]{4}-[A-Z0-9]{4}$/;

function normalizedStudyCode(value) {
  const raw = String(value || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
  if (raw.startsWith("TTA") && raw.length >= 11) {
    return `TTA-${raw.slice(3, 7)}-${raw.slice(7, 11)}`;
  }
  return String(value || "").trim().toUpperCase();
}

function isFullStudyCode(value) {
  return FULL_STUDY_CODE_PATTERN.test(normalizedStudyCode(value));
}

function rememberStudyCode(code, hint) {
  const full = isFullStudyCode(code) ? normalizedStudyCode(code) : "";
  if (full) {
    state.code = full;
    localStorage.setItem(storageKey("study_code"), full);
  }
  if (hint) {
    localStorage.setItem(storageKey("study_code_hint"), String(hint).trim().toUpperCase());
  }
  return full;
}

function resolveFullStudyCode(preferred, hint) {
  const expectedHint = String(
    hint
    || localStorage.getItem(storageKey("study_code_hint"))
    || state.bootstrap?.participant?.code_hint
    || "",
  ).trim().toUpperCase();
  for (const raw of [preferred, state.code, localStorage.getItem(storageKey("study_code"))]) {
    if (!isFullStudyCode(raw)) continue;
    const code = normalizedStudyCode(raw);
    if (expectedHint && !code.endsWith(expectedHint)) continue;
    return code;
  }
  return "";
}

function staticUrl(path) {
  const cleaned = String(path || "").replace(/^\/+/, "");
  return `${APP_BASE}/${cleaned}`.replace(/\/{2,}/g, "/");
}

function bindStaticAssetSources(root = document) {
  root.querySelectorAll("[data-static-src]").forEach((node) => {
    const relative = node.getAttribute("data-static-src");
    if (relative) node.setAttribute("src", staticUrl(relative));
  });
  root.querySelectorAll("img[src^='static/']").forEach((node) => {
    node.setAttribute("src", staticUrl(node.getAttribute("src")));
  });
}

function classifyClientDevice() {
  const userAgent = navigator.userAgent || "";
  const clientHintsMobile = navigator.userAgentData?.mobile === true;
  const ipadDesktopMode = /Macintosh/i.test(userAgent) && navigator.maxTouchPoints > 1;
  const tablet = ipadDesktopMode || /iPad|Tablet|Kindle|Silk|PlayBook/i.test(userAgent)
    || (/Android/i.test(userAgent) && !/Mobile/i.test(userAgent));
  const phone = !tablet && (
    clientHintsMobile
    || /iPhone|iPod|Windows Phone|IEMobile|Opera Mini|Android.+Mobile|Mobile Safari/i.test(userAgent)
  );
  if (tablet) return "tablet";
  if (phone) return "phone";
  if (window.matchMedia?.("(any-pointer: fine)").matches) return "desktop";
  return "unknown";
}

const CLIENT_DEVICE_CLASS = classifyClientDevice();

function studyDeviceId() {
  const key = storageKey("study_device_id");
  let value = localStorage.getItem(key);
  if (!value) {
    value = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    localStorage.setItem(key, value);
  }
  return value;
}

const state = {
  token: localStorage.getItem(storageKey("study_token")) || "",
  code: (() => {
    const stored = localStorage.getItem(storageKey("study_code")) || "";
    return isFullStudyCode(stored) ? normalizedStudyCode(stored) : "";
  })(),
  bootstrap: null,
  day: 1,
  sessionId: null,
  pendingInterruptChoice: false,
  interruptContinueChosen: false,
  quality: new Map(),
  qualityAttempts: new Map(),
  micStream: null,
  micDeviceId: localStorage.getItem(storageKey("microphone_device_id")) || "",
  audioContext: null,
  workletNode: null,
  timingSocket: null,
  timingCommandWaiters: new Map(),
  tapResultWaiters: new Map(),
  timingStateVersion: null,
  currentActivationId: null,
  timingReconnectInFlight: false,
  phaseTransitionInFlight: false,
  currentPhase: null,
  currentPolicyToken: null,
  pairSocket: null,
  pairHeartbeat: null,
  pairReconnectTimer: null,
  pairReconnectAttempt: 0,
  pairReconnectEnabled: false,
  pairReady: false,
  companionReady: false,
  companionDisconnectGraceTimer: null,
  pairRequiredGraceTimer: null,
  lastPairStatusMessage: null,
  lastPairStatusReceivedAtMs: 0,
  httpCompanionReady: false,
  httpCompanionReadyAtMs: 0,
  companionLivePollTimer: null,
  companionReadyFalseTimer: null,
  companionAttentionTimer: null,
  trainingPassed: false,
  trainingIndex: 0,
  trainingAttempts: 0,
  trainingErrors: 0,
  trainingStartedMs: null,
  trainingAdvanceTimer: null,
  trainingReviewTimer: null,
  trainingReviewDeadlineMs: null,
  trainingReviewUnlocked: false,
  trainingReviewFeedbackPrefix: "",
  eventQueue: [],
  eventFlushTimer: null,
  eventFlushInFlight: null,
  lastEventFlushError: null,
  demoAudioTimer: null,
  demoAudioTick: 0,
  demoPcmMode: "ambient",
  demoSpeechUntilMs: 0,
  demoSpeechSeed: 0,
  recorder: null,
  recognition: null,
  recognitionGeneration: 0,
  recognitionWanted: false,
  asrLastFinalMs: 0,
  asrLastError: "",
  asrRestartTimer: null,
  companionRelayActive: false,
  companionRelayRecognition: null,
  companionRelayParts: [],
  companionRelayInterim: "",
  companionRelayGeneration: 0,
  companionRelayStopToken: 0,
  companionRelayFinalizeTimer: null,
  companionRelayWatchdog: null,
  companionRelayResumeRecognition: false,
  companionRelayPcmChunks: [],
  companionRelayPcmSamples: 0,
  companionRelayFinalPromise: null,
  asrPausedForAgentSpeech: false,
  asrInterimTranscript: "",
  autoActivateCooldownUntilMs: 0,
  frozenUserTranscript: "",
  transcriptBuffer: [],
  serverAsrPcmChunks: [],
  serverAsrPcmSamples: 0,
  pendingMicVerification: null,
  decisionContextTarget: "experimental_agent",
  decisionContextAllowsReply: false,
  decisionContextVersion: 0,
  decisionContextRequestGeneration: 0,
  pendingDecisionTranscript: "",
  decisionFactorization: null,
  localLastVoiceMs: 0,
  agentState: "waiting",
  agentStateChangedAtMs: performance.now(),
  stateTransitionTapGuardMs: 500,
  currentAudio: null,
  currentTurnAbort: null,
  noticeAudioContext: null,
  agentTurnGeneration: 0,
  thinkingWatchdog: null,
  scheduleRunning: false,
  interactionEnding: false,
  sessionCreateInFlight: false,
  sessionActivated: false,
  startInFlight: false,
  questionnaireSubmitInFlight: false,
  endToEndCompletionReceipt: null,
  compensationSubmitInFlight: false,
  compensationSignatureDirty: false,
  compensationSignaturePadReady: false,
  pendingCloseout: null,
  pendingFinalQuestionnaireEntry: false,
  finalQuestionnaireRecovery: null,
  schedulePaused: false,
  pageHiddenSoftPaused: false,
  pageHiddenGraceTimer: null,
  pageHiddenGraceTickTimer: null,
  pageHiddenGraceDeadlineMs: 0,
  attentionGraceReason: null,
  scheduledRest: false,
  restRemainingSeconds: 0,
  companionWindowActive: false,
  companionInviteTimer: null,
  companionInviteTickTimer: null,
  companionInviteDeadlineMs: 0,
  companionInviteUsed: false,
  tapFeedbackClearTimer: null,
  tapFeedbackStickyUntilWaiting: false,
  tapDesyncHealInFlight: false,
  activationReleaseInFlight: null,
  activeBlockIndex: null,
  analysisEligible: true,
  pauseStarted: 0,
  scheduleAbort: false,
  returnToSetupInFlight: false,
  opportunities: 0,
  demo: new URLSearchParams(location.search).get("demo") === "1",
  selectedTopic: null,
  topicSaved: false,
  topicCatalog: null,
  previousTopicIds: new Set(),
  deviceId: studyDeviceId(),
  browserPlaybackProbe: null,
  capacity: null,
  capacityPollTimer: null,
  shareAppointment: null,
  sessionLock: null,
  groupSessionPassword: "",
  groupSessionUnlocked: false,
  publicEnrollmentEnabled: true,
  preferFutureAppointment: false,
  selectedAppointmentStartMs: null,
  appointmentPollTimer: null,
  appointmentCountdownTimer: null,
  shareSetupGuardTimer: null,
  shareSetupCountdownTimer: null,
  shareSetupDeadlineMs: 0,
  shareSetupServerOffsetMs: 0,
  shareSetupStatusInFlight: false,
  shareSetupExpiredHandled: false,
  currentViewId: null,
  participantSpeechLastMs: 0,
  interactionPresenceTimer: null,
  interactionPresenceCountdownStartedMs: 0,
  interactionPresenceRenderState: "hidden",
  interactionPresenceRenderSecond: null,
  interactionPresenceRenderTarget: null,
  companionAgentBusy: false,
};

const REQUIRED_QUALITY = ["network", "microphone", "reaction_latency", "browser"];
const INTERACTION_SPEECH_GRACE_MS = 30000;
const INTERACTION_RECOVERY_COUNTDOWN_MS = 30000;
const QUALITY_GATE_LABELS = {
  network: "网络稳定性检测",
  microphone: "麦克风与环境噪声检测",
  reaction_latency: "点击反应延迟检测",
  browser: "浏览器与声音播放检测",
};

const views = [
  "login-view",
  "appointment-view",
  "consent-view",
  "compensation-view",
  "setup-view",
  "study-view",
  "questionnaire-view",
  "complete-view",
];

function showView(id) {
  state.currentViewId = id;
  for (const view of views) $(`#${view}`).classList.toggle("hidden", view !== id);
  const appointmentLocked = id === "appointment-view";
  document.body.classList.toggle("share-appointment-locked", appointmentLocked);
  for (const control of document.querySelectorAll(".app-header button")) {
    const keepEnabled = control.id === "logout-btn";
    control.disabled = appointmentLocked && !keepEnabled;
  }
  if (id !== "appointment-view") {
    clearInterval(state.appointmentPollTimer);
    clearInterval(state.appointmentCountdownTimer);
    state.appointmentPollTimer = null;
    state.appointmentCountdownTimer = null;
  }
  window.scrollTo(0, 0);
  requestAnimationFrame(() => window.scrollTo(0, 0));
}

function clearShareSetupTiming() {
  clearInterval(state.shareSetupGuardTimer);
  clearInterval(state.shareSetupCountdownTimer);
  state.shareSetupGuardTimer = null;
  state.shareSetupCountdownTimer = null;
  state.shareSetupDeadlineMs = 0;
  const banner = $("#setup-deadline-banner");
  banner.classList.add("hidden");
  banner.dataset.urgent = "false";
}

function formatShareSetupStart(ms) {
  return new Intl.DateTimeFormat(I18N.isEnglish ? "en-GB" : "zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(new Date(ms));
}

function renderShareSetupCountdown() {
  if (!state.shareSetupDeadlineMs) return;
  const authoritativeNow = Date.now() + state.shareSetupServerOffsetMs;
  const remainingMs = Math.max(0, state.shareSetupDeadlineMs - authoritativeNow);
  const remainingSeconds = Math.ceil(remainingMs / 1000);
  const minutes = Math.floor(remainingSeconds / 60);
  const seconds = remainingSeconds % 60;
  $("#setup-deadline-countdown").textContent = (
    `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
  );
  $("#setup-deadline-banner").dataset.urgent = String(remainingMs <= 5 * 60 * 1000);
  if (remainingMs <= 0) void refreshShareSetupStatus();
}

function applyShareSetupTiming(appointment) {
  if (appointment?.setup_timed_out) {
    clearShareSetupTiming();
    handleShareSetupExpired();
    return;
  }
  const deadlineMs = Number(appointment?.setup_deadline_ms) || 0;
  if (!appointment?.setup_timer_active || !deadlineMs || state.sessionActivated) {
    clearShareSetupTiming();
    return;
  }
  const serverMs = Number(appointment.server_ms);
  if (Number.isFinite(serverMs) && serverMs > 0) {
    state.shareSetupServerOffsetMs = serverMs - Date.now();
  }
  state.shareSetupDeadlineMs = deadlineMs;
  state.shareSetupExpiredHandled = false;
  const startedMs = Number(appointment.setup_started_ms) || (deadlineMs - Number(appointment.setup_timeout_ms));
  $("#setup-deadline-start").textContent = I18N.isEnglish
    ? `Started: ${formatShareSetupStart(startedMs)}`
    : `开始时间：${formatShareSetupStart(startedMs)}`;
  $("#setup-deadline-banner").classList.remove("hidden");
  renderShareSetupCountdown();
  if (!state.shareSetupCountdownTimer) {
    state.shareSetupCountdownTimer = setInterval(renderShareSetupCountdown, 1000);
  }
}

function handleShareSetupExpired() {
  if (state.shareSetupExpiredHandled) return;
  state.shareSetupExpiredHandled = true;
  clearShareSetupTiming();
  try { state.timingSocket?.close(); } catch (_) { /* Already closed. */ }
  try { stopPairConnection(); } catch (_) { /* Pairing may not have started. */ }
  localStorage.removeItem(storageKey("study_token"));
  localStorage.removeItem(storageKey("study_code"));
  state.token = "";
  state.code = "";
  state.sessionId = null;
  state.sessionActivated = false;
  state.bootstrap = null;
  state.shareAppointment = null;
  setLogoutVisible(false);
  setConnection(I18N.t("准备时间已结束"), "error");
  showView("login-view");
  $("#login-error").textContent = I18N.t(
    "此前的电脑实验会话在 30 分钟内未完成准备并启动正式实验，现已结束。请重新报名并生成新编号。",
  );
  $("#setup-expired-overlay").classList.remove("hidden");
  setTimeout(() => $("#setup-expired-restart").focus(), 0);
}

async function refreshShareSetupStatus() {
  if (state.shareSetupStatusInFlight || !state.token || state.sessionActivated) return;
  state.shareSetupStatusInFlight = true;
  try {
    const result = await api("/api/share-fit/appointment");
    const appointment = result.appointment || result;
    state.shareAppointment = appointment;
    if (state.bootstrap) state.bootstrap.share_fit_appointment = appointment;
    applyShareSetupTiming(appointment);
    if (
      appointment.serial_required
      && appointment.global_external_holder_present
    ) {
      renderAppointmentOverlay(appointment);
    }
  } catch (_) {
    // The server status endpoint and activation endpoint remain authoritative.
  } finally {
    state.shareSetupStatusInFlight = false;
  }
}

function armShareSetupGuard() {
  clearInterval(state.shareSetupGuardTimer);
  state.shareSetupGuardTimer = null;
  if (!state.shareAppointment?.setup_timer_active || state.sessionActivated) return;
  void refreshShareSetupStatus();
  state.shareSetupGuardTimer = setInterval(refreshShareSetupStatus, 5000);
}

$("#setup-expired-restart").addEventListener("click", () => {
  $("#setup-expired-overlay").classList.add("hidden");
  $("#enrollment-contact-email").focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

function setConnection(label, kind = "") {
  const node = $("#connection-label");
  if (node) node.textContent = label;
  $("#connection-dot").className = `status-dot ${kind}`.trim();
  syncCodeHintEntryAffordances();
}

function connectionIsUnlinked() {
  const label = String($("#connection-label")?.textContent || "").trim();
  return !state.token && (label === "尚未连接" || label === "Not yet connected" || label === "尚未开始");
}

function syncCodeHintEntryAffordances() {
  const button = $("#connection-label");
  if (!button) return;
  const open = connectionIsUnlinked();
  button.classList.toggle("is-entry", open);
  button.setAttribute("aria-haspopup", open ? "dialog" : "false");
}

function setLogoutVisible(visible) {
  $("#logout-btn").classList.toggle("hidden", !visible);
}

function openCodeHintEntry() {
  if (!connectionIsUnlinked()) return;
  const overlay = $("#code-hint-entry-overlay");
  const input = $("#code-hint-entry-input");
  const error = $("#code-hint-entry-error");
  if (!overlay || !input) return;
  if (error) error.textContent = "";
  input.value = "";
  overlay.classList.remove("hidden");
  input.focus();
}

function closeCodeHintEntry() {
  $("#code-hint-entry-overlay")?.classList.add("hidden");
}

async function submitCodeHintEntry() {
  const input = $("#code-hint-entry-input");
  const error = $("#code-hint-entry-error");
  const button = $("#code-hint-entry-submit");
  const hint = String(input?.value || "").trim().toUpperCase();
  if (error) error.textContent = "";
  if (!/^[A-Z0-9]{4}$/.test(hint)) {
    if (error) error.textContent = I18N.t("请输入四位登录码尾号。");
    input?.focus();
    return;
  }
  if (button) button.disabled = true;
  try {
    const result = await api("/api/public/code-hint-entry", {
      method: "POST",
      body: { hint },
    });
    const resumedCode = resolveFullStudyCode(result.code, result.code_hint);
    if (resumedCode) rememberStudyCode(resumedCode, result.code_hint);
    else if (result.code_hint) {
      localStorage.setItem(
        storageKey("study_code_hint"),
        String(result.code_hint).trim().toUpperCase(),
      );
    }
    state.token = result.token;
    localStorage.setItem(storageKey("study_token"), result.token);
    closeCodeHintEntry();
    await loadBootstrap();
  } catch (err) {
    if (error) {
      error.textContent = err.message || I18N.t("未找到匹配的编号，请核对后重试。");
    }
  } finally {
    if (button) button.disabled = false;
  }
}

$("#connection-label")?.addEventListener("click", () => {
  if (connectionIsUnlinked()) openCodeHintEntry();
});
$("#code-hint-entry-submit")?.addEventListener("click", () => {
  void submitCodeHintEntry();
});
$("#code-hint-entry-cancel")?.addEventListener("click", closeCodeHintEntry);
$("#code-hint-entry-input")?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    void submitCodeHintEntry();
  }
});
$("#code-hint-entry-input")?.addEventListener("input", (event) => {
  event.target.value = String(event.target.value || "").toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 4);
});

async function logoutParticipant() {
  const confirmed = window.confirm(
    "确定退出当前编号吗？未完成的实验会立即结束并释放实验位置，已上传数据不会删除；如需再次参与，请重新生成编号。若已进入最终问卷，请先保存最终问卷凭据。",
  );
  if (!confirmed) return;

  const button = $("#logout-btn");
  button.disabled = true;
  try {
    await withTimeout(flushEvents(true), 3000, "pending events timed out");
  } catch (_) {
    // The authoritative exit call below still records the release request.
  }
  if (state.token) {
    try {
      const result = await withTimeout(
        api("/api/participants/me/exit", { method: "POST", keepalive: true }),
        8000,
        "服务器未在 8 秒内确认释放实验位置",
      );
      if (!result?.immediate_release_confirmed) {
        throw new Error("服务器没有返回实验位置释放凭证");
      }
    } catch (error) {
      window.alert(`暂时无法退出：${error.message}。编号仍保留在本页面，请检查网络后重试。`);
      button.disabled = false;
      return;
    }
  }
  try { state.timingSocket?.close(); } catch (_) { /* Already closed. */ }
  try { state.pairSocket?.close(); } catch (_) { /* Already closed. */ }
  localStorage.removeItem(storageKey("study_token"));
  localStorage.removeItem(storageKey("study_code"));
  state.token = "";
  state.code = "";
  window.location.reload();
}

$("#logout-btn").addEventListener("click", logoutParticipant);

function rememberedGroupSessionPassword() {
  return String(
    state.groupSessionPassword
    || sessionStorage.getItem(storageKey("group_session_password"))
    || "",
  ).trim();
}

function applyGroupSessionPassword(password) {
  const normalized = String(password || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
  state.groupSessionPassword = normalized;
  state.groupSessionUnlocked = Boolean(normalized);
  if (normalized) {
    sessionStorage.setItem(storageKey("group_session_password"), normalized);
  } else {
    sessionStorage.removeItem(storageKey("group_session_password"));
  }
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-TTA-Device-Class", CLIENT_DEVICE_CLASS);
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  const groupPassword = rememberedGroupSessionPassword();
  if (groupPassword) headers.set("X-Group-Session-Password", groupPassword);
  if (options.body && typeof options.body !== "string" && !(options.body instanceof Blob)) {
    headers.set("Content-Type", "application/json");
    options.body = JSON.stringify(options.body);
  }
  const requestPath = !options.method || options.method === "GET" ? I18N.apiPath(path) : path;
  const response = await fetch(`${APP_BASE}${requestPath}`, { ...options, headers, cache: "no-store" });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    let responseDetail = null;
    try {
      const payload = await response.json();
      responseDetail = payload.detail;
      if (typeof payload.detail === "string") detail = payload.detail;
      else if (payload.detail?.message) detail = payload.detail.message;
      else detail = JSON.stringify(payload.detail);
    } catch (_) {
      // Keep HTTP status text.
    }
    const error = new Error(detail);
    error.status = response.status;
    error.detail = responseDetail;
    throw error;
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response;
}

function wsUrl(path, params) {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const query = new URLSearchParams(params).toString();
  return `${scheme}//${location.host}${APP_BASE}${path}?${query}`;
}

function setEnrollButtonsDisabled(disabled) {
  const startBtn = $("#enroll-submit");
  const futureBtn = $("#enroll-future-appointment");
  if (disabled) {
    if (startBtn) startBtn.disabled = true;
    if (futureBtn) futureBtn.disabled = true;
    return;
  }
  applyEnrollmentAvailability();
}

function applyPublicEnrollmentClosed(closed) {
  state.publicEnrollmentEnabled = !closed;
  $("#public-enroll-open")?.classList.toggle("hidden", closed);
  $("#public-enroll-closed")?.classList.toggle("hidden", !closed);
  if (closed) setEnrollButtonsDisabled(true);
}

function applyEnrollmentAvailability() {
  const startBtn = $("#enroll-submit");
  const futureBtn = $("#enroll-future-appointment");
  const enrollmentClosed = state.publicEnrollmentEnabled === false;
  const capacityBlocked = Boolean(
    state.capacity && state.capacity.ok === false && !state.capacity.holding_slot
  );
  const lockBlocksStart = Boolean(state.sessionLock?.active) && !state.groupSessionUnlocked;
  if (startBtn) startBtn.disabled = enrollmentClosed || capacityBlocked || lockBlocksStart;
  if (futureBtn) futureBtn.disabled = enrollmentClosed || capacityBlocked;
}

async function enrollFromHomepage({ preferFutureAppointment = false } = {}) {
  const error = $("#login-error");
  error.textContent = "";
  const emailInput = $("#enrollment-contact-email");
  const contacts = {
    contact_email: emailInput.value.trim(),
    contact_phone: $("#enrollment-contact-phone").value.trim(),
    contact_wechat: $("#enrollment-contact-wechat").value.trim(),
  };
  if (!contacts.contact_email && !contacts.contact_phone && !contacts.contact_wechat) {
    error.textContent = I18N.t("报名时必须填写邮箱、电话号码或微信号中的至少一项。");
    emailInput.focus();
    return;
  }
  if (contacts.contact_email && !emailInput.validity.valid) {
    error.textContent = I18N.t("请输入有效的邮箱，或改填电话号码/微信号。");
    emailInput.focus();
    return;
  }
  setEnrollButtonsDisabled(true);
  try {
    state.preferFutureAppointment = Boolean(preferFutureAppointment);
    setConnection("正在建立匿名记录", "warn");
    const result = await api("/api/enroll", {
      method: "POST",
      body: {
        device_id: state.deviceId,
        ...contacts,
        intent: preferFutureAppointment ? "future_book" : "start_now",
        group_session_password: rememberedGroupSessionPassword(),
      },
    });
    rememberStudyCode(result.code, result.code_hint);
    state.token = result.token;
    localStorage.setItem(storageKey("study_token"), result.token);
    await loadBootstrap();
  } catch (err) {
    state.preferFutureAppointment = false;
    error.textContent = `${I18N.t("暂时无法开始：")}${err.message} ${I18N.t("如需帮助，请联系微信 study-contact（微信名：Research team）。")}`;
    setConnection("尚未开始", "error");
    setEnrollButtonsDisabled(false);
  }
}

$("#enroll-submit").addEventListener("click", () => enrollFromHomepage());
$("#enroll-future-appointment")?.addEventListener("click", () => (
  enrollFromHomepage({ preferFutureAppointment: true })
));

function formatLockClock(ms) {
  if (!ms) return "—";
  return new Date(Number(ms)).toLocaleString("zh-CN", {
    hour12: false,
    timeZone: "Asia/Shanghai",
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function syncGroupSessionLockBar() {
  const bar = $("#group-session-lock-bar");
  if (!bar) return;
  const lock = state.sessionLock || {};
  const onLogin = !$("#login-view")?.classList.contains("hidden");
  const show = Boolean(onLogin && (lock.active || lock.upcoming));
  bar.hidden = !show;
  bar.classList.toggle("hidden", !show);
  const copy = $("#group-session-lock-copy");
  if (!copy) return;
  const windows = Array.isArray(lock.windows) && lock.windows.length
    ? lock.windows
    : (lock.start_ms && lock.end_ms ? [{ start_ms: lock.start_ms, end_ms: lock.end_ms }] : []);
  const ranges = windows
    .map((row) => `${formatLockClock(row.start_ms)}–${formatLockClock(row.end_ms)}`)
    .join("，");
  if (state.groupSessionUnlocked && lock.active) {
    copy.textContent = `召集密码已确认。当前锁定至 ${formatLockClock(lock.end_ms)}，现在可以开始填写。`;
    return;
  }
  if (lock.active) {
    copy.textContent = `该时段已锁定集体实验（${ranges}）。外部不能直接开始；预约仍可选择锁定窗外的时间。持召集密码可在此开启。`;
    return;
  }
  if (lock.upcoming) {
    copy.textContent = `即将锁定集体实验（${ranges}）。锁定开始后需召集密码才能直接开始。`;
  }
}

async function refreshSessionLock() {
  try {
    const payload = await api("/api/session-lock");
    state.sessionLock = payload;
    if (state.groupSessionPassword && payload?.active) {
      try {
        await api("/api/session-lock/unlock", {
          method: "POST",
          body: { password: state.groupSessionPassword },
        });
        state.groupSessionUnlocked = true;
      } catch (_) {
        state.groupSessionUnlocked = false;
      }
    } else if (!payload?.active) {
      state.groupSessionUnlocked = false;
    }
  } catch (_) {
    state.sessionLock = state.sessionLock || null;
  }
  syncGroupSessionLockBar();
  applyEnrollmentAvailability();
  return state.sessionLock;
}

$("#group-session-unlock")?.addEventListener("click", async () => {
  const error = $("#group-session-lock-error");
  const input = $("#group-session-password");
  if (error) error.textContent = "";
  const password = String(input?.value || "").trim();
  if (!password) {
    if (error) error.textContent = "请输入召集密码。";
    input?.focus();
    return;
  }
  try {
    await api("/api/session-lock/unlock", {
      method: "POST",
      body: { password },
    });
    applyGroupSessionPassword(password);
    if (input) input.value = "";
    syncGroupSessionLockBar();
    applyEnrollmentAvailability();
  } catch (err) {
    state.groupSessionUnlocked = false;
    if (error) error.textContent = err.message || "召集密码不正确。";
  }
});

function normalizedFinalCredential(value) {
  return String(value || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
}

function displayedFinalCredential(value) {
  const normalized = normalizedFinalCredential(value);
  return normalized.length === 8
    ? `${normalized.slice(0, 4)}-${normalized.slice(4)}`
    : normalized;
}

$("#appointment-resume-voucher")?.addEventListener("input", (event) => {
  event.target.value = String(event.target.value || "").replace(/\D/g, "").slice(0, 4);
});

$("#appointment-resume-submit")?.addEventListener("click", async () => {
  const error = $("#appointment-resume-error");
  const button = $("#appointment-resume-submit");
  const voucher = String($("#appointment-resume-voucher")?.value || "").replace(/\D/g, "");
  const fullName = String($("#appointment-resume-name")?.value || "").trim();
  const contacts = {
    contact_email: $("#enrollment-contact-email").value.trim(),
    contact_phone: $("#enrollment-contact-phone").value.trim(),
    contact_wechat: $("#enrollment-contact-wechat").value.trim(),
  };
  error.textContent = "";
  if (!voucher && !fullName && !contacts.contact_email && !contacts.contact_phone && !contacts.contact_wechat) {
    error.textContent = I18N.t("请输入四位预约凭证，或填写已登记的联系方式/姓名。");
    ($("#appointment-resume-voucher") || $("#appointment-resume-name")).focus();
    return;
  }
  if (voucher && voucher.length !== 4) {
    error.textContent = I18N.t("预约凭证必须是四位数字。");
    $("#appointment-resume-voucher").focus();
    return;
  }
  if (contacts.contact_email) {
    const emailInput = $("#enrollment-contact-email");
    if (emailInput && !emailInput.validity.valid) {
      error.textContent = I18N.t("请输入有效的邮箱，或改填电话号码/微信号。");
      emailInput.focus();
      return;
    }
  }
  button.disabled = true;
  try {
    setConnection("正在核验预约", "warn");
    const result = await api("/api/share-fit/appointment/resume", {
      method: "POST",
      body: {
        device_id: state.deviceId,
        voucher,
        full_name: fullName,
        ...contacts,
      },
    });
    const resumedCode = resolveFullStudyCode(result.code, result.code_hint);
    if (resumedCode) rememberStudyCode(resumedCode, result.code_hint);
    else {
      state.code = "";
      if (result.code_hint) {
        localStorage.setItem(
          storageKey("study_code_hint"),
          String(result.code_hint).trim().toUpperCase(),
        );
      }
    }
    state.token = result.token;
    localStorage.setItem(storageKey("study_token"), result.token);
    await loadBootstrap();
  } catch (err) {
    error.textContent = `${I18N.t("暂时无法进入已预约实验：")}${err.message} ${I18N.t("如需帮助，请联系微信 study-contact（微信名：Research team）。")}`;
    setConnection("尚未开始", "error");
    button.disabled = false;
  }
});

$("#final-recovery-entry")?.addEventListener("input", (event) => {
  const normalized = normalizedFinalCredential(event.target.value).slice(0, 8);
  event.target.value = displayedFinalCredential(normalized);
});

$("#final-recovery-submit")?.addEventListener("click", async () => {
  const error = $("#final-recovery-error");
  const button = $("#final-recovery-submit");
  const credential = normalizedFinalCredential($("#final-recovery-entry").value);
  error.textContent = "";
  if (credential.length !== 8) {
    error.textContent = "请输入完整的 8 位受试者凭据。";
    return;
  }
  button.disabled = true;
  try {
    const result = await api("/api/final-questionnaire/recover", {
      method: "POST",
      body: { credential },
    });
    state.token = result.token;
    state.code = "";
    localStorage.setItem(storageKey("study_token"), result.token);
    localStorage.removeItem(storageKey("study_code"));
    await loadBootstrap();
  } catch (err) {
    error.textContent = err.message;
  } finally {
    button.disabled = false;
  }
});

function applyCapacityStatus(capacity) {
  state.capacity = capacity || null;
  const banner = $("#capacity-banner");
  const title = $("#capacity-banner-title");
  const copy = $("#capacity-banner-copy");
  const blocked = Boolean(capacity && capacity.ok === false && !capacity.holding_slot);
  if (banner) banner.classList.toggle("hidden", !blocked);
  if (blocked) {
    const minutes = Number(capacity.retry_after_minutes) || 30;
    if (title) title.textContent = I18N.t("当前同时使用人数较多");
    if (copy) {
      copy.textContent = (
        capacity.message_zh
        || capacity.message
        || I18N.t("为保证语音与网络稳定，请约 {n} 分钟后再来尝试。").replace("{n}", String(minutes))
      );
    }
    setEnrollButtonsDisabled(true);
  } else if (!state.token) {
    if (document.documentElement.dataset.appReady === "true") applyEnrollmentAvailability();
  }
  updateStartEligibility();
}

async function refreshCapacityStatus() {
  try {
    if (state.token) {
      const bootstrap = await api("/api/bootstrap");
      applyCapacityStatus(bootstrap.capacity);
      if (state.bootstrap) state.bootstrap.capacity = bootstrap.capacity;
      return bootstrap.capacity;
    }
    const payload = await api("/api/capacity");
    applyCapacityStatus(payload.capacity);
    return payload.capacity;
  } catch (_) {
    return state.capacity;
  }
}

function startCapacityPolling() {
  clearInterval(state.capacityPollTimer);
  state.capacityPollTimer = setInterval(() => {
    if (state.scheduleRunning) return;
    void refreshCapacityStatus();
    if (!state.token) void refreshSessionLock();
  }, 20000);
}

function appointmentDateParts(ms) {
  const formatter = new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  });
  return Object.fromEntries(
    formatter.formatToParts(new Date(Number(ms))).map((part) => [part.type, part.value]),
  );
}

function formatAppointmentTime(ms) {
  if (!Number.isFinite(Number(ms))) return "—";
  const parts = appointmentDateParts(ms);
  return `${parts.year}年${parts.month}月${parts.day}日 ${parts.hour}:${parts.minute}（北京时间）`;
}

function parseAppointmentInput(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : NaN;
}

function hideAppointmentOverlay() {
  $("#share-appointment-overlay").classList.add("hidden");
  $("#share-appointment-error").textContent = "";
  clearInterval(state.appointmentPollTimer);
  state.appointmentPollTimer = null;
}

function appointmentSlotLabel(ms) {
  if (I18N.isEnglish) {
    return new Intl.DateTimeFormat("en-GB", {
      timeZone: "Asia/Shanghai",
      month: "2-digit",
      day: "2-digit",
      weekday: "short",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23",
    }).format(new Date(Number(ms)));
  }
  const parts = appointmentDateParts(ms);
  const weekday = new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    weekday: "short",
  }).format(new Date(Number(ms)));
  return `${parts.month}月${parts.day}日 ${weekday} ${parts.hour}:${parts.minute}`;
}

function selectAppointmentSlot(startMs) {
  const selectedStartMs = Number(startMs);
  state.selectedAppointmentStartMs = Number.isFinite(selectedStartMs)
    ? selectedStartMs
    : null;
  $("#share-appointment-time").value = state.selectedAppointmentStartMs
    ? String(state.selectedAppointmentStartMs)
    : "";
  for (const button of document.querySelectorAll(".appointment-slot")) {
    const selected = Number(button.dataset.startMs) === state.selectedAppointmentStartMs;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-checked", String(selected));
  }
  $("#share-appointment-submit").disabled = !state.selectedAppointmentStartMs;
}

function renderAppointmentSlots(appointment) {
  const grid = $("#share-appointment-slots");
  const slots = Array.isArray(appointment?.time_slots) ? appointment.time_slots : [];
  const availableStarts = new Set(
    slots.filter((slot) => slot.available).map((slot) => Number(slot.start_ms)),
  );
  if (!availableStarts.has(Number(state.selectedAppointmentStartMs))) {
    state.selectedAppointmentStartMs = Number(
      slots.find((slot) => slot.available)?.start_ms || 0,
    ) || null;
  }
  grid.replaceChildren();
  for (const slot of slots) {
    const startMs = Number(slot.start_ms);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `appointment-slot ${slot.available ? "available" : "blocked"}`;
    button.dataset.startMs = String(startMs);
    button.textContent = appointmentSlotLabel(startMs);
    button.disabled = !slot.available;
    button.setAttribute("role", "radio");
    button.setAttribute("aria-checked", "false");
    if (slot.available) {
      button.addEventListener("click", () => selectAppointmentSlot(startMs));
    } else {
      button.title = slot.reason === "booked"
        ? "该时间块已被预约"
        : slot.reason === "operator_blocked"
          ? "该时间段已被设为不可预约"
          : slot.reason === "session_lock"
            ? "该时间处于集体实验锁定时段"
            : "该时间块处于预约缓冲范围";
    }
    grid.appendChild(button);
  }
  selectAppointmentSlot(state.selectedAppointmentStartMs);
}

function renderAppointmentOverlay(appointment = state.shareAppointment) {
  if (!appointment?.serial_required) return;
  clearInterval(state.appointmentPollTimer);
  clearInterval(state.appointmentCountdownTimer);
  state.appointmentPollTimer = null;
  state.appointmentCountdownTimer = null;
  state.shareAppointment = appointment;
  const serverMs = Number(appointment.server_ms) || Date.now();
  const horizonMs = Number(appointment.horizon_end_ms) || serverMs + (12 * 60 * 60 * 1000);
  renderAppointmentSlots(appointment);
  const optional = Boolean(appointment.can_book_future && !appointment.appointment_required);
  const copy = $("#share-appointment-copy");
  if (copy) {
    copy.textContent = optional
      ? I18N.t("当前实验位置空闲。你可以预约未来 12 小时内的时间，到点后再来；每个已预约开始时间的前后 1 小时内均不可再预约。")
      : I18N.t("当前已有参与者取得实验位置。每个已预约开始时间的前后 1 小时内均不可再预约；请在预约时间到达后 10 分钟内回到本页，否则该预约自动作废。");
  }
  $("#share-appointment-cancel")?.classList.toggle("hidden", !optional);
  $("#share-appointment-window").textContent =
    `可预约范围：现在至 ${formatAppointmentTime(horizonMs)}。时间块间隔 30 分钟；已预约时间及其前后 1 小时缓冲范围不可选择。`;
  const blocked = Array.isArray(appointment.blocked_start_ms)
    ? appointment.blocked_start_ms.filter((value) => Number(value) > 0)
    : [];
  $("#share-appointment-blocked").textContent = blocked.length
    ? `已预约时间：${blocked.map(formatAppointmentTime).join("；")}`
    : "当前没有其他已预约的开始时间。";
  $("#share-appointment-error").textContent = "";
  $("#share-appointment-overlay").classList.remove("hidden");
  state.appointmentPollTimer = setInterval(
    () => void refreshAppointmentStatus(),
    15000,
  );
}

function appointmentCountdownText(appointment) {
  const startMs = Number(appointment?.scheduled_start_ms);
  if (!startMs) return "预约时间尚未确定";
  const remaining = startMs - Date.now();
  if (remaining <= 0) {
    if (appointment?.predecessor_delay) {
      const waitUntil = Number(appointment.predecessor_deadline_ms || appointment.retry_after_ms);
      const retryRemaining = Math.max(0, waitUntil - Date.now());
      return retryRemaining > 0
        ? `上一位参与者最多还需 ${formatTime(retryRemaining / 1000)}，请到 ${formatAppointmentTime(waitUntil)} 后再输入凭证`
        : "上一位参与者时限已到，请输入四位凭证进入";
    }
    if (appointment?.arrival_reentry_required) {
      const entryDeadline = Number(appointment.entry_deadline_ms);
      const entryRemaining = Math.max(0, entryDeadline - Date.now());
      return entryRemaining > 0
        ? `实验位置已空闲，请在 ${formatTime(entryRemaining / 1000)} 内输入凭证进入`
        : "空闲后超过 15 分钟未进入，预约凭证即将失效";
    }
    if (Number(appointment?.arrived_ms) > 0) {
      return "已按时到达并验证凭证，正在准备进入实验";
    }
    const grace = Math.max(0, Number(appointment.grace_deadline_ms) - Date.now());
    if (grace > 0) return `预约时间已到，请输入四位凭证（迟到释放倒计时 ${formatTime(grace / 1000)}）`;
    return "预约位置正在释放，请刷新状态后重新预约";
  }
  const totalMinutes = Math.ceil(remaining / 60_000);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return hours > 0
    ? `距离预约开始还有 ${hours} 小时 ${minutes} 分钟`
    : `距离预约开始还有 ${minutes} 分钟`;
}

function renderAppointmentWaiting(appointment) {
  state.shareAppointment = appointment;
  hideAppointmentOverlay();
  showView("appointment-view");
  $("#appointment-wait-time").textContent = formatAppointmentTime(appointment.scheduled_start_ms);
  $("#appointment-voucher-display").textContent = appointment.voucher || "已发放";
  const gate = $("#appointment-arrival-gate");
  const delay = $("#appointment-delay-message");
  const due = Date.now() >= Number(appointment.scheduled_start_ms || 0);
  const stillInGrace = Date.now() <= Number(appointment.grace_deadline_ms || 0);
  const showFirstEntry = due && stillInGrace && !Number(appointment.arrived_ms);
  const showReentry = Boolean(appointment.arrival_reentry_required)
    && !appointment.predecessor_delay;
  const hideGate = appointment.predecessor_delay || !(showFirstEntry || showReentry);
  const wasHidden = gate.classList.contains("hidden");
  gate.classList.toggle("hidden", hideGate);
  delay.classList.toggle("hidden", !appointment.predecessor_delay);
  delay.textContent = appointment.predecessor_delay
    ? "你已按时到达，但上一位参与者的实验尚未完成。请等到上方显示的最晚结束时间后再输入凭证；本次预约不会因超过 10 分钟而取消。空闲后请在 15 分钟内进入，否则凭证失效。"
    : "";
  if (hideGate || wasHidden) {
    $("#appointment-voucher-entry").value = "";
  }
  const updateCountdown = () => {
    $("#appointment-countdown").textContent = appointmentCountdownText(state.shareAppointment);
  };
  updateCountdown();
  clearInterval(state.appointmentCountdownTimer);
  clearInterval(state.appointmentPollTimer);
  state.appointmentCountdownTimer = setInterval(updateCountdown, 1000);
  state.appointmentPollTimer = setInterval(() => void refreshAppointmentStatus(), 15000);
}

async function claimDueAppointmentArrival(appointment, voucher) {
  if (
    !appointment?.serial_required
    || appointment.status !== "booked"
    || Math.max(Number(appointment.server_ms) || 0, Date.now())
      < Number(appointment.scheduled_start_ms)
  ) {
    return appointment;
  }
  const result = await api("/api/share-fit/appointment/arrive", {
    method: "POST",
    body: { voucher: voucher || null },
  });
  return result.appointment || result;
}

function routeShareAppointmentError(error) {
  const appointment = error?.detail?.appointment;
  const code = String(error?.detail?.code || "");
  if (code === "share_setup_timed_out" || appointment?.setup_timed_out) {
    if (appointment) {
      state.shareAppointment = appointment;
      if (state.bootstrap) state.bootstrap.share_fit_appointment = appointment;
    }
    handleShareSetupExpired();
    return true;
  }
  if (code === "participant_already_complete") {
    void loadBootstrap();
    return true;
  }
  if (code === "share_participant_exited") {
    void loadBootstrap();
    return true;
  }
  if (
    !appointment?.serial_required
    || ![
      "share_serial_occupied",
      "share_serial_not_claimed",
      "share_appointment_required",
      "share_appointment_early",
      "share_appointment_released",
    ].includes(code)
  ) {
    return false;
  }
  state.shareAppointment = appointment;
  if (state.bootstrap) state.bootstrap.share_fit_appointment = appointment;
  hideStartFailureDialog();
  if (appointment.appointment_required) {
    renderAppointmentOverlay(appointment);
    return true;
  }
  if (!appointment.can_start_setup) {
    renderAppointmentWaiting(appointment);
    return true;
  }
  return false;
}

async function refreshAppointmentStatus() {
  const error = $("#appointment-wait-error");
  if (error) error.textContent = "";
  try {
    const result = await api("/api/share-fit/appointment");
    let appointment = result.appointment || result;
    state.shareAppointment = appointment;
    if (state.bootstrap) state.bootstrap.share_fit_appointment = appointment;
    if (
      appointment.serial_required
      && appointment.status === "booked"
      && Number(appointment.server_ms) >= Number(appointment.scheduled_start_ms)
    ) {
      const retryDue = !Number(appointment.retry_after_ms)
        || Number(appointment.server_ms) >= Number(appointment.retry_after_ms);
      if (
        appointment.credential_verified
        && retryDue
        && !appointment.arrival_reentry_required
        && !appointment.predecessor_delay
      ) {
        try {
          appointment = await claimDueAppointmentArrival(appointment, appointment.voucher);
          state.shareAppointment = appointment;
          if (state.bootstrap) state.bootstrap.share_fit_appointment = appointment;
        } catch (arrivalError) {
          if (routeShareAppointmentError(arrivalError)) return null;
          throw arrivalError;
        }
      }
    }
    if (appointment.appointment_required) {
      renderAppointmentOverlay(appointment);
      return appointment;
    }
    if (appointment.status === "booked" && !appointment.can_start_setup) {
      renderAppointmentWaiting(appointment);
      return appointment;
    }
    if (
      (state.preferFutureAppointment || !$("#share-appointment-overlay")?.classList.contains("hidden"))
      && appointment.can_book_future
    ) {
      renderAppointmentOverlay(appointment);
      return appointment;
    }
    if (appointment.can_start_setup || !appointment.serial_required) {
      await loadBootstrap();
      return appointment;
    }
    renderAppointmentWaiting(appointment);
    return appointment;
  } catch (err) {
    if (error) error.textContent = `暂时无法刷新预约状态：${err.message}`;
    return null;
  }
}

function showAppointmentSuccess(appointment) {
  $("#appointment-success-time").textContent = formatAppointmentTime(
    appointment.scheduled_start_ms,
  );
  $("#appointment-success-voucher").textContent = appointment.voucher || "——";
  $("#appointment-success-copy").textContent = "复制凭证";
  $("#appointment-success-overlay").classList.remove("hidden");
}

$("#appointment-success-copy").addEventListener("click", async () => {
  const voucher = $("#appointment-success-voucher").textContent.trim();
  try {
    await navigator.clipboard.writeText(voucher);
    $("#appointment-success-copy").textContent = "已复制";
  } catch (_) {
    $("#appointment-success-copy").textContent = `请手动记录 ${voucher}`;
  }
});

$("#appointment-success-confirm").addEventListener("click", async () => {
  $("#appointment-success-overlay").classList.add("hidden");
  await loadBootstrap();
});

$("#share-appointment-cancel")?.addEventListener("click", async () => {
  state.preferFutureAppointment = false;
  hideAppointmentOverlay();
  if (state.token) await loadBootstrap();
});

$("#share-appointment-submit").addEventListener("click", async () => {
  const button = $("#share-appointment-submit");
  const error = $("#share-appointment-error");
  error.textContent = "";
  const scheduledStartMs = parseAppointmentInput($("#share-appointment-time").value);
  if (!Number.isFinite(scheduledStartMs)) {
    error.textContent = "请选择有效的北京时间。";
    return;
  }
  button.disabled = true;
  try {
    const result = await api("/api/share-fit/appointment", {
      method: "POST",
      body: { scheduled_start_ms: scheduledStartMs },
    });
    const appointment = result.appointment || result;
    state.preferFutureAppointment = false;
    state.shareAppointment = appointment;
    if (state.bootstrap) state.bootstrap.share_fit_appointment = appointment;
    hideAppointmentOverlay();
    showAppointmentSuccess(appointment);
  } catch (err) {
    const appointment = err.detail?.appointment;
    if (appointment) {
      state.shareAppointment = appointment;
      if (state.bootstrap) state.bootstrap.share_fit_appointment = appointment;
      renderAppointmentOverlay(appointment);
    }
    error.textContent = err.message;
  } finally {
    button.disabled = false;
  }
});

$("#appointment-voucher-submit").addEventListener("click", async () => {
  const button = $("#appointment-voucher-submit");
  const error = $("#appointment-wait-error");
  const voucher = $("#appointment-voucher-entry").value.trim();
  error.textContent = "";
  if (!/^\d{4}$/.test(voucher)) {
    error.textContent = "请输入四位数字预约凭证；如已遗失，请联系微信 study-contact 获取。";
    return;
  }
  button.disabled = true;
  try {
    const appointment = await claimDueAppointmentArrival(state.shareAppointment, voucher);
    state.shareAppointment = appointment;
    if (state.bootstrap) state.bootstrap.share_fit_appointment = appointment;
    if (appointment.can_start_setup) await loadBootstrap();
    else renderAppointmentWaiting(appointment);
  } catch (err) {
    error.textContent = err.message;
  } finally {
    button.disabled = false;
  }
});

$("#appointment-refresh")?.addEventListener("click", () => void refreshAppointmentStatus());
$("#appointment-change")?.addEventListener("click", () => renderAppointmentOverlay());

async function loadBootstrap() {
  state.bootstrap = await api("/api/bootstrap");
  state.shareAppointment = state.bootstrap.share_fit_appointment || null;
  const pilot = state.bootstrap.pilot_only;
  const pilotBanner = $("#pilot-banner");
  pilotBanner.classList.toggle("hidden", !pilot);
  if (pilot) {
    pilotBanner.textContent = state.bootstrap.participant.analysis_excluded
      ? "仅用于个人调试：数据和音频与用户实验正式数据永久隔离，不进入用户实验正式统计"
      : "非正式数据模式：当前数据不能并入正式 Study A 或 Study B";
  }
  state.demo = state.demo && pilot;
  applyCapacityStatus(state.bootstrap.capacity);
  setConnection(`编号尾号 ${state.bootstrap.participant.code_hint}`, "online");
  setLogoutVisible(true);
  if (state.bootstrap.interrupt_checkpoint?.available && !state.bootstrap.operator_resume) {
    state.pendingInterruptChoice = true;
  }
  if (state.bootstrap.end_to_end_complete && !state.bootstrap.interrupt_checkpoint?.available) {
    state.endToEndCompletionReceipt = state.bootstrap.completion_receipt;
  }
  if (state.bootstrap.share_fit_appointment?.participant_exited) {
    clearShareSetupTiming();
    localStorage.removeItem(storageKey("study_token"));
    localStorage.removeItem(storageKey("study_code"));
    state.token = "";
    state.code = "";
    state.bootstrap = null;
    state.shareAppointment = null;
    setLogoutVisible(false);
    showView("login-view");
    $("#login-error").textContent = (
      "该匿名编号已经主动退出，未完成实验及预约已立即结束，实验位置也已释放。"
      + "如需再次参与，请点击“开始填写”生成新编号；此前已上传的数据会保留，但不会作为端到端完成数据进入用户实验汇总。"
    );
    return;
  }
  if (state.bootstrap.share_fit_appointment?.setup_timed_out) {
    handleShareSetupExpired();
    return;
  }
  state.shareSetupExpiredHandled = false;
  $("#setup-expired-overlay").classList.add("hidden");
  applyShareSetupTiming(state.shareAppointment);
  const finalRecovery = state.bootstrap.final_questionnaire_recovery;
  if (finalRecovery?.pending) {
    state.finalQuestionnaireRecovery = finalRecovery;
    state.sessionId = finalRecovery.session_id;
    state.sessionActivated = true;
    state.day = 1;
    if (!finalRecovery.model_ready) {
      state.pendingFinalQuestionnaireEntry = true;
      state.pendingCloseout = { endedEarly: false, enterFinalQuestionnaire: true };
      $("#questionnaire-title").textContent = "正在安全保存模型并进入最终问卷";
      $("#questionnaire-instructions").textContent = "模型尚未安全提交，因此实验位暂未释放。请保持页面打开并点击下方按钮重试。";
      $("#questionnaire-items").replaceChildren();
      $("#questionnaire-submit").disabled = true;
      showView("questionnaire-view");
      const error = new Error(finalRecovery.last_error || "模型提交尚未完成");
      error.status = 503;
      error.closeoutErrorCode = "final_questionnaire_entry_pending";
      showCloseoutFailure(error);
    } else if (finalRecovery.final_submitted_ms) {
      state.pendingCloseout = { endedEarly: false };
      try {
        await completeDay(false);
        state.pendingCloseout = null;
      } catch (error) {
        await showQuestionnaire("final");
        showCloseoutFailure(error);
      }
    } else {
      await showQuestionnaire("final");
    }
    return;
  }
  if (state.bootstrap.end_to_end_complete && !state.bootstrap.interrupt_checkpoint?.available) {
    setConnection("实验完成", "online");
    renderCompletion(false);
    return;
  }
  if (state.preferFutureAppointment && state.shareAppointment?.can_book_future) {
    showView("login-view");
    renderAppointmentOverlay(state.shareAppointment);
    return;
  }
  if (
    state.shareAppointment?.serial_required
    && state.shareAppointment.status === "booked"
    && !state.shareAppointment.can_start_setup
  ) {
    const due = Number(state.shareAppointment.server_ms)
      >= Number(state.shareAppointment.scheduled_start_ms);
    const retryDue = !Number(state.shareAppointment.retry_after_ms)
      || Number(state.shareAppointment.server_ms) >= Number(state.shareAppointment.retry_after_ms);
    if (!due || !state.shareAppointment.credential_verified || !retryDue) {
      renderAppointmentWaiting(state.shareAppointment);
      return;
    }
  }
  if (!state.bootstrap.consent_valid) {
    renderConsent(state.bootstrap.consent);
    showView("consent-view");
    return;
  }
  if (!state.bootstrap.compensation_registration_complete) {
    renderCompensationRegistration(state.bootstrap.compensation_registration);
    showView("compensation-view");
    return;
  }
  let appointment = state.shareAppointment;
  if (
    appointment?.serial_required
    && appointment.status === "booked"
    && Number(appointment.server_ms) >= Number(appointment.scheduled_start_ms)
  ) {
    const retryDue = !Number(appointment.retry_after_ms)
      || Number(appointment.server_ms) >= Number(appointment.retry_after_ms);
    if (!appointment.credential_verified || !retryDue) {
      renderAppointmentWaiting(appointment);
      return;
    }
    try {
      appointment = await claimDueAppointmentArrival(appointment, appointment.voucher);
      state.shareAppointment = appointment;
      state.bootstrap.share_fit_appointment = appointment;
    } catch (error) {
      if (routeShareAppointmentError(error)) return;
      throw error;
    }
  }
  if (appointment?.serial_required && appointment.appointment_required) {
    showView("login-view");
    renderAppointmentOverlay(appointment);
  } else if (
    appointment?.serial_required
    && appointment.status === "booked"
    && !appointment.can_start_setup
  ) {
    renderAppointmentWaiting(appointment);
  } else {
    // A questionnaire is part of the study session audit trail.  Enter the
    // setup flow first; a fresh participant creates the computer study
    // session there, and only then is the baseline questionnaire displayed.
    enterSetup();
  }
}

function responseExists(instrument) {
  return (state.bootstrap?.progress?.responses || []).some((row) => row.instrument === instrument);
}

function shouldShowDraftDocumentNotice(document) {
  const status = String(document?.ethics_status || "").trim().toLowerCase();
  if (status === "pending" || status === "approved") return false;
  if (status === "rejected") return true;
  return !document?.approved;
}

function renderConsent(consent) {
  $("#consent-title").textContent = consent.title;
  const container = $("#consent-document");
  container.replaceChildren();
  container.scrollTop = 0;
  if (shouldShowDraftDocumentNotice(consent)) {
    const notice = window.document.createElement("p");
    notice.className = "draft-notice";
    notice.textContent = consent.status_notice;
    container.appendChild(notice);
  }
  for (const section of consent.sections) {
    const heading = window.document.createElement("h2");
    heading.textContent = section.heading;
    container.appendChild(heading);
    for (const paragraphText of section.paragraphs) {
      const paragraph = window.document.createElement("p");
      paragraph.textContent = paragraphText;
      container.appendChild(paragraph);
    }
  }
  const choices = $("#consent-choices");
  choices.replaceChildren();
  for (const choice of consent.choices) {
    const label = document.createElement("label");
    label.className = "check-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.name = choice.id;
    input.required = Boolean(choice.required);
    const text = document.createElement("span");
    text.textContent = choice.label;
    label.append(input, text);
    choices.appendChild(label);
  }
  const typedNameInput = $("#consent-typed-name");
  const typedNameLabel = $("#consent-typed-name-label");
  const typedNameHelp = $("#consent-typed-name-help");
  if (typedNameLabel) {
    typedNameLabel.textContent = consent.typed_name_label || "参与者姓名（打字输入）";
  }
  if (typedNameHelp) {
    typedNameHelp.textContent = consent.typed_name_help
      || "请输入您的真实姓名。打字姓名与屏幕勾选确认共同构成电子知情同意记录。";
  }
  if (typedNameInput) {
    typedNameInput.value = "";
    typedNameInput.required = true;
  }
  const submit = $("#consent-submit");
  const updateConsentRead = () => {
    const reachedEnd = container.scrollHeight - container.scrollTop - container.clientHeight < 12;
    submit.disabled = !reachedEnd;
    $("#consent-scroll-hint").textContent = reachedEnd
      ? "文件已阅读到末尾，请勾选同意、打字输入姓名后确认。"
      : "请先阅读到文件末尾，再进行电子签署。";
  };
  container.onscroll = updateConsentRead;
  requestAnimationFrame(updateConsentRead);
}

async function downloadConsentCopy() {
  const response = await api("/api/consent-copy");
  const blob = await response.blob();
  const disposition = response.headers.get("content-disposition") || "";
  const matchedName = disposition.match(/filename="([^"]+)"/i);
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = matchedName?.[1] || "tap-to-adapt-consent.html";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}

async function downloadConsentEvidence() {
  const response = await api("/api/consent-evidence");
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
}

function consentClientNonce() {
  const bytes = new Uint8Array(16);
  window.crypto.getRandomValues(bytes);
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}

$("#consent-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = $("#consent-error");
  error.textContent = "";
  const consent = state.bootstrap.consent;
  const form = new FormData(event.target);
  const typedName = String(form.get("typed_name") || "").trim().replace(/\s+/g, " ");
  if (typedName.length < 2) {
    error.textContent = "请先打字输入参与者姓名（至少两个字符）。";
    $("#consent-typed-name")?.focus();
    return;
  }
  try {
    const choices = Object.fromEntries(
      consent.choices.map((choice) => [choice.id, form.get(choice.id) === "on"]),
    );
    await api("/api/consent", {
      method: "POST",
      body: {
        version: consent.version,
        document_hash: consent.sha256,
        ...choices,
        typed_name: typedName,
        affirmation: consent.required_affirmation,
        client_nonce: consentClientNonce(),
      },
    });
    try {
      await downloadConsentCopy();
    } catch (copyError) {
      error.textContent = `同意已记录，但副本下载失败：${copyError.message}。请联系研究负责人获取副本。`;
    }
    await loadBootstrap();
  } catch (err) {
    if (err.status === 409) {
      event.target.reset();
      try {
        await loadBootstrap();
        $("#consent-error").textContent =
          "知情同意书刚刚更新，系统已载入最新版并清空旧选择。请从头阅读后重新勾选；本次提交没有记录任何同意。";
      } catch (refreshError) {
        error.textContent =
          `知情同意书已更新，但自动刷新失败：${refreshError.message}。请刷新整个页面后重新阅读。`;
      }
      return;
    }
    error.textContent = `未能签署：${err.message}`;
  }
});

$("#download-consent-copy").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    await downloadConsentCopy();
  } catch (err) {
    window.alert(`副本下载失败：${err.message}`);
  } finally {
    button.disabled = false;
  }
});

$("#download-consent-evidence").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    await downloadConsentEvidence();
  } catch (err) {
    window.alert(`证据包下载失败：${err.message}`);
  } finally {
    button.disabled = false;
  }
});

function clearCompensationSignature() {
  const canvas = $("#compensation-signature");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  state.compensationSignatureDirty = false;
}

function setupCompensationSignaturePad() {
  if (state.compensationSignaturePadReady) return;
  state.compensationSignaturePadReady = true;
  const canvas = $("#compensation-signature");
  const context = canvas.getContext("2d");
  context.lineCap = "round";
  context.lineJoin = "round";
  context.lineWidth = 3.2;
  context.strokeStyle = "#17212b";
  let drawing = false;
  let travelled = 0;
  let previous = null;
  const point = (event) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (canvas.width / rect.width),
      y: (event.clientY - rect.top) * (canvas.height / rect.height),
    };
  };
  canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    drawing = true;
    travelled = 0;
    previous = point(event);
    context.beginPath();
    context.moveTo(previous.x, previous.y);
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drawing || !previous) return;
    event.preventDefault();
    const current = point(event);
    travelled += Math.hypot(current.x - previous.x, current.y - previous.y);
    context.lineTo(current.x, current.y);
    context.stroke();
    previous = current;
  });
  const finish = (event) => {
    if (!drawing) return;
    drawing = false;
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    if (travelled >= 18) state.compensationSignatureDirty = true;
    previous = null;
  };
  canvas.addEventListener("pointerup", finish);
  canvas.addEventListener("pointercancel", finish);
  clearCompensationSignature();
}

function renderCompensationRegistration(notice) {
  setupCompensationSignaturePad();
  clearCompensationSignature();
  $("#compensation-title").textContent = notice.title;
  const lead = $("#compensation-lead");
  if (lead) {
    lead.textContent = notice.lead
      || notice.purpose
      || lead.textContent;
  }
  const hideDraftNotice = !shouldShowDraftDocumentNotice(notice);
  $("#compensation-draft-notice").textContent = hideDraftNotice ? "" : notice.status_notice;
  $("#compensation-draft-notice").classList.toggle("hidden", hideDraftNotice);
  $("#compensation-purpose").textContent = notice.purpose;
  $("#compensation-contact-help").textContent = notice.contact_help;
  const protections = $("#compensation-protections");
  protections.replaceChildren();
  for (const text of notice.protections || []) {
    const item = document.createElement("li");
    item.textContent = text;
    protections.appendChild(item);
  }
  $("#compensation-choice-prompt").textContent = notice.choice_prompt;
  $("#compensation-provide-choice").textContent = notice.provide_choice_label;
  $("#compensation-decline-choice").textContent = notice.decline_choice_label;
  $("#compensation-decline-explanation").textContent =
    notice.decline_explanation;
  $("#compensation-affirmation").textContent = notice.separate_affirmation;
  const personalDebug = Boolean(state.bootstrap.participant.analysis_excluded);
  $("#compensation-debug-notice").classList.toggle("hidden", !personalDebug);
  $("#compensation-debug-copy").textContent = notice.debug_notice;
  $("#compensation-form").reset();
  $("#compensation-contact-email").value = "";
  $("#compensation-contact-phone").value = "";
  $("#compensation-contact-wechat").value = "";
  for (const radio of $$('input[name="compensation_route"]')) {
    radio.checked = false;
  }
  syncCompensationNameFromConsent();
  const contactOnly = Boolean(
    state.bootstrap.compensation_registration_decision
      && !state.bootstrap.compensation_contact_complete,
  );
  const contactAlreadySaved = Boolean(
    state.bootstrap.compensation_contact_complete,
  );
  $("#compensation-contact-card").classList.toggle("hidden", contactAlreadySaved);
  $("#compensation-contact-saved").classList.toggle("hidden", !contactAlreadySaved);
  $("#compensation-contact-only-panel").classList.toggle("hidden", !contactOnly);
  $("#compensation-route-choice").classList.toggle("hidden", contactOnly);
  $("#compensation-form").classList.add("hidden");
  $("#compensation-decline-panel").classList.add("hidden");
  $("#compensation-error").textContent = "";
}

function consentTypedName() {
  return String(state.bootstrap?.consent_typed_name || "").trim().replace(/\s+/g, " ");
}

function syncCompensationNameFromConsent() {
  const name = consentTypedName();
  const hidden = $("#compensation-full-name");
  if (hidden) hidden.value = name;
  const note = $("#compensation-name-reuse");
  if (!note) return;
  note.textContent = name
    ? I18N.t("姓名将使用知情同意页已填写的姓名：{name}").replace("{name}", name)
    : I18N.t("姓名将使用知情同意页已填写的姓名。");
}

function collectCompensationContacts(error) {
  const emailInput = $("#compensation-contact-email");
  const contacts = {
    contact_email: emailInput.value.trim(),
    contact_phone: $("#compensation-contact-phone").value.trim(),
    contact_wechat: $("#compensation-contact-wechat").value.trim(),
  };
  if (!contacts.contact_email && !contacts.contact_phone && !contacts.contact_wechat) {
    if (state.bootstrap?.compensation_contact_complete) return contacts;
    error.textContent = I18N.t("请至少填写邮箱、电话号码或微信号中的一项联系方式。");
    emailInput.focus();
    return null;
  }
  if (contacts.contact_email && !emailInput.validity.valid) {
    error.textContent = I18N.t("请输入有效的邮箱，或改填电话号码/微信号。");
    emailInput.focus();
    return null;
  }
  return contacts;
}

function renderCompensationRouteChoice() {
  const decision = $('input[name="compensation_route"]:checked')?.value || "";
  $("#compensation-form").classList.toggle("hidden", decision !== "provide");
  $("#compensation-decline-panel").classList.toggle(
    "hidden",
    decision !== "decline",
  );
  if (decision === "provide") {
    syncCompensationNameFromConsent();
  }
  $("#compensation-error").textContent = "";
}

$("#clear-compensation-signature").addEventListener(
  "click", clearCompensationSignature,
);
for (const radio of $$('input[name="compensation_route"]')) {
  radio.addEventListener("change", renderCompensationRouteChoice);
}

$("#compensation-contact-submit").addEventListener("click", async () => {
  if (state.compensationSubmitInFlight) return;
  const error = $("#compensation-error");
  error.textContent = "";
  const contacts = collectCompensationContacts(error);
  if (!contacts) return;
  const notice = state.bootstrap.compensation_registration;
  state.compensationSubmitInFlight = true;
  $("#compensation-contact-submit").disabled = true;
  try {
    await api("/api/compensation-contact", {
      method: "POST",
      body: {
        notice_version: notice.version,
        notice_hash: notice.sha256,
        ...contacts,
      },
    });
    await loadBootstrap();
  } catch (err) {
    if (err.status === 409) await loadBootstrap();
    else error.textContent = `${I18N.t("未能保存联系方式：")}${err.message}`;
  } finally {
    state.compensationSubmitInFlight = false;
    $("#compensation-contact-submit").disabled = false;
  }
});

$("#compensation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.compensationSubmitInFlight) return;
  const error = $("#compensation-error");
  error.textContent = "";
  const contacts = collectCompensationContacts(error);
  if (!contacts) return;
  if (!state.compensationSignatureDirty) {
    error.textContent = I18N.t("请先在签名框内完成手写电子签名。");
    $("#compensation-signature").focus();
    return;
  }
  const notice = state.bootstrap.compensation_registration;
  syncCompensationNameFromConsent();
  const fullName = consentTypedName() || $("#compensation-full-name").value.trim();
  if (fullName.length < 2) {
    error.textContent = I18N.t("缺少知情同意页姓名，请返回上一页重新签署后再登记报销信息。");
    return;
  }
  const signature = $("#compensation-signature")
    .toDataURL("image/png")
    .replace(/^data:image\/png;base64,/, "");
  state.compensationSubmitInFlight = true;
  $("#compensation-submit").disabled = true;
  try {
    await api("/api/compensation-identity", {
      method: "POST",
      body: {
        notice_version: notice.version,
        notice_hash: notice.sha256,
        document_type: $("#compensation-document-type").value,
        full_name: fullName,
        document_number: $("#compensation-document-number").value.trim(),
        signature_png_base64: signature,
        separate_consent: $("#compensation-separate-consent").checked,
        affirmation: notice.separate_affirmation,
        ...contacts,
      },
    });
    await loadBootstrap();
  } catch (err) {
    if (err.status === 409) {
      await loadBootstrap();
      $("#compensation-error").textContent =
        I18N.t("补偿身份信息说明已更新，请重新阅读并提交。");
    } else {
      error.textContent = `${I18N.t("未能保存补偿登记：")}${err.message}`;
    }
  } finally {
    state.compensationSubmitInFlight = false;
    $("#compensation-submit").disabled = false;
  }
});

$("#compensation-decline-submit").addEventListener("click", async () => {
  if (state.compensationSubmitInFlight) return;
  const error = $("#compensation-error");
  error.textContent = "";
  const contacts = collectCompensationContacts(error);
  if (!contacts) return;
  const notice = state.bootstrap.compensation_registration;
  state.compensationSubmitInFlight = true;
  $("#compensation-decline-submit").disabled = true;
  try {
    await api("/api/compensation-decline", {
      method: "POST",
      body: {
        notice_version: notice.version,
        notice_hash: notice.sha256,
        decision: "decline",
        affirmation: notice.decline_affirmation,
        ...contacts,
      },
    });
    await loadBootstrap();
  } catch (err) {
    if (err.status === 409) {
      await loadBootstrap();
      $("#compensation-error").textContent =
        I18N.t("补偿报销说明已更新，请重新阅读并选择。");
    } else {
      error.textContent = `${I18N.t("未能保存补偿选择：")}${err.message}`;
    }
  } finally {
    state.compensationSubmitInFlight = false;
    $("#compensation-decline-submit").disabled = false;
  }
});

function enterSetup() {
  showView("setup-view");
  const { openSession, completedSession } = configureSingleSession();
  const companionBase = `${location.origin}${APP_BASE}/companion/`;
  const companionCode = resolveFullStudyCode(
    state.code,
    state.bootstrap?.participant?.code_hint,
  );
  if (companionCode) rememberStudyCode(companionCode, state.bootstrap?.participant?.code_hint);
  const companionUrl = companionCode
    ? `${companionBase}${encodeURIComponent(companionCode)}${I18N.isEnglish ? "?lang=en" : ""}`
    : "";
  const companionLink = $("#companion-url");
  const companionCopy = $("#companion-url-copy");
  if (companionUrl) {
    companionLink.textContent = companionUrl;
    companionLink.href = companionUrl;
    companionCopy.dataset.url = companionUrl;
    companionCopy.disabled = false;
  } else {
    companionLink.textContent = I18N.t(
      "完整陪伴链接暂不可用。请使用首次报名的电脑，或联系微信 study-contact 获取完整编号（TTA-XXXX-XXXX），不要使用四位尾号链接。",
    );
    companionLink.removeAttribute("href");
    companionCopy.dataset.url = "";
    companionCopy.disabled = true;
  }
  $("#companion-url-copy-status").textContent = "";
  $("#companion-url-copy-status").classList.add("hidden");
  if (completedSession && !openSession && !state.bootstrap?.interrupt_checkpoint?.available) {
    renderCompletion(false);
    return;
  }
  if (!protocolUsesSingleSession()) {
    $("#session-error").textContent = "该编号绑定的是已停用的旧版多日流程，不能继续。请在管理端重新生成“仅用于个人调试”或当前实验阶段的新编号。";
    markStep("overview");
    return;
  }
  if (openSession) {
    restoreOpenSession(openSession);
    applyShareSetupTiming(state.shareAppointment);
    armShareSetupGuard();
    if (!responseExists("baseline")) {
      showQuestionnaire("baseline", {}, { afterSubmit: prepareDailySetup });
    } else {
      prepareDailySetup();
    }
    return;
  }
  checkBrowser();
  markStep("overview");
  startStudySession();
}

function protocolUsesSingleSession() {
  return Number(state.bootstrap?.client_protocol?.session_design?.total_sessions || 0) === 1;
}

function configureSingleSession() {
  const studySessions = (state.bootstrap.progress.sessions || [])
    .filter((session) => session.client_role === "study");
  const open = studySessions.find((session) => !["completed", "ended", "aborted"].includes(session.state));
  const completed = studySessions.find((session) => ["completed", "ended"].includes(session.state));
  state.day = open ? Number(open.day) : 1;
  state.previousTopicIds = new Set();
  restoreTopicSelectionFromHistory();
  return { openSession: open || null, completedSession: completed || null };
}

function restoreTopicSelectionFromHistory() {
  const topicHistory = state.bootstrap?.topic_history || [];
  const currentTopics = topicHistory
    .filter((row) => Number(row.day) === state.day)
    .sort((left, right) => Number(left.server_ms) - Number(right.server_ms));
  const currentTopic = currentTopics.at(-1);
  state.selectedTopic = currentTopic?.topic_id || null;
  state.topicSaved = Boolean(currentTopic);
  return currentTopic || null;
}

function restoreOpenSession(session) {
  state.sessionId = session.id;
  state.day = Number(session.day);
  state.sessionActivated = session.state === "active" || session.activated_ms != null;
  state.quality = new Map();
  state.qualityAttempts = new Map();
  const skipQualityRestore = Boolean(
    state.pendingInterruptChoice
    || state.bootstrap?.interrupt_checkpoint?.available
  );
  if (skipQualityRestore) {
    forceFreshQualityChecks();
    sendEvent("session.client_resumed", {
      day: state.day,
      demo: state.demo,
      session_state: session.state,
      restored_quality_kinds: [],
      quality_restore_skipped: true,
    }).catch((err) => {
      $("#session-error").textContent = `Session recovery event failed: ${err.message}`;
    });
    return;
  }
  const restoredChecks = new Set();
  const checkIds = {
    network: "network",
    microphone: "mic",
    reaction_latency: "reaction",
    browser: "browser",
  };
  for (const row of state.bootstrap.progress.quality || []) {
    if (row.session_id !== state.sessionId) continue;
    let metrics = {};
    try {
      metrics = typeof row.metrics === "string" ? JSON.parse(row.metrics) : (row.metrics || {});
    } catch (_) { /* A legacy quality row may not contain JSON metrics. */ }
    const currentBrowserVerification = row.kind !== "browser"
      || (
        metrics.verification_method === "typed_keyword_recall"
        && ["zh-blue-kite-v1", "en-blue-kite-v1"].includes(metrics.phrase_id)
      );
    const restoredPassed = Boolean(row.passed) && currentBrowserVerification;
    if (!state.quality.has(row.kind)) state.quality.set(row.kind, restoredPassed);
    if (row.kind === "browser" && restoredPassed) {
      const language = ["zh-CN", "en-US"].includes(metrics.test_language)
        ? metrics.test_language
        : "zh-CN";
      $("#browser-audio-language").value = language;
      $("#browser-audio-language").dataset.passedLanguage = language;
      $("#browser-audio-language").disabled = true;
    }
    const checkId = checkIds[row.kind];
    if (checkId && !restoredChecks.has(row.kind)) {
      setCheck(
        checkId,
        restoredPassed,
        restoredPassed ? "已通过（从本次记录恢复）" : "上次检测未通过，请重新检测",
      );
      if (row.kind === "microphone" && !restoredPassed) {
        showMicTroubleshooting(metrics);
      }
      restoredChecks.add(row.kind);
    }
    const attempt = Number(metrics?.attempt_number || 0);
    state.qualityAttempts.set(
      row.kind,
      Math.max(state.qualityAttempts.get(row.kind) || 0, attempt),
    );
  }
  sendEvent("session.client_resumed", {
    day: state.day,
    demo: state.demo,
    session_state: session.state,
    restored_quality_kinds: [...state.quality.keys()],
  }).catch((err) => {
    $("#session-error").textContent = `Session recovery event failed: ${err.message}`;
  });
}

function forceFreshQualityChecks() {
  state.quality = new Map();
  state.qualityAttempts = new Map();
  const labels = {
    network: "请重新完成网络检测",
    mic: "请重新完成麦克风检测",
    reaction: "请重新完成点击检测",
    browser: "请重新完成播放与听辨检测",
  };
  for (const [id, text] of Object.entries(labels)) {
    const item = $(`#${id}-check`);
    item?.classList.remove("pass", "fail");
    const value = $(`#${id}-value`);
    if (value) value.textContent = I18N.t(text);
  }
  const language = $("#browser-audio-language");
  if (language) {
    language.disabled = false;
    delete language.dataset.passedLanguage;
  }
}

function currentInterruptCheckpoint() {
  const local = loadScheduleProgress();
  if (local?.interrupted) {
    return {
      available: true,
      session_id: local.session_id || state.sessionId,
      cursor: Number(local.cursor || 0),
      stage_key: local.stage_key || null,
      active_elapsed_seconds: Number(local.active_elapsed_seconds || 0),
      opportunities: Number(local.opportunities || 0),
      reason: local.interrupted_reason || "",
    };
  }
  return state.bootstrap?.interrupt_checkpoint || null;
}

function interruptResumeCursor() {
  const checkpoint = currentInterruptCheckpoint();
  if (!checkpoint?.available) return null;
  const stages = scheduleStages();
  if (checkpoint.stage_key) {
    const index = stages.findIndex((stage) => stage.key === checkpoint.stage_key);
    if (index >= 0) return index;
  }
  const cursor = Number(checkpoint.cursor);
  return Number.isFinite(cursor) ? Math.max(0, cursor) : null;
}

function syncInterruptChoicePanel() {
  const panel = $("#interrupt-choice-panel");
  const startButton = $("#start-study-btn");
  const checkpoint = currentInterruptCheckpoint();
  const show = Boolean(state.pendingInterruptChoice && checkpoint?.available);
  panel?.classList.toggle("hidden", !show);
  if (startButton) startButton.classList.toggle("hidden", show);
  if (show) {
    const ready = startEligibilityBlockers().length === 0;
    $("#interrupt-restart-btn").disabled = !ready;
    $("#interrupt-continue-btn").disabled = !ready;
  }
}

function markStep(active) {
  const order = ["overview", "network", "phone", "training", "ready"];
  const activeIndex = order.indexOf(active);
  for (const item of $$(".step-item")) {
    const index = order.indexOf(item.dataset.step);
    item.classList.toggle("active", index === activeIndex);
    item.classList.toggle("done", index < activeIndex);
  }
}

async function startStudySession() {
  if (state.sessionCreateInFlight || state.sessionId) return;
  const error = $("#session-error");
  error.textContent = "";
  state.sessionCreateInFlight = true;
  try {
    const session = await api("/api/sessions/start", { method: "POST", body: { day: state.day } });
    state.sessionId = session.id;
    state.quality = new Map();
    state.qualityAttempts = new Map();
    restoreTopicSelectionFromHistory();
    state.sessionActivated = session.state === "active";
    if (session.share_fit_appointment) {
      state.shareAppointment = session.share_fit_appointment;
      if (state.bootstrap) {
        state.bootstrap.share_fit_appointment = session.share_fit_appointment;
      }
      applyShareSetupTiming(session.share_fit_appointment);
      armShareSetupGuard();
    }
    sendEvent(
      state.sessionActivated ? "session.client_resumed" : "session.client_started",
      { day: state.day, demo: state.demo },
    );
    if (!responseExists("baseline")) {
      await showQuestionnaire("baseline", {}, { afterSubmit: prepareDailySetup });
    } else {
      prepareDailySetup();
    }
  } catch (err) {
    if (routeShareAppointmentError(err)) {
      error.textContent = "";
      return;
    }
    error.textContent = `无法开始：${err.message}`;
  } finally {
    state.sessionCreateInFlight = false;
  }
}

function prepareDailySetup() {
  showView("setup-view");
  armShareSetupGuard();
  restoreTapTrainingCompletion();
  restoreDataCollectionPreference();
  const saved = loadScheduleProgress();
  if (saved?.interrupted || (Number(saved?.cursor || 0) > 0 && !saved?.interaction_complete) || state.bootstrap?.interrupt_checkpoint?.available) {
    if (state.bootstrap?.operator_resume) {
      /* Operator sidecar still owns the XSLX-style block-5 path. */
    } else {
      state.pendingInterruptChoice = true;
      forceFreshQualityChecks();
    }
    if (state.bootstrap?.participant?.code_hint) {
      setConnection(`编号尾号 ${state.bootstrap.participant.code_hint}`, "online");
    }
  }
  if (state.bootstrap?.operator_resume?.resume_block === 5) {
    $("#start-error").textContent = I18N.t("本次将从第 5/5 个统计块继续，请完成电脑四分钟与手机两分钟。");
  }
  $("#quality-panel").classList.remove("hidden");
  $("#phone-panel").classList.remove("hidden");
  refreshMicrophoneDevices(false).catch(() => {});
  markStep("network");
  state.pairReconnectEnabled = true;
  updateStartEligibility();
  connectPairSocket();
  startCompanionLivePolling();
  if (state.quality.get("browser") !== true) checkBrowser();
  window.scrollTo({ top: $("#quality-panel").offsetTop - 18, behavior: "smooth" });
}

async function saveQuality(kind, passed, metrics) {
  const attemptNumber = (state.qualityAttempts.get(kind) || 0) + 1;
  const attemptId = crypto.randomUUID();
  const result = await api("/api/quality", {
    method: "POST",
    body: {
      session_id: state.sessionId,
      kind,
      passed,
      metrics: { ...metrics, attempt_id: attemptId, attempt_number: attemptNumber },
    },
  });
  state.qualityAttempts.set(kind, Number(result.attempt_number || attemptNumber));
  state.quality.set(kind, result.passed === true);
  updateStartEligibility();
  return result;
}

function setCheck(id, passed, text) {
  const item = $(`#${id}-check`);
  item.classList.remove("pass", "fail");
  item.classList.add(passed ? "pass" : "fail");
  $(`#${id}-value`).textContent = text;
}

$("#network-btn").addEventListener("click", async () => {
  const button = $("#network-btn");
  button.disabled = true;
  try {
    $("#network-value").textContent = "正在发送 12 次测量请求";
    const sampleTimeoutMs = 2000;
    const rtts = [];
    const successfulRtts = [];
    let failures = 0;
    for (let index = 0; index < 12; index += 1) {
      const started = performance.now();
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), sampleTimeoutMs);
      try {
        await api(`/api/ping?sample=${index}`, { signal: controller.signal });
        const rtt = performance.now() - started;
        rtts.push(rtt);
        successfulRtts.push(rtt);
      } catch (_) {
        failures += 1;
        // Keep timeout-censored samples finite so the server can persist a failed probe.
        rtts.push(sampleTimeoutMs);
      } finally {
        clearTimeout(timeout);
      }
      await sleep(90);
    }
    rtts.sort((a, b) => a - b);
    successfulRtts.sort((a, b) => a - b);
    const median = percentile(successfulRtts, 0.5);
    const p95 = percentile(successfulRtts, 0.95);
    const jitter = successfulRtts.length
      ? Math.sqrt(successfulRtts.reduce((sum, value) => sum + (value - median) ** 2, 0) / successfulRtts.length)
      : Infinity;
    // Loss is its own gate. A single tolerated timeout must not also become the
    // latency P95/max and make `failures <= 1` impossible by construction.
    const QG = window.TTA_QUALITY_GATE;
    const networkMetrics = {
      median_ms: median,
      p95_ms: p95,
      jitter_ms: jitter,
      failures,
      samples: rtts,
      successful_samples_ms: successfulRtts,
      timeout_ms: sampleTimeoutMs,
    };
    const saved = await saveQuality("network", QG.evaluateNetwork(networkMetrics).passed, networkMetrics);
    setCheck(
      "network",
      saved.passed,
      QG.formatNetworkSummary(networkMetrics, saved.passed, saved.server_decision),
    );
  } catch (err) {
    setCheck(
      "network",
      false,
      window.TTA_QUALITY_GATE.formatSaveError(err, "network", state.qualityAttempts.get("network") || 0),
    );
  } finally {
    button.disabled = false;
  }
});

function stopMicStream() {
  if (state.micStream) {
    for (const track of state.micStream.getTracks()) track.stop();
  }
  state.micStream = null;
}

function selectedMicLabel() {
  const select = $("#mic-device-select");
  return select?.selectedOptions?.[0]?.textContent || "Windows / Edge 默认输入设备";
}

async function refreshMicrophoneDevices(requestPermission = false) {
  let temporaryStream = null;
  if (requestPermission && !state.micStream?.active) {
    temporaryStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  }
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices())
      .filter((device) => device.kind === "audioinput" && device.deviceId !== "default");
    const select = $("#mic-device-select");
    const preferred = state.micDeviceId;
    select.replaceChildren(new Option("Windows / Edge 默认输入设备", ""));
    devices.forEach((device, index) => {
      select.add(new Option(device.label || `麦克风 ${index + 1}`, device.deviceId));
    });
    if ([...select.options].some((option) => option.value === preferred)) {
      select.value = preferred;
    } else {
      state.micDeviceId = "";
      localStorage.removeItem(storageKey("microphone_device_id"));
    }
  } finally {
    if (temporaryStream) {
      for (const track of temporaryStream.getTracks()) track.stop();
    }
  }
}

async function ensureMic() {
  const currentDeviceId = state.micStream?.getAudioTracks?.()[0]?.getSettings?.().deviceId || "";
  if (
    state.micStream?.active
    && (!state.micDeviceId || currentDeviceId === state.micDeviceId)
  ) return state.micStream;
  stopMicStream();
  const audio = {
    channelCount: 1,
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  };
  if (state.micDeviceId) audio.deviceId = { exact: state.micDeviceId };
  state.micStream = await navigator.mediaDevices.getUserMedia({
    audio,
    video: false,
  });
  await refreshMicrophoneDevices(false);
  return state.micStream;
}

function hideMicTroubleshooting() {
  $("#mic-troubleshooting").classList.add("hidden");
}

function showMicTroubleshooting(metrics = {}) {
  const speech = Number(metrics.speech_rms_p90 || 0);
  const silentInput = speech < 0.00001;
  const asrFailed = metrics.asr_probe_passed === false && metrics.asr_probe_soft !== true;
  const clipping = Number(metrics.clipping_ratio || 0) >= 0.01;
  $("#mic-troubleshooting-title").textContent = silentInput
    ? "当前输入设备正在输出静音"
    : clipping
      ? "说话音量偏大"
      : asrFailed
        ? "语音识别未通过（不是音量不够）"
        : "麦克风或环境检测未通过";
  const heard = metrics.asr_probe_transcript
    ? `刚才识别到：“${metrics.asr_probe_transcript}”。`
    : "";
  $("#mic-troubleshooting-detail").textContent = silentInput
    ? `网页已连接“${selectedMicLabel()}”，但没有收到声音。通常是 Windows 默认输入设备切换、硬件静音或选中了虚拟设备。`
    : clipping
      ? `当前音量偏大。请离麦 30–90 厘米，用日常交谈的正常音量自然说一句完整的话。当前设备：${selectedMicLabel()}。`
      : asrFailed
        ? `${heard}请用日常交谈的正常音量自然说一句完整的话。当前设备：${selectedMicLabel()}。`
        : `当前使用“${selectedMicLabel()}”。请按以下步骤核对设备、输入音量和环境噪声后重试。`;
  $("#mic-troubleshooting").classList.remove("hidden");
}

$("#refresh-mic-devices").addEventListener("click", async () => {
  const button = $("#refresh-mic-devices");
  button.disabled = true;
  try {
    await refreshMicrophoneDevices(true);
    $("#mic-value").textContent = `已刷新麦克风列表，当前选择：${selectedMicLabel()}`;
  } catch (error) {
    $("#mic-value").textContent = `无法读取麦克风列表：${error.message}`;
    showMicTroubleshooting();
  } finally {
    button.disabled = false;
  }
});

$("#mic-device-select").addEventListener("change", () => {
  state.micDeviceId = $("#mic-device-select").value;
  if (state.micDeviceId) {
    localStorage.setItem(storageKey("microphone_device_id"), state.micDeviceId);
  } else {
    localStorage.removeItem(storageKey("microphone_device_id"));
  }
  stopMicStream();
  state.quality.delete("microphone");
  setCheck("mic", false, `已切换到“${selectedMicLabel()}”，请重新检测`);
  hideMicTroubleshooting();
  updateStartEligibility();
});

navigator.mediaDevices?.addEventListener?.("devicechange", () => {
  refreshMicrophoneDevices(false).catch(() => {});
});

async function sampleAudioMetrics(analyser, seconds) {
  const buffer = new Float32Array(analyser.fftSize);
  const rmsValues = [];
  let clipped = 0;
  let total = 0;
  const end = performance.now() + seconds * 1000;
  while (performance.now() < end) {
    analyser.getFloatTimeDomainData(buffer);
    let sum = 0;
    for (const value of buffer) {
      sum += value * value;
      if (Math.abs(value) >= 0.985) clipped += 1;
      total += 1;
    }
    rmsValues.push(Math.sqrt(sum / buffer.length));
    await sleep(50);
  }
  rmsValues.sort((a, b) => a - b);
  return { median: percentile(rmsValues, 0.5), p90: percentile(rmsValues, 0.9), clipping: total ? clipped / total : 0 };
}

function runOpenRecognitionProbe(timeoutMs = 9000) {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    return Promise.resolve({
      chars: 0,
      confidence: 0,
      error: "unsupported",
      transcript: "",
    });
  }
  return new Promise((resolve) => {
    const recognition = new Recognition();
    let settled = false;
    let bestText = "";
    let confidence = 0;
    const finish = (error = "") => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { recognition.stop(); } catch (_) { /* Already stopped. */ }
      resolve({
        chars: bestText.trim().length,
        confidence,
        error,
        transcript: bestText.trim(),
      });
    };
    recognition.lang = I18N.isEnglish ? "en-US" : "zh-CN";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 3;
    recognition.onresult = (event) => {
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const alternatives = event.results[index];
        for (let alt = 0; alt < alternatives.length; alt += 1) {
          const candidate = alternatives[alt];
          const text = String(candidate?.transcript || "");
          if (text.length >= bestText.length) {
            bestText = text;
            confidence = Number(candidate.confidence || 0);
          }
        }
        if (event.results[index].isFinal) finish();
      }
    };
    recognition.onerror = (event) => finish(event.error || "recognition_error");
    recognition.onend = () => finish(bestText ? "" : "ended_without_result");
    const timer = setTimeout(() => finish("timeout"), timeoutMs);
    try { recognition.start(); } catch (error) { finish(error.message || "start_failed"); }
  });
}

function pcmChunksToBase64(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const joined = new Int16Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.length;
  }
  const bytes = new Uint8Array(joined.buffer);
  let binary = "";
  for (let index = 0; index < bytes.length; index += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  }
  return btoa(binary);
}

async function transcribePcmChunks(chunks, context, signal = null) {
  if (!chunks?.length) throw new Error("没有可识别的音频");
  return api("/api/asr/transcribe", {
    method: "POST",
    body: {
      session_id: state.sessionId,
      pcm16_base64: pcmChunksToBase64(chunks),
      sample_rate_hz: 16000,
      language: I18N.isEnglish ? "en" : "zh",
      context,
    },
    ...(signal ? { signal } : {}),
  });
}

function appendRollingPcm(samples) {
  const chunk = samples.slice();
  state.serverAsrPcmChunks.push(chunk);
  state.serverAsrPcmSamples += chunk.length;
  const maximum = 15 * 16000;
  while (state.serverAsrPcmSamples > maximum && state.serverAsrPcmChunks.length > 1) {
    const removed = state.serverAsrPcmChunks.shift();
    state.serverAsrPcmSamples -= removed.length;
  }
}

function clearRollingPcm() {
  state.serverAsrPcmChunks = [];
  state.serverAsrPcmSamples = 0;
}

function appendCompanionRelayPcm(samples) {
  const chunk = samples.slice();
  state.companionRelayPcmChunks.push(chunk);
  state.companionRelayPcmSamples += chunk.length;
  const maximum = 15 * 16000;
  while (
    state.companionRelayPcmSamples > maximum
    && state.companionRelayPcmChunks.length > 1
  ) {
    const removed = state.companionRelayPcmChunks.shift();
    state.companionRelayPcmSamples -= removed.length;
  }
}

async function capturePcmWhile(source, context, task) {
  await context.audioWorklet.addModule(`${APP_BASE}/static/pcm-worklet.js`);
  const worklet = new AudioWorkletNode(context, "pcm-capture");
  const mute = context.createGain();
  const chunks = [];
  mute.gain.value = 0;
  worklet.port.onmessage = (event) => chunks.push(new Int16Array(event.data).slice());
  source.connect(worklet).connect(mute).connect(context.destination);
  try {
    await task();
  } finally {
    try { source.disconnect(worklet); } catch (_) { /* Already disconnected. */ }
    try { worklet.disconnect(); } catch (_) { /* Already disconnected. */ }
    try { mute.disconnect(); } catch (_) { /* Already disconnected. */ }
  }
  return chunks;
}

function hideMicAsrVerification() {
  $("#mic-asr-verification").classList.add("hidden");
  $("#mic-asr-transcript").textContent = "";
}

function presentMicAsrCandidate() {
  const pending = state.pendingMicVerification;
  const candidate = pending?.candidates?.[pending.candidateIndex];
  if (!candidate) return false;
  pending.metrics.asr_probe_transcript = candidate.transcript;
  pending.metrics.asr_probe_source = candidate.source;
  pending.metrics.asr_probe_model = candidate.model || null;
  pending.metrics.asr_probe_latency_ms = candidate.latency_ms || null;
  $("#mic-asr-transcript").textContent = candidate.transcript;
  $("#mic-asr-verification").classList.remove("hidden");
  $("#mic-value").textContent = `${pending.volumeStatus} · SNR ${round(pending.metrics.snr_db, 1)} dB · 请核对识别结果`;
  return true;
}

$("#mic-btn").addEventListener("click", async () => {
  const button = $("#mic-btn");
  button.disabled = true;
  hideMicTroubleshooting();
  hideMicAsrVerification();
  state.pendingMicVerification = null;
  try {
    if (state.demo) {
      $("#mic-value").textContent = "正在注入预演音频指标";
      await sleep(350);
      const metrics = {
        quiet_rms_median: 0.012,
        quiet_rms_p90: 0.018,
        speech_rms_p90: 0.18,
        snr_db: 23.5,
        clipping_ratio: 0,
        simulation: true,
      };
      setCheck("mic", true, "预演输入 · SNR 23.5 dB · 不可用于正式分析");
      await saveQuality("microphone", true, metrics);
      return;
    }
    const stream = await ensureMic();
    const context = state.audioContext || new AudioContext();
    state.audioContext = context;
    await withTimeout(context.resume(), 5000, "音频系统没有响应");
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    const track = stream.getAudioTracks()[0];
    const trackSettings = track.getSettings();
    const trackConstraints = track.getConstraints();
    $("#mic-value").textContent = "请保持安静 3 秒";
    const quiet = await sampleAudioMetrics(analyser, 3);
    $("#mic-value").textContent = "请用正常音量自然说一句完整的话，例如介绍今天做过的一件事";
    let speech;
    const pcmChunks = await capturePcmWhile(source, context, async () => {
      speech = await sampleAudioMetrics(analyser, 6);
    });
    try { source.disconnect(); } catch (_) { /* Already disconnected. */ }
    stopMicStream();
    $("#mic-value").textContent = "正在用后台语音识别核对刚才的发言";
    let serverProbe = null;
    let serverError = "";
    try {
      serverProbe = await transcribePcmChunks(pcmChunks, "preflight");
    } catch (error) {
      serverError = String(error.message || error);
    }
    const snr = 20 * Math.log10(Math.max(speech.p90, 1e-7) / Math.max(quiet.median, 1e-7));
    const acousticPassed = snr >= 8
      && quiet.p90 < 0.09
      && speech.clipping < 0.01
      && speech.p90 > 0.006;
    const volumeStatus = speech.p90 <= 0.006
      ? "正常说话音量偏小"
      : speech.clipping >= 0.01
        ? "说话音量偏大并出现削波（请改用正常说话音量）"
        : "正常说话音量可用";
    const metrics = {
      quiet_rms_median: quiet.median,
      quiet_rms_p90: quiet.p90,
      speech_rms_p90: speech.p90,
      snr_db: snr,
      clipping_ratio: speech.clipping,
      speech_volume_status: volumeStatus,
      instructed_distance_cm: [30, 90],
      asr_verification_version: "semantic-confirmation-v1",
      asr_pipeline_version: "server-whisper-only-v1",
      asr_probe_passed: false,
      asr_user_confirmed: false,
      asr_probe_chars: 0,
      asr_probe_confidence: null,
      asr_browser_error: "",
      asr_server_error: serverError,
      asr_probe_transcript: "",
      track_settings: trackSettings,
      constraints: trackConstraints,
    };
    const candidates = [];
    const addCandidate = (transcript, sourceName, detail = {}) => {
      const value = String(transcript || "").trim();
      if (!value || candidates.some((item) => item.transcript === value)) return;
      candidates.push({ transcript: value, source: sourceName, ...detail });
    };
    addCandidate(serverProbe?.text, "server_whisper", {
      model: serverProbe?.model,
      latency_ms: serverProbe?.latency_ms,
    });
    state.pendingMicVerification = {
      metrics,
      candidates,
      candidateIndex: 0,
      acousticPassed,
      volumeStatus,
    };
    if (!presentMicAsrCandidate()) {
      const saved = await saveQuality("microphone", false, metrics);
      setCheck("mic", false, "没有识别到语音，请确认麦克风后重新测试");
      showMicTroubleshooting({ ...metrics, server_decision: saved.server_decision });
      state.pendingMicVerification = null;
    }
  } catch (err) {
    setCheck(
      "mic",
      false,
      window.TTA_QUALITY_GATE.formatSaveError(err, "microphone", state.qualityAttempts.get("microphone") || 0),
    );
    showMicTroubleshooting({ error: err.message });
    await saveQuality("microphone", false, {
      error: err.message,
      asr_verification_version: "semantic-confirmation-v1",
      asr_user_confirmed: false,
    });
  } finally {
    button.disabled = false;
  }
});

$("#mic-asr-accept").addEventListener("click", async () => {
  const pending = state.pendingMicVerification;
  if (!pending) return;
  const accept = $("#mic-asr-accept");
  const reject = $("#mic-asr-reject");
  accept.disabled = true;
  reject.disabled = true;
  try {
    const transcript = String(pending.metrics.asr_probe_transcript || "").trim();
    pending.metrics.asr_user_confirmed = true;
    pending.metrics.asr_probe_passed = Boolean(transcript);
    pending.metrics.asr_probe_chars = transcript.length;
    const saved = await saveQuality("microphone", pending.acousticPassed, pending.metrics);
    setCheck(
      "mic",
      saved.passed,
      saved.passed
        ? `${pending.volumeStatus} · SNR ${round(pending.metrics.snr_db, 1)} dB · 语音识别大意正确`
        : window.TTA_QUALITY_GATE.formatMicrophoneSummary(
          pending.metrics,
          false,
          saved.server_decision,
        ),
    );
    hideMicAsrVerification();
    if (saved.passed) hideMicTroubleshooting();
    else showMicTroubleshooting(pending.metrics);
    state.pendingMicVerification = null;
  } finally {
    accept.disabled = false;
    reject.disabled = false;
  }
});

$("#mic-asr-reject").addEventListener("click", async () => {
  const pending = state.pendingMicVerification;
  if (!pending) return;
  pending.candidateIndex += 1;
  if (presentMicAsrCandidate()) {
    $("#mic-value").textContent = "已切换另一种识别结果，请再次核对大意";
    return;
  }
  pending.metrics.asr_user_confirmed = false;
  pending.metrics.asr_probe_passed = false;
  await saveQuality("microphone", false, pending.metrics);
  hideMicAsrVerification();
  setCheck("mic", false, "识别结果与刚才说的话明显不符，请重新测试");
  showMicTroubleshooting(pending.metrics);
  state.pendingMicVerification = null;
});

let reactionTrial = null;
$("#reaction-btn").addEventListener("click", async () => {
  $("#reaction-btn").disabled = true;
  if (state.demo) {
    const samples = [238, 251, 264, 279, 293, 307];
    setCheck("reaction", true, "预演点击反应中位数 272 ms · 不可用于正式分析");
    await saveQuality("reaction_latency", true, {
      median_ms: 271.5,
      samples_ms: samples,
      synthetic: true,
    });
    $("#reaction-btn").disabled = false;
    return;
  }
  $("#reaction-stage").classList.remove("hidden");
  const samples = [];
  try {
    for (let trial = 0; trial < 6; trial += 1) {
      $("#reaction-wait").classList.remove("hidden");
      $("#reaction-target").classList.add("hidden");
      await sleep(800 + Math.random() * 1200);
      reactionTrial = { shown: performance.now(), resolve: null };
      const clicked = new Promise((resolve) => { reactionTrial.resolve = resolve; });
      $("#reaction-wait").classList.add("hidden");
      $("#reaction-target").textContent = I18N.t(`点我（还差 ${6 - trial} 次）`);
      $("#reaction-target").classList.remove("hidden");
      samples.push(await clicked);
    }
    samples.sort((a, b) => a - b);
    const median = percentile(samples, 0.5);
    const QG = window.TTA_QUALITY_GATE;
    const reactionMetrics = { median_ms: median, samples_ms: samples };
    const saved = await saveQuality(
      "reaction_latency",
      QG.evaluateReaction(reactionMetrics).passed,
      reactionMetrics,
    );
    setCheck(
      "reaction",
      saved.passed,
      saved.passed
        ? `点击反应中位数 ${round(median)} ms · 范围 ${round(samples[0])}–${round(samples.at(-1))} ms`
        : QG.formatReactionSummary(reactionMetrics, false, saved.server_decision),
    );
  } catch (err) {
    setCheck(
      "reaction",
      false,
      window.TTA_QUALITY_GATE.formatSaveError(err, "reaction_latency", state.qualityAttempts.get("reaction_latency") || 0),
    );
  } finally {
    $("#reaction-stage").classList.add("hidden");
    $("#reaction-btn").disabled = false;
  }
});

$("#reaction-target").addEventListener("pointerdown", () => {
  if (!reactionTrial || !reactionTrial.resolve) return;
  const latency = performance.now() - reactionTrial.shown;
  const resolve = reactionTrial.resolve;
  reactionTrial.resolve = null;
  resolve(latency);
});

function browserCapabilityChecks() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  return {
    secure_context: window.isSecureContext,
    media_devices: Boolean(navigator.mediaDevices?.getUserMedia),
    audio_worklet: Boolean(window.AudioWorkletNode),
    websocket: Boolean(window.WebSocket),
    speech_recognition: Boolean(Recognition),
    media_recorder: Boolean(window.MediaRecorder),
    speech_synthesis: Boolean(window.speechSynthesis && window.SpeechSynthesisUtterance),
  };
}

function browserCapabilitiesPassed(checks) {
  return [
    "secure_context",
    "media_devices",
    "audio_worklet",
    "websocket",
    "media_recorder",
    "speech_synthesis",
  ]
    .every((name) => checks[name] === true);
}

function setBrowserCheckPending(text = "请选择语言并播放测试句") {
  const item = $("#browser-check");
  item.classList.remove("pass", "fail");
  $("#browser-value").textContent = I18N.t(text);
}

function resetBrowserPlaybackInput() {
  window.speechSynthesis?.cancel();
  state.browserPlaybackProbe = null;
  $("#browser-audio-answer").value = "";
  $("#browser-answer-row").classList.add("hidden");
  $("#browser-audio-verify").classList.add("hidden");
  window.TTA_PLAYBACK_TEST.resetVerifyButton($("#browser-audio-verify"));
}

function checkBrowser() {
  const checks = browserCapabilityChecks();
  const passed = browserCapabilitiesPassed(checks);
  if (passed) {
    if (state.quality.get("browser") !== true) {
      state.quality.delete("browser");
      setBrowserCheckPending();
      updateStartEligibility();
    }
    return;
  }
  setCheck(
    "browser",
    false,
    `${I18N.t("缺少：")}${Object.entries(checks).filter(([, value]) => !value).map(([key]) => key).join("、")}`,
  );
  if (state.sessionId) saveQuality("browser", false, {
    ...checks,
    user_agent: navigator.userAgent,
    device_id: state.deviceId,
  }).catch(() => {});
  else state.quality.set("browser", false);
  updateStartEligibility();
}

$("#browser-btn").addEventListener("click", async () => {
  const button = $("#browser-btn");
  const checks = browserCapabilityChecks();
  resetBrowserPlaybackInput();
  state.quality.delete("browser");
  updateStartEligibility();
  if (!browserCapabilitiesPassed(checks)) {
    checkBrowser();
    return;
  }
  button.disabled = true;
  $("#browser-value").textContent = I18N.t("正在播放测试句");
  try {
    const language = $("#browser-audio-language").value || "zh-CN";
    const test = window.TTA_PLAYBACK_TEST.selected(language);
    if (!state.sessionId) throw new Error("session_missing");
    const playback = await playStudyServerSpeech(test.text, { context: "quality_probe" });
    state.browserPlaybackProbe = { ...checks, ...playback, phrase_id: test.id };
    $("#browser-answer-row").classList.remove("hidden");
    $("#browser-audio-verify").classList.remove("hidden");
    window.TTA_PLAYBACK_TEST.syncVerifyButton(
      $("#browser-audio-answer"),
      $("#browser-audio-verify"),
    );
    $("#browser-audio-answer").focus();
    $("#browser-value").textContent = I18N.t("测试句已播放，请输入听到的内容");
  } catch (err) {
    const metrics = {
      ...checks,
      test_language: $("#browser-audio-language").value || "zh-CN",
      playback_api: "server_mp3",
      playback_error: err.message,
      typed_answer_stored: false,
      user_agent: navigator.userAgent,
      device_id: state.deviceId,
    };
    if (state.sessionId) await saveQuality("browser", false, metrics);
    else {
      state.quality.set("browser", false);
      updateStartEligibility();
    }
    setCheck("browser", false, I18N.t("测试句播放失败，请检查系统音量后重试"));
  } finally {
    button.disabled = false;
  }
});

$("#browser-audio-verify").addEventListener("click", async () => {
  if (!state.browserPlaybackProbe) {
    setBrowserCheckPending("请先播放测试句");
    return;
  }
  const button = $("#browser-audio-verify");
  button.disabled = true;
  const language = $("#browser-audio-language").value || "zh-CN";
  const evaluation = window.TTA_PLAYBACK_TEST.evaluate(language, $("#browser-audio-answer").value);
  const metrics = {
    ...state.browserPlaybackProbe,
    ...evaluation,
    test_language: language,
    verification_method: "typed_keyword_recall",
    playback_verified: evaluation.passed,
    typed_answer_stored: false,
    user_agent: navigator.userAgent,
    device_id: state.deviceId,
  };
  try {
    const saved = state.sessionId
      ? await saveQuality("browser", evaluation.passed, metrics)
      : { passed: evaluation.passed };
    if (!state.sessionId) {
      state.quality.set("browser", saved.passed === true);
      updateStartEligibility();
    }
    setCheck(
      "browser",
      saved.passed,
      I18N.t(saved.passed
        ? "听辨正确，浏览器与声音播放正常"
        : "内容未匹配，请重听后再输入"),
    );
    if (saved.passed) {
      $("#browser-audio-language").dataset.passedLanguage = language;
      $("#browser-audio-language").disabled = true;
    }
  } catch (err) {
    setCheck("browser", false, I18N.t("检测结果未能保存，请重试"));
  } finally {
    window.TTA_PLAYBACK_TEST.syncVerifyButton(
      $("#browser-audio-answer"),
      button,
    );
  }
});

$("#browser-audio-answer").addEventListener("input", () => {
  window.TTA_PLAYBACK_TEST.syncVerifyButton(
    $("#browser-audio-answer"),
    $("#browser-audio-verify"),
  );
});

$("#browser-audio-answer").addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  if (!$("#browser-audio-verify").disabled) $("#browser-audio-verify").click();
});

$("#browser-audio-language").addEventListener("change", async () => {
  const wasPassed = state.quality.get("browser") === true;
  if (wasPassed) {
    $("#browser-audio-language").value =
      $("#browser-audio-language").dataset.passedLanguage || "zh-CN";
    return;
  }
  resetBrowserPlaybackInput();
  state.quality.delete("browser");
  setBrowserCheckPending();
  updateStartEligibility();
});

const PAIR_DISCONNECT_GRACE_MS = 30000;
const PAIR_REQUIRED_GRACE_MS = 3000;
const COMPANION_READY_FALSE_DEBOUNCE_MS = 8000;
const SETUP_PAIR_BLIP_GRACE_MS = 12000;

function clearCompanionDisconnectGrace() {
  clearTimeout(state.companionDisconnectGraceTimer);
  state.companionDisconnectGraceTimer = null;
}

function clearCompanionReadyFalseDebounce() {
  clearTimeout(state.companionReadyFalseTimer);
  state.companionReadyFalseTimer = null;
}

function clearPairRequiredGrace() {
  clearTimeout(state.pairRequiredGraceTimer);
  state.pairRequiredGraceTimer = null;
}

function renderPairStatusLabel(pairReady, companionReady, message) {
  const bothRoles = Array.isArray(message?.roles)
    && message.roles.includes("study")
    && message.roles.includes("companion");
  const dayMismatch = bothRoles && message?.same_day_online !== true;
  $("#pair-dot").className = `status-dot ${pairReady ? "online" : "warn"}`;
  $("#pair-label").textContent = companionReady
    ? "同编号手机已连接，全部检测就绪"
    : pairReady
      ? "手机已连接，等待完成手机端检测"
      : dayMismatch
        ? "手机连接的不是当前实验会话"
        : "等待同编号手机连接";
}

function applyPairStatusMessage(message) {
  state.lastPairStatusMessage = message;
  state.lastPairStatusReceivedAtMs = performance.now();
  const nextPairReady = message.same_day_online === true && message.day === state.day;
  const nextCompanionReady = nextPairReady && message.companion_ready === true;
  const wasPairReady = state.pairReady;
  const wasCompanionReady = state.companionReady;

  if (nextPairReady && nextCompanionReady) {
    clearCompanionDisconnectGrace();
    clearPairRequiredGrace();
    clearCompanionReadyFalseDebounce();
    state.pairReady = true;
    state.companionReady = true;
  } else if ((wasPairReady || state.companionDisconnectGraceTimer) && !nextPairReady) {
    // Keep optimistic ready during grace so brief blips do not thrash steps
    // (setup used to bounce between 连接手机 / 练习 / 开始实验).
    if (!state.companionDisconnectGraceTimer) {
      const graceMs = state.scheduleRunning ? PAIR_DISCONNECT_GRACE_MS : SETUP_PAIR_BLIP_GRACE_MS;
      state.companionDisconnectGraceTimer = setTimeout(() => {
        state.companionDisconnectGraceTimer = null;
        const latest = state.lastPairStatusMessage || {};
        const stillPair = latest.same_day_online === true && latest.day === state.day;
        const stillReady = stillPair && latest.companion_ready === true;
        if (stillPair && stillReady) {
          state.pairReady = true;
          state.companionReady = true;
          renderPairStatusLabel(true, true, latest);
          if (!state.scheduleRunning) updateStartEligibility();
          return;
        }
        state.pairReady = stillPair;
        state.companionReady = stillReady;
        renderPairStatusLabel(state.pairReady, state.companionReady, latest);
        if (state.scheduleRunning && !stillPair) {
          pauseSchedule("companion_disconnected");
        } else if (!state.scheduleRunning) {
          updateStartEligibility();
        }
      }, graceMs);
    }
  } else if (
    !state.scheduleRunning
    && wasCompanionReady
    && nextPairReady
    && !nextCompanionReady
  ) {
    // Debounce ready→not-ready so setup steps do not flicker between 3 and 5.
    state.pairReady = true;
    state.companionReady = true;
    if (!state.companionReadyFalseTimer) {
      state.companionReadyFalseTimer = setTimeout(() => {
        state.companionReadyFalseTimer = null;
        const latest = state.lastPairStatusMessage || {};
        const stillPair = latest.same_day_online === true && latest.day === state.day;
        const stillReady = stillPair && latest.companion_ready === true;
        state.pairReady = stillPair;
        state.companionReady = stillReady;
        renderPairStatusLabel(state.pairReady, state.companionReady, latest);
        updateStartEligibility();
      }, COMPANION_READY_FALSE_DEBOUNCE_MS);
    }
  } else if (!state.companionDisconnectGraceTimer && !state.companionReadyFalseTimer) {
    clearCompanionReadyFalseDebounce();
    state.pairReady = nextPairReady;
    state.companionReady = nextCompanionReady;
  }

  if (state.scheduleRunning) {
    renderPairStatusLabel(state.pairReady, state.companionReady, message);
  } else {
    // Keep the visible pair copy aligned with the start gate (live status),
    // while sticky flags still damp step-rail flicker.
    const live = livePairSnapshotForStart();
    renderPairStatusLabel(live.pairReady, live.companionReady, message);
    updateStartEligibility();
  }
}

function handlePairRequiredFromServer() {
  // Server already applies a qualified-pair grace; this short client delay only
  // absorbs the race where the phone is mid-reconnect when the first
  // pair_required arrives.
  if (!state.scheduleRunning || state.scheduleAbort || state.interactionEnding) return;
  if (state.pairRequiredGraceTimer) return;
  state.pairRequiredGraceTimer = setTimeout(() => {
    state.pairRequiredGraceTimer = null;
    if (!state.scheduleRunning) return;
    const latest = state.lastPairStatusMessage || {};
    const recovered = latest.same_day_online === true
      && latest.day === state.day
      && latest.companion_ready === true;
    if (!recovered) pauseSchedule("pair_required_by_server");
  }, PAIR_REQUIRED_GRACE_MS);
}

function connectPairSocket() {
  if (!state.sessionId || !state.pairReconnectEnabled) return;
  if (state.pairSocket?.readyState === WebSocket.OPEN
    || state.pairSocket?.readyState === WebSocket.CONNECTING) return;
  clearTimeout(state.pairReconnectTimer);
  state.pairReconnectTimer = null;
  const socket = new WebSocket(wsUrl("/ws/pair", {
    token: state.token,
    session_id: state.sessionId,
  }));
  state.pairSocket = socket;
  socket.addEventListener("open", () => {
    state.pairReconnectAttempt = 0;
    const heartbeat = () => {
      pairSend({ type: "heartbeat", client_ms: Date.now(), role: "study" });
      // Re-announce companion window so a phone that missed the first notice
      // (or stuck on the primary overlay) still switches into talk UI.
      if (state.companionWindowActive) {
        pairSend({
          type: "notice",
          action: "companion_start",
          window: state.activeBlockIndex || 1,
          sync: true,
        });
      }
    };
    heartbeat();
    clearInterval(state.pairHeartbeat);
    state.pairHeartbeat = setInterval(heartbeat, 5000);
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "pair_status") {
      applyPairStatusMessage(message);
    }
    if (message.type === "background_state") {
      // Only pair_status.companion_ready is authoritative across the split
      // HTTP/realtime listeners; do not mark ready from a relay alone.
      queueEvent("companion.background_state", message);
    }
    if (message.type === "companion_state") {
      queueEvent("companion.chat_state", message);
      if (["listening_start", "utterance"].includes(message.action)) {
        noteParticipantSpeechActivity(
          message.action === "utterance" ? "companion_utterance" : "companion_listening_start",
          { transcript_chars: Number(message.transcript_chars || 0) },
        );
      }
      if (message.action === "utterance" || message.action === "tts_start") {
        state.companionAgentBusy = true;
        grantParticipantResponseWindow("companion_agent_busy");
      } else if (message.action === "tts_end" || message.action === "tts_failed") {
        state.companionAgentBusy = false;
        grantParticipantResponseWindow("companion_agent_finished");
      }
      if (message.action === "silence_invite") {
        markCompanionInviteFired("phone");
      } else if (message.action === "invite_reset" || message.action === "invite_activity") {
        if (state.companionWindowActive) armCompanionInviteUi(10000, "phone_activity");
      } else if (message.action === "window_end") {
        clearCompanionInviteUi();
      } else if (message.action === "phase_sync_request") {
        rebroadcastStudyPhaseToCompanion(message);
      }
      if (message.action === "tts_failed" && state.scheduleRunning) {
        pauseSchedule("companion_tts_server_failed");
      }
    }
    if (message.type === "phone_ptt") {
      handleCompanionPhonePtt(message);
    }
  });
  socket.addEventListener("close", () => {
    if (state.pairSocket !== socket) return;
    state.pairSocket = null;
    clearInterval(state.pairHeartbeat);
    state.pairHeartbeat = null;
    if (state.scheduleRunning) {
      // Desktop pair socket blips must not immediately mark steps / eject.
      applyPairStatusMessage({
        ...(state.lastPairStatusMessage || {}),
        same_day_online: false,
        companion_ready: false,
        roles: ["study"],
      });
    } else {
      state.pairReady = false;
      state.companionReady = false;
      $("#pair-dot").className = "status-dot warn";
      $("#pair-label").textContent = "手机连接已断开";
      updateStartEligibility();
    }
    schedulePairReconnect();
  });
}

function schedulePairReconnect() {
  if (!state.pairReconnectEnabled || !state.sessionId || state.pairReconnectTimer) return;
  const attempt = state.pairReconnectAttempt + 1;
  state.pairReconnectAttempt = attempt;
  const delayMs = Math.min(8000, 1000 * (2 ** Math.min(attempt - 1, 3)));
  $("#pair-label").textContent = `手机连接已断开，${Math.ceil(delayMs / 1000)} 秒后自动重连`;
  state.pairReconnectTimer = setTimeout(() => {
    state.pairReconnectTimer = null;
    if (!state.pairReconnectEnabled || !state.sessionId) return;
    $("#pair-label").textContent = "正在重新连接手机伴随端";
    connectPairSocket();
  }, delayMs);
}

function stopPairConnection() {
  disablePairReconnect();
  const socket = state.pairSocket;
  state.pairSocket = null;
  try { socket?.close(); } catch (_) { /* Already closed. */ }
}

function disablePairReconnect() {
  state.pairReconnectEnabled = false;
  state.pairReconnectAttempt = 0;
  clearTimeout(state.pairReconnectTimer);
  state.pairReconnectTimer = null;
  clearInterval(state.pairHeartbeat);
  state.pairHeartbeat = null;
}

function pairSend(message) {
  if (state.pairSocket?.readyState === WebSocket.OPEN) {
    state.pairSocket.send(JSON.stringify(message));
  }
}

function handleCompanionPhonePtt(message) {
  if (message.action === "start") {
    noteParticipantSpeechActivity("companion_ptt_start");
    startCompanionRelayAsr();
  }
  else if (message.action === "stop") stopCompanionRelayAsr();
}

function interactionPresenceStageActive() {
  return Boolean(
    state.scheduleRunning
    && !state.interactionEnding
    && !state.scheduleAbort
    && !state.schedulePaused
    && !state.scheduledRest
    && !state.phaseTransitionInFlight
    && state.currentPhase
    && !["rest", "complete"].includes(state.currentPhase),
  );
}

function interactionPresenceAgentBusy() {
  if (state.companionWindowActive) {
    // In the phone-companion window, only the companion's answer is a valid
    // listening interval. The computer agent speaking unexpectedly must not
    // hide participant inactivity; a successful desktop interrupt is tracked
    // separately as an explicit participant action.
    return state.companionAgentBusy;
  }
  return state.agentState === "thinking" || state.agentState === "speaking";
}

function interactionPresenceCopy(mode, seconds = null) {
  const companion = state.companionWindowActive;
  if (mode === "warning") {
    return {
      title: "互动即将超时",
      copy: I18N.isEnglish
        ? (companion
          ? `Within ${seconds} seconds, hold the phone button to keep talking. If the computer agent is responding at the wrong time, press Stop answering on the computer. Otherwise this session will exit.`
          : `Speak with the computer experimental system within ${seconds} seconds, or this session will exit.`)
        : (companion
          ? `请在 ${seconds} 秒内长按手机圆形按钮继续对话；若电脑实验系统正在不合时机地回答，可在电脑点击「停止回答」。否则本次实验将退出。`
          : `请在 ${seconds} 秒内继续与电脑实验系统说话，否则本次实验将退出。`),
    };
  }
  return {
    title: "请保持连续语音对话",
    copy: companion
      ? "请留在当前位置，持续与手机陪伴智能体自然对话；倾听它回答时无需说话。"
      : "请留在当前位置，持续与电脑实验系统自然对话；倾听它回答时无需说话。",
  };
}

function renderInteractionPresence(mode = "monitoring", seconds = null, { force = false } = {}) {
  const strip = $("#interaction-presence-strip");
  if (!strip) return;
  const visible = interactionPresenceStageActive();
  if (!visible) {
    strip.classList.add("hidden");
    state.interactionPresenceRenderState = "hidden";
    state.interactionPresenceRenderSecond = null;
    state.interactionPresenceRenderTarget = null;
    return;
  }
  const normalizedSecond = mode === "warning" ? Math.max(0, Number(seconds) || 0) : null;
  const normalizedTarget = state.companionWindowActive ? "companion" : "primary";
  if (
    !force
    && state.interactionPresenceRenderState === mode
    && state.interactionPresenceRenderSecond === normalizedSecond
    && state.interactionPresenceRenderTarget === normalizedTarget
  ) return;
  const copy = interactionPresenceCopy(mode, normalizedSecond);
  strip.dataset.state = mode;
  strip.setAttribute("role", mode === "warning" ? "alert" : "status");
  strip.setAttribute("aria-live", mode === "warning" ? "assertive" : "polite");
  $("#interaction-presence-title").textContent = I18N.t(copy.title);
  $("#interaction-presence-copy").textContent = I18N.t(copy.copy);
  const countdown = $("#interaction-presence-countdown");
  if (countdown) {
    countdown.textContent = String(normalizedSecond ?? "");
    countdown.classList.toggle("hidden", mode !== "warning");
    countdown.setAttribute("aria-hidden", String(mode !== "warning"));
  }
  strip.classList.remove("hidden");
  state.interactionPresenceRenderState = mode;
  state.interactionPresenceRenderSecond = normalizedSecond;
  state.interactionPresenceRenderTarget = normalizedTarget;
  pairSend({
    type: "notice",
    action: "interaction_presence",
    state: mode,
    seconds: normalizedSecond,
    target: normalizedTarget,
  });
}

function grantParticipantResponseWindow(reason = "stage_change") {
  if (!state.scheduleRunning) return;
  const wasWarning = state.interactionPresenceCountdownStartedMs > 0;
  state.participantSpeechLastMs = performance.now();
  state.interactionPresenceCountdownStartedMs = 0;
  if (wasWarning) {
    queueEvent("session.interaction_presence_recovered", {
      source: reason,
      by_participant_speech: false,
    });
  }
  renderInteractionPresence("monitoring", null, { force: wasWarning });
}

function noteParticipantSpeechActivity(source = "asr", metadata = {}) {
  if (!state.scheduleRunning || state.scheduledRest || state.interactionEnding) return;
  const wasWarning = state.interactionPresenceCountdownStartedMs > 0;
  state.participantSpeechLastMs = performance.now();
  state.interactionPresenceCountdownStartedMs = 0;
  if (wasWarning) {
    queueEvent("session.interaction_presence_recovered", {
      source,
      by_participant_speech: true,
      ...metadata,
    });
  }
  renderInteractionPresence("monitoring", null, { force: wasWarning });
}

function noteCompanionInterruptActivity(message = {}) {
  if (!state.companionWindowActive || !state.scheduleRunning || state.interactionEnding) return;
  const wasWarning = state.interactionPresenceCountdownStartedMs > 0;
  state.participantSpeechLastMs = performance.now();
  state.interactionPresenceCountdownStartedMs = 0;
  queueEvent(
    wasWarning
      ? "session.interaction_presence_recovered"
      : "session.interaction_presence_activity",
    {
      source: "companion_computer_interrupt",
      by_participant_speech: false,
      by_successful_interrupt: true,
      tap_id: message.tap_id || null,
    },
  );
  renderInteractionPresence("monitoring", null, { force: wasWarning });
}

function checkInteractionPresence() {
  if (!interactionPresenceStageActive()) {
    renderInteractionPresence("monitoring");
    return;
  }
  if (interactionPresenceAgentBusy()) {
    grantParticipantResponseWindow("agent_busy");
    return;
  }
  const now = performance.now();
  if (!state.participantSpeechLastMs) state.participantSpeechLastMs = now;
  const silentMs = now - state.participantSpeechLastMs;
  if (silentMs < INTERACTION_SPEECH_GRACE_MS) {
    renderInteractionPresence("monitoring");
    return;
  }
  if (!state.interactionPresenceCountdownStartedMs) {
    state.interactionPresenceCountdownStartedMs = now;
    const companion = state.companionWindowActive;
    queueEvent("session.interaction_presence_countdown_started", {
      silence_grace_ms: INTERACTION_SPEECH_GRACE_MS,
      countdown_ms: INTERACTION_RECOVERY_COUNTDOWN_MS,
      target: companion ? "companion" : "primary",
      activity_sources: companion
        ? ["companion_ptt_start", "companion_utterance", "tap.interrupt"]
        : ["speech_start", "asr_interim", "asr_final"],
      manual_tap_activity_policy: companion ? "successful_interrupt_only" : "excluded",
    });
  }
  const elapsed = now - state.interactionPresenceCountdownStartedMs;
  const remainingSeconds = Math.max(
    0,
    Math.ceil((INTERACTION_RECOVERY_COUNTDOWN_MS - elapsed) / 1000),
  );
  renderInteractionPresence("warning", remainingSeconds);
  if (elapsed < INTERACTION_RECOVERY_COUNTDOWN_MS) return;
  queueEvent("session.interaction_presence_expired", {
    silence_grace_ms: INTERACTION_SPEECH_GRACE_MS,
    countdown_ms: INTERACTION_RECOVERY_COUNTDOWN_MS,
    target: state.companionWindowActive ? "companion" : "primary",
    manual_tap_activity_policy: state.companionWindowActive
      ? "successful_interrupt_only"
      : "excluded",
  });
  pauseSchedule("participant_inactive");
}

function startInteractionPresenceMonitor() {
  clearInterval(state.interactionPresenceTimer);
  state.participantSpeechLastMs = performance.now();
  state.interactionPresenceCountdownStartedMs = 0;
  state.interactionPresenceRenderState = "hidden";
  state.interactionPresenceRenderSecond = null;
  state.interactionPresenceRenderTarget = null;
  state.interactionPresenceTimer = setInterval(checkInteractionPresence, 250);
}

function stopInteractionPresenceMonitor({ notifyPhone = true } = {}) {
  clearInterval(state.interactionPresenceTimer);
  state.interactionPresenceTimer = null;
  state.interactionPresenceCountdownStartedMs = 0;
  state.participantSpeechLastMs = 0;
  $("#interaction-presence-strip")?.classList.add("hidden");
  state.interactionPresenceRenderState = "hidden";
  state.interactionPresenceRenderSecond = null;
  state.interactionPresenceRenderTarget = null;
  if (notifyPhone) {
    pairSend({ type: "notice", action: "interaction_presence", state: "hidden" });
  }
}

function rebroadcastStudyPhaseToCompanion(message = {}) {
  if (!state.scheduleRunning) return;
  // Keep phone BGM/timeline aligned after reconnects or missed notices.
  pairSend({
    type: "background_state",
    action: "start",
    day: state.day,
    offset_seconds: 0,
    sync: true,
    rest_held: state.scheduledRest,
  });
  if (interactionPresenceStageActive()) {
    pairSend({
      type: "notice",
      action: "interaction_presence",
      state: state.interactionPresenceRenderState === "warning" ? "warning" : "monitoring",
      seconds: state.interactionPresenceRenderSecond,
      target: state.companionWindowActive ? "companion" : "primary",
      sync: true,
    });
  }
  if (state.scheduledRest) {
    pairSend({
      type: "notice",
      action: "rest_start",
      block: state.activeBlockIndex || Number(message.block) || 1,
      seconds: Math.max(1, Math.ceil(Number(state.restRemainingSeconds) || 60)),
      sync: true,
    });
    return;
  }
  if (state.companionWindowActive) {
    pairSend({
      type: "notice",
      action: "companion_start",
      window: state.activeBlockIndex || Number(message.window) || 1,
      sync: true,
    });
    return;
  }
  const primaryVisible = !$("#primary-attention-banner")?.classList.contains("hidden");
  if (primaryVisible) {
    pairSend({
      type: "notice",
      action: "primary_start",
      block: state.activeBlockIndex || Number(message.window) || 1,
      sync: true,
    });
  }
}

function startCompanionRelayAsr() {
  clearTimeout(state.companionRelayFinalizeTimer);
  clearTimeout(state.companionRelayWatchdog);
  state.companionRelayStopToken += 1;
  state.companionRelayGeneration += 1;
  state.companionRelayParts = [];
  state.companionRelayInterim = "";
  state.companionRelayPcmChunks = [];
  state.companionRelayPcmSamples = 0;
  state.companionRelayFinalPromise = null;
  state.companionRelayActive = true;
  state.companionRelayResumeRecognition = state.recognitionWanted;
  state.companionRelayRecognition = null;
  if (state.recognition) {
    state.recognitionWanted = false;
    const activeRecognition = state.recognition;
    state.recognition = null;
    try { activeRecognition.stop(); } catch (_) { /* Browser ASR is no longer used. */ }
  }
  pairSend({ type: "desktop_asr", action: "listening" });
  queueEvent("companion.desktop_asr_relay", {
    action: "start",
    browser_recognition_available: false,
    server_fallback_available: true,
    asr_pipeline: "server_whisper_only",
  });
  // Safety: never leave companionRelayActive stuck (blocks auto-activate).
  state.companionRelayWatchdog = setTimeout(() => {
    if (!state.companionRelayActive) return;
    queueEvent("companion.desktop_asr_relay", { action: "watchdog_clear" });
    void stopCompanionRelayAsr();
  }, 20000);
}

function stopCompanionRelayAsr() {
  if (state.companionRelayFinalPromise) return state.companionRelayFinalPromise;
  if (!state.companionRelayActive && !state.companionRelayRecognition) {
    return Promise.resolve("");
  }
  state.companionRelayActive = false;
  clearTimeout(state.companionRelayWatchdog);
  state.companionRelayWatchdog = null;
  const stopToken = state.companionRelayStopToken + 1;
  state.companionRelayStopToken = stopToken;
  const recognition = state.companionRelayRecognition;
  let finalized = false;
  let resolveFinal;
  const finalPromise = new Promise((resolve) => { resolveFinal = resolve; });
  state.companionRelayFinalPromise = finalPromise;

  const emitFinal = async () => {
    if (finalized || stopToken !== state.companionRelayStopToken) return;
    finalized = true;
    clearTimeout(state.companionRelayFinalizeTimer);
    state.companionRelayFinalizeTimer = null;
    state.companionRelayRecognition = null;
    state.companionRelayGeneration += 1;
    let transcript = "";
    let source = "server_whisper";
    let serverResult = null;
    let serverError = "";
    try {
      serverResult = await transcribePcmChunks(
        state.companionRelayPcmChunks,
        "companion",
      );
      transcript = String(serverResult.text || "").trim();
    } catch (error) {
      serverError = String(error.message || error);
    }
    pairSend({ type: "desktop_asr", action: "final", transcript });
    queueEvent("companion.desktop_asr_relay", {
      action: "final",
      transcript_chars: transcript.length,
      asr_source: source,
      asr_model: serverResult?.model || null,
      asr_latency_ms: serverResult?.latency_ms || null,
      asr_error: serverError || null,
    });
    state.companionRelayParts = [];
    state.companionRelayInterim = "";
    state.companionRelayPcmChunks = [];
    state.companionRelayPcmSamples = 0;
    // Always restore study ASR after phone PTT so chat text stays available.
    // Response timing itself does not depend on ASR.
    if (
      state.scheduleRunning
      && !state.schedulePaused
      && !state.scheduledRest
      && !state.asrPausedForAgentSpeech
      && !state.demo
    ) {
      state.recognitionWanted = true;
      startRecognition();
    }
    state.companionRelayResumeRecognition = false;
    state.companionRelayFinalPromise = null;
    resolveFinal(transcript);
  };

  clearTimeout(state.companionRelayFinalizeTimer);
  state.companionRelayFinalizeTimer = setTimeout(() => { void emitFinal(); }, 1500);
  if (!recognition) {
    void emitFinal();
    return finalPromise;
  }
  recognition.onend = () => {
    if (stopToken !== state.companionRelayStopToken) return;
    void emitFinal();
  };
  try {
    recognition.stop();
  } catch (_) {
    void emitFinal();
  }
  return finalPromise;
}

function updateStartEligibility() {
  const blockers = startEligibilityBlockers();
  const passed = REQUIRED_QUALITY.every((key) => state.quality.get(key) === true);
  const startButton = $("#start-study-btn");
  const busy = state.startInFlight || state.scheduleRunning;
  startButton.disabled = busy;
  startButton.setAttribute("aria-disabled", String(busy || blockers.length > 0));
  startButton.classList.toggle("blocked-action", !busy && blockers.length > 0);
  startButton.textContent = busy
    ? I18N.t("正在启动实验")
    : I18N.t(
      blockers.length > 0
        ? "开始完整实验"
        : (state.trainingPassed ? "练习通过，开始完整实验" : "开始点击练习"),
    );
  renderStartEligibility(blockers);
  const phoneBlocked = blockers.some(({ code }) => (
    code === "pair_not_connected" || code === "companion_not_ready"
  ));
  if (!phoneBlocked) {
    clearTimeout(state.companionAttentionTimer);
    state.companionAttentionTimer = null;
    $("#companion-link-instruction").classList.remove("attention-highlight");
  }
  // Never thrash setup step highlighting while the experiment is running.
  if (state.scheduleRunning) return;
  // Step rail may use sticky flags to avoid flicker; the start button does not.
  if (!state.sessionId) markStep("overview");
  else if (!passed) markStep("network");
  else if (!state.pairReady || !state.companionReady) markStep("phone");
  else if (!state.trainingPassed) markStep("training");
  else markStep("ready");
  syncInterruptChoicePanel();
}

function startEligibilityBlockers() {
  const blockers = [];
  if (!protocolUsesSingleSession()) {
    blockers.push({ code: "protocol_obsolete", label: "该编号绑定旧版多日流程，请重新生成当前版本编号" });
  }
  if (!state.sessionId) blockers.push({ code: "session_missing", label: "本次实验尚未创建" });
  for (const kind of REQUIRED_QUALITY) {
    const result = state.quality.get(kind);
    if (result !== true) {
      blockers.push({
        code: `quality_${kind}_${result === false ? "failed" : "pending"}`,
        label: `${QUALITY_GATE_LABELS[kind]}${result === false ? "未通过" : "尚未完成"}`,
      });
    }
  }
  // Start gate must require a live phone companion — never sticky/grace alone.
  const live = livePairSnapshotForStart();
  if (!live.pairReady) blockers.push({ code: "pair_not_connected", label: "同编号的手机伴随端尚未连接" });
  else if (!live.companionReady) blockers.push({ code: "companion_not_ready", label: "手机端检测尚未全部通过并完成连接" });
  if (state.capacity && state.capacity.ok === false && !state.capacity.holding_slot) {
    const minutes = Number(state.capacity.retry_after_minutes) || 30;
    blockers.push({
      code: "study_capacity_full",
      label: `当前同时使用人数较多，请约 ${minutes} 分钟后再来尝试`,
    });
  }
  return blockers;
}

function renderStartEligibility(blockers) {
  const status = $("#start-gate-status");
  const list = $("#start-gate-blockers");
  const ready = blockers.length === 0;
  status.dataset.ready = String(ready);
  status.dataset.blockers = blockers.map((blocker) => blocker.code).join(",");
  $("#start-study-btn").dataset.blockers = status.dataset.blockers;
  list.replaceChildren();
  for (const blocker of blockers) {
    const item = document.createElement("li");
    item.dataset.blockerCode = blocker.code;
    item.textContent = blocker.label;
    list.appendChild(item);
  }
  $("#start-gate-summary").textContent = ready
    ? (
      state.startInFlight
        ? "正在启动实验"
        : (
          state.trainingPassed
            ? "所有开始条件均已满足"
            : "设备与手机条件已满足；请先完成点击练习。"
        )
    )
    : `还需完成 ${blockers.length} 项条件：`;
  status.classList.toggle("ready", ready);
  status.dataset.trainingPassed = String(state.trainingPassed);
}

const TAP_TRAINING_SCENARIOS = [
  {
    id: "initial_wait_for_auto_response",
    agentState: "waiting",
    expected: "wait",
    titleZh: "刚说完，先等待 1～3 秒",
    titleEn: "You have just finished; wait 1–3 seconds",
    copyZh: "你刚说完问题并希望实验智能体回答；它仍显示“等待”，还没有经过建议的 1～3 秒等待时间。",
    copyEn: "You have just finished a question and want the experimental agent to answer. It still says “Waiting,” and the suggested 1–3-second wait has not yet passed.",
    correctZh: "正确：先不点击，等待 1～3 秒，让实验智能体有机会自行回答；按钮不需要每轮都点。",
    correctEn: "Correct: do not tap yet. Wait 1–3 seconds to give the experimental agent a chance to answer on its own; the button is not required every turn.",
    wrongZh: "不对：说完后先等待 1～3 秒；如果仍未开始且你希望它现在回答，再点击“现在回答”。",
    wrongEn: "Not quite: after you finish, wait 1–3 seconds. If it still has not started and you want an answer now, tap Answer now.",
  },
  {
    id: "manual_activate_after_wait",
    agentState: "waiting",
    expected: "tap",
    titleZh: "等待 1～3 秒，它仍未回答",
    titleEn: "After waiting 1–3 seconds, it has not answered",
    copyZh: "你已经说完问题并希望实验智能体回答；等待 1～3 秒后，它仍显示“等待”。",
    copyEn: "You finished a question and want the experimental agent to answer. After waiting 1–3 seconds, it still says “Waiting.”",
    correctZh: "正确：等待 1～3 秒仍未开始，而你希望它现在回答；此时点击一次“现在回答”。",
    correctEn: "Correct: 1–3 seconds have passed and it has not started, while you want an answer now. Tap Answer now once.",
    wrongZh: "不对：你已经等待了 1～3 秒，且希望它现在回答；仍无响应时应点击一次“现在回答”。",
    wrongEn: "Not quite: you have waited 1–3 seconds and want an answer now. If it still has not responded, tap Answer now once.",
  },
  {
    id: "retry_after_stale_wait",
    agentState: "waiting",
    expected: "retry",
    titleZh: "已经超过约 5 秒，仍未回答",
    titleEn: "More than about 5 seconds have passed with no answer",
    copyZh: "你说完后已经等待超过约 5 秒，实验智能体仍显示“等待”。",
    copyEn: "More than about 5 seconds have passed since you finished, and the experimental agent still says “Waiting.”",
    waitLabelZh: "这时不点击，并继续等待",
    waitLabelEn: "Do not tap and keep waiting",
    retryLabelZh: "这时不点击，然后我再重新说一遍",
    retryLabelEn: "Do not tap; I will say it again",
    correctZh: "正确：请重新说一遍，不要再为上一句话点击“现在回答”。",
    correctEn: "Correct: say it again. Do not tap Answer now for the previous utterance.",
    wrongZh: "不对：超过约 5 秒仍未开始回答时，应重新说一遍；不要再为上一句话点击“现在回答”。",
    wrongEn: "Not quite: if more than about 5 seconds pass with no answer, say it again. Do not tap Answer now for the previous utterance.",
  },
  {
    id: "appropriate_response",
    agentState: "speaking",
    expected: "wait",
    titleZh: "它正在合适地回答",
    titleEn: "It is answering appropriately",
    copyZh: "你刚问完问题，实验智能体按时开始回答，而且你希望听完。",
    copyEn: "You just finished a question. The experimental agent started on time, and you want to hear the answer.",
    correctZh: "正确：响应时机合适，而且你希望它继续；不要点击。",
    correctEn: "Correct: the timing is appropriate and you want it to continue. Do not tap.",
    wrongZh: "不对：点击会把一个合适的回答错误地标成需要“停止回答”；此时不要点击。",
    wrongEn: "Not quite: tapping would incorrectly mark an appropriate answer as needing Stop answering. Do not tap.",
  },
  {
    id: "premature_interruption",
    agentState: "thinking",
    expected: "tap",
    titleZh: "它过早准备回答",
    titleEn: "It prepares an answer too early",
    copyZh: "你还没开口，或话还没有说完，实验智能体就显示“思考中”。",
    copyEn: "You have not started speaking, or have not finished speaking, but the experimental agent changes to “Thinking.”",
    correctZh: "正确：你还没开口或没说完时，它不该开始准备回答；点击“停止回答”。",
    correctEn: "Correct: it should not prepare an answer before you start or finish speaking. Press Stop answering.",
    wrongZh: "不对：你还没开口或尚未说完时，智能体开始准备回答，应点击“停止回答”。",
    wrongEn: "Not quite: if the agent prepares an answer before you begin or finish speaking, press Stop answering.",
  },
  {
    id: "companion_intrusion",
    agentState: "speaking",
    expected: "tap",
    titleZh: "它闯入手机对话",
    titleEn: "It intrudes on the phone conversation",
    copyZh: "页面正处于两分钟手机陪伴智能体时段；电脑实验智能体突然开始回答。",
    copyEn: "The page is in the two-minute phone-companion period, but the computer experimental agent suddenly starts answering.",
    correctZh: "正确：手机对话时段内，电脑实验智能体的任何介入都应点击“停止回答”。",
    correctEn: "Correct: during the phone-companion period, tap Stop answering for any intervention from the computer experimental agent.",
    wrongZh: "不对：你现在的对话对象是手机陪伴智能体；电脑实验智能体介入时应点击“停止回答”。",
    wrongEn: "Not quite: you are speaking with the phone companion. Tap Stop answering on the computer experimental agent.",
  },
  {
    id: "no_reply_needed",
    agentState: "waiting",
    expected: "wait",
    titleZh: "这句话不需要回答",
    titleEn: "This utterance needs no answer",
    copyZh: "你对实验智能体说“好吧，就这样”，并不希望它继续回应；它保持等待。",
    copyEn: "You tell the experimental agent, “Okay, that is all,” and do not want another response. It remains waiting.",
    correctZh: "正确：你不需要回应，而且系统没有错误介入；不要点击。",
    correctEn: "Correct: you do not want another response and the system did not intervene incorrectly. Do not tap.",
    wrongZh: "不对：只有“希望它回答但没有回答”时才在等待状态点击；这里不要点击。",
    wrongEn: "Not quite: tap while waiting only when you want an answer and none starts. Do not tap here.",
  },
];

function trainingStorageKey() {
  return storageKey(`tap_training_passed_v6:${state.sessionId || state.code || "unknown"}`);
}

function restoreTapTrainingCompletion() {
  state.trainingPassed = sessionStorage.getItem(trainingStorageKey()) === "1";
}

function localizedTraining(scenario, field) {
  return scenario[`${field}${I18N.isEnglish ? "En" : "Zh"}`];
}

function labeledTrainingChoice(letter, text) {
  return `${letter}. ${text}`;
}

const TAP_TRAINING_REVIEW_MS = 5000;

function clearTrainingReviewTimer() {
  clearTimeout(state.trainingReviewTimer);
  state.trainingReviewTimer = null;
  state.trainingReviewDeadlineMs = null;
}

function unlockTrainingAnswers() {
  clearTrainingReviewTimer();
  state.trainingReviewUnlocked = true;
  $("#training-tap-btn").disabled = false;
  $("#training-wait-btn").disabled = false;
  $("#training-retry-btn").disabled = false;
  const feedback = $("#training-feedback");
  feedback.className = `training-feedback ${state.trainingReviewFeedbackPrefix ? "incorrect ready" : "ready"}`;
  feedback.textContent = state.trainingReviewFeedbackPrefix
    ? `${state.trainingReviewFeedbackPrefix} ${I18N.t("现在可以重新作答。")}`
    : I18N.t("现在可以作答");
}

function updateTrainingReviewCountdown() {
  const remainingSeconds = Math.max(
    0,
    Math.ceil((Number(state.trainingReviewDeadlineMs) - performance.now()) / 1000),
  );
  if (remainingSeconds <= 0) {
    unlockTrainingAnswers();
    return;
  }
  const feedback = $("#training-feedback");
  feedback.className = `training-feedback reviewing${state.trainingReviewFeedbackPrefix ? " incorrect" : ""}`;
  const countdownCopy = I18N.t(
    state.trainingReviewFeedbackPrefix
      ? "请阅读解释，{n} 秒后可重新选择。"
      : "请先审题，{n} 秒后可作答",
  ).replace("{n}", String(remainingSeconds));
  feedback.textContent = state.trainingReviewFeedbackPrefix
    ? `${state.trainingReviewFeedbackPrefix} ${countdownCopy}`
    : countdownCopy;
  state.trainingReviewTimer = setTimeout(updateTrainingReviewCountdown, 200);
}

function startTrainingReviewGate(feedbackPrefix = "") {
  clearTrainingReviewTimer();
  state.trainingReviewFeedbackPrefix = String(feedbackPrefix || "").trim();
  state.trainingReviewUnlocked = false;
  state.trainingReviewDeadlineMs = performance.now() + TAP_TRAINING_REVIEW_MS;
  $("#training-tap-btn").disabled = true;
  $("#training-wait-btn").disabled = true;
  $("#training-retry-btn").disabled = true;
  updateTrainingReviewCountdown();
}

function openTapTraining() {
  clearTimeout(state.trainingAdvanceTimer);
  state.trainingAdvanceTimer = null;
  clearTrainingReviewTimer();
  state.trainingReviewUnlocked = false;
  $("#training-intro").classList.remove("hidden");
  $("#training-trial").classList.add("hidden");
  $("#training-complete").classList.add("hidden");
  $("#tap-training-overlay").classList.remove("hidden");
  $("#training-feedback").textContent = "";
  sendEvent("training.opened", {
    training: true,
    excluded_from_analysis: true,
    scenario_count: TAP_TRAINING_SCENARIOS.length,
  }).catch(() => { /* Training remains usable if event logging is delayed. */ });
  requestAnimationFrame(() => $("#training-begin-btn").focus());
}

function closeTapTraining() {
  clearTimeout(state.trainingAdvanceTimer);
  state.trainingAdvanceTimer = null;
  clearTrainingReviewTimer();
  state.trainingReviewUnlocked = false;
  $("#tap-training-overlay").classList.add("hidden");
  updateStartEligibility();
  $("#start-study-btn").scrollIntoView({ behavior: "smooth", block: "center" });
  requestAnimationFrame(() => $("#start-study-btn").focus());
}

function renderTapTrainingScenario() {
  const scenario = TAP_TRAINING_SCENARIOS[state.trainingIndex];
  $("#training-progress-label").textContent =
    `${I18N.t("练习")} ${state.trainingIndex + 1}/${TAP_TRAINING_SCENARIOS.length}`;
  const agentState = $("#training-agent-state");
  const display = $("#training-scenario");
  const trainingStates = {
    waiting: { label: "等待", action: "点击“现在回答”按钮", image: staticUrl("static/emotions/wait.png") },
    thinking: { label: "思考中", action: "点击“停止回答”按钮", image: staticUrl("static/emotions/think.png") },
    speaking: { label: "正在回答", action: "点击“停止回答”按钮", image: staticUrl("static/emotions/speak.png") },
  };
  const presentation = trainingStates[scenario.agentState];
  display.dataset.state = scenario.agentState;
  agentState.dataset.state = scenario.agentState;
  agentState.textContent = I18N.t(presentation.label);
  $("#training-agent-emotion").src = presentation.image;
  $("#training-scenario-title").textContent = localizedTraining(scenario, "title");
  $("#training-scenario-copy").textContent = localizedTraining(scenario, "copy");
  $("#training-tap-btn").textContent = labeledTrainingChoice(
    "A",
    I18N.t(presentation.action),
  );
  $("#training-tap-btn").dataset.action = scenario.agentState === "waiting"
    ? "manual_activate"
    : "interrupt";
  const retryButton = $("#training-retry-btn");
  const hasRetryChoice = Boolean(scenario.retryLabelZh && scenario.retryLabelEn);
  $("#training-wait-btn").textContent = labeledTrainingChoice(
    "B",
    hasRetryChoice
      ? localizedTraining(scenario, "waitLabel")
      : I18N.t("这时不点击"),
  );
  retryButton.classList.toggle("hidden", !hasRetryChoice);
  retryButton.textContent = labeledTrainingChoice(
    "C",
    hasRetryChoice
      ? localizedTraining(scenario, "retryLabel")
      : I18N.t("这时不点击，然后我再重新说一遍"),
  );
  startTrainingReviewGate();
}

function beginTapTraining() {
  state.trainingIndex = 0;
  state.trainingAttempts = 0;
  state.trainingErrors = 0;
  state.trainingStartedMs = performance.now();
  $("#training-intro").classList.add("hidden");
  $("#training-complete").classList.add("hidden");
  $("#training-trial").classList.remove("hidden");
  sendEvent("training.started", {
    training: true,
    excluded_from_analysis: true,
    scenario_count: TAP_TRAINING_SCENARIOS.length,
  }).catch(() => { /* Training remains usable if event logging is delayed. */ });
  renderTapTrainingScenario();
}

function completeTapTraining() {
  clearTrainingReviewTimer();
  state.trainingReviewUnlocked = false;
  state.trainingPassed = true;
  sessionStorage.setItem(trainingStorageKey(), "1");
  $("#training-trial").classList.add("hidden");
  $("#training-complete").classList.remove("hidden");
  sendEvent("training.completed", {
    training: true,
    excluded_from_analysis: true,
    scenario_count: TAP_TRAINING_SCENARIOS.length,
    attempts: state.trainingAttempts,
    errors: state.trainingErrors,
    duration_ms: Math.round(performance.now() - state.trainingStartedMs),
  }).catch(() => { /* Completion remains valid locally if upload is delayed. */ });
  updateStartEligibility();
  requestAnimationFrame(() => $("#training-finish-btn").focus());
}

function answerTapTraining(choice) {
  if (!state.trainingReviewUnlocked) return;
  const scenario = TAP_TRAINING_SCENARIOS[state.trainingIndex];
  const correct = choice === scenario.expected;
  state.trainingAttempts += 1;
  if (!correct) state.trainingErrors += 1;
  sendEvent("training.answer", {
    training: true,
    excluded_from_analysis: true,
    scenario_id: scenario.id,
    scenario_index: state.trainingIndex,
    expected_action: scenario.expected,
    chosen_action: choice,
    correct,
    attempt: state.trainingAttempts,
  }).catch(() => { /* Training remains usable if event logging is delayed. */ });
  const feedback = $("#training-feedback");
  feedback.className = `training-feedback ${correct ? "correct" : "incorrect"}`;
  feedback.textContent = localizedTraining(scenario, correct ? "correct" : "wrong");
  if (!correct) {
    startTrainingReviewGate(localizedTraining(scenario, "wrong"));
    return;
  }

  state.trainingReviewUnlocked = false;
  $("#training-tap-btn").disabled = true;
  $("#training-wait-btn").disabled = true;
  $("#training-retry-btn").disabled = true;
  state.trainingAdvanceTimer = setTimeout(() => {
    state.trainingIndex += 1;
    if (state.trainingIndex >= TAP_TRAINING_SCENARIOS.length) completeTapTraining();
    else renderTapTrainingScenario();
  }, 950);
}

$("#training-begin-btn").addEventListener("click", beginTapTraining);
$("#training-tap-btn").addEventListener("click", () => answerTapTraining("tap"));
$("#training-wait-btn").addEventListener("click", () => answerTapTraining("wait"));
$("#training-retry-btn").addEventListener("click", () => answerTapTraining("retry"));
$("#training-close-btn").addEventListener("click", closeTapTraining);
$("#training-finish-btn").addEventListener("click", closeTapTraining);

function emphasizeCompanionLink() {
  const instruction = $("#companion-link-instruction");
  clearTimeout(state.companionAttentionTimer);
  instruction.classList.remove("attention-highlight");
  void instruction.offsetWidth;
  instruction.classList.add("attention-highlight");
  instruction.scrollIntoView({ behavior: "smooth", block: "center" });
  requestAnimationFrame(() => instruction.focus({ preventScroll: true }));
  $("#start-error").textContent = I18N.t("请先在手机上打开上方专属链接，并完成手机端检测。");
  state.companionAttentionTimer = setTimeout(() => {
    instruction.classList.remove("attention-highlight");
    state.companionAttentionTimer = null;
  }, 7000);
}

function showStartFailureDialog({ title, message, blockers = [], hint = "" }) {
  const overlay = $("#start-failure-overlay");
  const list = $("#start-failure-blockers");
  $("#start-failure-title").textContent = I18N.t(title || "无法启动实验");
  $("#start-failure-message").textContent = I18N.t(message || "");
  $("#start-failure-hint").textContent = hint ? I18N.t(hint) : "";
  list.replaceChildren();
  for (const blocker of blockers) {
    const item = document.createElement("li");
    item.textContent = I18N.t(blocker.label || blocker);
    list.appendChild(item);
  }
  $("#start-error").textContent = I18N.t(message || title || "无法启动实验");
  overlay.classList.remove("hidden");
  $("#start-failure-ok").focus();
}

function hideStartFailureDialog() {
  $("#start-failure-overlay")?.classList.add("hidden");
}

$("#start-failure-ok")?.addEventListener("click", () => hideStartFailureDialog());

function explainBlockedStart(blockers) {
  const phoneBlocked = blockers.some(({ code }) => (
    code === "pair_not_connected" || code === "companion_not_ready"
  ));
  if (phoneBlocked) {
    emphasizeCompanionLink();
    showStartFailureDialog({
      title: "手机端尚未就绪",
      message: "输入同一编号还不够：手机必须打开上方专属链接，完成全部检测，并点击启用声音后才会与电脑配对连接。",
      blockers,
      hint: "请用手机打开专属陪伴链接（微信内打开也可以）；看到手机页显示“已连接/就绪”后再点开始。",
    });
    return;
  }
  const status = $("#start-gate-status");
  status.scrollIntoView({ behavior: "smooth", block: "center" });
  showStartFailureDialog({
    title: "还不能开始实验",
    message: "请先完成下列仍未通过的条件，然后再点击开始。",
    blockers,
  });
}

const START_CONFIRM_COUNTDOWN_SEC = 5;

function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

function isExperimentFullscreen() {
  return fullscreenElement() === document.documentElement;
}

async function requestExperimentFullscreen() {
  if (isExperimentFullscreen()) return true;
  const root = document.documentElement;
  const request = root.requestFullscreen || root.webkitRequestFullscreen;
  if (typeof request !== "function") throw new Error("fullscreen_api_unavailable");
  await request.call(root, { navigationUI: "hide" });
  if (!isExperimentFullscreen()) throw new Error("fullscreen_not_entered");
  return true;
}

function dataCollectionStorageKey() {
  return storageKey("study_data_collection_opt_in");
}

function readDataCollectionOptIn() {
  return localStorage.getItem(dataCollectionStorageKey()) === "1";
}

function writeDataCollectionOptIn(optedIn) {
  localStorage.setItem(dataCollectionStorageKey(), optedIn ? "1" : "0");
}

function updateDataCollectionThanks() {
  const optedIn = $("#study-data-collection-opt-in").checked;
  $("#study-data-collection-thanks").classList.toggle("hidden", !optedIn);
}

function restoreDataCollectionPreference() {
  const checkbox = $("#study-data-collection-opt-in");
  if (!checkbox) return;
  checkbox.checked = readDataCollectionOptIn();
  updateDataCollectionThanks();
}

function openDataCollectionOverlay() {
  $("#data-collection-overlay").classList.remove("hidden");
  $("#data-collection-close").focus();
}

function closeDataCollectionOverlay() {
  $("#data-collection-overlay").classList.add("hidden");
}

function bindDataCollectionControls() {
  const checkbox = $("#study-data-collection-opt-in");
  const infoButton = $("#study-data-collection-info");
  const closeButton = $("#data-collection-close");
  if (!checkbox || !infoButton || !closeButton) return;

  infoButton.addEventListener("click", (event) => {
    event.preventDefault();
    openDataCollectionOverlay();
  });
  closeButton.addEventListener("click", closeDataCollectionOverlay);
  checkbox.addEventListener("change", () => {
    writeDataCollectionOptIn(checkbox.checked);
    updateDataCollectionThanks();
    queueEvent("data_collection.preference_changed", {
      opted_in: checkbox.checked,
      scope: "deidentified_academic_release",
    });
    if (checkbox.checked) openDataCollectionOverlay();
  });
}

function recordDataCollectionPreferenceForSession() {
  const optedIn = readDataCollectionOptIn();
  queueEvent("data_collection.preference", {
    opted_in: optedIn,
    scope: "deidentified_academic_release",
    blocks_experiment: false,
  });
  return optedIn;
}

function confirmStartStudyWithCountdown() {
  return new Promise((resolve) => {
    const overlay = $("#start-confirm-overlay");
    const okBtn = $("#start-confirm-ok");
    const cancelBtn = $("#start-confirm-cancel");
    const countdownEl = $("#start-confirm-countdown");
    const hintEl = $("#start-confirm-hint");
    let remaining = START_CONFIRM_COUNTDOWN_SEC;
    let timerId = null;
    let settled = false;

    function finish(result) {
      if (settled) return;
      settled = true;
      if (timerId) clearInterval(timerId);
      overlay.classList.add("hidden");
      document.removeEventListener("keydown", onKeyDown);
      resolve(result);
    }

    function onKeyDown(event) {
      if (event.key === "Escape") finish(false);
    }

    function renderWaiting() {
      countdownEl.textContent = String(remaining);
      okBtn.textContent = I18N.t(`进入全屏并开始实验（${remaining}）`);
      overlay.dataset.countdownState = "waiting";
      hintEl.textContent = I18N.t("请阅读上方说明，倒计时结束后方可开始");
    }

    function renderReady() {
      countdownEl.textContent = "0";
      okBtn.textContent = I18N.t("进入全屏并开始实验");
      okBtn.disabled = false;
      okBtn.removeAttribute("aria-disabled");
      overlay.dataset.countdownState = "ready";
      hintEl.textContent = I18N.t("倒计时已结束，可以开始");
    }

    okBtn.disabled = true;
    okBtn.setAttribute("aria-disabled", "true");
    renderWaiting();
    overlay.classList.remove("hidden");
    cancelBtn.focus();

    cancelBtn.onclick = () => finish(false);
    okBtn.onclick = async () => {
      if (okBtn.disabled) return;
      okBtn.disabled = true;
      okBtn.setAttribute("aria-disabled", "true");
      hintEl.textContent = I18N.t("正在进入全屏…");
      try {
        await requestExperimentFullscreen();
        finish(true);
      } catch (_) {
        okBtn.disabled = false;
        okBtn.removeAttribute("aria-disabled");
        hintEl.textContent = I18N.t("无法进入全屏，实验不能开始。请允许浏览器全屏后重试。");
      }
    };
    document.addEventListener("keydown", onKeyDown);

    timerId = setInterval(() => {
      remaining -= 1;
      if (remaining > 0) {
        renderWaiting();
        return;
      }
      clearInterval(timerId);
      timerId = null;
      renderReady();
    }, 1000);
  });
}

async function chooseInterruptPath(continueFromCheckpoint) {
  if (state.startInFlight || state.scheduleRunning) return;
  const blockers = startEligibilityBlockers();
  if (blockers.length > 0) {
    explainBlockedStart(blockers);
    return;
  }
  state.pendingInterruptChoice = false;
  state.interruptContinueChosen = Boolean(continueFromCheckpoint);
  syncInterruptChoicePanel();
  if (continueFromCheckpoint) {
    await sendEvent("session.interrupt_continue_chosen", {
      cursor: interruptResumeCursor(),
      stage_key: currentInterruptCheckpoint()?.stage_key || null,
      reason: currentInterruptCheckpoint()?.reason || null,
    });
  } else {
    try {
      await resetInteractionState("interrupt_restart_chosen");
    } catch (_) { /* Setup start still re-establishes server state. */ }
    clearScheduleProgress();
    await sendEvent("session.interrupt_restart_chosen", {
      reason: currentInterruptCheckpoint()?.reason || null,
    });
  }
  $("#start-study-btn")?.click();
}

$("#interrupt-restart-btn")?.addEventListener("click", () => {
  void chooseInterruptPath(false);
});
$("#interrupt-continue-btn")?.addEventListener("click", () => {
  void chooseInterruptPath(true);
});

$("#start-study-btn").addEventListener("click", async () => {
  if (state.startInFlight || state.scheduleRunning) return;
  if (state.pendingInterruptChoice && currentInterruptCheckpoint()?.available) {
    syncInterruptChoicePanel();
    return;
  }
  const error = $("#start-error");
  error.textContent = "";
  await refreshCompanionHttpPresence();
  const blockers = startEligibilityBlockers().filter(({ code }) => (
    code !== "pair_not_connected" && code !== "companion_not_ready"
  ));
  if (blockers.length > 0) {
    explainBlockedStart(startEligibilityBlockers());
    return;
  }
  if (!state.trainingPassed) {
    openTapTraining();
    return;
  }
  const confirmed = await confirmStartStudyWithCountdown();
  if (!confirmed) return;
  recordDataCollectionPreferenceForSession();
  state.startInFlight = true;
  $("#start-study-btn").disabled = true;
  showStudyLoading("正在连接计时模型与麦克风");
  try {
    await startExperimentRuntime();
    showStudyLoading("正在进入实验界面");
    showView("study-view");
    if (!state.topicSaved) await saveTopicSelection("free_topic", "free");
    await renderTopics();
    $("#topic-drawer").classList.remove("hidden");
    hideStudyLoading();
    await runSchedule();
  } catch (err) {
    hideStudyLoading();
    showView("setup-view");
    if (routeShareAppointmentError(err)) {
      error.textContent = "";
      return;
    }
    const capacityFull = err.status === 429 || err.detail?.code === "study_capacity_full";
    const detail = String(err.message || "未知错误");
    error.textContent = capacityFull ? detail : `无法启动：${detail}`;
    const liveBlockers = startEligibilityBlockers();
    showStartFailureDialog({
      title: capacityFull ? "请稍后再来" : "无法启动实验",
      message: detail,
      blockers: liveBlockers,
      hint: capacityFull
        ? "同时进行的人数较多时，系统会暂缓新实验以保持语音与网络稳定。请按提示约半小时后再试。"
        : liveBlockers.some(({ code }) => code === "pair_not_connected" || code === "companion_not_ready")
          ? "同编号登录后，还需完成手机检测并启用声音，两端 WebSocket 同时在线才会放行。"
          : "请按提示处理后重试；若反复失败，请联系微信 study-contact（微信名：Research team）。",
    });
  } finally {
    state.startInFlight = false;
    hideStudyLoading();
    updateStartEligibility();
  }
});

async function startExperimentRuntime() {
  if (state.scheduleRunning) throw new Error("今天的实验流程已经启动，请勿重复操作。");
  if (!isExperimentFullscreen()) {
    throw new Error("电脑实验系统未处于全屏，实验不能开始。请重新点击开始并允许全屏。");
  }
  const continueCursor = state.interruptContinueChosen ? interruptResumeCursor() : null;
  if (continueCursor != null) {
    const checkpoint = currentInterruptCheckpoint() || {};
    saveScheduleProgress({
      cursor: continueCursor,
      stage_key: checkpoint.stage_key || null,
      started: Boolean(checkpoint.active_elapsed_seconds),
      active_elapsed_seconds: Number(checkpoint.active_elapsed_seconds || 0),
      opportunities: Number(checkpoint.opportunities || state.opportunities || 0),
      interrupted: false,
    });
  } else {
    clearScheduleProgress();
    const resumeCursor = operatorResumeCursor();
    if (resumeCursor != null) {
      saveScheduleProgress({
        cursor: resumeCursor,
        stage_key: `block-${state.bootstrap.operator_resume.resume_block}-primary`,
        started: false,
      });
    }
  }
  const capacity = await refreshCapacityStatus();
  if (capacity && capacity.ok === false && !capacity.holding_slot) {
    const err = new Error(capacity.message_zh || capacity.message || "当前同时使用人数较多，请稍后再试");
    err.status = 429;
    err.detail = { code: "study_capacity_full", ...capacity };
    throw err;
  }
  // Always hit activate: server re-confirms a LIVE phone companion even when
  // this study session was activated earlier in the day.
  try {
    await api(`/api/sessions/${state.sessionId}/activate`, { method: "POST" });
  } catch (error) {
    if (error.status === 429 || error.detail?.code === "study_capacity_full") {
      applyCapacityStatus({
        ok: false,
        holding_slot: false,
        retry_after_minutes: error.detail?.retry_after_minutes || 30,
        message_zh: error.detail?.message || error.message,
        message: error.detail?.message || error.message,
        active_count: error.detail?.active_count,
        max_concurrent: error.detail?.max_concurrent,
      });
    }
    // Demote sticky UI immediately when server rejects missing phone.
    if (error.status === 409) {
      clearCompanionDisconnectGrace();
      clearCompanionReadyFalseDebounce();
      state.pairReady = false;
      state.companionReady = false;
      updateStartEligibility();
    }
    throw error;
  }
  state.sessionActivated = true;
  clearShareSetupTiming();
  await ensureAuthoritativePairReady(8000);
  const stream = state.demo ? null : await ensureMic();
  await connectTiming(stream);
  if (!state.demo) startRecognition();
  if (stream && state.bootstrap.consent_choices.raw_audio) startRecording(stream);
  state.eventFlushTimer = setInterval(flushEvents, 2000);
  pairSend({ type: "background_state", action: "start", day: state.day, offset_seconds: 0 });
  state.scheduleAbort = false;
  state.interactionEnding = false;
  clearRollingPcm();
  state.scheduleRunning = true;
  state.schedulePaused = false;
  state.scheduledRest = false;
  state.activeBlockIndex = null;
  state.analysisEligible = true;
  setPageVisibilityNoticeVisible(true);
  setConnection("实验进行中", "online");
  startInteractionPresenceMonitor();
  reevaluateAttentionGrace("runtime_started");
}

function httpCompanionPresenceFresh() {
  return Boolean(
    state.httpCompanionReady
    && (Date.now() - Number(state.httpCompanionReadyAtMs || 0)) < 30000
  );
}

async function refreshCompanionHttpPresence() {
  if (!state.sessionId || !state.token) return false;
  try {
    const live = await api(`/api/sessions/${state.sessionId}/pair-live`);
    const ready = live?.companion_ready === true || live?.same_day_online === true;
    state.httpCompanionReady = ready;
    state.httpCompanionReadyAtMs = ready ? Date.now() : 0;
    return ready;
  } catch (_) {
    return httpCompanionPresenceFresh();
  }
}

function startCompanionLivePolling() {
  if (state.companionLivePollTimer) return;
  void refreshCompanionHttpPresence().then(() => updateStartEligibility());
  state.companionLivePollTimer = setInterval(() => {
    if (state.scheduleRunning) return;
    void refreshCompanionHttpPresence().then(() => updateStartEligibility());
  }, 2000);
}

function livePairSnapshotForStart() {
  // Prefer the latest pair_status, but a fresh HTTP companion-ready ping is
  // enough to start when the two WebSockets landed on different listeners.
  const latest = state.lastPairStatusMessage;
  if (latest && typeof latest === "object") {
    const roles = Array.isArray(latest.roles) ? latest.roles : [];
    const pairReady = latest.same_day_online === true
      && latest.day === state.day
      && roles.includes("study")
      && roles.includes("companion");
    const companionReady = pairReady && latest.companion_ready === true;
    if (pairReady && companionReady) {
      return { pairReady: true, companionReady: true, latest };
    }
  }
  if (httpCompanionPresenceFresh()) {
    return { pairReady: true, companionReady: true, latest };
  }
  if (!latest || typeof latest !== "object") {
    return { pairReady: false, companionReady: false, latest: null };
  }
  const roles = Array.isArray(latest.roles) ? latest.roles : [];
  const pairReady = latest.same_day_online === true
    && latest.day === state.day
    && roles.includes("study")
    && roles.includes("companion");
  const companionReady = pairReady && latest.companion_ready === true;
  return { pairReady, companionReady, latest };
}

function authoritativePairSnapshot() {
  return livePairSnapshotForStart();
}

async function ensureAuthoritativePairReady(timeoutMs = 8000) {
  // Debounced UI state can look "ready" while the phone is already gone.
  clearCompanionReadyFalseDebounce();
  clearCompanionDisconnectGrace();
  const started = Date.now();
  let lastReadyNudgeMs = 0;
  while (Date.now() - started < timeoutMs) {
    const snap = livePairSnapshotForStart();
    if (
      snap.pairReady
      && snap.companionReady
      && state.pairSocket?.readyState === WebSocket.OPEN
    ) {
      state.pairReady = true;
      state.companionReady = true;
      return;
    }
    pairSend({ type: "heartbeat", client_ms: Date.now(), role: "study" });
    // Nudge at most once per 1.5s; denser request_ready storms made phones
    // force-reconnect and flicker between 重连中 / 可开始.
    if (Date.now() - lastReadyNudgeMs >= 1500) {
      lastReadyNudgeMs = Date.now();
      pairSend({ type: "notice", action: "request_ready", day: state.day });
    }
    await sleep(400);
  }
  const snap = livePairSnapshotForStart();
  state.pairReady = Boolean(snap.pairReady);
  state.companionReady = Boolean(snap.companionReady);
  updateStartEligibility();
  if (snap.pairReady && !snap.companionReady) {
    throw new Error(
      "手机端尚未向服务器确认就绪。请在手机上重新点击「完成检测并连接电脑」，看到已连接后再开始。",
    );
  }
  throw new Error(
    "手机伴随端未在线。请用专属链接打开手机陪伴页，完成检测并连接后，再开始实验。",
  );
}

async function connectTiming(stream) {
  clearInterval(state.demoAudioTimer);
  state.demoAudioTimer = null;
  try { state.workletNode?.disconnect(); } catch (_) { /* Already disconnected. */ }
  state.workletNode = null;
  await ensureAuthoritativePairReady(8000);
  const socket = new WebSocket(wsUrl("/ws/timing", { token: state.token, session_id: state.sessionId }));
  socket.binaryType = "arraybuffer";
  state.timingSocket = socket;
  let timingReady = false;
  let settled = false;
  socket.addEventListener("message", handleTimingMessage);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new Error("计时模型连接超时"));
    }, 20000);
    const fail = (message) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { socket.close(); } catch (_) { /* Already closed. */ }
      reject(new Error(message));
    };
    // Browsers often fire "error" before "close"; wait briefly for the close
    // code / timing_error payload so we do not mask a pair rejection as 1006.
    socket.addEventListener("error", () => {
      setTimeout(() => {
        if (!settled && !timingReady) {
          fail("计时模型连接失败：网络或服务暂时不可用，请确认手机陪伴页保持打开后重试");
        }
      }, 400);
    }, { once: true });
    socket.addEventListener("close", (event) => {
      if (timingReady || settled) return;
      if (event.code === 4401 || event.code === 403 || event.code === 1006) {
        fail("计时连接被拒绝：手机陪伴端未保持就绪。请只保留手机上一个陪伴页，重新点「完成检测并连接电脑」后再开始。");
        return;
      }
      fail(`计时模型连接失败（关闭码 ${event.code || "未知"}）`);
    }, { once: true });
    socket.addEventListener("message", (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "timing_ready") {
          timingReady = true;
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          resolve();
        }
        if (message.type === "timing_error") {
          const pairIssue = message.code === "pair_not_ready"
            || String(message.message || "").toLowerCase().includes("companion");
          fail(
            pairIssue
              ? "计时连接被拒绝：手机陪伴端未保持就绪。请只保留手机上一个陪伴页，重新点「完成检测并连接电脑」后再开始。"
              : `计时模型连接失败：${message.message || message.code || "未知错误"}`,
          );
        }
      } catch (_) {
        // Non-JSON frames are handled by the binary PCM path later.
      }
    });
  });
  socket.addEventListener("close", () => {
    if (state.timingSocket === socket) state.timingSocket = null;
    for (const waiter of state.timingCommandWaiters.values()) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error("计时连接在命令确认前断开"));
    }
    state.timingCommandWaiters.clear();
    for (const waiter of state.tapResultWaiters.values()) {
      clearTimeout(waiter.timer);
      waiter.reject(new Error("计时连接在点击确认前断开"));
    }
    state.tapResultWaiters.clear();
    if (state.scheduleRunning && !state.scheduleAbort) pauseSchedule("timing_socket_closed");
  });

  if (state.demo) {
    const samples = new Int16Array(1600);
    state.demoAudioTimer = setInterval(() => {
      if (socket.readyState !== WebSocket.OPEN || state.schedulePaused || state.scheduledRest) return;
      const controlled = state.demoPcmMode === "controlled";
      const speaking = controlled
        ? performance.now() < state.demoSpeechUntilMs
        : state.demoAudioTick % 40 < 18;
      const seed = state.demoSpeechSeed || 1;
      for (let index = 0; index < samples.length; index += 1) {
        if (!speaking) {
          samples[index] = 0;
          continue;
        }
        const absolute = index + state.demoAudioTick * samples.length;
        const fundamental = 155 + (seed % 75);
        const envelope = Math.min(1, index / 120, (samples.length - index) / 120);
        const voiced = Math.sin((2 * Math.PI * fundamental * absolute) / 16000)
          + 0.42 * Math.sin((2 * Math.PI * fundamental * 2.01 * absolute) / 16000)
          + 0.2 * Math.sin((2 * Math.PI * fundamental * 3.02 * absolute) / 16000);
        samples[index] = Math.round(2800 * envelope * voiced);
      }
      socket.send(samples.buffer.slice(0));
      state.demoAudioTick += 1;
    }, 100);
    return;
  }

  const context = state.audioContext || new AudioContext();
  state.audioContext = context;
  await withTimeout(context.resume(), 5000, "音频系统没有响应，请关闭其他音频应用后重试");
  await context.audioWorklet.addModule(`${APP_BASE}/static/pcm-worklet.js`);
  const source = context.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(context, "pcm-capture");
  const mute = context.createGain();
  mute.gain.value = 0;
  source.connect(worklet).connect(mute).connect(context.destination);
  worklet.port.onmessage = (event) => {
    const samples = new Int16Array(event.data);
    if (state.scheduleRunning && !state.schedulePaused && !state.scheduledRest) {
      appendRollingPcm(samples);
      if (state.companionRelayActive) appendCompanionRelayPcm(samples);
    }
    // Always keep the timing / online-learning model fed. Phone PTT may borrow
    // desktop Web Speech, but starving PCM here made activate fire without
    // follow-up and felt like “the model stopped responding”.
    if (socket.readyState === WebSocket.OPEN && !state.schedulePaused && !state.scheduledRest) {
      let energy = 0;
      for (let index = 0; index < samples.length; index += 1) {
        const normalized = samples[index] / 32768;
        energy += normalized * normalized;
      }
      const rms = Math.sqrt(energy / Math.max(1, samples.length));
      if (rms >= 0.012) {
        state.localLastVoiceMs = performance.now();
      }
      // v27: activation is acoustic-model only. Do not arm reply-needed /
      // decision_context from RMS or ASR endpointing — that reimposed
      // VAD-like delays. ASR text is consumed at activation for chat.
      socket.send(event.data);
    }
  };
  state.workletNode = worklet;
}

function handleTimingMessage(event) {
  const message = JSON.parse(event.data);
  if (message.type === "timing_error" && message.tap_id && state.tapResultWaiters.has(message.tap_id)) {
    const waiter = state.tapResultWaiters.get(message.tap_id);
    state.tapResultWaiters.delete(message.tap_id);
    clearTimeout(waiter.timer);
    waiter.reject(new Error(message.code || message.message || "tap rejected"));
    queueEvent("tap.client_error", {
      tap_id: message.tap_id,
      code: message.code || "timing_error",
      message: message.message || "",
    });
    return;
  }
  if (message.transition_id && state.timingCommandWaiters.has(message.transition_id)) {
    const waiter = state.timingCommandWaiters.get(message.transition_id);
    state.timingCommandWaiters.delete(message.transition_id);
    clearTimeout(waiter.timer);
    if (message.type === "phase_blocked") {
      waiter.reject(new Error(message.message || "阶段切换被状态机拒绝"));
    } else if (waiter.expected.has(message.type)) {
      waiter.resolve(message);
    } else {
      waiter.reject(new Error(`未预期的计时确认：${message.type}`));
    }
    return;
  }
  if (message.type === "prediction") {
    if (!message.skipped) queueEvent("model.prediction", message);
    if (message.activate
      && !state.phaseTransitionInFlight
      && state.agentState === "waiting"
      && !state.schedulePaused
      && !state.activationReleaseInFlight
      && !state.tapDesyncHealInFlight) {
      state.currentActivationId = message.activation_id || null;
      if (Number.isInteger(message.activation_state_version)) {
        state.timingStateVersion = message.activation_state_version;
      }
      if (performance.now() < state.autoActivateCooldownUntilMs) {
        queueEvent("agent.turn_skipped", {
          reason: "auto_activate_cooldown",
          cooldown_remaining_ms: Math.round(state.autoActivateCooldownUntilMs - performance.now()),
        });
        void releaseServerActivation("auto_activate_cooldown");
        return;
      }
      // Timing model / tap decide when to respond. ASR must not gate this.
      beginAgentTurn("automatic");
    }
  } else if (message.type === "tap_result") {
    const waiter = message.tap_id ? state.tapResultWaiters.get(message.tap_id) : null;
    if (waiter) {
      state.tapResultWaiters.delete(message.tap_id);
      clearTimeout(waiter.timer);
      waiter.resolve(message);
    }
    if (typeof message.server_activation_id === "string" || message.server_activation_id === null) {
      state.currentActivationId = message.server_activation_id;
    }
    if (Number.isInteger(message.server_state_version)) {
      state.timingStateVersion = message.server_state_version;
    }
    queueEvent("tap.result", message);
    if (message.action === "manual_activate") {
      showTapActionFeedback("manual_activate", true);
      beginAgentTurn("manual");
    } else if (message.action === "interrupt") {
      showTapActionFeedback("interrupt", true);
      noteCompanionInterruptActivity(message);
      stopAgentSpeech();
      setAgentState("waiting");
    } else if (message.action === "state_transition_mistouch") {
      showTapMistouch(
        message.intended_action,
        Number(message.state_age_ms),
        "server",
      );
    } else if (message.action === "conditional_tap_cancelled") {
      handleConditionalTapCancelled(message);
    }
  } else if (message.type === "state_ack") {
    if (typeof message.server_activation_id === "string" || message.server_activation_id === null) {
      state.currentActivationId = message.server_activation_id;
    }
    if (Number.isInteger(message.server_state_version)) {
      state.timingStateVersion = message.server_state_version;
    }
  } else if (message.type === "timing_ready") {
    state.currentActivationId = message.server_activation_id || null;
    if (Number.isInteger(message.server_state_version)) {
      state.timingStateVersion = message.server_state_version;
    }
    state.decisionFactorization = message.decision_factorization || null;
    state.stateTransitionTapGuardMs = Math.max(
      0,
      Number(message.state_transition_tap_guard_ms ?? 500),
    );
    queueEvent("model.ready", message);
  } else if (message.type === "timing_error") {
    queueEvent("model.error", { code: message.code || "timing_error" });
    pauseSchedule("timing_model_error");
  } else if (message.type === "pair_required") {
    handlePairRequiredFromServer();
  }
}

function sendTimingCommand(message, expectedTypes, timeoutMs = 5000) {
  if (state.timingSocket?.readyState !== WebSocket.OPEN) {
    return Promise.reject(new Error("计时连接尚未就绪"));
  }
  const transitionId = crypto.randomUUID();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      state.timingCommandWaiters.delete(transitionId);
      reject(new Error("计时状态确认超时"));
    }, timeoutMs);
    state.timingCommandWaiters.set(transitionId, {
      resolve,
      reject,
      timer,
      expected: new Set(expectedTypes),
    });
    state.timingSocket.send(JSON.stringify({ ...message, transition_id: transitionId }));
  });
}

async function setDecisionContext({
  conversationTarget,
  replyNeededProbability,
  source,
  textSha256 = null,
}) {
  const acknowledgement = await sendTimingCommand(
    {
      type: "decision_context",
      conversation_target: conversationTarget,
      reply_needed_probability: replyNeededProbability,
      source,
      text_sha256: textSha256,
    },
    ["decision_context_ack"],
  );
  if (acknowledgement.accepted === false) {
    queueEvent("decision_context.stale_ack", {
      conversation_target: conversationTarget,
      source,
      rejection_reason: acknowledgement.rejection_reason || "stale_context",
      server_agent_state: acknowledgement.server_agent_state || null,
    });
    return acknowledgement;
  }
  state.decisionContextTarget = conversationTarget;
  state.decisionContextAllowsReply =
    acknowledgement.context_allows_activation === true;
  state.decisionContextVersion = Number(acknowledgement.decision_context_version || 0);
  queueEvent("decision_context.ack", {
    conversation_target: conversationTarget,
    context_allows_reply: state.decisionContextAllowsReply,
    decision_context_version: state.decisionContextVersion,
    source,
    text_sha256: textSha256,
  });
  return acknowledgement;
}

async function preparePrimaryDecisionContext() {
  state.decisionContextRequestGeneration += 1;
  await setDecisionContext({
    conversationTarget: "experimental_agent",
    replyNeededProbability: 1.0,
    source: "primary_assumed",
  });
}

function showStudyLoading(detail = "") {
  const overlay = $("#study-loading-overlay");
  const detailNode = $("#study-loading-detail");
  if (detailNode) detailNode.textContent = detail || I18N.t("正在连接实验系统");
  overlay?.classList.remove("hidden");
}

function hideStudyLoading() {
  $("#study-loading-overlay")?.classList.add("hidden");
}

async function resetInteractionState(reason = "restart_after_interrupt") {
  if (!state.sessionId) return;
  try {
    await api(`/api/sessions/${state.sessionId}/reset-interaction-state`, {
      method: "POST",
      body: { reason },
    });
    state.transcriptBuffer = [];
    state.frozenUserTranscript = "";
    if ($("#live-transcript")) {
      $("#live-transcript").textContent = I18N.t("可以从右侧任选一个话题开始说");
    }
    queueEvent("interaction.reset", { reason });
  } catch (error) {
    queueEvent("interaction.reset_error", { reason, message: error.message });
  }
}

function clearThinkingWatchdog() {
  clearTimeout(state.thinkingWatchdog);
  state.thinkingWatchdog = null;
}

function armThinkingWatchdog(generation) {
  clearThinkingWatchdog();
  state.thinkingWatchdog = setTimeout(() => {
    if (generation !== state.agentTurnGeneration || state.agentState !== "thinking") return;
    queueEvent("agent.turn_timeout", { stage: "watchdog" });
    stopAgentSpeech();
    setAgentState("waiting");
    restoreLiveTranscriptAfterAgentTurn();
  }, 50000);
}

function finishAgentTurnIfStale(generation, nextState = "waiting") {
  if (generation !== state.agentTurnGeneration) return;
  if (state.agentState === "thinking" || state.agentState === "speaking") {
    setAgentState(nextState);
  }
}

async function classifyReplyNeeded(transcript) {
  if (!transcript || state.decisionContextTarget !== "experimental_agent") return;
  if (state.agentState !== "waiting") {
    state.pendingDecisionTranscript = transcript;
    queueEvent("decision_context.deferred", { reason: "agent_active", agent_state: state.agentState });
    return;
  }
  if (state.pendingDecisionTranscript === transcript) state.pendingDecisionTranscript = "";
  const generation = state.decisionContextRequestGeneration + 1;
  state.decisionContextRequestGeneration = generation;
  try {
    const prediction = await api("/api/reply-needed", {
      method: "POST",
      body: {
        session_id: state.sessionId,
        transcript,
      },
    });
    if (
      generation !== state.decisionContextRequestGeneration
      || state.decisionContextTarget !== "experimental_agent"
    ) return;
    await setDecisionContext({
      conversationTarget: "experimental_agent",
      replyNeededProbability: Number(prediction.probability),
      source: `asr:${prediction.model_version}`,
      textSha256: prediction.text_sha256,
    });
  } catch (error) {
    queueEvent("decision_context.classification_error", { message: error.message });
  }
}

async function setTimingPhase(
  phase,
  policyToken,
  {
    activeBlockIndex = state.activeBlockIndex,
    analysisEligible = !["rest", "complete"].includes(phase),
  } = {},
) {
  const previousPhase = state.currentPhase;
  const previousState = state.agentState;
  state.phaseTransitionInFlight = true;
  try {
    if (previousState !== "waiting") {
      stopAgentSpeech();
      setAgentState("waiting");
    }
    let acknowledgement;
    let timedOut = false;
    try {
      acknowledgement = await sendTimingCommand(
        {
          type: "phase",
          phase,
          policy_token: policyToken,
          drain_active: true,
          active_block_index: activeBlockIndex,
          analysis_eligible: analysisEligible,
        },
        ["phase_ack"],
      );
    } catch (error) {
      if (String(error?.message || error) !== "计时状态确认超时") throw error;
      timedOut = true;
      acknowledgement = {
        policy: null,
        drained_agent_state: previousState,
        server_activation_id: null,
        server_state_version: null,
      };
    }
    state.currentActivationId = acknowledgement.server_activation_id || null;
    if (Number.isInteger(acknowledgement.server_state_version)) {
      state.timingStateVersion = acknowledgement.server_state_version;
    }
    state.currentPhase = phase;
    state.currentPolicyToken = policyToken;
    state.activeBlockIndex = activeBlockIndex;
    state.analysisEligible = Boolean(
      analysisEligible && !["rest", "complete"].includes(phase),
    );
    if (state.scheduledRest && !["rest", "complete"].includes(phase)) {
      if (state.recorder?.state === "paused") state.recorder.resume();
      if (state.audioContext?.state === "suspended") await state.audioContext.resume();
      state.scheduledRest = false;
      if (!state.demo && !state.schedulePaused && !state.scheduleAbort) startRecognition();
    }
    queueEvent("phase.transition_ack", {
      from_phase: previousPhase,
      to_phase: phase,
      policy_token: policyToken,
      policy: acknowledgement.policy,
      drained_agent_state: acknowledgement.drained_agent_state || previousState,
      active_block_index: state.activeBlockIndex,
      analysis_eligible: state.analysisEligible,
      timed_out: timedOut,
    });
  } finally {
    state.phaseTransitionInFlight = false;
  }
}

function applyAgentState(next) {
  if (next !== state.agentState) {
    state.agentStateChangedAtMs = performance.now();
  }
  state.agentState = next;
  const labels = {
    waiting: {
      state: "等待",
      description: "实验系统尚未开始回答",
      action: "现在回答",
      hint: "说完后等待 1～3 秒；仍未回答时点击（超过约 5 秒请重新说一遍）",
      image: staticUrl("static/emotions/wait.png"),
      intent: "manual_activate",
    },
    thinking: {
      state: "思考中",
      description: "实验系统正在准备回答",
      action: "停止回答",
      hint: "您认为我开始回答您的时机不对时点击",
      image: staticUrl("static/emotions/think.png"),
      intent: "interrupt",
    },
    speaking: {
      state: "正在回答",
      description: "实验系统正在说话",
      action: "停止回答",
      hint: "您认为我开始回答您的时机不对时点击",
      image: staticUrl("static/emotions/speak.png"),
      intent: "interrupt",
    },
  };
  let presentation = labels[next];
  // During the phone-companion window, guide attention to the phone instead of
  // inviting a desktop “现在回答” activation.
  const companionGuide = Boolean(state.companionWindowActive && next === "waiting");
  if (companionGuide) {
    presentation = {
      ...presentation,
      action: "请与手机上的陪伴智能体互动",
      hint: "请转向手机与陪伴伙伴说话；电脑实验系统此时无需点击",
      intent: "companion_phone_guide",
    };
  }
  $("#tap-zone").dataset.state = next;
  $("#tap-zone").classList.toggle("companion-phone-focus", companionGuide);
  $("#agent-state").textContent = I18N.t(presentation.state);
  $("#agent-state-description").textContent = I18N.t(presentation.description);
  $("#agent-emotion").src = presentation.image;
  const tapBtn = $("#tap-action-btn");
  tapBtn.dataset.action = presentation.intent;
  $("#tap-action-label").textContent = I18N.t(presentation.action);
  $("#tap-instruction").textContent = I18N.t(presentation.hint);
  // Companion window + waiting: gray non-interactive guide, not a green activate.
  tapBtn.disabled = companionGuide;
  tapBtn.setAttribute("aria-disabled", String(companionGuide));
  tapBtn.classList.toggle("is-companion-guide", companionGuide);
}

function clearTapActionFeedback() {
  clearTimeout(state.tapFeedbackClearTimer);
  state.tapFeedbackClearTimer = null;
  state.tapFeedbackStickyUntilWaiting = false;
  const feedback = $("#tap-action-feedback");
  if (!feedback) return;
  feedback.textContent = "";
  feedback.classList.add("hidden");
  feedback.classList.remove("mistouch", "effective");
}

function scheduleClearTapActionFeedback(delayMs = 0) {
  clearTimeout(state.tapFeedbackClearTimer);
  state.tapFeedbackClearTimer = setTimeout(() => {
    state.tapFeedbackClearTimer = null;
    clearTapActionFeedback();
  }, Math.max(0, delayMs));
}

function showTapActionFeedback(action, effective) {
  const feedback = $("#tap-action-feedback");
  const actionLabel = action === "interrupt" ? "停止回答" : "现在回答";
  clearTimeout(state.tapFeedbackClearTimer);
  state.tapFeedbackClearTimer = null;
  feedback.textContent = effective
    ? `最近一次点击：${actionLabel}已生效`
    : `最近一次点击：误触，${actionLabel}未生效，也不会用于模型学习`;
  feedback.classList.toggle("mistouch", !effective);
  feedback.classList.toggle("effective", effective);
  feedback.classList.remove("hidden");
  // Keep effective "现在回答" only for the current activation round.
  // Interrupt / mistouch are momentary — clear after a short glance window.
  if (effective && action === "manual_activate") {
    state.tapFeedbackStickyUntilWaiting = true;
  } else if (effective && action === "interrupt") {
    state.tapFeedbackStickyUntilWaiting = false;
    scheduleClearTapActionFeedback(2000);
  } else {
    state.tapFeedbackStickyUntilWaiting = false;
    scheduleClearTapActionFeedback(2500);
  }
}

function showTapCancelledFeedback(reason) {
  const feedback = $("#tap-action-feedback");
  const syncing = [
    "expected_state_mismatch",
    "manual_requires_waiting",
    "expected_activation_mismatch",
    "expected_state_version_mismatch",
  ].includes(reason);
  clearTimeout(state.tapFeedbackClearTimer);
  state.tapFeedbackClearTimer = null;
  state.tapFeedbackStickyUntilWaiting = false;
  feedback.textContent = syncing
    ? "最近一次点击：状态不同步，正在恢复后重试"
    : "最近一次点击：未生效，请再点一次";
  feedback.classList.add("mistouch");
  feedback.classList.remove("effective", "hidden");
  scheduleClearTapActionFeedback(2500);
}

async function releaseServerActivation(reason) {
  if (state.activationReleaseInFlight) {
    try { await state.activationReleaseInFlight; } catch (_) { /* Prior release already logged. */ }
  }
  const run = (async () => {
    queueEvent("agent.activation_release", { reason });
    if (state.timingSocket?.readyState !== WebSocket.OPEN) {
      state.currentActivationId = null;
      return false;
    }
    try {
      const ack = await sendTimingCommand({ type: "state", state: "waiting" }, ["state_ack"]);
      state.currentActivationId = (
        typeof ack.server_activation_id === "string" || ack.server_activation_id === null
      )
        ? ack.server_activation_id
        : null;
      if (Number.isInteger(ack.server_state_version)) {
        state.timingStateVersion = ack.server_state_version;
      }
      if (state.agentState !== "waiting") {
        applyAgentState("waiting");
        resumeRecognitionAfterAgentSpeech();
      }
      return true;
    } catch (error) {
      state.currentActivationId = null;
      queueEvent("agent.activation_release_failed", {
        reason,
        message: error.message,
      });
      return false;
    }
  })();
  state.activationReleaseInFlight = run;
  try {
    return await run;
  } finally {
    if (state.activationReleaseInFlight === run) state.activationReleaseInFlight = null;
  }
}

function handleConditionalTapCancelled(message) {
  const reason = message.cancel_reason || "conditional_tap_cancelled";
  const desyncReasons = new Set([
    "expected_state_mismatch",
    "manual_requires_waiting",
    "expected_activation_mismatch",
    "expected_state_version_mismatch",
  ]);
  if (
    !desyncReasons.has(reason)
    || state.agentState !== "waiting"
    || state.currentTurnAbort
    || state.tapDesyncHealInFlight
    || message.intended_action !== "manual_activate"
  ) {
    // Keep non-desync cancels visible; desync heals silently then retries.
    if (!desyncReasons.has(reason)) showTapCancelledFeedback(reason);
    return;
  }
  state.tapDesyncHealInFlight = true;
  queueEvent("tap.desync_heal", { reason });
  releaseServerActivation(`tap_desync_${reason}`)
    .then(async (released) => {
      if (!released || state.agentState !== "waiting" || state.schedulePaused) return null;
      // Release bumps server/client state clocks; retrying inside the 500ms
      // transition guard was logged as mistouch even though the first tap was valid.
      await sleep(state.stateTransitionTapGuardMs + 80);
      if (state.agentState !== "waiting" || state.schedulePaused) return null;
      state.agentStateChangedAtMs = performance.now() - state.stateTransitionTapGuardMs - 1;
      return sendTapRequest({
        intended_action: "manual_activate",
        expected_state: "waiting",
        expected_activation_id: state.currentActivationId || "",
        expected_state_version: state.timingStateVersion,
        feedback_attribution: "acoustic_personalizer",
      });
    })
    .catch((error) => {
      queueEvent("tap.desync_heal_failed", { message: error.message, reason });
      showTapCancelledFeedback("client_not_ready");
    })
    .finally(() => {
      state.tapDesyncHealInFlight = false;
    });
}

function showTapMistouch(action, stateAgeMs, source = "client") {
  showTapActionFeedback(action, false);
  queueEvent("tap.mistouch", {
    intended_action: action,
    state_age_ms: Number.isFinite(stateAgeMs) ? stateAgeMs : null,
    state_transition_tap_guard_ms: state.stateTransitionTapGuardMs,
    source,
    agent_state: state.agentState,
    active_block_index: state.activeBlockIndex,
    analysis_eligible: state.analysisEligible,
  });
}

function shouldAcceptAsrResults() {
  return state.scheduleRunning
    && !state.schedulePaused
    && !state.scheduledRest
    && !state.demo
    && state.agentState === "waiting"
    && !state.asrPausedForAgentSpeech;
}

function pauseRecognitionForAgentSpeech() {
  if (state.asrPausedForAgentSpeech) return;
  state.asrPausedForAgentSpeech = true;
  state.recognitionWanted = false;
  state.recognitionGeneration += 1;
  clearTimeout(state.asrRestartTimer);
  state.asrRestartTimer = null;
  const recognition = state.recognition;
  state.recognition = null;
  try { recognition?.stop(); } catch (_) { /* Already stopped. */ }
}

function resumeRecognitionAfterAgentSpeech() {
  if (!state.asrPausedForAgentSpeech) return;
  state.asrPausedForAgentSpeech = false;
  if (state.scheduleRunning && !state.schedulePaused && !state.scheduledRest && !state.demo) {
    startRecognition();
  }
}

function setAgentSubtitle(text) {
  const node = $("#agent-subtitle");
  if (!node) return;
  const value = String(text || "").trim();
  node.textContent = value;
  node.classList.toggle("hidden", !value);
  node.classList.toggle("active", Boolean(value));
  if (value) {
    requestAnimationFrame(() => {
      node.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
    });
  }
}

function clearAgentSubtitle() {
  setAgentSubtitle("");
}

function restoreLiveTranscriptAfterAgentTurn() {
  const hint = I18N.t("可以从右侧任选一个话题开始说");
  $("#live-transcript").textContent = state.frozenUserTranscript || hint;
  state.frozenUserTranscript = "";
  clearAgentSubtitle();
}

function notifyCompanionExperimentalBarge(next, previous) {
  if (!state.companionWindowActive) return;
  if (next === "thinking" || next === "speaking") {
    if (previous !== "thinking" && previous !== "speaking") {
      pairSend({ type: "notice", action: "experimental_barge", agent_state: next });
      queueEvent("companion.experimental_barge_notice", { agent_state: next });
    } else if (next !== previous) {
      pairSend({ type: "notice", action: "experimental_barge", agent_state: next });
    }
    return;
  }
  if (next === "waiting" && (previous === "thinking" || previous === "speaking")) {
    pairSend({ type: "notice", action: "experimental_barge_clear" });
    queueEvent("companion.experimental_barge_clear", {});
  }
}

function setAgentState(next) {
  const previous = state.agentState;
  applyAgentState(next);
  if (next === "thinking" || next === "speaking") {
    pauseRecognitionForAgentSpeech();
  } else if (next === "waiting" && (previous === "thinking" || previous === "speaking")) {
    // Drop any agent TTS/echo captured while the response was playing.  The
    // next utterance starts a fresh rolling ASR context.
    clearRollingPcm();
    resumeRecognitionAfterAgentSpeech();
    restoreLiveTranscriptAfterAgentTurn();
    state.pendingDecisionTranscript = "";
    // End of one activation round — drop sticky "最近一次点击" feedback.
    if (state.tapFeedbackStickyUntilWaiting) {
      scheduleClearTapActionFeedback(350);
    }
    if (!state.companionWindowActive) {
      grantParticipantResponseWindow("experimental_agent_finished");
    }
  }
  notifyCompanionExperimentalBarge(next, previous);
  if (state.timingSocket?.readyState === WebSocket.OPEN) {
    state.timingSocket.send(JSON.stringify({ type: "state", state: next }));
  }
}

function sendTapRequest(intent = {}) {
  const tapId = crypto.randomUUID();
  const conditionalIntent = Object.hasOwn(intent, "intended_action");
  const conditionalSnapshotComplete = (
    !conditionalIntent
    || (
      ["manual_activate", "interrupt"].includes(intent.intended_action)
      && ["waiting", "thinking", "speaking"].includes(intent.expected_state)
      && Object.hasOwn(intent, "expected_activation_id")
      && Number.isInteger(intent.expected_state_version)
    )
  );
  const accepted = Boolean(
    state.scheduleRunning
    && !state.interactionEnding
    && !state.schedulePaused
    && !state.scheduledRest
    && state.timingSocket?.readyState === WebSocket.OPEN
    && conditionalSnapshotComplete
  );
  queueEvent("tap.pointer", {
    tap_id: tapId,
    accepted,
    agent_state: state.agentState,
    schedule_running: state.scheduleRunning,
    schedule_paused: state.schedulePaused,
    scheduled_rest: state.scheduledRest,
    active_block_index: state.activeBlockIndex,
    analysis_eligible: state.analysisEligible,
    socket_state: state.timingSocket?.readyState ?? null,
    visible: document.visibilityState === "visible",
    intended_action: intent.intended_action || null,
    intent_reason: intent.intent_reason || null,
    expected_state: Object.hasOwn(intent, "expected_state") ? intent.expected_state : null,
    expected_activation_id: Object.hasOwn(intent, "expected_activation_id") ? intent.expected_activation_id : null,
    expected_state_version: Object.hasOwn(intent, "expected_state_version") ? intent.expected_state_version : null,
  });
  if (!accepted) {
    return Promise.resolve({
      type: "tap_result",
      tap_id: tapId,
      action: "client_tap_cancelled",
      accepted: false,
      cancel_reason: conditionalSnapshotComplete ? "client_not_ready" : "conditional_snapshot_incomplete",
    });
  }
  const payload = {
    type: "tap",
    tap_id: tapId,
    pointer_agent_state: state.agentState,
    client_ms: Date.now(),
    monotonic_ms: performance.now(),
    visible: document.visibilityState === "visible",
  };
  for (const key of ["intended_action", "expected_state", "expected_activation_id", "expected_state_version", "feedback_attribution"]) {
    if (Object.hasOwn(intent, key)) payload[key] = intent[key];
  }
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      state.tapResultWaiters.delete(tapId);
      reject(new Error("点击状态确认超时"));
    }, 6000);
    state.tapResultWaiters.set(tapId, { resolve, reject, timer });
    try {
      state.timingSocket.send(JSON.stringify(payload));
    } catch (error) {
      state.tapResultWaiters.delete(tapId);
      clearTimeout(timer);
      reject(error);
    }
  });
}

window.tapStudyConditionalTap = (intent) => sendTapRequest(intent || {});
window.tapStudySetPcmMode = (mode) => {
  if (!state.demo || !["ambient", "controlled"].includes(mode)) return false;
  state.demoPcmMode = mode;
  if (mode === "controlled") state.demoSpeechUntilMs = 0;
  return true;
};
window.tapStudyQueueVoicedPcm = (request = {}) => {
  if (!state.demo || state.timingSocket?.readyState !== WebSocket.OPEN || state.schedulePaused) return false;
  const durationMs = Math.max(300, Math.min(4000, Number(request.duration_ms) || 900));
  state.demoPcmMode = "controlled";
  state.demoSpeechSeed = Number(request.seed) || 1;
  state.demoSpeechUntilMs = Math.max(state.demoSpeechUntilMs, performance.now() + durationMs);
  queueEvent("simulation.pcm_burst", {
    duration_ms: durationMs,
    expected_frames: Math.ceil(durationMs / 100),
    frame_samples: 1600,
    sample_rate_hz: 16000,
    scheduler_hz: 10,
    seed: state.demoSpeechSeed,
  });
  return true;
};

$("#tap-action-btn").addEventListener("pointerdown", (event) => {
  event.preventDefault();
  const expectedState = state.agentState;
  if (state.companionWindowActive && expectedState === "waiting") {
    const feedback = $("#tap-action-feedback");
    if (feedback) {
      clearTimeout(state.tapFeedbackClearTimer);
      feedback.textContent = I18N.t("请与手机上的陪伴智能体互动：长按手机圆形按钮说话。");
      feedback.classList.remove("hidden", "mistouch", "effective");
      scheduleClearTapActionFeedback(2500);
    }
    return;
  }
  const intendedAction = expectedState === "waiting" ? "manual_activate" : "interrupt";
  const stateAgeMs = performance.now() - state.agentStateChangedAtMs;
  if (stateAgeMs < state.stateTransitionTapGuardMs) {
    showTapMistouch(intendedAction, stateAgeMs);
    return;
  }
  // v27: reply-needed no longer gates timing; taps always train the acoustic path.
  const tapIntent = {
    intended_action: intendedAction,
    expected_state: expectedState,
    expected_activation_id: state.currentActivationId || "",
    expected_state_version: state.timingStateVersion,
    feedback_attribution: "acoustic_personalizer",
  };
  if (intendedAction === "interrupt") {
    queueEvent("agent.turn_cancel_requested", {
      reason: "user_interrupt",
      expected_state: expectedState,
      expected_activation_id: tapIntent.expected_activation_id,
      generation: state.agentTurnGeneration,
    });
    // Cancel the local chat/TTS pipeline before waiting for the server tap
    // acknowledgement. This closes the race where /api/chat resolves after
    // pointerdown and advances thinking -> speaking before tap_result arrives.
    stopAgentSpeech();
  }
  sendTapRequest(tapIntent).then((result) => {
    if (result?.action === "client_tap_cancelled") {
      showTapCancelledFeedback(result.cancel_reason || "client_not_ready");
    }
  }).catch((error) => {
    queueEvent("tap.client_error", { message: error.message });
    if (
      intendedAction === "interrupt"
      && (state.agentState === "thinking" || state.agentState === "speaking")
    ) {
      setAgentState("waiting");
    }
  });
});

function scheduleRecognitionRestart(delayMs = 250) {
  clearTimeout(state.asrRestartTimer);
  state.asrRestartTimer = setTimeout(() => {
    state.asrRestartTimer = null;
    if (
      state.recognitionWanted
      && state.scheduleRunning
      && !state.schedulePaused
      && !state.scheduledRest
      && !state.asrPausedForAgentSpeech
      && !state.demo
    ) {
      startRecognition();
    }
  }, delayMs);
}

function startRecognition() {
  // Transcripts are produced only by backend Whisper on captured PCM.
  state.recognitionWanted = true;
  state.recognitionGeneration += 1;
  const previous = state.recognition;
  state.recognition = null;
  try { previous?.stop(); } catch (_) { /* Browser SpeechRecognition is not used. */ }
}

function currentUserTranscript() {
  return [...state.transcriptBuffer, state.asrInterimTranscript]
    .filter(Boolean)
    .join(" ")
    .trim();
}

async function beginAgentTurn(source) {
  if (state.agentState !== "waiting" || state.schedulePaused || state.interactionEnding) return;
  // Design: response timing is decided only by the acoustic timing model and
  // user taps. ASR may supply chat text, but empty/missing ASR must never
  // delay, skip, or cancel an activate.
  if (state.companionRelayActive) {
    void stopCompanionRelayAsr();
  }
  let transcript = "";
  let asrSource = "server_whisper";
  const pcmSnapshot = state.serverAsrPcmChunks.map((chunk) => chunk.slice());
  const generation = state.agentTurnGeneration + 1;
  state.agentTurnGeneration = generation;
  const controller = new AbortController();
  state.currentTurnAbort = controller;
  const chatTimeout = setTimeout(() => controller.abort(), 45000);
  setAgentState("thinking");
  armThinkingWatchdog(generation);
  state.opportunities += 1;
  saveScheduleProgress({});
  clearAgentSubtitle();

  $("#live-transcript").textContent = I18N.t("正在识别刚才的发言");
  try {
    const result = await transcribePcmChunks(pcmSnapshot, "primary", controller.signal);
    transcript = String(result.text || "").trim();
    queueEvent("asr.server_result", {
      context: "primary",
      transcript_chars: transcript.length,
      model: result.model || null,
      language: result.language || null,
      latency_ms: result.latency_ms || null,
      duration_seconds: result.duration_seconds || null,
      signal_rms: result.signal_rms ?? null,
      audio_persisted: false,
    });
  } catch (error) {
    state.asrLastError = String(error.message || error);
    queueEvent("asr.server_error", {
      context: "primary",
      message: state.asrLastError,
    });
  }
  if (!transcript) {
    queueEvent("agent.turn_without_asr_transcript", {
      source,
      asr_last_error: state.asrLastError || null,
      ms_since_asr_final: state.asrLastFinalMs ? performance.now() - state.asrLastFinalMs : null,
      recognition_alive: false,
    });
  }
  state.frozenUserTranscript = transcript;
  state.transcriptBuffer = [];
  state.asrInterimTranscript = "";
  $("#live-transcript").textContent = transcript || I18N.t("系统正在回答，此处仅显示你的发言");
  queueEvent("agent.turn_start", {
    source,
    transcript_chars: transcript.length,
    topic_id: state.selectedTopic,
    asr_source: transcript ? asrSource : "unavailable",
  });
  try {
    if (!transcript) {
      const clarification = I18N.isEnglish
        ? "Sorry, I did not catch that. Could you please say it again?"
        : "抱歉，我刚才没有听清，可以再说一遍吗？";
      queueEvent("agent.turn_clarification", {
        source,
        reason: "browser_and_server_asr_empty",
      });
      clearTimeout(chatTimeout);
      setAgentSubtitle(clarification);
      await playAgentSpeech(clarification, generation, controller.signal);
      return;
    }
    const result = await api("/api/chat", {
      method: "POST",
      body: {
        session_id: state.sessionId,
        source: "primary",
        transcript,
      },
      signal: controller.signal,
    });
    clearTimeout(chatTimeout);
    if (generation !== state.agentTurnGeneration || state.agentState !== "thinking") return;
    setAgentSubtitle(result.text);
    await playAgentSpeech(result.text, generation, controller.signal);
  } catch (err) {
    clearTimeout(chatTimeout);
    if (err.name === "AbortError" || generation !== state.agentTurnGeneration) {
      queueEvent("agent.turn_cancelled", {
        source,
        stage: "request",
        reason: err.name === "AbortError" ? "timeout_or_abort" : "state_changed",
      });
      finishAgentTurnIfStale(generation);
      restoreLiveTranscriptAfterAgentTurn();
      return;
    }
    queueEvent("agent.turn_error", { message: err.message });
    finishAgentTurnIfStale(generation);
    restoreLiveTranscriptAfterAgentTurn();
  } finally {
    clearThinkingWatchdog();
    if (generation === state.agentTurnGeneration) state.currentTurnAbort = null;
  }
}

function studyAgentLanguage() {
  return $("#browser-audio-language").dataset.passedLanguage
    || $("#browser-audio-language").value
    || (I18N.isEnglish ? "en-US" : "zh-CN");
}

async function ensureStudyAudioContext() {
  if (!state.audioContext) {
    state.audioContext = new AudioContext();
  }
  if (state.audioContext.state === "suspended") {
    await state.audioContext.resume();
  }
  return state.audioContext;
}

function linkAbortSignal(parentSignal, timeoutMs) {
  const controller = new AbortController();
  const onParentAbort = () => controller.abort(parentSignal?.reason || new DOMException("Aborted", "AbortError"));
  if (parentSignal?.aborted) {
    onParentAbort();
    return controller.signal;
  }
  parentSignal?.addEventListener("abort", onParentAbort, { once: true });
  const timer = setTimeout(() => {
    controller.abort(new DOMException("tts_server_timeout", "AbortError"));
  }, timeoutMs);
  controller.signal.addEventListener("abort", () => {
    clearTimeout(timer);
    parentSignal?.removeEventListener("abort", onParentAbort);
  }, { once: true });
  return controller.signal;
}

function agentTextDwellMs(text) {
  const chars = String(text || "").trim().length;
  return Math.max(2500, Math.min(8000, 1800 + chars * 220));
}

async function sleepAbortable(ms, signal) {
  if (signal?.aborted) {
    throw new DOMException("Aborted", "AbortError");
  }
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

async function playStudyServerSpeech(text, options = {}) {
  const {
    generation = null,
    signal = null,
    context = "agent",
    requireAgentState = null,
    timeoutMs = 8000,
  } = options;
  if (!state.sessionId) throw new Error("session_missing");
  const requestedAt = performance.now();
  const requestSignal = linkAbortSignal(signal, timeoutMs);
  const response = await api("/api/tts", {
    method: "POST",
    body: { session_id: state.sessionId, text },
    signal: requestSignal,
  });
  const provider = response.headers.get("X-TTS-Provider") || "server";
  const voice = response.headers.get("X-TTS-Voice") || "";
  const synthesisLatencyMs = Number(response.headers.get("X-TTS-Latency-Ms") || 0);
  const data = await (await response.blob()).arrayBuffer();
  if (!data.byteLength) throw new Error("tts_empty_audio");
  const audioContext = await ensureStudyAudioContext();
  const audioBuffer = await audioContext.decodeAudioData(data.slice(0));
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  const gain = audioContext.createGain();
  gain.gain.value = 0.92;
  source.connect(gain).connect(audioContext.destination);
  const playback = { kind: "server_web_audio", source, gain, settle: null, generation };
  if (context === "agent") state.currentAudio = playback;

  queueEvent("tts.playback_start", {
    provider,
    voice,
    voice_lang: studyAgentLanguage(),
    synthesis_latency_ms: synthesisLatencyMs,
    request_to_play_ms: Math.round(performance.now() - requestedAt),
    fallback: false,
    playback_api: "server_web_audio",
    text_chars: String(text || "").trim().length,
    context,
  });

  const outcome = await new Promise((resolve, reject) => {
    playback.settle = resolve;
    const onAbort = () => {
      try { source.stop(0); } catch (_) { /* Already stopped. */ }
      resolve({ type: "cancelled" });
    };
    signal?.addEventListener("abort", onAbort, { once: true });
    source.onended = () => {
      signal?.removeEventListener("abort", onAbort);
      resolve({ type: "ended" });
    };
    try {
      source.start(0);
    } catch (err) {
      signal?.removeEventListener("abort", onAbort);
      reject(err);
    }
  });

  if (state.currentAudio === playback) state.currentAudio = null;

  if (
    outcome.type === "cancelled"
    || signal?.aborted
    || (generation != null && generation !== state.agentTurnGeneration)
    || (requireAgentState != null && state.agentState !== requireAgentState)
  ) {
    return {
      cancelled: true,
      language: studyAgentLanguage(),
      playback_api: "server_web_audio",
      provider,
      voice_name: voice,
      voice_lang: studyAgentLanguage(),
      text_chars: String(text || "").trim().length,
    };
  }

  queueEvent("tts.playback_end", { fallback: false, provider });
  return {
    cancelled: false,
    language: studyAgentLanguage(),
    playback_api: "server_web_audio",
    provider,
    voice_name: voice,
    voice_lang: studyAgentLanguage(),
    matching_voice_available: Boolean(voice),
    text_chars: String(text || "").trim().length,
  };
}

async function playBrowserSpeechFallback(text, generation, signal) {
  const spoken = String(text || "").trim();
  if (!spoken) throw new Error("speech_synthesis_empty_text");
  const language = studyAgentLanguage();
  const requestedAt = performance.now();
  const helper = window.TTA_PLAYBACK_TEST?.speakText;
  if (typeof helper !== "function" && !(window.speechSynthesis && window.SpeechSynthesisUtterance)) {
    throw new Error("speech_synthesis_unavailable");
  }

  queueEvent("tts.playback_start", {
    provider: "speechSynthesis",
    voice: "",
    voice_lang: language,
    synthesis_latency_ms: 0,
    request_to_play_ms: Math.round(performance.now() - requestedAt),
    fallback: true,
    playback_api: "speechSynthesis",
    text_chars: spoken.length,
    context: "agent",
  });

  const playback = {
    kind: "speechSynthesis",
    settle: null,
    generation,
  };
  state.currentAudio = playback;

  try {
    if (signal?.aborted || generation !== state.agentTurnGeneration) {
      return { cancelled: true, provider: "speechSynthesis", playback_api: "speechSynthesis" };
    }
    if (typeof helper === "function") {
      const result = await Promise.race([
        helper(spoken, language, 12000),
        new Promise((_, reject) => {
          const onAbort = () => reject(new DOMException("Aborted", "AbortError"));
          if (signal?.aborted) onAbort();
          else signal?.addEventListener("abort", onAbort, { once: true });
        }),
      ]);
      if (
        signal?.aborted
        || generation !== state.agentTurnGeneration
        || state.agentState !== "speaking"
      ) {
        window.speechSynthesis?.cancel();
        return { cancelled: true, ...result, fallback: true };
      }
      queueEvent("tts.playback_end", {
        fallback: true,
        provider: "speechSynthesis",
      });
      return { cancelled: false, ...result, fallback: true };
    }

    // Inline fallback when playback helper is unavailable.
    const synthesis = window.speechSynthesis;
    synthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(spoken);
    utterance.lang = language;
    utterance.rate = language === "en-US" ? 0.96 : 0.94;
    utterance.volume = 0.9;
    await new Promise((resolve, reject) => {
      let settled = false;
      const finish = (error = null) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        if (error) reject(error);
        else resolve();
      };
      const onAbort = () => {
        try { synthesis.cancel(); } catch (_) { /* ignore */ }
        finish(new DOMException("Aborted", "AbortError"));
      };
      utterance.onend = () => finish();
      utterance.onerror = (event) => finish(new Error(event.error || "speech_synthesis_failed"));
      const timer = setTimeout(() => {
        synthesis.cancel();
        finish(new Error("speech_synthesis_timeout"));
      }, 12000);
      signal?.addEventListener("abort", onAbort, { once: true });
      synthesis.speak(utterance);
    });
    if (
      signal?.aborted
      || generation !== state.agentTurnGeneration
      || state.agentState !== "speaking"
    ) {
      return { cancelled: true, provider: "speechSynthesis", playback_api: "speechSynthesis", fallback: true };
    }
    queueEvent("tts.playback_end", { fallback: true, provider: "speechSynthesis" });
    return {
      cancelled: false,
      language,
      playback_api: "speechSynthesis",
      provider: "speechSynthesis",
      fallback: true,
      text_chars: spoken.length,
    };
  } finally {
    if (state.currentAudio === playback) state.currentAudio = null;
  }
}

async function dwellAgentTextThenWait(text, generation, signal) {
  const dwellMs = agentTextDwellMs(text);
  queueEvent("tts.text_only_fallback", {
    reason: "server_and_browser_tts_failed",
    dwell_ms: dwellMs,
    text_chars: String(text || "").trim().length,
  });
  setAgentSubtitle(text);
  try {
    await sleepAbortable(dwellMs, signal);
  } catch (err) {
    if (err.name === "AbortError") {
      return { cancelled: true };
    }
    throw err;
  }
  if (generation !== state.agentTurnGeneration) {
    return { cancelled: true };
  }
  return { cancelled: false, dwell_ms: dwellMs };
}

async function playAgentSpeech(text, generation, signal) {
  const spoken = String(text || "").trim();
  try {
    if (generation !== state.agentTurnGeneration || state.agentState !== "thinking") {
      queueEvent("agent.turn_cancelled", { stage: "tts", reason: "state_changed" });
      finishAgentTurnIfStale(generation);
      restoreLiveTranscriptAfterAgentTurn();
      return;
    }
    setAgentState("speaking");
    setAgentSubtitle(spoken);

    // 1) Preferred: server TTS (hard timeout so a hung provider cannot stall the study).
    try {
      const result = await playStudyServerSpeech(spoken, {
        generation,
        signal,
        context: "agent",
        requireAgentState: "speaking",
        timeoutMs: 8000,
      });
      if (result.cancelled) {
        queueEvent("agent.turn_cancelled", { stage: "tts", reason: "state_changed" });
      }
      return;
    } catch (serverErr) {
      if (serverErr.name === "AbortError" || generation !== state.agentTurnGeneration) {
        if (signal?.aborted || generation !== state.agentTurnGeneration) {
          queueEvent("agent.turn_cancelled", { stage: "tts", reason: "state_changed" });
          return;
        }
      }
      queueEvent("tts.server_failed", {
        message: serverErr.message || "server_tts_failed",
        fallback_used: true,
        provider: "server",
        next: "speechSynthesis",
      });
    }

    if (generation !== state.agentTurnGeneration || state.agentState !== "speaking") {
      queueEvent("agent.turn_cancelled", { stage: "tts", reason: "state_changed" });
      return;
    }

    // 2) Browser built-in TTS.
    try {
      const browserResult = await playBrowserSpeechFallback(spoken, generation, signal);
      if (browserResult.cancelled) {
        queueEvent("agent.turn_cancelled", { stage: "tts", reason: "state_changed" });
      } else {
        queueEvent("tts.browser_fallback_ok", {
          provider: "speechSynthesis",
          text_chars: spoken.length,
        });
      }
      return;
    } catch (browserErr) {
      if (browserErr.name === "AbortError" || generation !== state.agentTurnGeneration) {
        queueEvent("agent.turn_cancelled", { stage: "tts", reason: "state_changed" });
        return;
      }
      queueEvent("tts.browser_failed", {
        message: browserErr.message || "browser_tts_failed",
        fallback_used: true,
        next: "text_only",
      });
    }

    if (generation !== state.agentTurnGeneration || state.agentState !== "speaking") {
      queueEvent("agent.turn_cancelled", { stage: "tts", reason: "state_changed" });
      return;
    }

    // 3) Last resort: keep the reply text visible, then return to waiting.
    const dwell = await dwellAgentTextThenWait(spoken, generation, signal);
    if (dwell.cancelled) {
      queueEvent("agent.turn_cancelled", { stage: "tts_text_dwell", reason: "state_changed" });
    }
  } finally {
    if (generation === state.agentTurnGeneration && state.agentState === "speaking") {
      setAgentState("waiting");
      restoreLiveTranscriptAfterAgentTurn();
    }
  }
}

function stopPrimaryTurnIntake() {
  state.recognitionWanted = false;
  try { state.recognition?.stop(); } catch (_) { /* Already stopped. */ }
  stopAgentSpeech();
}

function stopAgentSpeech() {
  state.agentTurnGeneration += 1;
  clearThinkingWatchdog();
  window.speechSynthesis?.cancel();
  if (state.currentTurnAbort) {
    state.currentTurnAbort.abort();
    state.currentTurnAbort = null;
  }
  const playback = state.currentAudio;
  if (playback?.source?.stop) {
    try { playback.source.stop(0); } catch (_) { /* Already stopped. */ }
  }
  if (playback?.audio instanceof HTMLAudioElement) {
    playback.audio.pause();
    playback.audio.currentTime = 0;
    if (playback.objectUrl) URL.revokeObjectURL(playback.objectUrl);
  }
  if (typeof playback?.settle === "function") playback.settle({ type: "cancelled" });
  state.currentAudio = null;
}

function startRecording(stream) {
  const preferred = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"];
  const mimeType = preferred.find((type) => MediaRecorder.isTypeSupported(type));
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType, audioBitsPerSecond: 48000 } : undefined);
  recorder.ondataavailable = async (event) => {
    if (!event.data.size) return;
    try {
      await api(`/api/audio/${state.sessionId}/study_stream`, {
        method: "POST",
        headers: { "Content-Type": event.data.type.split(";", 1)[0] },
        body: event.data,
      });
    } catch (err) {
      queueEvent("audio.upload_error", { message: err.message, bytes: event.data.size });
    }
  };
  recorder.start(60000);
  state.recorder = recorder;
}

async function topicCatalog() {
  if (!state.topicCatalog) state.topicCatalog = await api("/api/topics");
  return state.topicCatalog;
}

async function saveTopicSelection(topicId, categoryId) {
  const previousTopicId = state.selectedTopic;
  if (state.topicSaved && previousTopicId === topicId) return;
  const eventType = state.topicSaved ? "topic.changed" : "topic.selected";
  state.topicSaved = false;
  updateStartEligibility();
  try {
    await sendEvent(eventType, {
      topic_id: topicId,
      category_id: categoryId,
      reused_topic_id: state.previousTopicIds.has(topicId),
      ...(eventType === "topic.changed" ? {
        from_topic_id: previousTopicId,
        user_initiated: true,
      } : {}),
    });
  } catch (err) {
    if (err.detail?.code === "topic_already_selected") {
      state.selectedTopic = topicId;
      state.topicSaved = true;
      updateStartEligibility();
      return;
    }
    throw err;
  }
  state.selectedTopic = topicId;
  state.topicSaved = true;
  updateStartEligibility();
}

async function renderTopics() {
  const topics = await topicCatalog();
    const container = $("#topic-list");
    container.replaceChildren();
    for (const category of topics.categories) {
      const section = document.createElement("section");
      section.className = "topic-category";
      const heading = document.createElement("h3");
      heading.textContent = category.label || (I18N.isEnglish ? category.label_en : category.label_zh);
      section.appendChild(heading);
      for (const prompt of category.prompts) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "topic-button";
        button.textContent = prompt.text;
        button.classList.toggle("active", prompt.id === state.selectedTopic);
        button.addEventListener("click", async () => {
          for (const peer of $$(".topic-button")) peer.classList.toggle("active", peer === button);
          try {
            await saveTopicSelection(prompt.id, category.id);
          } catch (err) {
            queueEvent("topic.selection_error", { message: err.message });
          }
        });
        section.appendChild(button);
      }
      container.appendChild(section);
    }
}

$("#topics-toggle").addEventListener("click", () => $("#topic-drawer").classList.toggle("hidden"));

function phaseDurations() {
  if (state.demo) {
    return { frozenMin: 8, frozenMax: 16, frozenTarget: 2, candidate: 10, primary: 12, companion: 8, finalPrimary: 6, micro: 10 };
  }
  return { frozenMin: 300, frozenMax: 420, frozenTarget: 12, candidate: 90, primary: 480, companion: 120, finalPrimary: 60, micro: 180 };
}

function policySequence() {
  const daySequence = state.bootstrap.participant.probe_schedule[state.day - 1] || "ABC";
  return [...daySequence];
}

function operatorResumeCursor() {
  const resume = state.bootstrap?.operator_resume;
  if (!resume || Number(resume.resume_block) !== 5) return null;
  if (resume.session_id && state.sessionId && resume.session_id !== state.sessionId) return null;
  const stages = scheduleStages();
  const index = stages.findIndex((stage) => stage.key === `block-${resume.resume_block}-primary`);
  return index >= 0 ? index : null;
}

function scheduleProgressKey() {
  return state.sessionId ? storageKey(`schedule_progress:${state.sessionId}`) : "";
}

function loadScheduleProgress() {
  const key = scheduleProgressKey();
  if (!key) return null;
  try {
    const value = JSON.parse(localStorage.getItem(key) || "null");
    return value?.session_id === state.sessionId ? value : null;
  } catch (_) {
    return null;
  }
}

function saveScheduleProgress(patch) {
  const key = scheduleProgressKey();
  if (!key) return;
  const current = loadScheduleProgress() || { session_id: state.sessionId, cursor: 0 };
  localStorage.setItem(key, JSON.stringify({
    ...current,
    ...patch,
    session_id: state.sessionId,
    opportunities: state.opportunities,
    saved_ms: Date.now(),
  }));
}

function clearScheduleProgress() {
  const key = scheduleProgressKey();
  if (key) localStorage.removeItem(key);
}

function scheduleStages() {
  const blockSchedule = state.bootstrap?.client_protocol?.block_schedule || {};
  if (blockSchedule.enabled) {
    const blocks = Math.max(1, Number(blockSchedule.active_blocks || 1));
    const primarySeconds = Number(blockSchedule.primary_seconds || 240);
    const companionSeconds = Number(blockSchedule.companion_seconds || 120);
    const restSeconds = Number(blockSchedule.rest_seconds || 60);
    const activePhase = state.bootstrap.client_protocol.personalisation_phase || "learning";
    const stages = [];
    for (let block = 1; block <= blocks; block += 1) {
      stages.push({
        kind: "timed",
        key: `block-${block}-primary`,
        label: "与实验系统对话",
        detail: `第 ${block}/${blocks} 个六分钟统计块 · 前四分钟`,
        seconds: primarySeconds,
        phase: activePhase,
        policyToken: "A",
        activeBlockIndex: block,
      });
      stages.push({
        kind: "companion",
        key: `block-${block}-companion`,
        seconds: companionSeconds,
        window: block,
        totalWindows: blocks,
        activeBlockIndex: block,
        phase: activePhase,
        policyToken: "A",
      });
      if (block < blocks || blockSchedule.rest_after_final_block) {
        stages.push({
          kind: "rest",
          key: `block-${block}-rest`,
          seconds: restSeconds,
          block: block,
          totalBlocks: blocks,
        });
      }
    }
    return stages;
  }
  const d = phaseDurations();
  const candidateProbes = state.day === 1
    ? (state.bootstrap.participant.candidate_probes || [])
    : [];
  const stages = [];
  if (candidateProbes.length) {
    for (const [index, probe] of candidateProbes.entries()) {
      stages.push({
        kind: "timed",
        key: `opening-${index + 1}`,
        label: "开场交流",
        detail: `交流时段 ${index + 1}/${candidateProbes.length}`,
        seconds: d.candidate,
        phase: "candidate_probe",
        policyToken: probe.probe_token,
      });
    }
  } else {
    stages.push({ kind: "frozen", durations: d, key: "frozen" });
  }
  for (let cycle = 0; cycle < 3; cycle += 1) {
    stages.push({
      kind: "timed",
      key: `learning-${cycle + 1}`,
      label: "自由选题交流",
      detail: `交流时段 ${cycle + 1}/4`,
      seconds: d.primary,
      phase: "learning",
      policyToken: "A",
    });
    stages.push({
      kind: "companion",
      key: `companion-${cycle + 1}`,
      seconds: d.companion,
      window: cycle + 1,
    });
  }
  stages.push({
    kind: "timed",
    key: "learning-4",
    label: "自由选题交流",
    detail: "交流时段 4/4",
    seconds: d.finalPrimary,
    phase: "learning",
    policyToken: "A",
  });
  for (const [index, policyToken] of policySequence().entries()) {
    stages.push({
      kind: "timed",
      key: `micro-${index + 1}-${policyToken}`,
      label: "收尾交流",
      detail: `交流时段 ${index + 1}/3`,
      seconds: d.micro,
      phase: "micro_probe",
      policyToken,
    });
  }
  return stages;
}

async function runSchedule() {
  const stages = scheduleStages();
  const saved = loadScheduleProgress();
  state.opportunities = Number(saved?.opportunities || 0);
  let cursor = Math.min(stages.length, Math.max(0, Number(saved?.cursor || 0)));
  for (; cursor < stages.length && !state.scheduleAbort; cursor += 1) {
    const stage = stages[cursor];
    const resume = saved?.cursor === cursor && saved?.stage_key === stage.key ? saved : null;
    saveScheduleProgress({ cursor, stage_key: stage.key, started: Boolean(resume?.started) });
    if (stage.kind === "frozen") await runFrozenPhase(stage.durations, resume);
    else if (stage.kind === "companion") {
      await runCompanionWindow(
        stage.seconds,
        stage.window,
        resume,
        stage.totalWindows,
        stage.activeBlockIndex,
        stage.phase,
        stage.policyToken,
      );
    } else if (stage.kind === "rest") {
      await runScheduledRest(stage, resume);
    } else {
      await runTimedPhase({ ...stage, resume });
    }
    if (state.scheduleAbort) break;
    saveScheduleProgress({
      cursor: cursor + 1,
      stage_key: null,
      started: false,
      completed_seconds: 0,
      active_elapsed_seconds: 0,
    });
  }
  if (!state.scheduleAbort) {
    const progress = loadScheduleProgress();
    if (progress?.interrupted) {
      await returnToSetupAfterInterrupt(progress.interrupted_reason || "manual");
      return;
    }
    state.interactionEnding = true;
    await setTimingPhase("complete", "A");
    saveScheduleProgress({
      cursor: stages.length,
      stage_key: "questionnaire",
      started: false,
      interaction_complete: true,
      ended_early: Boolean(saved?.ended_early),
    });
    await finishInteraction(Boolean(saved?.ended_early));
  }
}

async function runFrozenPhase(durations, resume = null) {
  let activeElapsed = Number(resume?.active_elapsed_seconds || 0);
  const startOpportunities = Number(resume?.start_opportunities ?? state.opportunities);
  await setTimingPhase("frozen", "A");
  updatePhaseText("开场交流", "继续自由选题并按真实感受自然对话");
  if (resume?.started) {
    await sendEvent("phase.resume", { phase: "frozen", policy_token: "A", active_elapsed_seconds: activeElapsed });
  } else {
    await sendEvent("phase.start", { phase: "frozen", policy_token: "A", target_opportunities: durations.frozenTarget });
  }
  saveScheduleProgress({ started: true, active_elapsed_seconds: activeElapsed, start_opportunities: startOpportunities });
  let last = performance.now();
  while (!state.scheduleAbort) {
    if (state.schedulePaused) {
      await waitWhilePaused();
      last = performance.now();
      continue;
    }
    const now = performance.now();
    activeElapsed += (now - last) / 1000;
    last = now;
    const opportunities = state.opportunities - startOpportunities;
    $("#phase-timer").textContent = formatTime(activeElapsed);
    saveScheduleProgress({ active_elapsed_seconds: activeElapsed, start_opportunities: startOpportunities });
    if ((activeElapsed >= durations.frozenMin && opportunities >= durations.frozenTarget) || activeElapsed >= durations.frozenMax) break;
    await sleep(200);
  }
  await sendEvent("phase.end", {
    phase: "frozen",
    elapsed_seconds: activeElapsed,
    opportunities: state.opportunities - startOpportunities,
  });
}

async function ensureTimingCaptureReady(reason) {
  if (state.recorder?.state === "paused") {
    try { state.recorder.resume(); } catch (_) { /* Recorder may already be live. */ }
  }
  if (state.audioContext && state.audioContext.state !== "running") {
    try {
      await state.audioContext.resume();
    } catch (error) {
      queueEvent("timing.audio_resume_failed", {
        reason,
        message: String(error?.message || error),
        audio_context_state: state.audioContext.state,
      });
    }
  }
  queueEvent("timing.audio_ready", {
    reason,
    audio_context_state: state.audioContext?.state || null,
    recorder_state: state.recorder?.state || null,
    scheduled_rest: state.scheduledRest,
  });
}

function clearCompanionInviteUi() {
  clearTimeout(state.companionInviteTimer);
  clearInterval(state.companionInviteTickTimer);
  state.companionInviteTimer = null;
  state.companionInviteTickTimer = null;
  state.companionInviteDeadlineMs = 0;
  const panel = $("#companion-invite-countdown");
  if (panel) {
    panel.classList.add("hidden");
    panel.dataset.state = "idle";
  }
}

function renderCompanionInviteCountdown() {
  const panel = $("#companion-invite-countdown");
  const secondsEl = $("#companion-invite-seconds");
  if (!panel || !secondsEl) return;
  if (!state.companionWindowActive || state.companionInviteUsed || !state.companionInviteDeadlineMs) {
    panel.classList.add("hidden");
    return;
  }
  const remaining = Math.max(0, Math.ceil((state.companionInviteDeadlineMs - performance.now()) / 1000));
  secondsEl.textContent = String(remaining);
  panel.dataset.state = "counting";
  panel.classList.remove("hidden");
  const label = $("#companion-invite-label");
  if (label) {
    label.textContent = I18N.t("若暂未开口，手机陪伴伙伴将在 {n} 秒后主动打招呼").replace("{n}", String(remaining));
  }
}

function markCompanionInviteFired(source = "local") {
  state.companionInviteUsed = true;
  clearTimeout(state.companionInviteTimer);
  clearInterval(state.companionInviteTickTimer);
  state.companionInviteTimer = null;
  state.companionInviteTickTimer = null;
  state.companionInviteDeadlineMs = 0;
  const panel = $("#companion-invite-countdown");
  const label = $("#companion-invite-label");
  if (panel && label) {
    panel.dataset.state = "fired";
    panel.classList.remove("hidden");
    label.textContent = I18N.t("手机陪伴伙伴正在主动打招呼（与电脑实验系统不是同一个）");
  }
  queueEvent("companion.invite_countdown", { action: "fired", source });
}

function armCompanionInviteUi(delayMs = 10000, source = "local") {
  clearCompanionInviteUi();
  if (!state.companionWindowActive || state.companionInviteUsed) return;
  state.companionInviteDeadlineMs = performance.now() + Math.max(1000, delayMs);
  renderCompanionInviteCountdown();
  state.companionInviteTickTimer = setInterval(renderCompanionInviteCountdown, 250);
  state.companionInviteTimer = setTimeout(() => {
    state.companionInviteTimer = null;
    // Phone owns the spoken invite; desktop only mirrors the countdown.
    if (!state.companionInviteUsed) {
      const panel = $("#companion-invite-countdown");
      const label = $("#companion-invite-label");
      if (panel && label) {
        panel.dataset.state = "waiting_phone";
        panel.classList.remove("hidden");
        label.textContent = I18N.t("等待手机陪伴伙伴主动打招呼…");
      }
    }
  }, Math.max(1000, delayMs));
  queueEvent("companion.invite_countdown", { action: "armed", delay_ms: delayMs, source });
}

async function runTimedPhase({
  label,
  detail,
  seconds,
  phase,
  policyToken,
  resume = null,
  activeBlockIndex = null,
}) {
  if (state.scheduleAbort) return;
  state.scheduledRest = false;
  await ensureTimingCaptureReady("timed_phase_start");
  await setTimingPhase(phase, policyToken, {
    activeBlockIndex,
    analysisEligible: true,
  });
  await ensureTimingCaptureReady("timed_phase_after_ack");
  if (
    !state.demo
    && !state.recognition
    && !state.companionRelayActive
    && !state.asrPausedForAgentSpeech
    && !state.schedulePaused
  ) {
    startRecognition();
  }
  updatePhaseText(label, detail);
  grantParticipantResponseWindow("primary_phase_start");
  const primaryBanner = $("#primary-attention-banner");
  const primaryCountdown = $("#primary-countdown");
  primaryBanner?.classList.remove("hidden");
  const completedBefore = Math.min(seconds, Number(resume?.completed_seconds || 0));
  if (activeBlockIndex != null) {
    pairSend({
      type: "notice",
      action: "primary_start",
      block: activeBlockIndex,
      seconds: seconds - completedBefore,
    });
  }
  if (resume?.started) {
    await sendEvent("phase.resume", { phase, policy_token: policyToken, completed_seconds: completedBefore, detail });
  } else {
    await sendEvent("phase.start", { phase, policy_token: policyToken, planned_seconds: seconds, detail });
  }
  saveScheduleProgress({ started: true, completed_seconds: completedBefore });
  const completedNow = await countdown(
    seconds - completedBefore,
    primaryCountdown || $("#phase-timer"),
    (value) => {
      const remainingLabel = formatTime(Math.max(0, (seconds - completedBefore) - value));
      if ($("#phase-timer") && primaryCountdown) {
        $("#phase-timer").textContent = remainingLabel;
      }
      saveScheduleProgress({ completed_seconds: completedBefore + value });
    },
  );
  primaryBanner?.classList.add("hidden");
  stopPrimaryTurnIntake();
  await sendEvent("phase.end", {
    phase,
    policy_token: policyToken,
    completed_seconds: completedBefore + completedNow,
    detail,
  });
}

async function runCompanionWindow(
  seconds,
  index,
  resume = null,
  totalWindows = 3,
  activeBlockIndex = null,
  phase = null,
  policyToken = "A",
) {
  if (state.scheduleAbort) return;
  // Keep the experimental acoustic timing model online for the whole companion
  // window (PCM + predictions + barge-in). Never mark analysis_eligible false here.
  const timingPhase = phase || state.currentPhase || "learning";
  stopPrimaryTurnIntake();
  await setTimingPhase(timingPhase, policyToken || "A", {
    activeBlockIndex: activeBlockIndex ?? state.activeBlockIndex,
    analysisEligible: true,
  });
  state.analysisEligible = true;
  state.scheduledRest = false;
  await ensureTimingCaptureReady("companion_window_enter");
  stopAgentSpeech();
  await releaseServerActivation("companion_window_enter");
  if (state.agentState !== "waiting") {
    setAgentState("waiting");
  }
  clearAgentSubtitle();
  $("#primary-attention-banner")?.classList.add("hidden");
  queueEvent("companion.window_timing_online", {
    window: index,
    analysis_eligible: true,
    timing_model: "online",
    addressee_gate: "disabled",
    reply_needed_gate: "disabled",
  });
  updatePhaseText("与手机陪伴伙伴对话", `第 ${index}/${totalWindows} 个六分钟统计块 · 后两分钟`);
  $("#companion-attention-banner").classList.remove("hidden");
  state.companionWindowActive = true;
  state.companionInviteUsed = false;
  applyAgentState(state.agentState || "waiting");
  grantParticipantResponseWindow("companion_phase_start");
  // Study ASR can stay available for barge-in chat text; phone PTT temporarily
  // borrows desktop recognition via companion relay.
  if (
    !state.demo
    && !state.recognition
    && !state.companionRelayActive
    && !state.asrPausedForAgentSpeech
    && !state.schedulePaused
  ) {
    startRecognition();
  }
  const completedBefore = Math.min(seconds, Number(resume?.completed_seconds || 0));
  pairSend({ type: "notice", action: "companion_start", window: index, seconds: seconds - completedBefore });
  armCompanionInviteUi(10000, "companion_window_start");
  if (resume?.started) {
    await sendEvent("companion.window_resume", { window: index, completed_seconds: completedBefore });
  } else {
    await sendEvent("companion.window_start", { window: index, planned_seconds: seconds });
  }
  saveScheduleProgress({ started: true, completed_seconds: completedBefore });
  const completedNow = await countdown(seconds - completedBefore, $("#companion-countdown"), (value) => {
    saveScheduleProgress({ completed_seconds: completedBefore + value });
  });
  state.companionWindowActive = false;
  state.companionAgentBusy = false;
  applyAgentState(state.agentState || "waiting");
  clearCompanionInviteUi();
  pairSend({ type: "notice", action: "experimental_barge_clear" });
  pairSend({ type: "notice", action: "companion_stop", window: index });
  $("#companion-attention-banner").classList.add("hidden");
  await ensureTimingCaptureReady("companion_window_end");
  if (
    !state.demo
    && !state.recognition
    && !state.companionRelayActive
    && !state.asrPausedForAgentSpeech
    && !state.schedulePaused
  ) {
    startRecognition();
  }
  await sendEvent("companion.window_end", { window: index, completed_seconds: completedBefore + completedNow });
}

async function runScheduledRest(stage, resume = null) {
  if (state.scheduleAbort) return;
  state.scheduledRest = true;
  state.interactionPresenceCountdownStartedMs = 0;
  renderInteractionPresence("monitoring", null, { force: true });
  state.analysisEligible = false;
  state.activeBlockIndex = stage.block;
  state.recognitionWanted = false;
  try { state.recognition?.stop(); } catch (_) { /* Already stopped. */ }
  stopAgentSpeech();
  setAgentState("waiting");
  await setTimingPhase("rest", "A", {
    activeBlockIndex: stage.block,
    analysisEligible: false,
  });
  if (state.recorder?.state === "recording") state.recorder.pause();
  // Keep AudioContext running across rest. Suspending it often leaves the PCM
  // worklet silent after resume, so block 2+ never receives model.prediction.
  // PCM upload is already gated by state.scheduledRest.
  updatePhaseText("标准化休息", `第 ${stage.block}/${stage.totalBlocks} 个六分钟统计块结束`);
  $("#rest-overlay").classList.remove("hidden");
  const completedBefore = Math.min(stage.seconds, Number(resume?.completed_seconds || 0));
  const remainingSeconds = Math.max(0, stage.seconds - completedBefore);
  state.restRemainingSeconds = remainingSeconds;
  pairSend({
    type: "notice",
    action: "rest_start",
    block: stage.block,
    seconds: remainingSeconds,
    total_blocks: stage.totalBlocks,
  });
  if (resume?.started) {
    await sendEvent("rest.resume", {
      block: stage.block,
      completed_seconds: completedBefore,
      excluded_from_analysis: true,
    });
  } else {
    await sendEvent("rest.start", {
      block: stage.block,
      planned_seconds: stage.seconds,
      excluded_from_analysis: true,
    });
  }
  saveScheduleProgress({ started: true, completed_seconds: completedBefore });
  const completedNow = await countdown(
    remainingSeconds,
    $("#rest-countdown"),
    (value) => {
      state.restRemainingSeconds = Math.max(0, remainingSeconds - value);
      saveScheduleProgress({ completed_seconds: completedBefore + value });
    },
  );
  playRestEndChime();
  await sendEvent("rest.end", {
    block: stage.block,
    completed_seconds: completedBefore + completedNow,
    excluded_from_analysis: true,
  });
  state.scheduledRest = false;
  state.restRemainingSeconds = 0;
  pairSend({ type: "notice", action: "rest_end", block: stage.block });
  await ensureTimingCaptureReady("rest_end");
  $("#rest-overlay").classList.add("hidden");
}

function updatePhaseText(label, detail) {
  $("#phase-label").textContent = label;
  $("#phase-detail").textContent = detail;
}

function setPageVisibilityNoticeVisible(visible) {
  const notice = $("#page-visibility-notice");
  if (!notice) return;
  notice.textContent = I18N.t("全程保持电脑实验系统全屏；退出全屏或离开页面共用 30 秒离开时限");
  notice.classList.toggle("hidden", !visible);
}

async function ensureNoticeAudioContext() {
  if (!state.noticeAudioContext) {
    state.noticeAudioContext = new AudioContext();
  }
  if (state.noticeAudioContext.state === "suspended") {
    await state.noticeAudioContext.resume();
  }
  return state.noticeAudioContext;
}

function playRestEndChime() {
  void (async () => {
    try {
      const ctx = await ensureNoticeAudioContext();
      const start = ctx.currentTime;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.07, start + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.55);
      gain.connect(ctx.destination);

      const tone = (frequency, offset, duration) => {
        const oscillator = ctx.createOscillator();
        oscillator.type = "sine";
        oscillator.frequency.setValueAtTime(frequency, start + offset);
        oscillator.connect(gain);
        oscillator.start(start + offset);
        oscillator.stop(start + offset + duration);
      };
      tone(784, 0, 0.16);
      tone(988, 0.14, 0.2);
      queueEvent("rest.chime_played", { kind: "soft_two_tone" });
    } catch (error) {
      queueEvent("rest.chime_failed", { message: error.message });
    }
  })();
}

async function countdown(seconds, target = $("#phase-timer"), onProgress = null) {
  const planned = Math.max(0, Number(seconds) || 0);
  const startedMs = Date.now();
  const deadlineMs = startedMs + planned * 1000;
  let last = performance.now();
  let elapsed = 0;
  while (!state.scheduleAbort) {
    const graceMs = state.pageHiddenSoftPaused ? PAGE_HIDDEN_GRACE_MS : 0;
    const remainingMs = deadlineMs + graceMs - Date.now();
    if (remainingMs <= 0) break;
    if (state.schedulePaused) {
      while (state.schedulePaused && !state.scheduleAbort) {
        const pausedGraceMs = state.pageHiddenSoftPaused ? PAGE_HIDDEN_GRACE_MS : 0;
        if (Date.now() >= deadlineMs + pausedGraceMs) break;
        await sleep(250);
      }
      last = performance.now();
      continue;
    }
    const now = performance.now();
    elapsed += (now - last) / 1000;
    last = now;
    target.textContent = formatTime(Math.max(0, remainingMs / 1000));
    if (onProgress) onProgress(Math.min(planned, Math.max(0, elapsed)));
    await sleep(200);
  }
  return Math.min(planned, Math.max(0, (Date.now() - startedMs) / 1000));
}

const PAGE_HIDDEN_GRACE_MS = 30000;

function activeAttentionViolationReason() {
  if (document.visibilityState !== "visible") return "page_hidden";
  if (!isExperimentFullscreen()) return "fullscreen_exited";
  return null;
}

function clearPageHiddenGraceTimers() {
  clearTimeout(state.pageHiddenGraceTimer);
  clearInterval(state.pageHiddenGraceTickTimer);
  state.pageHiddenGraceTimer = null;
  state.pageHiddenGraceTickTimer = null;
  state.pageHiddenGraceDeadlineMs = 0;
}

function hideVisibilityGraceOverlay() {
  $("#visibility-grace-overlay")?.classList.add("hidden");
}

function renderVisibilityGraceCountdown() {
  const secondsEl = $("#visibility-grace-seconds");
  if (!secondsEl || !state.pageHiddenGraceDeadlineMs) return;
  const remaining = Math.max(0, Math.ceil((state.pageHiddenGraceDeadlineMs - performance.now()) / 1000));
  secondsEl.textContent = String(remaining);
}

function showVisibilityGraceOverlay(reason = state.attentionGraceReason) {
  const overlay = $("#visibility-grace-overlay");
  if (!overlay) return;
  const fullscreenExited = reason === "fullscreen_exited";
  const title = $("#visibility-grace-title");
  const prefix = $("#visibility-grace-prefix");
  const suffix = $("#visibility-grace-suffix");
  if (title) {
    title.textContent = I18N.t(fullscreenExited ? "电脑实验系统已退出全屏" : "页面暂时离开");
  }
  if (prefix) {
    prefix.textContent = I18N.t(
      fullscreenExited ? "退出全屏已计入离开时间。请在" : "检测到实验页不可见。请在",
    );
  }
  if (suffix) suffix.textContent = I18N.t("秒内恢复全屏并返回本页，即可继续当前进度。");
  $("#restore-fullscreen-btn")?.classList.toggle("hidden", !fullscreenExited);
  const helper = overlay.querySelector(".helper-text");
  if (helper) helper.textContent = I18N.t("超过时限后需要从头重新开始。");
  overlay.classList.remove("hidden");
  renderVisibilityGraceCountdown();
}

function beginPageHiddenGrace(reason = "page_hidden") {
  if (!state.scheduleRunning || state.interactionEnding || state.scheduleAbort) return;
  state.attentionGraceReason = reason;
  if (state.pageHiddenSoftPaused) {
    showVisibilityGraceOverlay(reason);
    queueEvent("session.attention_violation_changed", {
      reason,
      visibility: document.visibilityState,
      fullscreen: isExperimentFullscreen(),
      deadline_ms_remaining: Math.max(
        0,
        Math.round(state.pageHiddenGraceDeadlineMs - performance.now()),
      ),
    });
    return;
  }
  state.pageHiddenSoftPaused = true;
  state.schedulePaused = true;
  state.recognitionWanted = false;
  try { state.recognition?.stop(); } catch (_) { /* Already stopped. */ }
  stopAgentSpeech();
  setAgentState("waiting");
  pairSend({ type: "notice", action: "pause", reason });
  queueEvent("session.visibility_grace_start", {
    reason,
    grace_ms: PAGE_HIDDEN_GRACE_MS,
    visibility: document.visibilityState,
    fullscreen: isExperimentFullscreen(),
  });
  state.pageHiddenGraceDeadlineMs = performance.now() + PAGE_HIDDEN_GRACE_MS;
  showVisibilityGraceOverlay();
  clearInterval(state.pageHiddenGraceTickTimer);
  state.pageHiddenGraceTickTimer = setInterval(renderVisibilityGraceCountdown, 250);
  clearTimeout(state.pageHiddenGraceTimer);
  state.pageHiddenGraceTimer = setTimeout(() => {
    state.pageHiddenGraceTimer = null;
    void finalizePageHiddenGraceExpired(state.attentionGraceReason || reason);
  }, PAGE_HIDDEN_GRACE_MS);
}

function cancelPageHiddenGraceAndResume() {
  if (!state.pageHiddenSoftPaused || state.scheduleAbort || state.interactionEnding) return;
  const remainingViolation = activeAttentionViolationReason();
  if (remainingViolation) {
    state.attentionGraceReason = remainingViolation;
    showVisibilityGraceOverlay(remainingViolation);
    return;
  }
  clearPageHiddenGraceTimers();
  state.pageHiddenSoftPaused = false;
  state.attentionGraceReason = null;
  state.schedulePaused = false;
  grantParticipantResponseWindow("attention_resume");
  hideVisibilityGraceOverlay();
  pairSend({ type: "notice", action: "resume" });
  queueEvent("session.visibility_grace_resume", {
    visibility: document.visibilityState,
    fullscreen: isExperimentFullscreen(),
  });
  if (
    !state.demo
    && !state.recognition
    && !state.companionRelayActive
    && !state.asrPausedForAgentSpeech
    && !state.scheduledRest
    && state.scheduleRunning
  ) {
    startRecognition();
  }
}

async function finalizePageHiddenGraceExpired(reason = "page_hidden") {
  if (!state.pageHiddenSoftPaused) return;
  clearPageHiddenGraceTimers();
  state.pageHiddenSoftPaused = false;
  hideVisibilityGraceOverlay();
  queueEvent("session.visibility_grace_expired", {
    reason,
    grace_ms: PAGE_HIDDEN_GRACE_MS,
    visibility: document.visibilityState,
    fullscreen: isExperimentFullscreen(),
  });
  pauseSchedule(reason);
}

function pauseSchedule(reason = "manual") {
  if (!state.scheduleRunning || state.interactionEnding || state.scheduleAbort) return;
  clearPageHiddenGraceTimers();
  clearCompanionDisconnectGrace();
  clearPairRequiredGrace();
  clearCompanionReadyFalseDebounce();
  state.pageHiddenSoftPaused = false;
  state.attentionGraceReason = null;
  hideVisibilityGraceOverlay();
  state.scheduleAbort = true;
  state.schedulePaused = true;
  stopInteractionPresenceMonitor();
  state.recognitionWanted = false;
  try { state.recognition?.stop(); } catch (_) { /* Already stopped. */ }
  stopAgentSpeech();
  setAgentState("waiting");
  const progress = loadScheduleProgress() || {};
  saveScheduleProgress({
    interrupted: true,
    interrupted_reason: reason,
    interrupted_ms: Date.now(),
  });
  pairSend({ type: "notice", action: "pause", reason, terminal: true });
  queueEvent("session.interrupted", {
    reason,
    visibility: document.visibilityState,
    restart_required: false,
    cursor: Number(progress.cursor || 0),
    stage_key: progress.stage_key || null,
    active_elapsed_seconds: Number(progress.active_elapsed_seconds || 0),
    opportunities: Number(progress.opportunities || state.opportunities || 0),
  });
  void returnToSetupAfterInterrupt(reason);
}

function passiveExitReasonMessage(reason) {
  const messages = {
    page_hidden: "实验页面离开或被切换到后台超过 30 秒。",
    fullscreen_exited: "电脑实验系统退出全屏超过 30 秒。",
    timing_socket_closed: "计时与模型网络连接意外中断。",
    timing_model_error: "计时模型发生运行错误。",
    pair_required_by_server: "电脑端与手机端未能保持同时在线。",
    companion_disconnected: "手机陪伴端连接中断且未在允许时间内恢复。",
    companion_tts_server_failed: "手机陪伴智能体语音生成失败。",
    tts_server_failed: "电脑实验系统语音生成失败。",
    participant_inactive: "连续约 30 秒未检测到有效互动，醒目的 30 秒恢复倒计时结束后仍未恢复互动。",
    manual: "实验流程被中断。",
  };
  return messages[reason]
    || (String(reason).includes("pair") || String(reason).includes("companion")
      ? "电脑端与手机端连接中断。"
      : "实验流程发生意外中断。");
}

function showPassiveExitOverlay(reason, { resetDeferred = false } = {}) {
  const overlay = $("#pause-overlay");
  const reasonNode = $("#pause-reason");
  if (!overlay || !reasonNode) return;
  const base = I18N.t(passiveExitReasonMessage(reason));
  const suffix = I18N.t(
    resetDeferred
      ? "当前网络尚未恢复，请检查网络后重新准备。"
      : "请重新完成网络、播放、点击和语音识别检测后再选择下一步。",
  );
  reasonNode.textContent = `${base} ${suffix}`;
  overlay.dataset.reason = String(reason || "unknown");
  overlay.classList.remove("hidden");
  const resumeBtn = $("#resume-btn");
  if (resumeBtn) resumeBtn.textContent = I18N.t("去完成检测");
  resumeBtn?.focus();
}

function dismissPassiveExitOverlay() {
  const overlay = $("#pause-overlay");
  if (!overlay) return;
  overlay.classList.add("hidden");
  delete overlay.dataset.reason;
}

async function returnToSetupAfterInterrupt(reason) {
  if (state.returnToSetupInFlight) return;
  state.returnToSetupInFlight = true;
  clearPageHiddenGraceTimers();
  state.pageHiddenSoftPaused = false;
  state.attentionGraceReason = null;
  hideVisibilityGraceOverlay();
  const restartMessage = passiveExitReasonMessage(reason);

  state.interactionEnding = true;
  state.scheduleAbort = true;
  state.scheduleRunning = false;
  state.schedulePaused = false;
  state.scheduledRest = false;
  state.restRemainingSeconds = 0;
  state.analysisEligible = false;
  stopInteractionPresenceMonitor();
  setPageVisibilityNoticeVisible(false);
  $("#pause-overlay").classList.add("hidden");
  $("#rest-overlay").classList.add("hidden");
  $("#companion-attention-banner").classList.add("hidden");
  state.companionWindowActive = false;
  state.recognitionWanted = false;
  try { state.recognition?.stop(); } catch (_) { /* Already stopped. */ }
  stopAgentSpeech();
  pairSend({ type: "notice", action: "experimental_barge_clear" });
  pairSend({ type: "notice", action: "rest_end" });
  pairSend({ type: "notice", action: "companion_stop" });
  pairSend({ type: "background_state", action: "stop" });
  disablePairReconnect();
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
  clearInterval(state.eventFlushTimer);
  clearInterval(state.demoAudioTimer);
  state.demoAudioTimer = null;
  try { state.workletNode?.disconnect(); } catch (_) { /* Already disconnected. */ }
  state.workletNode = null;
  try { state.timingSocket?.close(); } catch (_) { /* Already closed. */ }
  state.timingSocket = null;
  try {
    await flushEvents();
  } catch (_) {
    // Returning to setup should not trap the participant when the network is unavailable.
  }
  let resetDeferred = false;
  queueEvent("session.returned_to_setup", { reason: reason || null, keep_checkpoint: true });
  state.pendingInterruptChoice = true;
  state.interruptContinueChosen = false;
  forceFreshQualityChecks();
  state.pairReconnectEnabled = true;
  connectPairSocket();
  startCompanionLivePolling();
  showView("setup-view");
  markStep("network");
  const codeHint = state.bootstrap?.participant?.code_hint;
  setConnection(codeHint ? `编号尾号 ${codeHint}` : "尚未连接", codeHint ? "online" : "offline");
  if (restartMessage) {
    $("#start-error").textContent = `${I18N.t(restartMessage)} ${I18N.t("请重新完成网络、播放、点击和语音识别检测后再选择下一步。")}`;
  }
  state.interactionEnding = false;
  state.returnToSetupInFlight = false;
  updateStartEligibility();
  showPassiveExitOverlay(reason, { resetDeferred });
}

async function waitWhilePaused() {
  while (state.schedulePaused && !state.scheduleAbort) await sleep(250);
}

$("#resume-btn").addEventListener("click", dismissPassiveExitOverlay);

function reevaluateAttentionGrace(source = "attention_change") {
  if (!state.scheduleRunning || state.interactionEnding) return;
  const reason = activeAttentionViolationReason();
  queueEvent("session.attention_state", {
    source,
    reason,
    visibility: document.visibilityState,
    fullscreen: isExperimentFullscreen(),
  });
  if (reason) beginPageHiddenGrace(reason);
  else cancelPageHiddenGraceAndResume();
}

document.addEventListener("visibilitychange", () => {
  queueEvent("page.visibility", { state: document.visibilityState });
  reevaluateAttentionGrace("visibilitychange");
});

document.addEventListener("fullscreenchange", () => {
  reevaluateAttentionGrace("fullscreenchange");
});

document.addEventListener("webkitfullscreenchange", () => {
  reevaluateAttentionGrace("webkitfullscreenchange");
});

$("#restore-fullscreen-btn")?.addEventListener("click", async () => {
  const button = $("#restore-fullscreen-btn");
  button.disabled = true;
  try {
    await requestExperimentFullscreen();
    reevaluateAttentionGrace("restore_fullscreen_button");
  } catch (_) {
    $("#visibility-grace-prefix").textContent = I18N.t("仍未进入全屏。请允许浏览器全屏，并在");
  } finally {
    button.disabled = false;
  }
});

window.addEventListener("pageshow", (event) => {
  if (!event.persisted || !state.scheduleRunning || state.interactionEnding) return;
  reevaluateAttentionGrace("pageshow");
});

$("#end-early-btn").addEventListener("click", async () => {
  if (!confirm("确定提前结束本次实验吗？已完成的数据仍会保存。")) return;
  queueEvent("session.ended_early", {});
  await finishInteraction(true);
});

async function finishInteraction(endedEarly = false) {
  if (!state.scheduleRunning) return;
  state.interactionEnding = true;
  saveScheduleProgress({
    cursor: scheduleStages().length,
    stage_key: "questionnaire",
    started: false,
    interaction_complete: true,
    ended_early: endedEarly,
  });
  state.scheduleRunning = false;
  state.scheduleAbort = true;
  state.scheduledRest = false;
  state.restRemainingSeconds = 0;
  state.analysisEligible = false;
  stopInteractionPresenceMonitor();
  setPageVisibilityNoticeVisible(false);
  $("#rest-overlay").classList.add("hidden");
  $("#companion-attention-banner")?.classList.add("hidden");
  state.companionWindowActive = false;
  state.recognitionWanted = false;
  try { state.recognition?.stop(); } catch (_) { /* Already stopped. */ }
  stopAgentSpeech();
  pairSend({ type: "notice", action: "experimental_barge_clear" });
  pairSend({ type: "notice", action: "rest_end" });
  pairSend({ type: "notice", action: "companion_stop" });
  pairSend({ type: "background_state", action: "stop" });
  disablePairReconnect();
  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
  clearInterval(state.eventFlushTimer);
  clearInterval(state.demoAudioTimer);
  state.demoAudioTimer = null;
  try { state.workletNode?.disconnect(); } catch (_) { /* Already disconnected. */ }
  state.workletNode = null;
  try { state.timingSocket?.close(); } catch (_) { /* Already closed. */ }
  state.timingSocket = null;
  await flushEvents();
  if (endedEarly) {
    state.pendingCloseout = { endedEarly: true };
    try {
      await completeDay(true);
      state.pendingCloseout = null;
    } catch (error) {
      $("#questionnaire-title").textContent = "正在确认提前结束";
      $("#questionnaire-instructions").textContent = "无需填写问卷。请保持页面打开并重试结束确认。";
      $("#questionnaire-items").replaceChildren();
      showView("questionnaire-view");
      showCloseoutFailure(error);
    }
    return;
  }
  try {
    await enterFinalQuestionnaire();
    await showQuestionnaire("final");
  } catch (error) {
    state.pendingFinalQuestionnaireEntry = true;
    state.pendingCloseout = { endedEarly: false, enterFinalQuestionnaire: true };
    $("#questionnaire-title").textContent = "正在安全保存模型并进入最终问卷";
    $("#questionnaire-instructions").textContent = "实验位会在模型安全保存后自动释放。请保持页面打开并重试，不要关闭页面。";
    $("#questionnaire-items").replaceChildren();
    $("#questionnaire-submit").disabled = true;
    showView("questionnaire-view");
    error.closeoutFailed = true;
    error.closeoutErrorCode = error.detail?.code || "final_questionnaire_entry_failed";
    showCloseoutFailure(error);
  }
}

async function enterFinalQuestionnaire() {
  await flushEvents();
  const result = await api(`/api/sessions/${state.sessionId}/enter-final-questionnaire`, {
    method: "POST",
  });
  state.finalQuestionnaireRecovery = {
    session_id: state.sessionId,
    credential: result.credential,
    entered_ms: result.entered_ms,
    model_finalized_ms: result.model_ready ? Date.now() : null,
    chain_committed_ms: result.model_ready ? Date.now() : null,
    final_submitted_ms: null,
    pending: true,
  };
  state.pendingFinalQuestionnaireEntry = false;
  return result;
}

async function showQuestionnaire(instrument, extraAnswers = {}, options = {}) {
  const questionnaire = await api(`/api/questionnaires/${instrument}`);
  const presentationOrder = questionnairePresentationOrder(instrument, questionnaire);
  state.currentQuestionnaire = {
    instrument,
    questionnaire,
    extraAnswers,
    options,
    presentationOrder,
  };
  hideCloseoutFailure();
  const credentialCard = $("#final-questionnaire-credential-card");
  const credential = state.finalQuestionnaireRecovery?.credential
    || state.bootstrap?.final_questionnaire_recovery?.credential;
  credentialCard?.classList.toggle("hidden", instrument !== "final" || !credential);
  if (instrument === "final" && credential) {
    $("#final-questionnaire-credential").textContent = displayedFinalCredential(credential);
  }
  $("#questionnaire-submit").disabled = false;
  $("#questionnaire-title").textContent = questionnaire.title;
  $("#questionnaire-instructions").textContent = questionnaire.instructions;
  const container = $("#questionnaire-items");
  container.replaceChildren();
  const entries = questionnaireEntries(questionnaire);
  for (const key of presentationOrder) {
    const entry = entries.get(key);
    if (!entry) continue;
    if (entry.kind === "item") {
      container.appendChild(renderLikertItem(entry.value, questionnaire.scale));
    } else if (entry.kind === "field") {
      container.appendChild(renderChoiceField(entry.value));
    } else {
      container.appendChild(renderOpenItem(entry.value));
    }
  }
  for (const followup of questionnaire.followups || []) {
    const row = renderOpenItem({ ...followup, required: false });
    row.id = `followup-${followup.id}`;
    row.classList.add("conditional-question-item", "hidden");
    row.setAttribute("aria-hidden", "true");
    const textarea = row.querySelector("textarea");
    textarea.disabled = true;
    container.appendChild(row);
    for (const input of container.querySelectorAll(`[name="${followup.when.item}"]`)) {
      input.addEventListener("change", () => updateConditionalFollowups(questionnaire));
    }
  }
  updateConditionalFollowups(questionnaire);
  showView("questionnaire-view");
}

$("#copy-final-questionnaire-credential")?.addEventListener("click", async () => {
  const credential = state.finalQuestionnaireRecovery?.credential
    || state.bootstrap?.final_questionnaire_recovery?.credential;
  const status = $("#copy-final-questionnaire-credential-status");
  if (!credential) return;
  try {
    await navigator.clipboard.writeText(displayedFinalCredential(credential));
    status.textContent = "凭据已复制。";
  } catch (_) {
    status.textContent = `请手动保存：${displayedFinalCredential(credential)}`;
  }
});

function questionnaireFollowupApplies(followup, answers = null) {
  const rawValue = answers
    ? answers[followup.when.item]
    : document.querySelector(`[name="${followup.when.item}"]:checked`)?.value;
  const value = Number(rawValue);
  if (!Number.isFinite(value)) return false;
  return (followup.when.lte !== undefined && value <= Number(followup.when.lte))
    || (followup.when.gte !== undefined && value >= Number(followup.when.gte));
}

function updateConditionalFollowups(questionnaire) {
  for (const followup of questionnaire.followups || []) {
    const row = document.getElementById(`followup-${followup.id}`);
    const textarea = row?.querySelector("textarea");
    if (!row || !textarea) continue;
    const applies = questionnaireFollowupApplies(followup);
    row.classList.toggle("hidden", !applies);
    row.setAttribute("aria-hidden", applies ? "false" : "true");
    textarea.disabled = !applies;
    textarea.required = applies;
  }
}

function questionnaireEntries(questionnaire) {
  const entries = new Map();
  for (const item of questionnaire.items || []) {
    entries.set(`item:${item.id}`, { kind: "item", value: item });
  }
  for (const field of questionnaire.fields || []) {
    entries.set(`field:${field.id}`, { kind: "field", value: field });
  }
  for (const item of questionnaire.open_items || []) {
    entries.set(`open:${item.id}`, { kind: "open", value: item });
  }
  return entries;
}

function questionnaireSeed(text) {
  let value = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    value ^= text.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

function seededQuestionnaireRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let mixed = value;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function questionnairePresentationOrder(instrument, questionnaire) {
  const order = [...questionnaireEntries(questionnaire).keys()];
  const responseInstrument = instrument === "daily" ? `daily_day_${state.day}` : instrument;
  const participantKey = state.bootstrap?.participant?.neutral_id
    || state.bootstrap?.participant?.code_hint
    || "anonymous";
  const random = seededQuestionnaireRandom(questionnaireSeed(
    `${participantKey}|${questionnaire.version}|${responseInstrument}`,
  ));
  for (let index = order.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    [order[index], order[target]] = [order[target], order[index]];
  }
  return order;
}

function closeQuestionTipPanels(except = null) {
  for (const panel of document.querySelectorAll(".question-tip-panel")) {
    if (except && panel === except) continue;
    panel.classList.add("hidden");
    const button = panel.parentElement?.querySelector?.(`.question-tip-btn[aria-controls="${panel.id}"]`);
    if (button) button.setAttribute("aria-expanded", "false");
  }
}

document.addEventListener("click", (event) => {
  if (event.target.closest(".question-tip-btn") || event.target.closest(".question-tip-panel")) return;
  closeQuestionTipPanels();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeQuestionTipPanels();
});

function renderChoiceField(field) {
  const fieldset = document.createElement("fieldset");
  fieldset.className = "question-item";
  const legend = document.createElement("legend");
  legend.className = "question-item-legend";
  const title = document.createElement("span");
  title.className = "question-item-title";
  title.textContent = field.text;
  legend.appendChild(title);
  if (field.helper) {
    const tipBtn = document.createElement("button");
    tipBtn.type = "button";
    tipBtn.className = "question-tip-btn";
    tipBtn.textContent = I18N.t("查看提示");
    tipBtn.setAttribute("aria-expanded", "false");
    tipBtn.setAttribute("aria-label", I18N.t("查看提示"));
    const tipId = `question-tip-${field.id}`;
    tipBtn.setAttribute("aria-controls", tipId);
    legend.appendChild(tipBtn);
    fieldset.appendChild(legend);
    const panel = document.createElement("div");
    panel.id = tipId;
    panel.className = "question-tip-panel hidden";
    panel.setAttribute("role", "note");
    panel.textContent = field.helper;
    tipBtn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const willOpen = panel.classList.contains("hidden");
      closeQuestionTipPanels(panel);
      panel.classList.toggle("hidden", !willOpen);
      tipBtn.setAttribute("aria-expanded", willOpen ? "true" : "false");
    });
    fieldset.appendChild(panel);
  } else {
    fieldset.appendChild(legend);
  }
  const choices = document.createElement("div");
  choices.className = "check-stack";
  for (const option of field.options || []) {
    const label = document.createElement("label");
    label.className = "check-row";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = field.id;
    input.value = option.value;
    input.required = Boolean(field.required);
    label.append(input, document.createTextNode(option.label));
    choices.appendChild(label);
  }
  fieldset.appendChild(choices);
  return fieldset;
}

function renderLikertItem(item, scale) {
  const fieldset = document.createElement("fieldset");
  fieldset.className = "question-item";
  const legend = document.createElement("legend");
  legend.textContent = item.text;
  fieldset.appendChild(legend);
  const row = document.createElement("div");
  row.className = "likert";
  for (let value = scale.min; value <= scale.max; value += 1) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = item.id;
    input.value = String(value);
    input.required = true;
    label.append(input, document.createTextNode(String(value)));
    row.appendChild(label);
  }
  const labels = document.createElement("div");
  labels.className = "scale-labels";
  labels.innerHTML = `<span>${scale.min_label}</span><span>${scale.max_label}</span>`;
  fieldset.append(row, labels);
  return fieldset;
}

const OPEN_TEXT_MIN_CHARS = 20;
const OPEN_TEXT_MIN_WORDS = 12;

function countOpenTextCjkChars(text) {
  return (String(text || "").match(/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/g) || []).length;
}

function countOpenTextEnglishWords(text) {
  return (String(text || "").match(/[A-Za-z]+(?:['’][A-Za-z]+)?/g) || []).length;
}

function openTextMeetsMinimum(text) {
  const value = String(text || "").trim();
  if (!value) return false;
  return countOpenTextCjkChars(value) >= OPEN_TEXT_MIN_CHARS
    || countOpenTextEnglishWords(value) >= OPEN_TEXT_MIN_WORDS;
}

function renderOpenItem(item) {
  const row = document.createElement("div");
  row.className = "question-item";
  const label = document.createElement("label");
  label.className = "field-label";
  label.htmlFor = `open-${item.id}`;
  label.textContent = item.text;
  const textarea = document.createElement("textarea");
  textarea.id = `open-${item.id}`;
  textarea.name = item.id;
  textarea.required = Boolean(item.required);
  const helper = document.createElement("p");
  helper.className = "helper-text open-item-min-hint";
  helper.textContent = I18N.t("每个填空至少填写 20 个汉字，或至少 12 个英文单词。");
  row.append(label, textarea, helper);
  return row;
}

$("#questionnaire-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.questionnaireSubmitInFlight || state.pendingCloseout) return;
  const error = $("#questionnaire-error");
  error.textContent = "";
  hideCloseoutFailure();
  const {
    instrument,
    questionnaire,
    extraAnswers,
    options,
    presentationOrder,
  } = state.currentQuestionnaire;
  if (!state.sessionId) {
    error.textContent = I18N.isEnglish
      ? "This questionnaire is not yet linked to the computer study session. Refresh the page and try again."
      : "问卷尚未绑定电脑端实验会话。请刷新页面后重试。";
    return;
  }
  const data = new FormData(event.target);
  const answers = { ...extraAnswers };
  for (const [key, value] of data.entries()) answers[key] = /^\d+$/.test(value) ? Number(value) : value;

  if (instrument === "final") {
    for (const openItem of questionnaire.open_items || []) {
      const value = String(answers[openItem.id] || "").trim();
      answers[openItem.id] = value;
      if (!value) {
        if (openItem.required) {
          error.textContent = I18N.t("请完整填写所有必答填空题后再提交。");
          return;
        }
        continue;
      }
      if (!openTextMeetsMinimum(value)) {
        error.textContent = I18N.t(
          "填空题每个空至少填写 20 个汉字，或至少 12 个英文单词。请补充后再提交。",
        );
        document.getElementById(`open-${openItem.id}`)?.focus();
        return;
      }
    }
    for (const followup of questionnaire.followups || []) {
      const applies = questionnaireFollowupApplies(followup, answers);
      if (!applies) {
        delete answers[followup.id];
        continue;
      }
      const response = String(answers[followup.id] || "").trim();
      answers[followup.id] = response;
      if (!response) {
        error.textContent = "请完成当前评分触发的补充问题后再提交。";
        document.getElementById(`open-${followup.id}`)?.focus();
        return;
      }
      if (!openTextMeetsMinimum(response)) {
        error.textContent = I18N.t(
          "填空题每个空至少填写 20 个汉字，或至少 12 个英文单词。请补充后再提交。",
        );
        document.getElementById(`open-${followup.id}`)?.focus();
        return;
      }
    }
  }

  state.questionnaireSubmitInFlight = true;
  $("#questionnaire-submit").disabled = true;
  try {
    const name = instrument === "daily" ? `daily_day_${state.day}` : instrument;
    await api("/api/responses", {
      method: "POST",
      body: {
        session_id: state.sessionId,
        instrument: name,
        version: questionnaire.version,
        answers,
        locale: I18N.locale,
        question_order: presentationOrder,
      },
    });
    if (instrument === "baseline") {
      state.bootstrap.progress.responses.push({ instrument: "baseline", version: questionnaire.version });
      if (options.afterSubmit) options.afterSubmit();
    } else if (instrument === "daily" && state.day === 3 && !extraAnswers.ended_early) {
      await showQuestionnaire("final");
    } else {
      if (instrument === "final") markFinalQuestionnaireSubmitted(Date.now());
      state.pendingCloseout = { endedEarly: Boolean(extraAnswers.ended_early) };
      await completeDay(state.pendingCloseout.endedEarly);
      state.pendingCloseout = null;
    }
  } catch (err) {
    if (err.closeoutFailed) showCloseoutFailure(err);
    else error.textContent = `提交失败：${err.message}`;
  } finally {
    state.questionnaireSubmitInFlight = false;
    if (!state.pendingCloseout) $("#questionnaire-submit").disabled = false;
  }
});

async function completeDay(endedEarly = false) {
  await flushEvents();
  const endpoint = endedEarly ? "abort" : "end";
  let closeout = null;
  try {
    closeout = await api(`/api/sessions/${state.sessionId}/${endpoint}`, { method: "POST" });
    if (!endedEarly && closeout?.end_to_end_complete !== true) {
      const error = new Error("服务器未返回端到端完成确认，请重试完成确认。");
      error.detail = { code: "end_to_end_completion_receipt_missing" };
      throw error;
    }
  } catch (error) {
    error.closeoutFailed = true;
    error.closeoutEndpoint = endpoint;
    error.closeoutErrorCode = error.detail?.code
      || `session_${endpoint}_http_${Number(error.status) || "network"}`;
    throw error;
  }
  state.endToEndCompletionReceipt = endedEarly
    ? null
    : closeout.completion_receipt;
  if (!endedEarly) {
    markFinalQuestionnaireSubmitted(closeout?.completion_receipt?.final_response_ms);
  }
  clearScheduleProgress();
  state.timingSocket?.close();
  stopPairConnection();
  if (endedEarly) {
    setConnection("实验已退出，正在返回开始界面", "warn");
    localStorage.removeItem(storageKey("study_token"));
    localStorage.removeItem(storageKey("study_code"));
    state.token = "";
    state.code = "";
    window.location.reload();
    return;
  }
  setConnection("实验完成", "online");
  renderCompletion(false);
}

function markFinalQuestionnaireSubmitted(submittedMs) {
  const ms = Number(submittedMs) || Date.now();
  const recovery = state.bootstrap?.final_questionnaire_recovery;
  if (recovery) {
    recovery.pending = false;
    recovery.final_submitted_ms = Number(recovery.final_submitted_ms) || ms;
  }
  if (state.finalQuestionnaireRecovery) {
    state.finalQuestionnaireRecovery.pending = false;
    state.finalQuestionnaireRecovery.final_submitted_ms =
      Number(state.finalQuestionnaireRecovery.final_submitted_ms) || ms;
  }
  if (state.endToEndCompletionReceipt && !Number(state.endToEndCompletionReceipt.final_response_ms)) {
    state.endToEndCompletionReceipt.final_response_ms = ms;
  }
}

function finalQuestionnaireAlreadySubmitted() {
  const receipt = state.endToEndCompletionReceipt || state.bootstrap?.completion_receipt;
  if (Number(receipt?.final_response_ms) > 0) return true;
  const recovery = state.finalQuestionnaireRecovery || state.bootstrap?.final_questionnaire_recovery;
  return Number(recovery?.final_submitted_ms) > 0;
}

function renderCompletion(endedEarly = false) {
  const endToEndComplete = !endedEarly && Boolean(state.endToEndCompletionReceipt);
  const questionnaireSubmitted = !endedEarly && finalQuestionnaireAlreadySubmitted();
  $("#complete-view").dataset.endToEndComplete = String(endToEndComplete);
  $("#complete-view").dataset.finalQuestionnaireSubmitted = String(questionnaireSubmitted);
  $("#complete-title").textContent = endedEarly
    ? "本次实验已提前结束"
    : questionnaireSubmitted
      ? "本次实验已完成"
      : "计时实验已完成，最终问卷待提交";
  $("#complete-copy").textContent = endedEarly
    ? "已完成的数据已经保存。请联系研究团队确认按正式规则应得的补偿。"
    : questionnaireSubmitted
      ? "最终问卷与实验数据已提交。请按下方说明领取40元实验报酬。"
      : "计时实验数据已保存，但最终问卷尚未提交。请使用首页的受试者凭据继续填写最终问卷后再领取报酬。";
  $("#completion-code").textContent = endedEarly
    ? `本次记录不会标记为完整完成，请联系研究团队 · 编号尾号 ${state.bootstrap.participant.code_hint}`
    : questionnaireSubmitted
      ? `完成记录 · 编号尾号 ${state.bootstrap.participant.code_hint}`
      : `计时已完成，最终问卷待提交 · 编号尾号 ${state.bootstrap.participant.code_hint}`;
  $("#compensation-code-hint").textContent =
    state.bootstrap.participant.code_hint;
  const codeHint = state.bootstrap.participant.code_hint;
  $("#compensation-claim-detail").textContent = I18N.isEnglish
    ? (
      state.bootstrap.compensation_registration_decision === "decline"
        ? `Send participant-code suffix ${codeHint}. You chose not to submit identity details; the research team will arrange an alternative payment method. Do not send an identity-card photograph through WeChat.`
        : `Send participant-code suffix ${codeHint}. Do not send an identity-card photograph through WeChat; the submitted details are stored in a separate encrypted reimbursement vault.`
    )
    : (
      state.bootstrap.compensation_registration_decision === "decline"
        ? `请发送你的参与者编号尾号 ${codeHint}。你已选择不提交身份信息；研究团队将安排替代发放方式。不要通过微信发送身份证照片。`
        : `请发送你的参与者编号尾号 ${codeHint}。不要通过微信发送身份证照片；已提交的信息保存在独立加密报销库中。`
    );
  $("#compensation-claim").classList.toggle("hidden", endedEarly || !questionnaireSubmitted);
  $("#download-portable-bundle-btn").textContent = I18N.isEnglish
    ? "Download my personal model and memory bundle"
    : "下载我的个人模型与记忆数据包";
  showView("complete-view");
}

$("#copy-wechat-btn").addEventListener("click", async () => {
  const status = $("#copy-wechat-status");
  try {
    await navigator.clipboard.writeText("study-contact");
    status.textContent = I18N.t("微信号已复制。");
  } catch (_) {
    status.textContent = I18N.t("请手动复制微信号：study-contact（微信名：Research team）");
  }
});

$("#download-portable-bundle-btn").addEventListener("click", async () => {
  const button = $("#download-portable-bundle-btn");
  const status = $("#portable-bundle-status");
  button.disabled = true;
  status.textContent = I18N.isEnglish
    ? "Preparing your verified personal bundle…"
    : "正在生成并校验你的个人数据包…";
  try {
    const response = await fetch("api/participants/me/portable-bundle", {
      headers: { Authorization: `Bearer ${state.token}` },
      cache: "no-store",
    });
    if (!response.ok) {
      let detail = null;
      try { detail = await response.json(); } catch (_) {}
      throw new Error(
        detail?.detail?.message
        || detail?.detail
        || (I18N.isEnglish ? "Download failed" : "下载失败")
      );
    }
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/i);
    const filename = match?.[1] || "tap-to-adapt-personal.zip";
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
    status.textContent = I18N.isEnglish
      ? "Download complete. Import this ZIP in the local Tap-to-Adapt project."
      : "下载完成。可在 Tap-to-Adapt 本地项目中直接导入这个 ZIP。";
  } catch (error) {
    status.textContent = error.message || (I18N.isEnglish ? "Download failed" : "下载失败");
  } finally {
    button.disabled = false;
  }
});

$("#companion-url-copy").addEventListener("click", async () => {
  const status = $("#companion-url-copy-status");
  const url = $("#companion-url-copy").dataset.url || $("#companion-url").href || "";
  if (!url || url === "#") return;
  status.classList.remove("hidden");
  try {
    await navigator.clipboard.writeText(url);
    status.textContent = I18N.t("链接已复制。");
  } catch (_) {
    status.textContent = I18N.t("请手动复制上方链接。");
  }
});

function showCloseoutFailure(error) {
  const status = $("#closeout-status");
  const code = error.closeoutErrorCode || "session_closeout_failed";
  status.classList.remove("hidden");
  status.dataset.state = "closeout_failed";
  status.dataset.errorCode = code;
  status.dataset.httpStatus = String(error.status || "");
  $("#closeout-error-code").textContent = code;
  $("#closeout-error-detail").textContent = error.message;
  $("#questionnaire-form").dataset.state = "closeout_failed";
  $("#questionnaire-submit").disabled = true;
  $("#retry-closeout-btn").disabled = false;
}

function hideCloseoutFailure() {
  const status = $("#closeout-status");
  status.classList.add("hidden");
  status.dataset.state = "";
  status.dataset.errorCode = "";
  status.dataset.httpStatus = "";
  $("#questionnaire-form").dataset.state = "";
}

$("#retry-closeout-btn").addEventListener("click", async () => {
  if (!state.pendingCloseout) return;
  const button = $("#retry-closeout-btn");
  const status = $("#closeout-status");
  button.disabled = true;
  status.dataset.state = "retrying_closeout";
  $("#closeout-error-detail").textContent = "正在重新确认本次实验完成状态，请保持页面打开。";
  try {
    if (state.pendingCloseout.enterFinalQuestionnaire) {
      await enterFinalQuestionnaire();
      state.pendingCloseout = null;
      hideCloseoutFailure();
      await showQuestionnaire("final");
      return;
    }
    await completeDay(state.pendingCloseout.endedEarly);
    state.pendingCloseout = null;
    hideCloseoutFailure();
  } catch (error) {
    showCloseoutFailure(error);
  } finally {
    if (state.pendingCloseout) button.disabled = false;
  }
});

function queueEvent(eventType, payload) {
  if (!state.sessionId) return;
  const training = payload?.training === true;
  state.eventQueue.push({
    event_type: eventType,
    client_ms: Date.now(),
    monotonic_ms: performance.now(),
    payload: {
      ...(payload || {}),
      active_block_index: state.activeBlockIndex,
      analysis_eligible: Boolean(
        !training && state.analysisEligible && !state.scheduledRest
      ),
      scheduled_rest: state.scheduledRest,
    },
  });
  if (state.eventQueue.length >= 80) flushEvents();
}

function sendEvent(eventType, payload) {
  queueEvent(eventType, payload);
  return flushEvents(true);
}

async function postEventBatch(events) {
  try {
    await api("/api/events/batch", {
      method: "POST",
      body: { session_id: state.sessionId, events },
    });
  } catch (err) {
    if (err.status === 413 && events.length > 1) {
      const mid = Math.ceil(events.length / 2);
      await postEventBatch(events.slice(0, mid));
      await postEventBatch(events.slice(mid));
      return;
    }
    throw err;
  }
}

async function flushEvents(throwOnError = false) {
  if (!state.eventQueue.length || !state.sessionId) return;
  if (state.eventFlushInFlight) {
    await state.eventFlushInFlight;
    if (state.eventQueue.length) return flushEvents(throwOnError);
    if (throwOnError && state.lastEventFlushError) {
      const error = state.lastEventFlushError;
      state.lastEventFlushError = null;
      throw error;
    }
    return;
  }
  const pending = state.eventQueue.splice(0, state.eventQueue.length);
  const flush = async () => {
    while (pending.length && state.sessionId) {
      const events = pending.splice(0, 100);
      try {
        await postEventBatch(events);
      } catch (err) {
        state.eventQueue.unshift(...events, ...pending);
        pending.length = 0;
        state.lastEventFlushError = err;
        break;
      }
    }
  };
  state.eventFlushInFlight = flush();
  try {
    await state.eventFlushInFlight;
  } finally {
    state.eventFlushInFlight = null;
  }
  if (throwOnError && state.lastEventFlushError) {
    const error = state.lastEventFlushError;
    state.lastEventFlushError = null;
    throw error;
  }
  if (state.eventQueue.length) void flushEvents(false);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function withTimeout(promise, milliseconds, message) {
  return Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(message)), milliseconds)),
  ]);
}

function percentile(values, fraction) {
  if (!values.length) return Infinity;
  const index = Math.min(values.length - 1, Math.max(0, Math.ceil(values.length * fraction) - 1));
  return values[index];
}

function round(value, digits = 0) {
  if (!Number.isFinite(value)) return "∞";
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function formatTime(seconds) {
  const total = Math.max(0, Math.ceil(seconds));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

async function initializeApplication() {
  bindStaticAssetSources();
  const enrollSubmit = $("#enroll-submit");
  if (!enrollSubmit) throw new Error("public enrollment control is missing");
  const deviceGate = $("#study-device-gate");
  if (CLIENT_DEVICE_CLASS !== "desktop") {
    $("#study-device-gate-title").textContent = "无法开始，需要在电脑端打开实验系统";
    $("#study-device-gate-message").textContent = "当前设备不是可用于主实验的笔记本或台式电脑。手机和平板不能作为主实验设备。";
    document.querySelector("main")?.setAttribute("inert", "");
    setEnrollButtonsDisabled(true);
    document.documentElement.dataset.appReady = "device-blocked";
    return;
  }
  deviceGate.classList.add("hidden");
  const isTest = APP_BASE.includes("tap-test");
  $("#instance-notice").textContent = isTest
    ? I18N.t("当前为测试系统：测试数据与正式实验完全隔离，不进入正式分析。")
    : I18N.t("当前为正式实验入口。开始前请完整阅读知情同意和报销信息说明。");
  try {
    const health = await api("/health");
    applyPublicEnrollmentClosed(health?.public_enrollment === false);
  } catch (_) {
    // Keep the stored-session / resume paths usable if health is briefly down.
  }
  if (state.token) {
    try {
      await withTimeout(loadBootstrap(), 10000, "saved session bootstrap timed out");
    } catch (_) {
      localStorage.removeItem(storageKey("study_token"));
      localStorage.removeItem(storageKey("study_code"));
      state.token = "";
      state.code = "";
      setLogoutVisible(false);
      showView("login-view");
    }
  }
  bindDataCollectionControls();
  try {
    await refreshCapacityStatus();
  } catch (_) {
    // Capacity checks are advisory before login; keep the page usable offline.
  }
  applyGroupSessionPassword(sessionStorage.getItem(storageKey("group_session_password")) || "");
  try {
    await refreshSessionLock();
  } catch (_) {
    // Lock status is advisory before login; keep the page usable offline.
  }
  if (!(state.capacity && state.capacity.ok === false && !state.capacity.holding_slot)) {
    applyEnrollmentAvailability();
  }
  startCapacityPolling();
  document.documentElement.dataset.appReady = "true";
  window.dispatchEvent(new CustomEvent("tap-study:ready"));
}

initializeApplication().catch((error) => {
  document.documentElement.dataset.appReady = "error";
  console.error("Tap-to-Adapt application initialization failed", error);
});
