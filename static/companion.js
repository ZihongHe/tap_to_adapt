/**
 * @file Mobile companion controller: device readiness, pairing, push-to-talk input, speech playback, and background-audio scheduling. Mobile push-to-talk delimits dialogue input; desktop taps provide timing corrections.
 */

"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function resolveAppBase() {
  try {
    const script = document.querySelector('script[src*="static/companion.js"]');
    if (script?.src) {
      const path = new URL(script.src, location.href).pathname.replace(/\/static\/companion\.js$/i, "");
      if (path) return path;
    }
  } catch (_) {
    // Fall through.
  }
  try {
    const baseHref = document.querySelector("base")?.href;
    if (baseHref) {
      const path = new URL(baseHref, location.href).pathname.replace(/\/$/, "");
      if (path) return path;
    }
  } catch (_) {
    // Fall through.
  }
  const match = location.pathname.match(/^(.*?)\/companion(?:\/|$)/i);
  return match ? match[1] : "";
}

const APP_BASE = resolveAppBase();
const I18N = window.TTA_I18N || {
  locale: "zh-CN",
  isEnglish: false,
  t: (value) => value,
  apiPath: (value) => value,
};
const COMPANION_TOKEN_KEY = `tta:${APP_BASE || "/"}:companion_token`;
const COMPANION_OWNER_KEY = `tta:${APP_BASE || "/"}:companion_owner`;
const COMPANION_TAB_ID = (
  window.crypto?.randomUUID?.()
  || `tab-${Date.now()}-${Math.random().toString(16).slice(2)}`
);

function classifyClientDevice() {
  const userAgent = navigator.userAgent || "";
  const clientHintsMobile = navigator.userAgentData?.mobile === true;
  const pcWechat = /WindowsWechat|MacWechat/i.test(userAgent);
  const ipadDesktopMode = /Macintosh/i.test(userAgent) && navigator.maxTouchPoints > 1;
  const explicitIpad = /iPad/i.test(userAgent) || ipadDesktopMode;
  // Mobile WeChat / in-app browsers: treat as phone even when UA omits "Mobile"
  // (common on some Android WeChat builds). PC WeChat stays blocked.
  if (/MicroMessenger/i.test(userAgent) && !pcWechat && !explicitIpad) {
    return "phone";
  }
  const wechatOrInAppMobile = /MicroMessenger|MQQBrowser|UCBrowser|baidubrowser/i.test(userAgent)
    && /Mobile|iPhone|iPod|Android/i.test(userAgent)
    && !pcWechat;
  const tablet = explicitIpad || /Tablet|Kindle|Silk|PlayBook/i.test(userAgent)
    || (/Android/i.test(userAgent) && !/Mobile/i.test(userAgent) && !/MicroMessenger/i.test(userAgent));
  const phone = !tablet && (
    clientHintsMobile
    || wechatOrInAppMobile
    || /iPhone|iPod|Windows Phone|IEMobile|Opera Mini|Android.+Mobile|Mobile Safari/i.test(userAgent)
  );
  if (tablet) return "tablet";
  if (phone) return "phone";
  if (window.matchMedia?.("(any-pointer: fine)").matches) return "desktop";
  return "unknown";
}

const CLIENT_DEVICE_CLASS = classifyClientDevice();
// Assigned after DOM-ready deferred load; helpers close over this binding.
let phoneLoginSubmit = null;

const state = {
  token: sessionStorage.getItem(COMPANION_TOKEN_KEY) || "",
  code: "",
  day: 1,
  bootstrap: null,
  sessionId: null,
  socket: null,
  context: null,
  backgroundAudio: $("#background-audio"),
  backgroundSource: null,
  backgroundGain: null,
  speechGain: null,
  companionWindow: false,
  phonePrimaryCountdownTimer: null,
  phonePrimaryDeadlineMs: 0,
  phonePrimaryPausedRemainingMs: null,
  phonePrimaryCountdownPaused: false,
  backgroundShouldPlay: false,
  backgroundPlaybackHeld: false,
  backgroundRestHeld: false,
  backgroundPttHeld: false,
  backgroundRecoveryTimer: null,
  backgroundLoadedUrl: "",
  pairMessageChain: Promise.resolve(),
  recognition: null,
  ttsPlaying: false,
  wakeLock: null,
  heartbeat: null,
  listening: false,
  micStream: null,
  phoneMicPassed: false,
  phoneMicMetrics: null,
  phoneQuality: new Map(),
  phoneQualityAttempts: new Map(),
  phoneBrowserProbe: null,
  phoneReadyAnnounced: false,
  companionEnvironment: { status: "pending", ok: false, message: "", reason: "" },
  desktopAsrRelayPending: false,
  desktopAsrRelayTimer: null,
  companionInviteTimer: null,
  companionInviteTickTimer: null,
  companionInviteDeadlineMs: 0,
  companionInviteUsed: false,
  companionWindowNumber: 0,
  capacity: null,
  capacityPollTimer: null,
  recognitionStartToken: 0,
  recognitionStartPending: false,
  localAsrFinal: "",
  localAsrInterim: "",
  pairSuperseded: false,
  pairClosingByPage: false,
  phaseSyncLastRequestMs: 0,
  // Primary = desktop ASR relay; sticky failover to phone-local ASR after
  // desktop relay errors / timeout / empty transcript in this companion window.
  asrFailoverToPhone: false,
  talkPointerDown: false,
  phoneRestDeadlineMs: 0,
  phoneRestPausedRemainingMs: null,
  phoneRestCountdownPaused: false,
  phoneRestCountdownTimer: null,
  interactionPresenceState: "hidden",
  interactionPresenceSeconds: null,
  interactionPresenceTarget: "primary",
};

function requestStudyPhaseSync(reason = "manual") {
  if (!state.sessionId || state.pairSuperseded || state.pairClosingByPage) return;
  if (state.socket?.readyState !== WebSocket.OPEN) return;
  const now = Date.now();
  // Avoid storms while pair_status heartbeats arrive every few seconds.
  if (now - Number(state.phaseSyncLastRequestMs || 0) < 8000) return;
  state.phaseSyncLastRequestMs = now;
  const restVisible = !$("#phone-rest-overlay")?.classList.contains("hidden");
  const primaryVisible = !$("#phone-primary-overlay")?.classList.contains("hidden");
  send({
    type: "companion_state",
    action: "phase_sync_request",
    expected: state.companionWindow
      ? "companion"
      : (restVisible ? "rest" : (primaryVisible ? "primary" : "any")),
    window: state.companionWindowNumber || 1,
    reason,
  });
}

const COMPANION_PHASE_COPY = {
  waiting: {
    title: "等待电脑端开始",
    copy: "背景声将在电脑端开始实验后自动播放。",
    chat: "电脑端开始实验后，请先专心与电脑端互动。",
    talkEnabled: false,
  },
  primary: {
    title: "正在模拟环境声音",
    copy: "不要看我（手机陪伴伙伴）。请专心和电脑上的实验系统互动。请不要关闭手机陪伴链接，也不要离开本页。",
    chat: "正在模拟环境声音。请专心和电脑实验系统互动；我是另一个个体。请不要关闭手机陪伴链接。",
    talkEnabled: false,
  },
  companion: {
    title: "现在与手机陪伴伙伴交谈",
    copy: "我是手机陪伴伙伴，不是电脑实验系统。长按下方圆形按钮对我说话，松开后等待回答。",
    chat: "手机陪伴伙伴时段已开始（与电脑实验系统不是同一个）。请长按下方圆形按钮说话。",
    talkEnabled: true,
  },
  rest: {
    title: "请休息一分钟",
    copy: "可以放松眼睛和声音，请不要继续对话或点击屏幕。请不要离开或关闭当前手机页面。背景声将在休息期间暂停，休息结束后从原位置继续播放。",
    chat: "标准化休息已开始。请休息一分钟，不要继续对话或点击屏幕，也不要离开或关闭当前手机页面。",
    talkEnabled: false,
  },
};

function setPhoneTalkHint(mode = "idle") {
  const hint = $("#phone-talk-hint");
  const button = $("#phone-talk-btn");
  if (!hint || !button) return;
  if (mode === "listening") {
    const listeningCopy = shouldUseDesktopAsrRelay()
      ? I18N.t("正在由电脑麦克风识别：请保持原位、面向手机说话，松手发送")
      : I18N.t("正在听，松手发送");
    hint.textContent = listeningCopy;
    button.setAttribute("aria-label", listeningCopy);
    return;
  }
  if (mode === "disabled") {
    hint.textContent = I18N.t("手机对话时段开始后可长按圆形按钮说话");
    button.setAttribute("aria-label", I18N.t("长按说话"));
    return;
  }
  const idleCopy = shouldUseDesktopAsrRelay()
    ? I18N.t("长按说话（请保持原位，面向手机自然说；声音由电脑麦克风识别，无需转向电脑）")
    : I18N.t("长按下方圆形按钮说话");
  hint.textContent = idleCopy;
  button.setAttribute("aria-label", I18N.t("长按说话"));
}

function renderPhoneContinuity(phase = null) {
  const strip = $("#phone-continuity-strip");
  if (!strip) return;
  const currentPhase = phase || (state.companionWindow ? "companion" : "primary");
  const visible = ["primary", "companion"].includes(currentPhase)
    && state.interactionPresenceState !== "hidden";
  strip.classList.toggle("hidden", !visible);
  if (!visible) return;
  const warning = state.interactionPresenceState === "warning";
  const target = state.interactionPresenceTarget === "companion" || currentPhase === "companion"
    ? "手机陪伴智能体"
    : "电脑实验系统";
  strip.dataset.state = warning ? "warning" : "monitoring";
  strip.setAttribute("role", warning ? "alert" : "status");
  strip.setAttribute("aria-live", warning ? "assertive" : "polite");
  $("#phone-continuity-title").textContent = I18N.t(
    warning ? "互动即将超时" : "请保持连续语音对话",
  );
  const remaining = Math.max(0, Number(state.interactionPresenceSeconds) || 0);
  $("#phone-continuity-copy").textContent = I18N.isEnglish
    ? (warning
      ? (target === "手机陪伴智能体"
        ? `Within ${remaining} seconds, hold the phone button to speak or tap Stop answering on the computer when its response timing is wrong. Otherwise, this session will become invalid.`
        : `Speak with the computer experimental system within ${remaining} seconds, or this session will become invalid.`)
      : `Stay in place and keep speaking naturally with the ${target === "手机陪伴智能体" ? "phone companion" : "computer experimental system"}. You do not need to speak while listening.`)
    : (warning
      ? (target === "手机陪伴智能体"
        ? `请在 ${remaining} 秒内长按手机说话；若电脑实验系统响应时机不对，也可在电脑点击「停止回答」。否则本次实验将失效。`
        : `请在 ${remaining} 秒内继续与电脑实验系统说话，否则本次实验将失效。`)
      : `请留在当前位置，持续与${target}自然对话；倾听回答时无需说话。`);
  const countdown = $("#phone-continuity-countdown");
  if (countdown) {
    countdown.textContent = String(remaining);
    countdown.classList.toggle("hidden", !warning);
    countdown.setAttribute("aria-hidden", String(!warning));
  }
}

function phonePassiveExitReasonMessage(reason) {
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

function showPhonePassiveExit(reason) {
  const overlay = $("#phone-passive-exit-overlay");
  const reasonNode = $("#phone-passive-exit-reason");
  if (!overlay || !reasonNode) return;
  reasonNode.textContent = `${I18N.t(phonePassiveExitReasonMessage(reason))} ${I18N.t("本次实验需要从头重新开始。")}`;
  overlay.dataset.reason = String(reason || "unknown");
  overlay.classList.remove("hidden");
  $("#phone-passive-exit-ok")?.focus();
}

function hidePhonePassiveExit() {
  const overlay = $("#phone-passive-exit-overlay");
  if (!overlay) return;
  overlay.classList.add("hidden");
  delete overlay.dataset.reason;
}

function companionPhaseCopy(phase) {
  if (phase !== "companion") {
    return COMPANION_PHASE_COPY[phase] || COMPANION_PHASE_COPY.waiting;
  }
  if (Number(state.companionWindowNumber) > 1) {
    return {
      title: "现在与手机陪伴伙伴交谈",
      copy: "我又来了～长按下方圆形按钮继续聊，松开后等待回答。",
      chat: "我又来了～我是陪伴系统智能体，咱们再继续聊聊。",
      talkEnabled: true,
    };
  }
  return COMPANION_PHASE_COPY.companion;
}

function setCompanionPhase(phase, { announceChat = false } = {}) {
  const copy = companionPhaseCopy(phase);
  const enteredCompanion = phase === "companion" && !state.companionWindow;
  state.companionWindow = phase === "companion";
  $("#phone-mode-title").textContent = I18N.t(copy.title);
  $("#phone-mode-copy").textContent = I18N.t(copy.copy);
  $("#phone-look-cue")?.classList.toggle("is-visible", phase === "companion");
  $("#phone-talk-btn").disabled = !copy.talkEnabled;
  if (!copy.talkEnabled) {
    stopRecognition();
    $("#phone-talk-btn").classList.remove("listening");
    setPhoneTalkHint("disabled");
    clearCompanionInvite();
  } else {
    setPhoneTalkHint("idle");
  }
  if (announceChat && copy.chat) addChatLine(I18N.t(copy.chat), "system");
  renderPhoneContinuity(phase);
  if (enteredCompanion) {
    state.companionInviteUsed = false;
    state.asrFailoverToPhone = false;
    armCompanionInvite(10000);
  } else if (phase !== "companion") {
    clearCompanionInvite();
    state.asrFailoverToPhone = false;
  }
}

function clearCompanionInvite() {
  clearTimeout(state.companionInviteTimer);
  clearInterval(state.companionInviteTickTimer);
  state.companionInviteTimer = null;
  state.companionInviteTickTimer = null;
  state.companionInviteDeadlineMs = 0;
  const panel = $("#phone-invite-countdown");
  if (panel) {
    panel.classList.add("hidden");
    panel.dataset.state = "idle";
  }
}

function renderPhoneInviteCountdown() {
  const panel = $("#phone-invite-countdown");
  const label = $("#phone-invite-label");
  if (!panel || !label) return;
  if (!state.companionWindow || state.companionInviteUsed || !state.companionInviteDeadlineMs) {
    if (!state.companionInviteUsed) panel.classList.add("hidden");
    return;
  }
  const remaining = Math.max(0, Math.ceil((state.companionInviteDeadlineMs - performance.now()) / 1000));
  panel.dataset.state = "counting";
  panel.classList.remove("hidden");
  label.textContent = I18N.t("若暂未开口，我将在 {n} 秒后主动打招呼").replace("{n}", String(remaining));
}

function armCompanionInvite(delayMs = 10000) {
  clearTimeout(state.companionInviteTimer);
  clearInterval(state.companionInviteTickTimer);
  state.companionInviteTimer = null;
  state.companionInviteTickTimer = null;
  if (!state.companionWindow || state.companionInviteUsed) return;
  state.companionInviteDeadlineMs = performance.now() + Math.max(1000, delayMs);
  renderPhoneInviteCountdown();
  state.companionInviteTickTimer = setInterval(renderPhoneInviteCountdown, 250);
  state.companionInviteTimer = setTimeout(() => {
    state.companionInviteTimer = null;
    void tryCompanionSilenceInvite();
  }, Math.max(1000, delayMs));
}

function noteCompanionUserActivity() {
  if (!state.companionWindow || state.companionInviteUsed) return;
  armCompanionInvite(10000);
  send({ type: "companion_state", action: "invite_activity", delay_ms: 10000 });
}

async function tryCompanionSilenceInvite() {
  if (!state.companionWindow || state.companionInviteUsed) return;
  if (state.listening || state.ttsPlaying || state.desktopAsrRelayPending) {
    armCompanionInvite(2000);
    return;
  }
  state.companionInviteUsed = true;
  clearTimeout(state.companionInviteTimer);
  clearInterval(state.companionInviteTickTimer);
  state.companionInviteTimer = null;
  state.companionInviteTickTimer = null;
  state.companionInviteDeadlineMs = 0;
  const panel = $("#phone-invite-countdown");
  const label = $("#phone-invite-label");
  if (panel && label) {
    panel.dataset.state = "fired";
    panel.classList.remove("hidden");
    label.textContent = I18N.t("我（手机陪伴伙伴）正在主动打招呼");
  }
  const returning = Number(state.companionWindowNumber) > 1;
  const invite = I18N.isEnglish
    ? (returning
      ? "I'm back—your phone companion agent. Let's keep chatting; hold the button below whenever you want to talk."
      : "Hi, I'm your phone companion—not the computer experiment system. Hold the button below whenever you want to chat.")
    : (returning
      ? "我又来了～我是陪伴系统智能体，咱们再继续聊聊。想说的时候按住下方按钮就行。"
      : "嗨，我是手机陪伴伙伴，不是电脑上的实验系统。想聊的时候，按住下方按钮对我说话就行。");
  addChatLine(`陪伴伙伴：${invite}`, "assistant");
  send({ type: "companion_state", action: "silence_invite", delay_ms: 10000 });
  try {
    await playSpeech(invite);
  } catch (_) {
    // Invite TTS failure should not block the companion window.
  }
}

function formatCountdownClock(totalSeconds) {
  const value = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function stopPhonePrimaryOverlayCountdown() {
  if (state.phonePrimaryCountdownTimer) {
    clearInterval(state.phonePrimaryCountdownTimer);
    state.phonePrimaryCountdownTimer = null;
  }
}

function phonePrimaryRemainingSeconds() {
  if (state.phonePrimaryCountdownPaused) {
    return Math.max(0, Math.ceil((state.phonePrimaryPausedRemainingMs || 0) / 1000));
  }
  if (!state.phonePrimaryDeadlineMs) return 0;
  return Math.max(0, Math.ceil((state.phonePrimaryDeadlineMs - performance.now()) / 1000));
}

function renderPhonePrimaryOverlayClock() {
  const countdown = $("#phone-primary-countdown");
  const copy = $("#phone-primary-overlay-copy");
  if (!countdown || !copy) return;
  const remaining = phonePrimaryRemainingSeconds();
  const clock = formatCountdownClock(remaining);
  countdown.textContent = clock;
  if (remaining <= 0) {
    copy.textContent = I18N.t("电脑端进入手机对话时段后将自动切换，请稍候");
  } else {
    copy.textContent = I18N.t("请 {time} 后再来和我互动").replace("{time}", clock);
  }
}

function markPhonePrimaryOverlayAwaitingCompanion() {
  renderPhonePrimaryOverlayClock();
  send({
    type: "companion_state",
    action: "phase_sync_request",
    expected: "companion",
    window: state.companionWindowNumber || 1,
  });
}

function pausePhonePrimaryOverlayCountdown() {
  if (state.phonePrimaryCountdownPaused || !state.phonePrimaryDeadlineMs) return;
  state.phonePrimaryPausedRemainingMs = Math.max(0, state.phonePrimaryDeadlineMs - performance.now());
  state.phonePrimaryCountdownPaused = true;
  renderPhonePrimaryOverlayClock();
}

function resumePhonePrimaryOverlayCountdown() {
  if (!state.phonePrimaryCountdownPaused) return;
  state.phonePrimaryDeadlineMs = performance.now() + Math.max(0, state.phonePrimaryPausedRemainingMs || 0);
  state.phonePrimaryCountdownPaused = false;
  state.phonePrimaryPausedRemainingMs = null;
  renderPhonePrimaryOverlayClock();
}

function showPhonePrimaryOverlay(totalSeconds) {
  const overlay = $("#phone-primary-overlay");
  const countdown = $("#phone-primary-countdown");
  const copy = $("#phone-primary-overlay-copy");
  if (!overlay || !countdown || !copy) return;
  stopPhonePrimaryOverlayCountdown();
  const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  state.phonePrimaryCountdownPaused = false;
  state.phonePrimaryPausedRemainingMs = null;
  state.phonePrimaryDeadlineMs = performance.now() + seconds * 1000;
  $("#phone-primary-overlay-title").textContent = I18N.t(
    "正在模拟环境声音，不要看我，专心和电脑互动",
  );
  renderPhonePrimaryOverlayClock();
  document.body.classList.add("phone-primary-focus");
  overlay.classList.remove("hidden");
  if (seconds <= 0) {
    markPhonePrimaryOverlayAwaitingCompanion();
    return;
  }
  state.phonePrimaryCountdownTimer = setInterval(() => {
    if (state.phonePrimaryCountdownPaused) return;
    renderPhonePrimaryOverlayClock();
    if (phonePrimaryRemainingSeconds() <= 0) {
      stopPhonePrimaryOverlayCountdown();
      markPhonePrimaryOverlayAwaitingCompanion();
    }
  }, 250);
}

function hidePhonePrimaryOverlay() {
  stopPhonePrimaryOverlayCountdown();
  state.phonePrimaryDeadlineMs = 0;
  state.phonePrimaryPausedRemainingMs = null;
  state.phonePrimaryCountdownPaused = false;
  document.body.classList.remove("phone-primary-focus");
  $("#phone-primary-overlay")?.classList.add("hidden");
}

function stopPhoneRestOverlayCountdown() {
  if (state.phoneRestCountdownTimer) {
    clearInterval(state.phoneRestCountdownTimer);
    state.phoneRestCountdownTimer = null;
  }
}

function phoneRestRemainingSeconds() {
  if (state.phoneRestCountdownPaused) {
    return Math.max(0, Math.ceil((state.phoneRestPausedRemainingMs || 0) / 1000));
  }
  if (!state.phoneRestDeadlineMs) return 0;
  return Math.max(0, Math.ceil((state.phoneRestDeadlineMs - performance.now()) / 1000));
}

function renderPhoneRestOverlayClock() {
  const countdown = $("#phone-rest-countdown");
  if (!countdown) return;
  countdown.textContent = formatCountdownClock(phoneRestRemainingSeconds());
}

function showPhoneRestOverlay(totalSeconds) {
  const overlay = $("#phone-rest-overlay");
  const countdown = $("#phone-rest-countdown");
  if (!overlay || !countdown) return;
  hidePhonePrimaryOverlay();
  hidePhoneBargeBanner();
  stopPhoneRestOverlayCountdown();
  const seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  state.phoneRestCountdownPaused = false;
  state.phoneRestPausedRemainingMs = null;
  state.phoneRestDeadlineMs = performance.now() + seconds * 1000;
  setCompanionPhase("rest", { announceChat: false });
  renderPhoneRestOverlayClock();
  document.body.classList.add("phone-rest-focus");
  overlay.classList.remove("hidden");
  if (seconds <= 0) return;
  state.phoneRestCountdownTimer = setInterval(() => {
    if (state.phoneRestCountdownPaused) return;
    renderPhoneRestOverlayClock();
    if (phoneRestRemainingSeconds() <= 0) {
      stopPhoneRestOverlayCountdown();
    }
  }, 250);
}

function hidePhoneRestOverlay() {
  stopPhoneRestOverlayCountdown();
  state.phoneRestDeadlineMs = 0;
  state.phoneRestPausedRemainingMs = null;
  state.phoneRestCountdownPaused = false;
  document.body.classList.remove("phone-rest-focus");
  $("#phone-rest-overlay")?.classList.add("hidden");
}

function pausePhoneRestOverlayCountdown() {
  if (state.phoneRestCountdownPaused || !state.phoneRestDeadlineMs) return;
  state.phoneRestPausedRemainingMs = Math.max(0, state.phoneRestDeadlineMs - performance.now());
  state.phoneRestCountdownPaused = true;
  renderPhoneRestOverlayClock();
}

function resumePhoneRestOverlayCountdown() {
  if (!state.phoneRestCountdownPaused) return;
  state.phoneRestDeadlineMs = performance.now() + Math.max(0, state.phoneRestPausedRemainingMs || 0);
  state.phoneRestCountdownPaused = false;
  state.phoneRestPausedRemainingMs = null;
  renderPhoneRestOverlayClock();
}

function formatAsrError(errorCode) {
  const code = String(errorCode || "").trim();
  if (code === "service-not-allowed") {
    return I18N.t("语音识别不可用：请允许麦克风，并用系统浏览器（Chrome/Safari）打开，不要使用微信内置浏览器。");
  }
  if (code === "not-allowed" || code === "permission-denied") {
    return I18N.t("麦克风权限被拒绝，请在浏览器设置中允许此站点使用麦克风。");
  }
  if (code === "audio-capture") {
    return I18N.t("无法访问麦克风，请关闭占用麦克风的其他应用后重试。");
  }
  if (code === "no-speech") {
    return I18N.t("没有检测到语音，请靠近手机并再次按住说话。");
  }
  return I18N.t("语音识别异常：") + (code || I18N.t("未知错误"));
}

const IN_APP_BROWSER_SIGNATURES = Object.freeze([
  { pattern: /MicroMessenger/i, label: "微信" },
  { pattern: /QQ\//i, label: "QQ" },
  { pattern: /Weibo/i, label: "微博" },
  { pattern: /DingTalk/i, label: "钉钉" },
  { pattern: /AlipayClient/i, label: "支付宝" },
]);

function detectInAppBrowser() {
  const userAgent = navigator.userAgent || "";
  for (const signature of IN_APP_BROWSER_SIGNATURES) {
    if (signature.pattern.test(userAgent)) {
      return { blocked: true, label: signature.label };
    }
  }
  return { blocked: false, label: "" };
}

function companionUsesDesktopAsrRelay() {
  // Primary path: desktop Chrome Web Speech relayed to the phone. Phone-local
  // ASR is the backup after a desktop relay failure (see asrFailoverToPhone).
  return true;
}

function phoneLocalAsrLikelyBlocked() {
  // In-app browsers often expose SpeechRecognition but reject with
  // service-not-allowed. Prefer staying on desktop relay instead of surfacing
  // a phone-mic / WeChat error that does not apply to the primary path.
  try {
    if (detectInAppBrowser().blocked) return true;
  } catch (_) {
    return true;
  }
  return false;
}

function canUsePhoneLocalAsr() {
  return false;
}

function shouldUseDesktopAsrRelay() {
  // Default always desktop relay. Sticky phone failover only when phone ASR is
  // actually usable; otherwise keep retrying the computer path.
  if (state.asrFailoverToPhone && canUsePhoneLocalAsr()) return false;
  return companionUsesDesktopAsrRelay();
}

function showDesktopAsrRetryMessage(detail = "") {
  const base = I18N.t("电脑端识别暂时失败，请确认电脑实验页保持打开，再按住说话重试。");
  $("#phone-active-error").textContent = detail ? `${base}（${detail}）` : base;
}

function failoverDesktopAsrToPhone(reason = "desktop_asr_failed") {
  clearTimeout(state.desktopAsrRelayTimer);
  state.desktopAsrRelayTimer = null;
  state.desktopAsrRelayPending = false;
  setPhoneAsrLive("");
  // Phone local ASR is backup only when it can actually run. Otherwise stay on
  // desktop primary and never show the WeChat/mic “语音识别不可用” copy.
  if (!canUsePhoneLocalAsr()) {
    state.asrFailoverToPhone = false;
    state.listening = false;
    $("#phone-talk-btn").classList.remove("listening");
    setPhoneTalkHint(state.companionWindow ? "idle" : "disabled");
    showDesktopAsrRetryMessage();
    send({
      type: "companion_state",
      action: "asr_failover_blocked",
      from: "desktop_relay",
      reason: String(reason || "desktop_asr_failed"),
      phone_local_available: false,
    });
    if (state.socket?.readyState !== WebSocket.OPEN) {
      connectSocket({ force: false });
    }
    return;
  }
  if (state.asrFailoverToPhone && !state.listening && !state.talkPointerDown) {
    return;
  }
  state.asrFailoverToPhone = true;
  send({
    type: "companion_state",
    action: "asr_failover",
    from: "desktop_relay",
    to: "phone_local",
    reason: String(reason || "desktop_asr_failed"),
  });
  // Still holding the talk button: switch to phone ASR without another press.
  if (state.talkPointerDown && state.companionWindow) {
    $("#phone-active-error").textContent = I18N.t(
      "电脑端识别失败，已切换到手机识别，请继续按住说话。",
    );
    state.listening = false;
    $("#phone-talk-btn").classList.remove("listening");
    void startLocalPhoneAsr({ failover: true });
    return;
  }
  state.listening = false;
  $("#phone-talk-btn").classList.remove("listening");
  setPhoneTalkHint(state.companionWindow ? "idle" : "disabled");
  $("#phone-active-error").textContent = I18N.t(
    "电脑端识别失败，已切换到手机识别，请再按住说话。",
  );
}

function recommendedSystemBrowsers() {
  const userAgent = navigator.userAgent || "";
  if (/iPhone|iPod|iPad/i.test(userAgent)) return "Safari";
  if (/Android/i.test(userAgent)) return "Chrome";
  return "Safari / Chrome";
}

function showCompanionBrowserGate(inAppLabel) {
  const browsers = recommendedSystemBrowsers();
  $("#companion-browser-gate-title").textContent = I18N.t("请改用 {browser} 打开").replace("{browser}", browsers);
  $("#companion-browser-gate-message").textContent = I18N.t(
    "当前在{label}内置浏览器中打开，无法参与手机陪伴实验。",
  ).replace("{label}", inAppLabel);
  $("#companion-browser-gate-help").textContent = I18N.t(
    "请点击右上角「…」选择「在浏览器中打开」，或复制链接到 {browser} 后再进入。",
  ).replace("{browser}", browsers);
  $("#companion-browser-gate").classList.remove("hidden");
  document.querySelector("header")?.classList.add("hidden");
  document.querySelector("main")?.classList.add("hidden");
  $("#phone-pilot-banner")?.classList.add("hidden");
  document.querySelector("main")?.setAttribute("inert", "");
  setCompanionLoginSubmitEnabled(false);
  state.companionEnvironment = {
    status: "failed",
    ok: false,
    message: companionEnvironmentFailureMessage("in_app_browser", inAppLabel),
    reason: "in_app_browser",
  };
}

function companionEnvironmentFailureMessage(reason, inAppLabel = "") {
  if (reason === "in_app_browser") {
    const template = I18N.t(
      "当前在{label}内置浏览器中打开，无法使用按住说话功能。请用系统浏览器（Safari/Chrome）打开此链接，不要使用微信内置浏览器。",
    );
    return template.replace("{label}", inAppLabel || I18N.t("应用"));
  }
  if (reason === "speech_recognition_unavailable") {
    return formatAsrError("service-not-allowed");
  }
  if (/^[a-z-]+$/.test(String(reason || ""))) {
    return formatAsrError(reason);
  }
  return formatAsrError("service-not-allowed");
}

function updateCompanionEnvironmentUI() {
  if (state.companionEnvironment.reason === "in_app_browser") return;
  const env = state.companionEnvironment;
  const relayInfo = env.reason === "desktop_asr_relay_in_app" || env.reason === "desktop_asr_relay";
  for (const id of ["phone-login-environment-warning", "phone-setup-environment-warning"]) {
    const box = document.getElementById(id);
    if (!box) continue;
    const message = box.querySelector("[data-env-message]");
    if (env.status === "checking") {
      box.classList.remove("hidden", "is-failed", "is-relay-info");
      box.classList.add("is-checking");
      if (message) message.textContent = I18N.t("正在检查浏览器环境…");
      continue;
    }
    if (relayInfo && env.ok) {
      box.classList.remove("hidden", "is-checking", "is-failed");
      box.classList.add("is-relay-info");
      if (message) message.textContent = env.message;
      continue;
    }
    if (!env.ok) {
      box.classList.remove("hidden", "is-checking", "is-relay-info");
      box.classList.add("is-failed");
      if (message) message.textContent = env.message;
      continue;
    }
    box.classList.add("hidden");
    box.classList.remove("is-checking", "is-failed", "is-relay-info");
  }
}

function setCompanionLoginSubmitEnabled(enabled) {
  if (!phoneLoginSubmit) return;
  if (enabled) {
    phoneLoginSubmit.disabled = false;
    phoneLoginSubmit.removeAttribute("disabled");
  } else {
    phoneLoginSubmit.disabled = true;
  }
}

function updateCompanionLoginSubmitEnabled() {
  if (CLIENT_DEVICE_CLASS !== "phone") return;
  const capacityBlocked = Boolean(
    state.capacity && state.capacity.ok === false && !state.capacity.holding_slot
  );
  // Deep-linked codes should stay actionable even while the environment probe
  // is finishing; WeChat sometimes leaves disabled buttons stuck otherwise.
  const hasCode = Boolean(normalizedCode($("#phone-code")?.value || ""));
  // If the deep link already filled a code, keep Continue clickable even while
  // the environment probe is still settling (WeChat can leave disabled stuck).
  setCompanionLoginSubmitEnabled(!capacityBlocked && (hasCode || state.companionEnvironment.ok));
}

async function initializeCompanionEnvironmentGate() {
  state.companionEnvironment = {
    status: "checking",
    ok: false,
    message: I18N.t("正在检查浏览器环境…"),
    reason: "",
  };
  updateCompanionEnvironmentUI();
  updateCompanionLoginSubmitEnabled();
  updatePhoneSetupReadiness();

  const inApp = detectInAppBrowser();
  state.companionEnvironment = {
    status: "ready",
    ok: true,
    message: inApp.blocked
      ? I18N.t("当前在{label}内置浏览器中，按住说话将由电脑端识别；失败时再尝试手机识别。").replace(
        "{label}",
        inApp.label,
      )
      : I18N.t("按住说话时，由电脑端麦克风采集；浏览器识别失败时由后台语音识别自动接手。"),
    reason: inApp.blocked ? "desktop_asr_relay_in_app" : "desktop_asr_relay",
  };
  updateCompanionEnvironmentUI();
  updateCompanionLoginSubmitEnabled();
  updatePhoneSetupReadiness();
}

async function acquireCompanionMicStream() {
  if (state.micStream?.getAudioTracks?.().some((track) => track.readyState === "live")) {
    return state.micStream;
  }
  state.micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });
  return state.micStream;
}

function releaseCompanionMicStream() {
  try {
    state.micStream?.getTracks?.().forEach((track) => {
      try { track.stop(); } catch (_) { /* Already stopped. */ }
    });
  } catch (_) { /* Ignore teardown races. */ }
  state.micStream = null;
}

function probeSpeechRecognition(timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      reject(new Error("speech_recognition_unavailable"));
      return;
    }
    const recognition = new Recognition();
    recognition.lang = I18N.isEnglish ? "en-US" : "zh-CN";
    recognition.continuous = false;
    recognition.interimResults = false;
    let settled = false;
    let started = false;
    let intentionalStop = false;
    const finish = (error = null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve(true);
    };
    recognition.onstart = () => {
      started = true;
      intentionalStop = true;
      try { recognition.stop(); } catch (_) { /* Probe only needs a successful start. */ }
    };
    recognition.onerror = (event) => {
      const code = String(event.error || "speech_recognition_failed");
      if ((intentionalStop || started) && code === "aborted") {
        intentionalStop = false;
        finish();
        return;
      }
      finish(new Error(code));
    };
    recognition.onend = () => {
      if (started) finish();
      else if (!settled) finish(new Error("speech_recognition_ended_before_start"));
    };
    const timer = setTimeout(() => {
      intentionalStop = true;
      try { recognition.stop(); } catch (_) { /* Timed probe should not leave recognition running. */ }
      if (!started && !settled) finish(new Error("speech_recognition_timeout"));
    }, timeoutMs);
    try {
      recognition.start();
    } catch (error) {
      finish(error);
    }
  });
}

const REQUIRED_PHONE_QUALITY = [
  "phone_network",
  "phone_microphone",
  "phone_reaction_latency",
  "phone_browser",
];
const COMPANION_MEDIA_VOLUME_PERCENT = 50;
const PHONE_VOLUME_CALIBRATION_OUTPUT_GAIN = 0.5;
const COMPANION_PHONE_PLACEMENT_CM = 50;
const COMPANION_PHONE_PLACEMENT_TOLERANCE_CM = 5;
const COMPANION_STANDARD_BACKGROUND_GAIN = 1;
const COMPANION_STANDARD_SPEECH_GAIN = 1;
const PHONE_QUALITY_LABELS = {
  phone_network: "手机网络稳定性检测",
  phone_microphone: "手机麦克风与环境噪声检测",
  phone_reaction_latency: "手机触摸反应延迟检测",
  phone_browser: "手机浏览器与最低音量硬校准",
};

function companionPlaybackStandard() {
  return {
    media_volume_target_percent: COMPANION_MEDIA_VOLUME_PERCENT,
    phone_placement_target_cm: COMPANION_PHONE_PLACEMENT_CM,
    phone_placement_range_cm: [
      COMPANION_PHONE_PLACEMENT_CM - COMPANION_PHONE_PLACEMENT_TOLERANCE_CM,
      COMPANION_PHONE_PLACEMENT_CM + COMPANION_PHONE_PLACEMENT_TOLERANCE_CM,
    ],
    background_element_volume: 1,
    background_gain: COMPANION_STANDARD_BACKGROUND_GAIN,
    speech_gain: COMPANION_STANDARD_SPEECH_GAIN,
    playback_path: "media_element_plus_web_audio_gain",
  };
}

function applyStandardCompanionPlayback() {
  if (!state.backgroundAudio) return;
  state.backgroundAudio.volume = 1;
  if (state.backgroundGain) state.backgroundGain.gain.value = COMPANION_STANDARD_BACKGROUND_GAIN;
  if (state.speechGain) state.speechGain.gain.value = COMPANION_STANDARD_SPEECH_GAIN;
}

function showPhoneView(id) {
  for (const view of ["phone-login-view", "phone-setup-view", "phone-active-view"]) {
    $(`#${view}`).classList.toggle("hidden", view !== id);
  }
}

function setPhoneLogoutVisible(visible) {
  $("#phone-logout-btn").classList.toggle("hidden", !visible);
}

async function logoutCompanion() {
  const confirmed = window.confirm(
    "确定退出当前编号吗？未完成的电脑与手机实验会立即结束并释放实验位置，已上传数据不会删除；如需再次参与，请重新生成编号。",
  );
  if (!confirmed) return;

  const button = $("#phone-logout-btn");
  button.disabled = true;
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

  clearInterval(state.heartbeat);
  stopRecognition();
  state.backgroundAudio.pause();
  state.socket?.close();
  sessionStorage.removeItem(COMPANION_TOKEN_KEY);
  state.token = "";
  window.location.reload();
}

$("#phone-logout-btn")?.addEventListener("click", logoutCompanion);

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-TTA-Device-Class", CLIENT_DEVICE_CLASS);
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  if (options.body && typeof options.body !== "string" && !(options.body instanceof Blob)) {
    headers.set("Content-Type", "application/json");
    options.body = JSON.stringify(options.body);
  }
  const requestPath = !options.method || options.method === "GET" ? I18N.apiPath(path) : path;
  const response = await fetch(`${APP_BASE}${requestPath}`, { ...options, headers, cache: "no-store" });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const value = await response.json();
      if (typeof value.detail === "string") detail = value.detail;
      else if (value.detail?.message) detail = value.detail.message;
      else detail = JSON.stringify(value.detail);
    } catch (_) {
      // Keep status text.
    }
    throw new Error(detail);
  }
  return (response.headers.get("content-type") || "").includes("application/json") ? response.json() : response;
}

function normalizedCode(value) {
  const raw = value.toUpperCase().replace(/[^A-Z0-9]/g, "");
  if (raw.startsWith("TTA") && raw.length >= 11) return `TTA-${raw.slice(3, 7)}-${raw.slice(7, 11)}`;
  return value.trim().toUpperCase();
}

function isFullParticipantCode(value) {
  return /^TTA-[A-Z0-9]{4}-[A-Z0-9]{4}$/.test(normalizedCode(value));
}

function updatePhoneCodeBadge(code = state.code) {
  const badge = $("#phone-code-badge");
  if (!badge) return;
  const value = String(code || "").trim().toUpperCase();
  if (!value) {
    badge.textContent = "";
    badge.classList.add("hidden");
    return;
  }
  badge.textContent = value;
  badge.title = `当前实验编号 ${value}`;
  badge.classList.remove("hidden");
}

function directCodeFromLocation() {
  const embedded = String(
    document.querySelector('meta[name="tta-participant-code"]')?.getAttribute("content") || "",
  ).trim();
  if (embedded) return normalizedCode(embedded);
  const params = new URLSearchParams(location.search);
  const queryCode = String(params.get("code") || params.get("participant_code") || "").trim();
  if (queryCode) return normalizedCode(queryCode);
  const hashCode = decodeURIComponent(location.hash.replace(/^#\/?/, "").trim());
  if (hashCode) return normalizedCode(hashCode);
  const prefix = `${APP_BASE}/companion/`;
  if (!location.pathname.startsWith(prefix)) return "";
  const pathCode = location.pathname.slice(prefix.length).split("/").filter(Boolean)[0] || "";
  return pathCode ? normalizedCode(decodeURIComponent(pathCode)) : "";
}

function scrubDirectCodeFromLocation() {
  const params = new URLSearchParams();
  if (I18N.isEnglish) params.set("lang", "en");
  const query = params.toString();
  window.history.replaceState(
    null,
    "",
    `${APP_BASE}/companion${query ? `?${query}` : ""}`,
  );
}

function applyDirectCompanionCode(directCode) {
  const input = $("#phone-code");
  const helper = input?.closest(".form-row")?.querySelector(".helper-text");
  const label = input?.closest(".form-row")?.querySelector("label[for='phone-code']");
  if (!input) return;
  input.value = directCode;
  input.readOnly = true;
  input.classList.add("code-input-autofilled");
  input.setAttribute("aria-readonly", "true");
  if (label) label.textContent = I18N.t("参与者编号（已从链接填入）");
  if (helper) {
    helper.textContent = I18N.t("编号已从专属陪伴链接自动填入，无需手动输入。");
  }
  updatePhoneCodeBadge(directCode);
}

function wsUrl(path, params) {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}${APP_BASE}${path}?${new URLSearchParams(params)}`;
}

let companionLoginInFlight = false;

async function performCompanionLogin({ auto = false } = {}) {
  const error = $("#phone-login-error");
  if (companionLoginInFlight) return;
  if (CLIENT_DEVICE_CLASS !== "phone") {
    error.textContent = I18N.t("无法开始，需要在手机端打开陪伴系统");
    error?.scrollIntoView({ block: "center", behavior: "smooth" });
    return;
  }
  const capacityBlocked = Boolean(
    state.capacity && state.capacity.ok === false && !state.capacity.holding_slot
  );
  if (capacityBlocked) {
    error.textContent = state.capacity.message_zh || state.capacity.message || I18N.t("当前同时使用人数较多");
    error?.scrollIntoView({ block: "center", behavior: "smooth" });
    return;
  }
  state.code = normalizedCode($("#phone-code").value);
  if (!state.code) {
    error.textContent = I18N.t("请输入参与者编号");
    error?.scrollIntoView({ block: "center", behavior: "smooth" });
    return;
  }
  if (!isFullParticipantCode(state.code)) {
    error.textContent = I18N.t(
      "请使用完整编号（TTA-XXXX-XXXX）打开专属陪伴链接，不要只使用四位尾号。",
    );
    error?.scrollIntoView({ block: "center", behavior: "smooth" });
    return;
  }
  updatePhoneCodeBadge(state.code);
  companionLoginInFlight = true;
  setCompanionLoginSubmitEnabled(false);
  error.textContent = auto
    ? I18N.t("正在使用专属链接自动连接…")
    : I18N.t("正在连接…");
  try {
    const login = await api("/api/login", { method: "POST", body: { code: state.code, role: "companion" } });
    state.token = login.token;
    sessionStorage.setItem(COMPANION_TOKEN_KEY, login.token);
    state.bootstrap = await api("/api/bootstrap");
    if (!state.bootstrap.consent_valid) throw new Error("请先在电脑端完成知情同意");
    updatePhoneCodeBadge(state.code);
    configureCompanionSession();
    const companionSession = await api("/api/sessions/start", {
      method: "POST",
      body: { day: state.day },
    });
    state.sessionId = companionSession.id;
    restorePhoneQuality();
    const pilotBanner = $("#phone-pilot-banner");
    pilotBanner.classList.toggle("hidden", !state.bootstrap.pilot_only);
    if (state.bootstrap.pilot_only) {
      pilotBanner.textContent = state.bootstrap.participant.analysis_excluded
        ? "仅用于个人调试：数据与正式实验永久隔离"
        : "非正式数据模式";
    }
    const license = state.bootstrap.background;
    $("#license-note").innerHTML = `${escapeHtml(license.attribution)} <a href="${escapeAttribute(license.license_url)}" target="_blank" rel="noreferrer">CC BY-NC-SA 4.0</a>`;
    setPhoneLogoutVisible(true);
    state.phoneReadyAnnounced = false;
    updatePhoneSetupReadiness();
    showPhoneView("phone-setup-view");
    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
    connectSocket();
    startHeartbeat();
  } catch (err) {
    error.textContent = `无法连接：${err.message}`;
    updateCompanionLoginSubmitEnabled();
    error?.scrollIntoView({ block: "center", behavior: "smooth" });
  } finally {
    companionLoginInFlight = false;
  }
}

window.__ttaCompanionLogin = (options) => performCompanionLogin(options || { auto: false });

$("#phone-login-form")?.addEventListener("submit", (event) => {
  event.preventDefault();
  event.stopPropagation();
  void performCompanionLogin({ auto: false });
  return false;
});

function bootCompanionLoginUi() {
  // Run as early as possible so a later missing-element listener cannot leave
  // WeChat users stuck on a disabled Continue button.
  try {
    phoneLoginSubmit = $("#phone-login-submit");
    if (!phoneLoginSubmit) throw new Error("companion login submit control is missing");
    phoneLoginSubmit.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      void performCompanionLogin({ auto: false });
    });
    if (CLIENT_DEVICE_CLASS !== "phone") {
      const gate = $("#companion-device-gate");
      gate?.classList.remove("hidden");
      const title = $("#companion-device-gate-title");
      const message = $("#companion-device-gate-message");
      if (title) title.textContent = I18N.t("无法开始，需要在手机端打开陪伴系统");
      if (message) {
        message.textContent = I18N.t(
          "当前设备不是智能手机。请用手机打开专属陪伴链接；微信内打开也可以，电脑和平板不行。",
        );
      }
      const english = document.querySelector("#companion-device-gate .device-role-gate-english");
      if (english) {
        english.textContent = "Open the companion link on a smartphone. WeChat in-app browser is fine; computers and tablets are not.";
      }
      document.querySelector("main")?.setAttribute("inert", "");
      setCompanionLoginSubmitEnabled(false);
      document.documentElement.dataset.appReady = "device-blocked";
      return;
    }
    $("#companion-device-gate")?.classList.add("hidden");
    document.querySelector("main")?.removeAttribute("inert");
    const earlyCode = directCodeFromLocation() || normalizedCode($("#phone-code")?.value || "");
    if (earlyCode) applyDirectCompanionCode(earlyCode);
    updateCompanionLoginSubmitEnabled();
    const readyError = $("#phone-login-error");
    if (readyError && /准备连接/.test(readyError.textContent || "")) {
      readyError.textContent = "";
    }
    if (earlyCode) {
      scrubDirectCodeFromLocation();
      void performCompanionLogin({ auto: true });
    }
    void initializeCompanionEnvironmentGate().finally(async () => {
      try { await refreshPhoneCapacityStatus(); } catch (_) { /* Keep login usable. */ }
      startPhoneCapacityPolling();
      updateCompanionLoginSubmitEnabled();
      document.documentElement.dataset.appReady = "true";
      window.dispatchEvent(new CustomEvent("tap-study:companion-ready"));
    });
  } catch (err) {
    console.error("companion login boot failed", err);
    const btn = $("#phone-login-submit");
    if (btn) {
      btn.disabled = false;
      btn.removeAttribute("disabled");
      btn.addEventListener("click", (event) => {
        event.preventDefault();
        void performCompanionLogin({ auto: false });
      });
    }
    const error = $("#phone-login-error");
    if (error) error.textContent = I18N.t("自动连接准备失败，请点击继续");
    document.documentElement.dataset.appReady = "boot-fallback";
  }
}

bootCompanionLoginUi();

function configureCompanionSession() {
  if (Number(state.bootstrap?.client_protocol?.session_design?.total_sessions || 0) !== 1) {
    throw new Error("该编号绑定的是已停用的旧版多日流程，请在电脑端换用当前版本的新编号");
  }
  const studySessions = (state.bootstrap.progress.sessions || [])
    .filter((session) => session.client_role === "study");
  const open = studySessions.find((session) => !["completed", "ended", "aborted"].includes(session.state));
  if (!open) throw new Error("请先在电脑端创建本次实验，再连接手机伴随端");
  state.day = Number(open.day);
}

function restorePhoneQuality() {
  state.phoneQuality = new Map();
  state.phoneQualityAttempts = new Map();
  const restored = new Set();
  for (const row of state.bootstrap?.progress?.quality || []) {
    if (row.session_id !== state.sessionId || !REQUIRED_PHONE_QUALITY.includes(row.kind)) continue;
    let metrics = {};
    try {
      metrics = typeof row.metrics === "string" ? JSON.parse(row.metrics) : (row.metrics || {});
    } catch (_) { /* Legacy rows may not contain structured metrics. */ }
    const phoneVolumePreparationOnly = row.kind === "phone_browser"
      && !row.passed
      && metrics.user_confirmed_media_volume_minimum !== true
      && !metrics.phrase_id;
    if (phoneVolumePreparationOnly) {
      state.phoneQualityAttempts.set(
        row.kind,
        Math.max(state.phoneQualityAttempts.get(row.kind) || 0, Number(metrics?.attempt_number || 0)),
      );
      continue;
    }
    const currentBrowserVerification = row.kind !== "phone_browser"
      || (
        metrics.verification_method === "typed_keyword_recall"
        && ["zh-blue-kite-v1", "en-blue-kite-v1"].includes(metrics.phrase_id)
        && metrics.probe_context === "minimum_volume_gate"
        && metrics.volume_gate_hard === true
        && Number(metrics.output_gain) === PHONE_VOLUME_CALIBRATION_OUTPUT_GAIN
        && Number(metrics.media_volume_min_percent) === COMPANION_MEDIA_VOLUME_PERCENT
        && metrics.user_confirmed_media_volume_minimum === true
      );
    const restoredPassed = Boolean(row.passed) && currentBrowserVerification;
    if (!restored.has(row.kind)) {
      state.phoneQuality.set(row.kind, restoredPassed);
      if (row.kind === "phone_browser" && restoredPassed) {
        const language = ["zh-CN", "en-US"].includes(metrics.test_language)
          ? metrics.test_language
          : "zh-CN";
        $("#phone-audio-language").value = language;
        $("#phone-audio-language").dataset.passedLanguage = language;
        $("#phone-audio-language").disabled = true;
      }
      if (row.kind === "phone_browser" && metrics.user_confirmed_media_volume_minimum === true) {
        $("#volume-confirm").checked = true;
      }
      const checkId = {
        phone_network: "phone-network",
        phone_microphone: "phone-mic",
        phone_reaction_latency: "phone-reaction",
        phone_browser: "phone-browser",
      }[row.kind];
      setPhoneCheck(
        checkId,
        restoredPassed,
        restoredPassed ? "已通过（从本次记录恢复）" : "上次检测未通过，请重新检测",
      );
      restored.add(row.kind);
    }
    if (row.kind === "phone_microphone" && restoredPassed) {
      state.phoneMicMetrics = metrics;
    }
    state.phoneQualityAttempts.set(
      row.kind,
      Math.max(state.phoneQualityAttempts.get(row.kind) || 0, Number(metrics?.attempt_number || 0)),
    );
  }
  state.phoneMicPassed = state.phoneQuality.get("phone_microphone") === true;
  syncPhoneVolumeConfirmationUI();
}

async function savePhoneQuality(kind, passed, metrics) {
  const attemptNumber = (state.phoneQualityAttempts.get(kind) || 0) + 1;
  const result = await api("/api/quality", {
    method: "POST",
    body: {
      session_id: state.sessionId,
      kind,
      passed,
      metrics: {
        ...metrics,
        attempt_id: crypto.randomUUID(),
        attempt_number: attemptNumber,
        client_role: "companion",
      },
    },
  });
  state.phoneQualityAttempts.set(kind, Number(result.attempt_number || attemptNumber));
  state.phoneQuality.set(kind, result.passed === true);
  updatePhoneSetupReadiness();
  return result;
}

function setPhoneCheck(id, passed, text) {
  const item = $(`#${id}-check`);
  item.classList.remove("pass", "fail");
  item.classList.add(passed ? "pass" : "fail");
  $(`#${id}-value`).textContent = text;
}

function resetPhoneCheck(id, text) {
  const item = $(`#${id}-check`);
  item.classList.remove("pass", "fail");
  $(`#${id}-value`).textContent = text;
}

function syncPhoneVolumeConfirmationUI() {
  const confirm = $("#volume-confirm");
  const button = $("#phone-browser-btn");
  if (!confirm || !button) return;
  const passed = state.phoneQuality.get("phone_browser") === true;
  if (passed) {
    confirm.checked = true;
    confirm.disabled = true;
    button.disabled = true;
    button.textContent = I18N.t("最低音量校准已通过");
    return;
  }
  confirm.disabled = false;
  button.disabled = !confirm.checked;
  button.textContent = I18N.t("播放最低音量校准句");
}

function phoneSetupBlockers() {
  const blockers = [];
  if (state.companionEnvironment.status === "checking") {
    blockers.push("正在检查浏览器与语音识别环境");
  } else if (!state.companionEnvironment.ok) {
    blockers.push("浏览器与语音识别环境未通过");
  }
  for (const kind of REQUIRED_PHONE_QUALITY) {
    const result = state.phoneQuality.get(kind);
    if (result !== true) {
      blockers.push(`${PHONE_QUALITY_LABELS[kind]}${result === false ? "未通过" : "尚未完成"}`);
    }
  }
  if (!$("#volume-confirm").checked) {
    blockers.push("尚未确认手机媒体音量至少为 50% 与固定摆放位置");
  }
  return blockers;
}

function updatePhoneSetupReadiness() {
  const blockers = phoneSetupBlockers();
  const ready = blockers.length === 0 && state.companionEnvironment.ok;
  $("#enable-phone-audio").disabled = !ready;
  const status = $("#phone-gate-status");
  status.dataset.ready = String(ready);
  const list = $("#phone-gate-blockers");
  list.replaceChildren();
  for (const label of blockers) {
    const item = document.createElement("li");
    item.textContent = label;
    list.appendChild(item);
  }
  $("#phone-gate-summary").textContent = ready
    ? "手机端全部检测已通过，可以连接电脑"
    : `连接前还需完成 ${blockers.length} 项条件：`;
  status.classList.toggle("ready", ready);
}

$("#volume-confirm")?.addEventListener("change", () => {
  resetPhonePlaybackInput();
  state.phoneQuality.delete("phone_browser");
  $("#phone-audio-language").disabled = false;
  delete $("#phone-audio-language").dataset.passedLanguage;
  resetPhoneCheck(
    "phone-browser",
    $("#volume-confirm").checked
      ? I18N.t("已确认系统媒体音量至少为 50%，请播放低增益校准句")
      : I18N.t("先把系统媒体音量调到 50% 以上，并勾选本卡内确认框"),
  );
  syncPhoneVolumeConfirmationUI();
  updatePhoneSetupReadiness();
});

$("#phone-network-btn")?.addEventListener("click", async () => {
  const button = $("#phone-network-btn");
  button.disabled = true;
  try {
    $("#phone-network-value").textContent = "正在发送 12 次测量请求";
    const timeoutMs = 2000;
    const samples = [];
    const successful = [];
    let failures = 0;
    for (let index = 0; index < 12; index += 1) {
      const started = performance.now();
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);
      try {
        await api(`/api/ping?sample=phone-${index}`, { signal: controller.signal });
        const rtt = performance.now() - started;
        samples.push(rtt);
        successful.push(rtt);
      } catch (_) {
        failures += 1;
        samples.push(timeoutMs);
      } finally {
        clearTimeout(timeout);
      }
      await sleep(90);
    }
    successful.sort((left, right) => left - right);
    const median = successful.length ? percentile(successful, 0.5) : timeoutMs;
    const p95 = successful.length ? percentile(successful, 0.95) : timeoutMs;
    const jitter = successful.length
      ? Math.sqrt(successful.reduce((sum, value) => sum + (value - median) ** 2, 0) / successful.length)
      : Infinity;
    const QG = window.TTA_QUALITY_GATE;
    const networkMetrics = {
      median_ms: median,
      p95_ms: p95,
      jitter_ms: jitter,
      failures,
      samples,
      successful_samples_ms: successful,
      timeout_ms: timeoutMs,
    };
    const localPassed = QG.evaluateNetwork(networkMetrics).passed;
    const saved = await savePhoneQuality("phone_network", localPassed, networkMetrics);
    setPhoneCheck(
      "phone-network",
      saved.passed,
      QG.formatNetworkSummary(networkMetrics, saved.passed, saved.server_decision),
    );
  } catch (err) {
    state.phoneQuality.set("phone_network", false);
    setPhoneCheck(
      "phone-network",
      false,
      window.TTA_QUALITY_GATE.formatSaveError(err, "phone_network", state.phoneQualityAttempts.get("phone_network") || 0),
    );
    updatePhoneSetupReadiness();
  } finally {
    button.disabled = false;
  }
});

async function samplePhoneAudio(analyser, seconds, onLevel) {
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
    const rms = Math.sqrt(sum / buffer.length);
    rmsValues.push(rms);
    onLevel?.(rms);
    await sleep(50);
  }
  rmsValues.sort((left, right) => left - right);
  return {
    median: percentile(rmsValues, 0.5),
    p90: percentile(rmsValues, 0.9),
    clipping: total ? clipped / total : 0,
  };
}

$("#phone-mic-test")?.addEventListener("click", async () => {
  const button = $("#phone-mic-test");
  const output = $("#phone-mic-value");
  const meter = $(".volume-meter");
  button.disabled = true;
  state.phoneMicPassed = false;
  updatePhoneSetupReadiness();
  let stream = null;
  let context = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: false,
    });
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    context = new AudioContextClass();
    await context.resume();
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    context.createMediaStreamSource(stream).connect(analyser);
    output.textContent = "请保持安静 2 秒";
    const quiet = await samplePhoneAudio(analyser, 2, () => {
      $("#phone-volume-bar").style.width = "2%";
      meter.setAttribute("aria-valuenow", "2");
    });
    output.textContent = "请保持 30–90 厘米距离，用正常音量连续说：手机语音检测一二三";
    const speech = await samplePhoneAudio(analyser, 4, (rms) => {
      const level = Math.max(2, Math.min(100, Math.round(rms * 420)));
      $("#phone-volume-bar").style.width = `${level}%`;
      meter.setAttribute("aria-valuenow", String(level));
    });
    const snr = 20 * Math.log10(Math.max(speech.p90, 1e-7) / Math.max(quiet.median, 1e-7));
    const QG = window.TTA_QUALITY_GATE || {};
    const micRules = QG.PHONE_MICROPHONE || {
      snr_db_min: 6,
      speech_rms_p90_min: 0.0035,
      clipping_ratio_max: 0.01,
    };
    const localEval = typeof QG.evaluateMicrophone === "function"
      ? QG.evaluateMicrophone({
        snr_db: snr,
        quiet_rms_p90: quiet.p90,
        speech_rms_p90: speech.p90,
        clipping_ratio: speech.clipping,
      }, micRules)
      : {
        passed: snr >= micRules.snr_db_min
          && speech.p90 > micRules.speech_rms_p90_min
          && speech.clipping < micRules.clipping_ratio_max,
      };
    const passed = Boolean(localEval.passed);
    state.phoneMicMetrics = {
      quiet_rms_median: quiet.median,
      quiet_rms_p90: quiet.p90,
      speech_rms_p90: speech.p90,
      snr_db: snr,
      clipping_ratio: speech.clipping,
      instructed_distance_cm: [30, 90],
      phone_placement_target_cm: COMPANION_PHONE_PLACEMENT_CM,
      phone_placement_range_cm: [
        COMPANION_PHONE_PLACEMENT_CM - COMPANION_PHONE_PLACEMENT_TOLERANCE_CM,
        COMPANION_PHONE_PLACEMENT_CM + COMPANION_PHONE_PLACEMENT_TOLERANCE_CM,
      ],
      passed,
    };
    const saved = await savePhoneQuality("phone_microphone", passed, {
      ...state.phoneMicMetrics,
      track_settings: stream.getAudioTracks()[0]?.getSettings?.() || {},
    });
    state.phoneMicPassed = saved.passed;
    setPhoneCheck(
      "phone-mic",
      saved.passed,
      saved.passed
        ? `正常说话音量可用 · SNR ${snr.toFixed(1)} dB`
        : (typeof QG.formatMicrophoneSummary === "function"
          ? QG.formatMicrophoneSummary(state.phoneMicMetrics, false, saved.server_decision)
          : "正常说话音量检测未通过"),
    );
  } catch (err) {
    state.phoneQuality.set("phone_microphone", false);
    setPhoneCheck("phone-mic", false, `无法检测手机麦克风：${err.message}`);
    try {
      await savePhoneQuality("phone_microphone", false, { error: err.message });
    } catch (_) { /* The visible error remains the actionable result. */ }
  } finally {
    stream?.getTracks().forEach((track) => track.stop());
    await context?.close().catch(() => {});
    button.disabled = false;
    updatePhoneSetupReadiness();
  }
});

let phoneReactionTrial = null;
$("#phone-reaction-btn")?.addEventListener("click", async () => {
  const button = $("#phone-reaction-btn");
  button.disabled = true;
  const stage = $("#phone-reaction-stage");
  const wait = $("#phone-reaction-wait");
  const target = $("#phone-reaction-target");
  try {
    stage.classList.remove("hidden");
    const samples = [];
    for (let trial = 0; trial < 6; trial += 1) {
      wait.classList.remove("hidden");
      target.classList.add("hidden");
      await sleep(700 + Math.random() * 900);
      phoneReactionTrial = { shown: performance.now(), resolve: null };
      const clicked = new Promise((resolve) => { phoneReactionTrial.resolve = resolve; });
      wait.classList.add("hidden");
      target.textContent = I18N.t(`轻触我（还差 ${6 - trial} 次）`);
      target.classList.remove("hidden");
      samples.push(await clicked);
    }
    samples.sort((left, right) => left - right);
    const median = percentile(samples, 0.5);
    const QG = window.TTA_QUALITY_GATE;
    const reactionMetrics = {
      median_ms: median,
      samples_ms: samples,
      pointer_type: "touch_or_pointer",
    };
    const saved = await savePhoneQuality(
      "phone_reaction_latency",
      QG.evaluateReaction(reactionMetrics).passed,
      reactionMetrics,
    );
    setPhoneCheck(
      "phone-reaction",
      saved.passed,
      saved.passed
        ? `触摸反应中位数 ${Math.round(median)} ms · 范围 ${Math.round(samples[0])}–${Math.round(samples.at(-1))} ms`
        : QG.formatReactionSummary(reactionMetrics, false, saved.server_decision),
    );
  } catch (err) {
    state.phoneQuality.set("phone_reaction_latency", false);
    setPhoneCheck(
      "phone-reaction",
      false,
      window.TTA_QUALITY_GATE.formatSaveError(err, "phone_reaction_latency", state.phoneQualityAttempts.get("phone_reaction_latency") || 0),
    );
    updatePhoneSetupReadiness();
  } finally {
    phoneReactionTrial = null;
    stage.classList.add("hidden");
    button.disabled = false;
  }
});

$("#phone-reaction-target")?.addEventListener("pointerdown", () => {
  if (!phoneReactionTrial?.resolve) return;
  const resolve = phoneReactionTrial.resolve;
  phoneReactionTrial.resolve = null;
  resolve(performance.now() - phoneReactionTrial.shown);
});

function phoneBrowserChecks() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  return {
    secure_context: window.isSecureContext,
    media_devices: Boolean(navigator.mediaDevices?.getUserMedia),
    audio_context: Boolean(window.AudioContext || window.webkitAudioContext),
    websocket: Boolean(window.WebSocket),
    speech_recognition: Boolean(Recognition),
    audio_element: Boolean(window.HTMLAudioElement),
  };
}

const PHONE_BROWSER_REQUIRED = Object.freeze([
  "secure_context",
  "media_devices",
  "audio_context",
  "websocket",
  "audio_element",
]);

function phoneBrowserCapabilitiesPassed(checks) {
  return PHONE_BROWSER_REQUIRED.every((name) => checks[name] === true);
}

async function playCompanionServerSpeech(text, options = {}) {
  const {
    context = "quality_probe",
    outputGain = 1,
  } = options;
  if (!state.sessionId) throw new Error("session_missing");
  const language = $("#phone-audio-language").value || "zh-CN";
  const requestedAt = performance.now();
  const response = await api("/api/tts", {
    method: "POST",
    body: { session_id: state.sessionId, text },
  });
  const provider = response.headers.get("X-TTS-Provider") || "server";
  const voice = response.headers.get("X-TTS-Voice") || "";
  const synthesisLatencyMs = Number(response.headers.get("X-TTS-Latency-Ms") || 0);
  const objectUrl = URL.createObjectURL(await response.blob());
  const audio = new Audio(objectUrl);
  audio.setAttribute("playsinline", "");
  audio.preload = "auto";
  const requestedGain = Math.max(0, Math.min(1, Number(outputGain) || 0));
  audio.volume = requestedGain;
  try {
    await audio.play();
    await new Promise((resolve, reject) => {
      audio.onended = () => resolve();
      audio.onerror = () => reject(new Error("playback_failed"));
    });
  } finally {
    audio.pause();
    audio.currentTime = 0;
    URL.revokeObjectURL(objectUrl);
  }
  const observedGain = Number(audio.volume);
  return {
    language,
    playback_api: "server_web_audio",
    provider,
    voice_name: voice,
    voice_lang: language,
    matching_voice_available: Boolean(voice),
    text_chars: String(text || "").trim().length,
    synthesis_latency_ms: synthesisLatencyMs,
    request_to_play_ms: Math.round(performance.now() - requestedAt),
    probe_context: context,
    // iOS WeChat often leaves HTMLAudioElement.volume readable as 1 even
    // after a 0.5 write.  The server gate checks the requested calibration
    // gain; observed volume is kept only as telemetry.
    output_gain: requestedGain,
    output_gain_requested: requestedGain,
    output_gain_observed: Number.isFinite(observedGain) ? observedGain : null,
  };
}

function resetPhonePlaybackInput() {
  state.phoneBrowserProbe = null;
  $("#phone-audio-answer").value = "";
  $("#phone-answer-row").classList.add("hidden");
  $("#phone-audio-verify").classList.add("hidden");
  window.TTA_PLAYBACK_TEST.resetVerifyButton($("#phone-audio-verify"));
}

$("#phone-browser-btn")?.addEventListener("click", async () => {
  const button = $("#phone-browser-btn");
  if (!$("#volume-confirm").checked) {
    resetPhoneCheck("phone-browser", I18N.t("请先把系统媒体音量调到 50% 以上，并勾选本卡内确认框"));
    syncPhoneVolumeConfirmationUI();
    updatePhoneSetupReadiness();
    return;
  }
  button.disabled = true;
  resetPhonePlaybackInput();
  state.phoneQuality.delete("phone_browser");
  updatePhoneSetupReadiness();
  try {
    const checks = phoneBrowserChecks();
    const capabilityPassed = phoneBrowserCapabilitiesPassed(checks);
    if (!capabilityPassed) {
      const saved = await savePhoneQuality("phone_browser", false, {
        ...checks,
        playback_verified: false,
        typed_answer_stored: false,
        media_volume_min_percent: COMPANION_MEDIA_VOLUME_PERCENT,
        user_confirmed_media_volume_minimum: true,
        user_agent: navigator.userAgent,
      });
      setPhoneCheck(
        "phone-browser",
        saved.passed,
        `缺少：${Object.entries(checks).filter(([, value]) => !value).map(([key]) => key).join("、")}`,
      );
      return;
    }
    const language = $("#phone-audio-language").value || "zh-CN";
    const test = window.TTA_PLAYBACK_TEST.selected(language);
    $("#phone-browser-value").textContent = I18N.t("正在播放测试句");
    const playback = await playCompanionServerSpeech(test.text, {
      context: "minimum_volume_gate",
      outputGain: PHONE_VOLUME_CALIBRATION_OUTPUT_GAIN,
    });
    state.phoneBrowserProbe = {
      ...checks,
      ...playback,
      phrase_id: test.id,
      volume_gate_hard: true,
      media_volume_min_percent: COMPANION_MEDIA_VOLUME_PERCENT,
      user_confirmed_media_volume_minimum: true,
    };
    $("#phone-answer-row").classList.remove("hidden");
    $("#phone-audio-verify").classList.remove("hidden");
    window.TTA_PLAYBACK_TEST.syncVerifyButton(
      $("#phone-audio-answer"),
      $("#phone-audio-verify"),
    );
    $("#phone-audio-answer").focus();
    $("#phone-browser-value").textContent = I18N.t("低增益校准句已播放，请完整输入听到的内容");
  } catch (err) {
    try {
      await savePhoneQuality("phone_browser", false, {
        ...phoneBrowserChecks(),
        test_language: $("#phone-audio-language").value || "zh-CN",
        playback_api: "server_web_audio",
        playback_error: err.message,
        playback_verified: false,
        typed_answer_stored: false,
        media_volume_min_percent: COMPANION_MEDIA_VOLUME_PERCENT,
        user_confirmed_media_volume_minimum: true,
        user_agent: navigator.userAgent,
      });
    } catch (_) {
      state.phoneQuality.set("phone_browser", false);
      updatePhoneSetupReadiness();
    }
    setPhoneCheck("phone-browser", false, I18N.t(err.message || "最低音量校准失败，请提高系统媒体音量后重试"));
  } finally {
    button.disabled = false;
  }
});

$("#phone-audio-verify")?.addEventListener("click", async () => {
  if (!state.phoneBrowserProbe) {
    $("#phone-browser-value").textContent = I18N.t("请先播放测试句");
    return;
  }
  const button = $("#phone-audio-verify");
  button.disabled = true;
  const language = $("#phone-audio-language").value || "zh-CN";
  const evaluation = window.TTA_PLAYBACK_TEST.evaluate(language, $("#phone-audio-answer").value);
  try {
    const saved = await savePhoneQuality("phone_browser", evaluation.passed, {
      ...state.phoneBrowserProbe,
      ...evaluation,
      test_language: language,
      verification_method: "typed_keyword_recall",
      playback_verified: evaluation.passed,
      typed_answer_stored: false,
      user_agent: navigator.userAgent,
    });
    const QG = window.TTA_QUALITY_GATE;
    setPhoneCheck(
      "phone-browser",
      saved.passed,
      I18N.t(
        QG.formatPhoneBrowserSummary(
          saved.passed,
          saved.server_decision,
          evaluation,
        ),
      ),
    );
    if (saved.passed) {
      $("#phone-audio-language").dataset.passedLanguage = language;
      $("#phone-audio-language").disabled = true;
      syncPhoneVolumeConfirmationUI();
    }
  } catch (err) {
    setPhoneCheck("phone-browser", false, I18N.t("检测结果未能保存，请重试"));
  } finally {
    window.TTA_PLAYBACK_TEST.syncVerifyButton(
      $("#phone-audio-answer"),
      button,
    );
  }
});

$("#phone-audio-answer")?.addEventListener("input", () => {
  window.TTA_PLAYBACK_TEST.syncVerifyButton(
    $("#phone-audio-answer"),
    $("#phone-audio-verify"),
  );
});

$("#phone-audio-answer")?.addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  event.preventDefault();
  if (!$("#phone-audio-verify").disabled) $("#phone-audio-verify").click();
});

$("#phone-audio-language")?.addEventListener("change", async () => {
  const wasPassed = state.phoneQuality.get("phone_browser") === true;
  if (wasPassed) {
    $("#phone-audio-language").value =
      $("#phone-audio-language").dataset.passedLanguage || "zh-CN";
    return;
  }
  resetPhonePlaybackInput();
  state.phoneQuality.delete("phone_browser");
  resetPhoneCheck(
    "phone-browser",
    $("#volume-confirm").checked
      ? I18N.t("已确认系统媒体音量至少为 50%，请播放低增益校准句")
      : I18N.t("先把系统媒体音量调到 50% 以上，并勾选本卡内确认框"),
  );
  syncPhoneVolumeConfirmationUI();
  updatePhoneSetupReadiness();
});

$("#enable-phone-audio")?.addEventListener("click", async () => {
  const error = $("#phone-setup-error");
  const button = $("#enable-phone-audio");
  error.textContent = "";
  if (!state.companionEnvironment.ok) {
    error.textContent = state.companionEnvironment.message || formatAsrError("service-not-allowed");
    return;
  }
  const blockers = phoneSetupBlockers();
  if (blockers.length > 0) {
    error.textContent = `无法连接：${blockers.join("；")}`;
    updatePhoneSetupReadiness();
    return;
  }
  if (button) {
    button.disabled = true;
    button.textContent = I18N.t("正在连接电脑…");
  }
  try {
    await initializeAudio();
    // Mark ready before (re)connecting so the open/heartbeat path never
    // announces ready:false after the user has enabled audio.
    state.phoneReadyAnnounced = true;
    if (!companionUsesDesktopAsrRelay()) {
      await acquireCompanionMicStream();
      releaseCompanionMicStream();
    }
    if (button) button.textContent = I18N.t("正在建立配对连接…");
    // Force=true claims this tab as the only live companion owner.
    await ensurePairSocketOpen(15000, { force: true });
    await requestWakeLock();
    // Announce once; do not pulse/wait-loop on companion_ready (that caused
    // pair_status storms and made the desktop setup steps flicker).
    announcePhoneReadiness(true);
    startCompanionReadyHttpPulse();
    await sleep(250);
    if (state.socket?.readyState !== WebSocket.OPEN) {
      throw new Error("配对连接已断开，请重试启用声音");
    }
    $("#phone-day-value").textContent = "本次实验";
    setCompanionPhase("waiting");
    showPhoneView("phone-active-view");
  } catch (err) {
    state.phoneReadyAnnounced = false;
    const message = String(err.message || "");
    if (message === "background_unavailable" || message.startsWith("media_error_")) {
      error.textContent = `准备失败：${backgroundLoadFailureMessage(err)}`;
    } else if (
      message === "speech_recognition_unavailable"
      || message.startsWith("speech_recognition_")
      || /^[a-z-]+$/.test(message)
    ) {
      error.textContent = `准备失败：${formatAsrError(message === "speech_recognition_unavailable" ? "service-not-allowed" : message)}`;
    } else {
      error.textContent = `准备失败：${message}`;
    }
    updatePhoneSetupReadiness();
  } finally {
    if (button) button.textContent = I18N.t("完成检测并连接电脑");
    updatePhoneSetupReadiness();
  }
});

function resolveStaticAssetUrl(path) {
  const normalized = String(path || "").startsWith("/") ? String(path) : `/${path}`;
  const joined = `${APP_BASE || ""}${normalized}`.replace(/\/{2,}/g, "/");
  return new URL(joined, location.origin).href;
}

function backgroundTrackForDay(background, day) {
  const tracks = background?.tracks || [];
  const direct = tracks.find((item) => item.day === day);
  if (direct) return direct;
  if (Number(background?.single_session_day_fallback ?? 1) === 1) {
    return tracks.find((item) => item.day === 1) || null;
  }
  return null;
}

function backgroundMediaCandidates(track, background) {
  const paths = [];
  const push = (path) => {
    if (path) paths.push(path);
  };
  // Phone WebViews (including in-app browsers) decode MP3 more reliably than M4A.
  if (CLIENT_DEVICE_CLASS !== "desktop") {
    push(track.fallback_url);
    push(track.url);
  } else {
    push(track.url);
    push(track.fallback_url);
  }
  const version = String(background?.version || "").trim();
  return [...new Set(paths.map((path) => {
    const url = new URL(resolveStaticAssetUrl(path));
    if (version) url.searchParams.set("v", version);
    return url.href;
  }))];
}

function backgroundLoadFailureMessage(error) {
  const code = Number(String(error?.message || "").match(/media_error_(\d+)/)?.[1] || 0);
  if (code === 4) return I18N.t("背景声格式不受支持，请改用系统浏览器打开");
  return I18N.t("背景声文件无法加载");
}

function bindBackgroundAudioTelemetry(audio) {
  audio.addEventListener("pause", () => {
    send({
      type: "background_state",
      action: "pause_event",
      current_time: audio.currentTime,
      should_play: state.backgroundShouldPlay,
      playback_held: state.backgroundPlaybackHeld,
      rest_held: state.backgroundRestHeld,
      ptt_held: state.backgroundPttHeld,
      tts_playing: state.ttsPlaying,
    });
    scheduleUnexpectedBackgroundRecovery("pause_event");
  });
  audio.addEventListener("stalled", () => send({
    type: "background_state",
    action: "stalled",
    current_time: audio.currentTime,
  }));
  audio.addEventListener("ended", () => send({
    type: "background_state",
    action: "ended",
    current_time: audio.currentTime,
  }));
}

function replaceBackgroundAudioElement() {
  const previous = state.backgroundAudio;
  if (!previous?.parentNode) return;
  const replacement = document.createElement("audio");
  replacement.id = "background-audio";
  replacement.preload = "auto";
  replacement.setAttribute("playsinline", "");
  previous.replaceWith(replacement);
  state.backgroundAudio = replacement;
  bindBackgroundAudioTelemetry(replacement);
}

function resetBackgroundAudioGraph() {
  try { state.backgroundSource?.disconnect(); } catch (_) { /* Already disconnected. */ }
  try { state.backgroundGain?.disconnect(); } catch (_) { /* Already disconnected. */ }
  try { state.speechGain?.disconnect(); } catch (_) { /* Already disconnected. */ }
  if (state.context && state.context.state !== "closed") {
    void state.context.close().catch(() => {});
  }
  state.backgroundSource = null;
  state.backgroundGain = null;
  state.speechGain = null;
  state.context = null;
  if (state.backgroundAudio?.dataset.mediaGraphLocked === "true") {
    replaceBackgroundAudioElement();
  }
}

async function waitForBackgroundMediaLoad(audio, urls) {
  let lastError = null;
  for (const url of urls) {
    try {
      await new Promise((resolve, reject) => {
        const cleanup = () => {
          audio.removeEventListener("canplaythrough", ready);
          audio.removeEventListener("canplay", ready);
          audio.removeEventListener("loadeddata", ready);
          audio.removeEventListener("error", failed);
        };
        const ready = () => {
          cleanup();
          resolve();
        };
        const failed = () => {
          cleanup();
          reject(new Error(`media_error_${audio.error?.code || 0}`));
        };
        audio.addEventListener("canplaythrough", ready, { once: true });
        audio.addEventListener("canplay", ready, { once: true });
        audio.addEventListener("loadeddata", ready, { once: true });
        audio.addEventListener("error", failed, { once: true });
        audio.src = url;
        audio.load();
      });
      return url;
    } catch (error) {
      lastError = error;
      audio.removeAttribute("src");
      audio.load();
    }
  }
  throw lastError || new Error("background_unavailable");
}

async function initializeAudio() {
  const track = backgroundTrackForDay(state.bootstrap.background, state.day);
  if (!track) throw new Error("没有找到今天的背景声轨道");
  const candidates = backgroundMediaCandidates(track, state.bootstrap.background);
  if (!candidates.length) throw new Error("没有找到今天的背景声轨道");

  resetBackgroundAudioGraph();
  await withTimeout(
    waitForBackgroundMediaLoad(state.backgroundAudio, candidates),
    30000,
    "背景声下载超时，请检查手机网络",
  );

  const context = new AudioContext();
  await withTimeout(context.resume(), 5000, "手机音频系统没有响应，请关闭其他音频应用后重试");
  state.context = context;
  // Loop as a safety net if the ~45min track ends before the session does.
  state.backgroundAudio.loop = true;
  state.backgroundLoadedUrl = state.backgroundAudio.currentSrc || candidates[0] || "";
  state.backgroundSource = context.createMediaElementSource(state.backgroundAudio);
  state.backgroundAudio.dataset.mediaGraphLocked = "true";
  state.backgroundGain = context.createGain();
  state.speechGain = context.createGain();
  applyStandardCompanionPlayback();
  state.backgroundSource.connect(state.backgroundGain).connect(context.destination);
  await withTimeout(state.backgroundAudio.play(), 8000, "浏览器阻止了声音播放，请再次长按启用");
  await sleep(700);
  state.backgroundAudio.pause();
  state.backgroundAudio.currentTime = 0;
  state.backgroundShouldPlay = false;
  $("#background-status").textContent = "已就绪";
}

async function ensureBackgroundPlaying(reason = "ensure") {
  if (
    !state.backgroundShouldPlay
    || state.backgroundPlaybackHeld
    || state.backgroundRestHeld
    || state.backgroundPttHeld
    || !state.backgroundAudio
  ) {
    return false;
  }
  try {
    if (!state.context || state.context.state === "closed") {
      // Graph was torn down (or never built). Rebuild it without rewinding an
      // active experiment; initializeAudio itself performs a 0-position probe.
      const resumeAtSeconds = Number(state.backgroundAudio.currentTime || 0);
      await initializeAudio();
      state.backgroundShouldPlay = true;
      if (resumeAtSeconds > 0 && Number.isFinite(resumeAtSeconds)) {
        state.backgroundAudio.currentTime = resumeAtSeconds;
      }
    } else if (state.context.state !== "running") {
      await state.context.resume();
    }
    applyStandardCompanionPlayback();
    if (state.backgroundAudio.paused) {
      await state.backgroundAudio.play();
    }
    $("#background-status").textContent = "播放中";
    return true;
  } catch (error) {
    $("#background-status").textContent = "恢复播放中…";
    send({
      type: "background_state",
      action: "play_failed",
      reason,
      message: String(error?.message || error || "play_failed"),
    });
    return false;
  }
}

function scheduleUnexpectedBackgroundRecovery(reason = "unexpected_pause", delayMs = 160) {
  clearTimeout(state.backgroundRecoveryTimer);
  state.backgroundRecoveryTimer = null;
  if (
    !state.backgroundShouldPlay
    || state.backgroundPlaybackHeld
    || state.backgroundRestHeld
    || state.backgroundPttHeld
    || !state.backgroundAudio?.paused
  ) {
    return;
  }
  state.backgroundRecoveryTimer = setTimeout(() => {
    state.backgroundRecoveryTimer = null;
    void ensureBackgroundPlaying(reason);
  }, Math.max(0, Number(delayMs) || 0));
}

let lastPhoneReadyAnnounceMs = 0;
let companionReadyHttpTimer = null;

function startCompanionReadyHttpPulse() {
  if (companionReadyHttpTimer) return;
  companionReadyHttpTimer = setInterval(() => {
    if (state.phoneReadyAnnounced) announcePhoneReadiness();
  }, 5000);
}

function announcePhoneReadiness(force = false) {
  if (!state.phoneReadyAnnounced) return;
  const now = Date.now();
  // Throttle: unthrottled ready pulses caused pair_status storms and desktop step flicker.
  if (!force && now - lastPhoneReadyAnnounceMs < 5000) return;
  lastPhoneReadyAnnounceMs = now;
  send({
    type: "background_state",
    action: "ready",
    day: state.day,
    ready: true,
    phone_microphone: state.phoneMicMetrics,
    phone_quality: Object.fromEntries(state.phoneQuality),
    playback_standard: companionPlaybackStandard(),
  });
  if (state.sessionId && state.token) {
    void api(`/api/sessions/${state.sessionId}/companion-ready`, { method: "POST" }).catch(() => {
      // Pair WebSocket remains the primary path; HTTP is the split-listener fallback.
    });
  }
}

let pairReconnectAttempt = 0;
let pairReconnectTimer = null;
let pairConnectStartedAtMs = 0;
let companionOwnerChannel = null;

function claimCompanionTabOwnership() {
  const payload = JSON.stringify({
    tabId: COMPANION_TAB_ID,
    sessionId: state.sessionId || "",
    at: Date.now(),
  });
  try {
    localStorage.setItem(COMPANION_OWNER_KEY, payload);
  } catch (_) {
    // Private mode may block localStorage; BroadcastChannel still helps.
  }
  try {
    if (!companionOwnerChannel) {
      companionOwnerChannel = new BroadcastChannel("tta-companion-owner");
      companionOwnerChannel.onmessage = (event) => {
        const data = event?.data || {};
        if (data.type === "takeover" && data.tabId && data.tabId !== COMPANION_TAB_ID) {
          markPairSuperseded("已被新的陪伴页取代，请关闭本页，只保留最新打开的那一页");
        }
      };
    }
    companionOwnerChannel.postMessage({
      type: "takeover",
      tabId: COMPANION_TAB_ID,
      sessionId: state.sessionId || "",
      at: Date.now(),
    });
  } catch (_) {
    // BroadcastChannel unavailable in some WebViews.
  }
}

function isCompanionTabOwner() {
  try {
    const raw = localStorage.getItem(COMPANION_OWNER_KEY);
    if (!raw) return true;
    const parsed = JSON.parse(raw);
    return !parsed?.tabId || parsed.tabId === COMPANION_TAB_ID;
  } catch (_) {
    return true;
  }
}

function markPairSuperseded(reason) {
  state.pairSuperseded = true;
  clearCompanionPairReconnect();
  pairReconnectAttempt = 0;
  if (state.socket) {
    const stale = state.socket;
    state.socket = null;
    try { stale.close(4000, "superseded"); } catch (_) { /* Already closed. */ }
  }
  setHeaderStatus("请关闭本页", "error");
  $("#desktop-status").textContent = I18N.t(
    reason || "已被新的陪伴页取代，请关闭本页，只保留最新打开的那一页",
  );
}

function clearCompanionPairReconnect() {
  clearTimeout(pairReconnectTimer);
  pairReconnectTimer = null;
}

function scheduleCompanionPairReconnect(delayMs, statusText) {
  clearCompanionPairReconnect();
  if (!state.sessionId || state.pairSuperseded || state.pairClosingByPage) return;
  if (!isCompanionTabOwner()) {
    markPairSuperseded("已被新的陪伴页取代，请关闭本页，只保留最新打开的那一页");
    return;
  }
  if (document.visibilityState === "hidden") {
    setHeaderStatus("页面在后台，暂停重连", "warn");
    $("#desktop-status").textContent = I18N.t("回到前台后自动重连");
    return;
  }
  $("#desktop-status").textContent = statusText || I18N.t("重连中…");
  pairReconnectTimer = setTimeout(() => {
    pairReconnectTimer = null;
    if (state.pairSuperseded || state.pairClosingByPage) return;
    connectSocket();
  }, delayMs);
}

async function ensurePairSocketOpen(timeoutMs = 15000, { force = false } = {}) {
  if (state.pairSuperseded && !force) {
    throw new Error("当前页已被新的陪伴页取代，请关闭本页，只保留最新打开的陪伴页");
  }
  if (force) {
    state.pairSuperseded = false;
    state.pairClosingByPage = false;
    claimCompanionTabOwnership();
  }
  clearCompanionPairReconnect();
  pairReconnectAttempt = 0;
  if (state.socket?.readyState === WebSocket.OPEN) return;
  // Prefer waiting out an in-flight connect over force-killing it (force thrash).
  if (state.socket?.readyState !== WebSocket.CONNECTING || force) {
    connectSocket({ force });
  }
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (state.pairSuperseded) {
      throw new Error("当前页已被新的陪伴页取代，请关闭本页，只保留最新打开的陪伴页");
    }
    if (state.socket?.readyState === WebSocket.OPEN) return;
    const connectingTooLong = state.socket?.readyState === WebSocket.CONNECTING
      && pairConnectStartedAtMs > 0
      && Date.now() - pairConnectStartedAtMs > 5000;
    if (
      !state.socket
      || state.socket.readyState === WebSocket.CLOSED
      || state.socket.readyState === WebSocket.CLOSING
      || connectingTooLong
    ) {
      connectSocket({ force: Boolean(force || connectingTooLong) });
    }
    await sleep(120);
  }
  throw new Error("手机与电脑的配对连接未能建立，请关闭其它重复陪伴页后重试");
}

function connectSocket({ force = false } = {}) {
  // Desktop/tablet companion tabs were fighting the phone for /ws/pair and
  // clearing companion_ready — never open the pair socket off-phone.
  if (CLIENT_DEVICE_CLASS !== "phone") {
    clearCompanionPairReconnect();
    setHeaderStatus("请用手机打开陪伴页", "error");
    $("#desktop-status").textContent = I18N.t("电脑端请关闭此页");
    return;
  }
  if (state.pairSuperseded || state.pairClosingByPage) return;
  if (!force && !isCompanionTabOwner()) {
    markPairSuperseded("已被新的陪伴页取代，请关闭本页，只保留最新打开的那一页");
    return;
  }
  if (state.socket?.readyState === WebSocket.OPEN) {
    if (state.phoneReadyAnnounced) announcePhoneReadiness(true);
    return;
  }
  if (!force && state.socket?.readyState === WebSocket.CONNECTING) return;
  if (!state.sessionId) return;
  if (!force && document.visibilityState === "hidden") {
    scheduleCompanionPairReconnect(2000, I18N.t("回到前台后自动重连"));
    return;
  }
  if (force && state.socket) {
    const stale = state.socket;
    state.socket = null;
    try { stale.close(); } catch (_) { /* Already closed. */ }
  }
  clearCompanionPairReconnect();
  claimCompanionTabOwnership();
  const socket = new WebSocket(wsUrl("/ws/pair", {
    token: state.token,
    session_id: state.sessionId,
  }));
  state.socket = socket;
  pairConnectStartedAtMs = Date.now();
  socket.addEventListener("open", () => {
    if (state.pairSuperseded || state.socket !== socket) {
      try { socket.close(4000, "superseded"); } catch (_) { /* Ignore. */ }
      return;
    }
    pairReconnectAttempt = 0;
    pairConnectStartedAtMs = 0;
    claimCompanionTabOwnership();
    setHeaderStatus(
      state.phoneReadyAnnounced ? "已连接" : "已配对，正在检测",
      state.phoneReadyAnnounced ? "online" : "warn",
    );
    $("#desktop-status").textContent = state.phoneReadyAnnounced
      ? I18N.t("同步就绪中")
      : I18N.t("等待");
    if (state.phoneReadyAnnounced) announcePhoneReadiness(true);
    // Pair blips used to pause BGM forever; resume if the study still wants audio.
    void ensureBackgroundPlaying("pair_reopen");
    requestStudyPhaseSync("pair_open");
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    // Serialize async handlers so a failed BGM play() cannot race/drop notices.
    state.pairMessageChain = state.pairMessageChain
      .catch(() => {})
      .then(async () => {
        if (state.pairSuperseded || state.socket !== socket) return;
        if (message.type === "pair_status") {
          const bothRoles = Array.isArray(message.roles)
            && message.roles.includes("study")
            && message.roles.includes("companion");
          const sameDay = message.same_day_online && message.day === state.day;
          $("#desktop-status").textContent = sameDay
            ? (message.companion_ready
              ? I18N.t("同编号在线·可开始")
              : I18N.t("同编号在线·待确认就绪"))
            : bothRoles
              ? I18N.t("会话不一致")
              : (state.phoneReadyAnnounced ? I18N.t("等待电脑端连接") : I18N.t("等待"));
          if (state.phoneReadyAnnounced && sameDay && message.companion_ready) {
            setHeaderStatus("已连接", "online");
            // Desktop may already be mid-block while this page still shows
            // “等待电脑端开始”; ask study to rebroadcast the live phase.
            if (!state.companionWindow) {
              requestStudyPhaseSync("pair_status_ready");
            }
          } else if (state.phoneReadyAnnounced && sameDay && !message.companion_ready) {
            setHeaderStatus("已配对，同步就绪中", "warn");
            announcePhoneReadiness(); // throttled
          }
        }
        if (message.type === "background_state") await handleBackgroundCommand(message);
        if (message.type === "notice") await handleNotice(message);
        if (message.type === "desktop_asr") handleDesktopAsrRelay(message);
      });
  });
  socket.addEventListener("close", (event) => {
    if (state.socket !== socket) return;
    state.socket = null;
    pairConnectStartedAtMs = 0;
    // A short signalling reconnect must not interrupt the continuous
    // background track. The experiment itself decides when audio stops.
    // Replaced by a newer companion page: stay dead so two tabs cannot thrash.
    if (event.code === 4001 || state.pairSuperseded) {
      markPairSuperseded("已被新的陪伴页取代，请关闭本页，只保留最新打开的那一页");
      return;
    }
    if (state.pairClosingByPage) {
      setHeaderStatus("页面已关闭连接", "warn");
      $("#desktop-status").textContent = I18N.t("页面关闭");
      return;
    }
    setHeaderStatus("连接断开", "error");
    pairReconnectAttempt += 1;
    const delayMs = Math.min(12000, 1600 * (2 ** Math.min(pairReconnectAttempt - 1, 3)));
    scheduleCompanionPairReconnect(delayMs, I18N.t("重连中…"));
  });
}

window.addEventListener("storage", (event) => {
  if (event.key !== COMPANION_OWNER_KEY || !event.newValue) return;
  try {
    const parsed = JSON.parse(event.newValue);
    if (parsed?.tabId && parsed.tabId !== COMPANION_TAB_ID) {
      markPairSuperseded("已被新的陪伴页取代，请关闭本页，只保留最新打开的那一页");
    }
  } catch (_) {
    // Ignore malformed ownership payloads.
  }
});

window.addEventListener("pagehide", () => {
  state.pairClosingByPage = true;
  clearCompanionPairReconnect();
  if (state.socket) {
    const stale = state.socket;
    state.socket = null;
    try { stale.close(1000, "pagehide"); } catch (_) { /* Already closed. */ }
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible" || !state.sessionId) return;
  if (state.pairSuperseded || state.pairClosingByPage) return;
  if (!isCompanionTabOwner()) {
    markPairSuperseded("已被新的陪伴页取代，请关闭本页，只保留最新打开的那一页");
    return;
  }
  if (
    !state.socket
    || state.socket.readyState === WebSocket.CLOSED
    || state.socket.readyState === WebSocket.CLOSING
  ) {
    connectSocket();
  }
});

async function handleBackgroundCommand(message) {
  try {
    if (message.action === "start") {
      const alreadyStarted = state.backgroundShouldPlay;
      state.backgroundShouldPlay = true;
      // Phase synchronization may update a rest hold, but duplicate start
      // commands must never rewind or clear an active PTT/rest hold.
      if (!alreadyStarted || message.sync) {
        state.backgroundRestHeld = Boolean(message.rest_held);
      }
      if (!alreadyStarted && !message.sync) {
        state.backgroundPlaybackHeld = false;
        state.backgroundRestHeld = false;
        state.backgroundPttHeld = false;
      }
      // Only a genuinely new experiment may set an initial offset. Reconnects,
      // duplicate starts and 4 -> 2 minute phase handoffs preserve playhead.
      if (!alreadyStarted) {
        state.backgroundAudio.currentTime = Number(message.offset_seconds || 0);
      }
      if (state.backgroundRestHeld) state.backgroundAudio.pause();
      await ensureBackgroundPlaying("start");
      if (!message.sync) setCompanionPhase("primary", { announceChat: true });
    } else if (message.action === "stop") {
      state.backgroundShouldPlay = false;
      state.backgroundPlaybackHeld = false;
      state.backgroundRestHeld = false;
      state.backgroundPttHeld = false;
      state.backgroundAudio.pause();
      $("#background-status").textContent = "已停止";
    }
  } catch (error) {
    send({
      type: "background_state",
      action: "play_failed",
      reason: message.action || "background_command",
      message: String(error?.message || error || "background_command_failed"),
    });
  }
}

function formatPhoneBargeCopyHtml() {
  const raw = I18N.t(
    "若接话不合时机，请到电脑点击「停止回答」打断。不要在手机上操作打断。"
  );
  return raw
    .replace(/「停止回答」/g, "「<span class=\"phrase-nowrap\">停止回答</span>」")
    .replace(/Stop answering/g, "<span class=\"phrase-nowrap\">Stop answering</span>");
}

function showPhoneBargeBanner(agentState = "thinking") {
  const banner = $("#phone-barge-banner");
  if (!banner || !state.companionWindow) return;
  const title = $("#phone-barge-title");
  const copy = $("#phone-barge-copy");
  const speaking = agentState === "speaking";
  if (title) {
    title.textContent = speaking
      ? I18N.t("电脑实验系统正在说话")
      : I18N.t("电脑实验系统正在插话");
  }
  if (copy) {
    copy.innerHTML = formatPhoneBargeCopyHtml();
  }
  const wasHidden = banner.classList.contains("hidden");
  banner.classList.remove("hidden");
  document.body.classList.add("phone-barge-focus");
  if (wasHidden) {
    navigator.vibrate?.([180, 80, 180]);
  }
}

function hidePhoneBargeBanner() {
  $("#phone-barge-banner")?.classList.add("hidden");
  document.body.classList.remove("phone-barge-focus");
}

async function handleNotice(message) {
  if (message.action === "interaction_presence") {
    state.interactionPresenceState = ["monitoring", "warning"].includes(message.state)
      ? message.state
      : "hidden";
    state.interactionPresenceSeconds = Number.isFinite(Number(message.seconds))
      ? Number(message.seconds)
      : null;
    state.interactionPresenceTarget = message.target === "companion" ? "companion" : "primary";
    renderPhoneContinuity();
    return;
  }
  if (message.action === "request_ready") {
    if (state.phoneReadyAnnounced && !state.pairSuperseded && !state.pairClosingByPage) {
      // Never force-kill a live/connecting socket from desktop nudges — that
      // thrash is what made the phone flicker between 重连中 and 可开始.
      if (state.socket?.readyState === WebSocket.OPEN) {
        announcePhoneReadiness(true);
      } else if (state.socket?.readyState !== WebSocket.CONNECTING) {
        connectSocket({ force: false });
      }
    }
    return;
  }
  if (message.action === "primary_start") {
    hidePhonePassiveExit();
    hidePhoneRestOverlay();
    hidePhoneBargeBanner();
    setCompanionPhase("primary", { announceChat: !message.sync });
    showPhonePrimaryOverlay(message.seconds);
    void ensureBackgroundPlaying("primary_start");
  } else if (message.action === "companion_start") {
    const alreadyCompanion = state.companionWindow;
    hidePhonePassiveExit();
    hidePhoneRestOverlay();
    hidePhonePrimaryOverlay();
    hidePhoneBargeBanner();
    state.companionWindowNumber = Math.max(1, Number(message.window) || 1);
    setCompanionPhase("companion", { announceChat: !alreadyCompanion && !message.sync });
    void ensureBackgroundPlaying("companion_start");
    if (!alreadyCompanion) {
      navigator.vibrate?.([150, 80, 150]);
      send({ type: "companion_state", action: "window_start", window: message.window });
    }
  } else if (message.action === "companion_stop") {
    hidePhoneBargeBanner();
    hidePhonePrimaryOverlay();
    // Between companion and the one-minute rest notice, keep talk disabled.
    // rest_start will present the full rest overlay with countdown.
    if ($("#phone-rest-overlay")?.classList.contains("hidden")) {
      setCompanionPhase("rest", { announceChat: !message.sync });
    }
    void ensureBackgroundPlaying("companion_stop");
    send({ type: "companion_state", action: "window_end", window: message.window });
  } else if (message.action === "rest_start") {
    hidePhoneBargeBanner();
    hidePhonePrimaryOverlay();
    const seconds = Math.max(0, Math.floor(Number(message.seconds) || 60));
    const alreadyResting = !$("#phone-rest-overlay")?.classList.contains("hidden");
    showPhoneRestOverlay(seconds);
    if (!alreadyResting && !message.sync) {
      addChatLine(I18N.t(COMPANION_PHASE_COPY.rest.chat), "system");
      navigator.vibrate?.([120, 60, 120]);
    }
    pauseBackgroundForRest();
  } else if (message.action === "rest_end") {
    hidePhoneRestOverlay();
    setCompanionPhase("waiting", { announceChat: !message.sync });
    resumeBackgroundAfterRest();
  } else if (message.action === "experimental_barge") {
    showPhoneBargeBanner(message.agent_state || "thinking");
  } else if (message.action === "experimental_barge_clear") {
    hidePhoneBargeBanner();
  } else if (message.action === "pause") {
    state.interactionPresenceState = "hidden";
    renderPhoneContinuity("rest");
    pausePhonePrimaryOverlayCountdown();
    pausePhoneRestOverlayCountdown();
    state.backgroundPlaybackHeld = true;
    state.backgroundAudio.pause();
    stopRecognition();
    $("#background-status").textContent = "已暂停";
    if (message.terminal === true) showPhonePassiveExit(message.reason);
  } else if (message.action === "resume") {
    resumePhonePrimaryOverlayCountdown();
    resumePhoneRestOverlayCountdown();
    state.backgroundPlaybackHeld = false;
    state.backgroundShouldPlay = true;
    await ensureBackgroundPlaying("resume");
  }
}

function pauseBackgroundForRest() {
  state.backgroundRestHeld = true;
  state.backgroundPttHeld = false;
  state.talkPointerDown = false;
  stopRecognition();
  state.backgroundAudio?.pause();
  $("#background-status").textContent = "休息中暂停";
}

function resumeBackgroundAfterRest() {
  state.backgroundRestHeld = false;
  void ensureBackgroundPlaying("rest_end");
}

function pauseBackgroundForPtt() {
  if (!state.backgroundShouldPlay || state.backgroundRestHeld) return;
  state.backgroundPttHeld = true;
  state.backgroundAudio?.pause();
  $("#background-status").textContent = "说话时暂停";
}

function resumeBackgroundAfterPtt() {
  if (!state.backgroundPttHeld) return;
  state.backgroundPttHeld = false;
  void ensureBackgroundPlaying("phone_ptt_end");
}

$("#phone-talk-btn")?.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  if (!state.companionWindow) return;
  state.talkPointerDown = true;
  pauseBackgroundForPtt();
  if (event.currentTarget?.setPointerCapture) {
    try { event.currentTarget.setPointerCapture(event.pointerId); } catch (_) { /* Ignore unsupported capture. */ }
  }
  void startRecognition();
});

$("#phone-talk-btn")?.addEventListener("pointerup", (event) => {
  event.preventDefault();
  state.talkPointerDown = false;
  if (event.currentTarget?.releasePointerCapture) {
    try { event.currentTarget.releasePointerCapture(event.pointerId); } catch (_) { /* Ignore unsupported capture. */ }
  }
  stopRecognition();
});

$("#phone-talk-btn")?.addEventListener("pointercancel", (event) => {
  event.preventDefault();
  state.talkPointerDown = false;
  stopRecognition();
});

$("#phone-talk-btn")?.addEventListener("contextmenu", (event) => {
  event.preventDefault();
});

$("#phone-passive-exit-ok")?.addEventListener("click", () => {
  hidePhonePassiveExit();
  setCompanionPhase("waiting");
  $("#phone-mode-copy").textContent = I18N.t("请在电脑端重新完成准备，等待下一次开始。");
});

function flushLocalAsrTranscript() {
  const transcript = [state.localAsrFinal, state.localAsrInterim]
    .filter(Boolean)
    .join(" ")
    .trim();
  state.localAsrFinal = "";
  state.localAsrInterim = "";
  return transcript;
}

async function startRecognition() {
  if (
    state.listening
    || state.recognitionStartPending
    || state.ttsPlaying
    || !state.companionWindow
  ) {
    return;
  }
  noteCompanionUserActivity();
  // Primary: desktop ASR relay. Phone-local only after an explicit usable failover.
  if (shouldUseDesktopAsrRelay()) {
    if (state.socket?.readyState === WebSocket.OPEN) {
      startDesktopRelayPtt();
      return;
    }
    // Pair blip — reopen and ask for another press; do not dump into phone ASR.
    connectSocket({ force: false });
    $("#phone-active-error").textContent = I18N.t(
      "正在重新连接电脑端识别，连接恢复后请再按住说话。",
    );
    return;
  }
  await startLocalPhoneAsr({ failover: state.asrFailoverToPhone });
}

async function startLocalPhoneAsr({ failover = false } = {}) {
  if (
    state.listening
    || state.recognitionStartPending
    || state.ttsPlaying
    || !state.companionWindow
  ) {
    return;
  }
  state.asrFailoverToPhone = false;
  showDesktopAsrRetryMessage();
  if (state.socket?.readyState !== WebSocket.OPEN) connectSocket({ force: false });
}

function stopRecognition() {
  resumeBackgroundAfterPtt();
  // Desktop relay PTT (primary path) — wait for desktop final via stop.
  if (state.desktopAsrRelayPending) {
    stopDesktopRelayPtt(true);
    return;
  }
  const recognition = state.recognition;
  if (!recognition) {
    // Cancel a start that has not attached recognition yet.
    state.recognitionStartToken += 1;
    state.recognitionStartPending = false;
    if (state.listening) resetTalkButton();
    return;
  }
  state.recognitionStartPending = false;
  // Wait for onend so interim/final transcripts can flush.
  try {
    recognition.stop();
  } catch (_) {
    if (state.recognition === recognition) state.recognition = null;
    const transcript = flushLocalAsrTranscript();
    resetTalkButton();
    if (transcript) void respondToTranscript(transcript);
  }
}

function startDesktopRelayPtt() {
  $("#phone-active-error").textContent = "";
  state.listening = true;
  state.desktopAsrRelayPending = true;
  $("#phone-talk-btn").classList.add("listening");
  setPhoneTalkHint("listening");
  send({ type: "phone_ptt", action: "start" });
  send({ type: "companion_state", action: "listening_start", mode: "desktop_relay" });
  clearTimeout(state.desktopAsrRelayTimer);
  state.desktopAsrRelayTimer = setTimeout(() => {
    if (!state.desktopAsrRelayPending) return;
    // Timed out waiting for desktop ASR — fail over to phone-local backup.
    try { send({ type: "phone_ptt", action: "stop" }); } catch (_) { /* Ignore. */ }
    failoverDesktopAsrToPhone("desktop_relay_timeout");
  }, 8000);
}

function stopDesktopRelayPtt(sendStop = true) {
  if (!state.listening && !state.desktopAsrRelayPending) return;
  state.listening = false;
  state.desktopAsrRelayPending = false;
  clearTimeout(state.desktopAsrRelayTimer);
  state.desktopAsrRelayTimer = null;
  $("#phone-talk-btn").classList.remove("listening");
  setPhoneTalkHint(state.companionWindow ? "idle" : "disabled");
  if (sendStop) send({ type: "phone_ptt", action: "stop" });
  send({ type: "companion_state", action: "listening_end", mode: "desktop_relay" });
  // Keep live text until final arrives; only clear when explicitly idle-stopping
  // without an in-flight final (handled in handleDesktopAsrRelay).
}

function setPhoneAsrLive(text) {
  const live = $("#phone-asr-live");
  if (!live) return;
  const value = String(text || "").trim();
  live.textContent = value
    ? `${I18N.t("电脑正在识别：")}${value}`
    : "";
  live.classList.toggle("hidden", !value);
}

function handleDesktopAsrRelay(message) {
  if (message.action === "listening") {
    setPhoneAsrLive("");
    $("#phone-active-error").textContent = "";
    // Desktop acknowledged the relay; extend the wait for finals a bit.
    clearTimeout(state.desktopAsrRelayTimer);
    state.desktopAsrRelayTimer = setTimeout(() => {
      if (!state.desktopAsrRelayPending) return;
      try { send({ type: "phone_ptt", action: "stop" }); } catch (_) { /* Ignore. */ }
      failoverDesktopAsrToPhone("desktop_relay_listening_timeout");
    }, 12000);
    return;
  }
  if (message.action === "interim") {
    setPhoneAsrLive(message.transcript || "");
    return;
  }
  if (message.action === "final") {
    stopDesktopRelayPtt(false);
    setPhoneAsrLive("");
    const transcript = String(message.transcript || "").trim();
    if (transcript) {
      // Desktop primary succeeded; keep using it for later turns.
      state.asrFailoverToPhone = false;
      void respondToTranscript(transcript);
      return;
    }
    // Empty transcript = desktop path failed for this utterance → phone backup.
    failoverDesktopAsrToPhone("desktop_relay_empty_final");
    return;
  }
  if (message.action === "error") {
    const code = String(message.error || "");
    // Chrome fires aborted when the desktop relay session is stopped/restarted;
    // that is expected and must not surface as a phone-side failure.
    if (code === "aborted" || code === "no-speech") return;
    try { send({ type: "phone_ptt", action: "stop" }); } catch (_) { /* Ignore. */ }
    failoverDesktopAsrToPhone(code || "desktop_relay_error");
  }
}

function resetTalkButton() {
  state.listening = false;
  $("#phone-talk-btn").classList.remove("listening");
  setPhoneTalkHint(state.companionWindow ? "idle" : "disabled");
  send({ type: "companion_state", action: "listening_end" });
}

async function respondToTranscript(transcript) {
  if (!transcript) return;
  state.companionInviteUsed = true;
  clearCompanionInvite();
  addChatLine(`你：${transcript}`, "user");
  send({ type: "companion_state", action: "utterance", transcript_chars: transcript.length });
  try {
    const response = await api("/api/chat", {
      method: "POST",
      body: { session_id: state.sessionId, source: "companion", transcript },
    });
    addChatLine(`陪伴伙伴：${response.text}`, "assistant");
    await playSpeech(response.text);
  } catch (err) {
    $("#phone-active-error").textContent = `对话失败：${err.message}`;
  }
}

async function playSpeech(text) {
  const before = state.backgroundAudio.currentTime;
  state.ttsPlaying = true;
  try {
    const response = await api("/api/tts", {
      method: "POST",
      body: { session_id: state.sessionId, text },
    });
    const data = await (await response.blob()).arrayBuffer();
    const audioBuffer = await state.context.decodeAudioData(data.slice(0));
    const source = state.context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(state.speechGain).connect(state.context.destination);
    send({
      type: "companion_state",
      action: "tts_start",
      mode: "server_web_audio",
      provider: response.headers.get("X-TTS-Provider") || "iflytek",
      voice: response.headers.get("X-TTS-Voice") || "",
      synthesis_latency_ms: Number(response.headers.get("X-TTS-Latency-Ms") || 0),
      background_time: before,
      fallback: false,
    });
    source.start();
    // Some mobile WebViews pause an HTMLMediaElement when WebAudio speech
    // takes audio focus. Restore the already-authorized background element so
    // speech and background remain mixed during the companion reply.
    scheduleUnexpectedBackgroundRecovery("companion_tts_started", 80);
    await new Promise((resolve) => { source.onended = resolve; });
  } catch (err) {
    send({
      type: "companion_state",
      action: "tts_failed",
      message: err.message,
      fallback_used: false,
    });
    $("#phone-active-error").textContent = `服务端语音暂时不可用：${err.message}。电脑端已暂停，请稍后重试。`;
    throw err;
  } finally {
    state.ttsPlaying = false;
    await ensureBackgroundPlaying("companion_tts_ended");
  }
  send({
    type: "companion_state",
    action: "tts_end",
    background_time_before: before,
    background_time_after: state.backgroundAudio.currentTime,
    background_paused: state.backgroundAudio.paused,
    background_should_play: state.backgroundShouldPlay,
    background_playback_held: state.backgroundPlaybackHeld,
    background_rest_held: state.backgroundRestHeld,
    background_ptt_held: state.backgroundPttHeld,
  });
}

function addChatLine(text, kind) {
  const paragraph = document.createElement("p");
  paragraph.className = `chat-line ${kind}`;
  paragraph.textContent = text;
  $("#phone-chat-log").appendChild(paragraph);
  $("#phone-chat-log").scrollTop = $("#phone-chat-log").scrollHeight;
}

function send(message) {
  if (state.socket?.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify({ ...message, client_ms: Date.now(), visibility: document.visibilityState }));
  }
}

function startHeartbeat() {
  clearInterval(state.heartbeat);
  state.heartbeat = setInterval(() => {
    $("#background-time").textContent = formatTime(state.backgroundAudio?.currentTime || 0);
    if (
      state.backgroundShouldPlay
      && !state.backgroundPlaybackHeld
      && !state.backgroundRestHeld
      && !state.backgroundPttHeld
      && state.backgroundAudio?.paused
    ) {
      void ensureBackgroundPlaying("heartbeat");
    }
    if (
      !state.companionWindow
      && !$("#phone-primary-overlay")?.classList.contains("hidden")
      && phonePrimaryRemainingSeconds() <= 0
    ) {
      markPhonePrimaryOverlayAwaitingCompanion();
    }
    send({
      type: "heartbeat",
      role: "companion",
      background_time: state.backgroundAudio?.currentTime || 0,
      background_paused: Boolean(state.backgroundAudio?.paused),
      background_should_play: state.backgroundShouldPlay,
      background_rest_held: state.backgroundRestHeld,
      background_ptt_held: state.backgroundPttHeld,
      background_playhead_preserved: true,
      audio_context: state.context?.state,
    });
  }, 5000);
}

async function requestWakeLock() {
  try {
    state.wakeLock = await navigator.wakeLock?.request("screen");
  } catch (_) {
    // Heartbeats still reveal screen suspension.
  }
}

document.addEventListener("visibilitychange", async () => {
  send({ type: "heartbeat", role: "companion", visibility_change: document.visibilityState });
  if (document.visibilityState === "visible") {
    if (!state.wakeLock) await requestWakeLock();
    void ensureBackgroundPlaying("visibility_visible");
  }
});

bindBackgroundAudioTelemetry(state.backgroundAudio);

function setHeaderStatus(text, kind) {
  $("#phone-header-status").textContent = text;
  $("#phone-dot").className = `status-dot ${kind}`;
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

function formatTime(seconds) {
  const value = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function percentile(values, fraction) {
  if (!values.length) return Infinity;
  const index = Math.min(values.length - 1, Math.max(0, Math.ceil(values.length * fraction) - 1));
  return values[index];
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

function escapeAttribute(value) {
  return String(value).replace(/["&<>]/g, (char) => ({ '"': "&quot;", "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char]));
}

sessionStorage.removeItem(COMPANION_TOKEN_KEY);
state.token = "";

function applyPhoneCapacityStatus(capacity) {
  state.capacity = capacity || null;
  const banner = $("#phone-capacity-banner");
  const title = $("#phone-capacity-banner-title");
  const copy = $("#phone-capacity-banner-copy");
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
  }
  updateCompanionLoginSubmitEnabled();
}

async function refreshPhoneCapacityStatus() {
  try {
    const payload = await withTimeout(api("/api/capacity"), 2500, "capacity-timeout");
    applyPhoneCapacityStatus(payload.capacity);
    return payload.capacity;
  } catch (_) {
    return state.capacity;
  }
}

function startPhoneCapacityPolling() {
  clearInterval(state.capacityPollTimer);
  state.capacityPollTimer = setInterval(() => {
    void refreshPhoneCapacityStatus();
  }, 20000);
}

// Login UI is booted earlier via bootCompanionLoginUi() so later listeners cannot block it.
