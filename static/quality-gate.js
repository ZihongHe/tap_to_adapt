/**
 * @file Shared browser quality gates for audio capabilities, microphone measurements, playback, and reaction-time checks. Results are submitted to server-side validation before study entry.
 */

(function initQualityGate(global) {
  const NETWORK = Object.freeze({
    sample_count: 12,
    minimum_successful: 11,
    maximum_failures: 1,
    median_ms_max: 250,
    p95_ms_max: 800,
    jitter_ms_max: 200,
  });

  const MICROPHONE = Object.freeze({
    snr_db_min: 8,
    quiet_rms_p90_max: 0.09,
    speech_rms_p90_min: 0.006,
    clipping_ratio_max: 0.01,
  });

  // Phone mics are quieter / noisier at conversational distance; keep desktop stricter.
  const PHONE_MICROPHONE = Object.freeze({
    snr_db_min: 6,
    quiet_rms_p90_max: 0.09,
    speech_rms_p90_min: 0.0035,
    clipping_ratio_max: 0.01,
  });

  const REACTION = Object.freeze({
    minimum_samples: 6,
    median_ms_min: 100,
    median_ms_max: 900,
  });

  const KIND_LABELS = Object.freeze({
    network: "网络稳定性检测",
    phone_network: "手机网络稳定性检测",
    microphone: "麦克风与环境噪声检测",
    phone_microphone: "手机麦克风与环境噪声检测",
    reaction_latency: "点击反应延迟检测",
    phone_reaction_latency: "手机点击反应延迟检测",
    browser: "浏览器与声音播放检测",
    phone_browser: "手机浏览器与最低音量硬校准",
  });

  function round(value) {
    return Math.round(Number(value));
  }

  function round1(value) {
    return Math.round(Number(value) * 10) / 10;
  }

  function successfulCount(metrics) {
    if (Array.isArray(metrics.successful_samples_ms)) return metrics.successful_samples_ms.length;
    return Math.max(0, NETWORK.sample_count - Number(metrics.failures || 0));
  }

  function evaluateNetwork(metrics) {
    const successful = successfulCount(metrics);
    const failures = Number(metrics.failures || 0);
    const median = Number(metrics.median_ms);
    const p95 = Number(metrics.p95_ms);
    const jitter = Number(metrics.jitter_ms);
    const issues = [];
    if (successful < NETWORK.minimum_successful) {
      issues.push({
        label: "成功次数",
        actual: `${successful}/12`,
        expected: `≥ ${NETWORK.minimum_successful}/12`,
      });
    }
    if (failures > NETWORK.maximum_failures) {
      issues.push({
        label: "丢失",
        actual: `${failures}/12`,
        expected: `≤ ${NETWORK.maximum_failures}/12`,
      });
    }
    if (Number.isFinite(median) && median > NETWORK.median_ms_max) {
      issues.push({
        label: "中位延迟",
        actual: `${round(median)} ms`,
        expected: `≤ ${NETWORK.median_ms_max} ms`,
      });
    }
    if (Number.isFinite(p95) && p95 > NETWORK.p95_ms_max) {
      issues.push({
        label: "P95 延迟",
        actual: `${round(p95)} ms`,
        expected: `≤ ${NETWORK.p95_ms_max} ms`,
      });
    }
    if (Number.isFinite(jitter) && jitter > NETWORK.jitter_ms_max) {
      issues.push({
        label: "抖动",
        actual: `${round(jitter)} ms`,
        expected: `≤ ${NETWORK.jitter_ms_max} ms`,
      });
    }
    return { passed: issues.length === 0, issues };
  }

  function evaluateMicrophone(metrics, thresholds = MICROPHONE) {
    const rules = thresholds || MICROPHONE;
    const issues = [];
    const snr = Number(metrics.snr_db);
    const quiet = Number(metrics.quiet_rms_p90);
    const speech = Number(metrics.speech_rms_p90);
    const clipping = Number(metrics.clipping_ratio);
    if (Number.isFinite(quiet) && quiet >= rules.quiet_rms_p90_max) {
      issues.push({
        label: "环境噪声",
        actual: quiet.toFixed(3),
        expected: `< ${rules.quiet_rms_p90_max}`,
      });
    }
    if (Number.isFinite(speech) && speech <= rules.speech_rms_p90_min) {
      issues.push({
        label: "正常说话音量",
        actual: speech.toFixed(3),
        expected: `> ${rules.speech_rms_p90_min}`,
      });
    }
    if (Number.isFinite(clipping) && clipping >= rules.clipping_ratio_max) {
      issues.push({
        label: "削波比例",
        actual: clipping.toFixed(3),
        expected: `< ${rules.clipping_ratio_max}`,
      });
    }
    if (Number.isFinite(snr) && snr < rules.snr_db_min) {
      issues.push({
        label: "信噪比 SNR",
        actual: `${round1(snr)} dB`,
        expected: `≥ ${rules.snr_db_min} dB`,
      });
    }
    const semanticConfirmation = metrics.asr_verification_version === "semantic-confirmation-v1";
    // New pages require the participant to confirm that the displayed
    // transcript preserves the meaning. Legacy pages retain soft ASR so a hot
    // update cannot invalidate an already-running setup.
    if (semanticConfirmation && (
      metrics.asr_user_confirmed !== true
      || !String(metrics.asr_probe_transcript || "").trim()
    )) {
      issues.push({
        label: "语音识别语义确认",
        actual: "未通过",
        expected: "确认大意正确",
      });
    } else if (!semanticConfirmation) {
      const acousticOk = issues.length === 0;
      if (metrics.asr_probe_passed === false && !acousticOk) {
        issues.push({
          label: "语音识别",
          actual: "未通过",
          expected: "通过",
        });
      }
    }
    return { passed: issues.length === 0, issues };
  }

  function evaluateReaction(metrics) {
    const samples = Array.isArray(metrics.samples_ms) ? metrics.samples_ms.length : 0;
    const median = Number(metrics.median_ms);
    const issues = [];
    if (samples < REACTION.minimum_samples) {
      issues.push({
        label: "有效点击",
        actual: `${samples} 次`,
        expected: `≥ ${REACTION.minimum_samples} 次`,
      });
    }
    if (Number.isFinite(median) && median < REACTION.median_ms_min) {
      issues.push({
        label: "反应中位数",
        actual: `${round(median)} ms`,
        expected: `≥ ${REACTION.median_ms_min} ms`,
      });
    }
    if (Number.isFinite(median) && median > REACTION.median_ms_max) {
      issues.push({
        label: "反应中位数",
        actual: `${round(median)} ms`,
        expected: `≤ ${REACTION.median_ms_max} ms`,
      });
    }
    return { passed: issues.length === 0, issues };
  }

  function formatIssues(issues) {
    if (!issues?.length) return "";
    return issues
      .map((item) => `${item.label} ${item.actual}（要求 ${item.expected}）`)
      .join("；");
  }

  function formatNetworkSummary(metrics, passed, serverDecision) {
    const summary = `中位 ${round(metrics.median_ms)} ms · P95 ${round(metrics.p95_ms)} ms · 抖动 ${round(metrics.jitter_ms)} ms · 丢失 ${Number(metrics.failures || 0)}/12`;
    if (passed) return summary;
    const serverIssues = serverDecision?.issues;
    if (serverIssues?.length) return `未通过：${formatIssues(serverIssues)}`;
    const evaluation = evaluateNetwork(metrics);
    const detail = formatIssues(evaluation.issues);
    return detail ? `未通过：${detail}` : summary;
  }

  function formatMicrophoneSummary(metrics, passed, serverDecision) {
    const summary = `SNR ${round1(metrics.snr_db)} dB · 环境噪声 ${Number(metrics.quiet_rms_p90 || 0).toFixed(3)} · 说话音量 ${Number(metrics.speech_rms_p90 || 0).toFixed(3)}`;
    if (passed) return summary;
    const serverIssues = serverDecision?.issues;
    if (serverIssues?.length) return `未通过：${formatIssues(serverIssues)}`;
    const evaluation = evaluateMicrophone(metrics);
    const detail = formatIssues(evaluation.issues);
    return detail ? `未通过：${detail}` : summary;
  }

  function formatReactionSummary(metrics, passed, serverDecision) {
    const summary = `反应中位数 ${round(metrics.median_ms)} ms`;
    if (passed) return summary;
    const serverIssues = serverDecision?.issues;
    if (serverIssues?.length) return `未通过：${formatIssues(serverIssues)}`;
    const evaluation = evaluateReaction(metrics);
    const detail = formatIssues(evaluation.issues);
    return detail ? `未通过：${detail}` : summary;
  }

  const PHONE_BROWSER_MISSING = Object.freeze({
    keyword_recall: "听辨内容未匹配，请重听后完整输入听到的句子",
    playback_verified: "听辨内容未匹配，请重听后完整输入听到的句子",
    answer_length: "听辨内容过短，请完整输入听到的句子",
    calibration_output_gain: "校准播放增益未按 0.5 记录，不是句子输错。请强制刷新陪伴页后重新播放并核对；若仍失败，请用系统浏览器打开陪伴链接",
    minimum_volume_probe: "还没有播放最低音量校准句，请先点击播放",
    hard_volume_gate: "最低音量硬校准记录不完整，请重新播放校准句",
    media_volume_minimum: "请将系统媒体音量调到至少 50%",
    minimum_confirmed: "请勾选“系统媒体音量已调到至少 50%”",
    phrase_id: "校准句记录不完整，请重新播放并核对",
    test_language: "校准语言记录不完整，请重新播放并核对",
    verification_method: "听辨核对方式记录不完整，请重新播放并核对",
    playback_api: "声音播放接口记录不完整，请重新播放并核对",
    speech_recognition: "当前手机浏览器未提供语音识别接口，请刷新后重试",
    secure_context: "当前页面不是安全上下文，请通过陪伴链接重新打开",
    media_devices: "当前浏览器无法访问媒体设备",
    audio_context: "当前浏览器无法创建音频上下文",
    websocket: "当前浏览器不支持 WebSocket",
    audio_element: "当前浏览器无法播放音频",
  });

  function formatPhoneBrowserSummary(passed, serverDecision, evaluation) {
    if (passed) return "最低音量硬校准通过，可以连接电脑";
    const missing = Array.isArray(serverDecision?.missing)
      ? serverDecision.missing
      : [];
    const reasons = [...new Set(
      missing.map((name) => PHONE_BROWSER_MISSING[name] || `未满足条件：${name}`),
    )];
    if (reasons.length) return `未通过：${reasons.join("；")}`;
    if (evaluation && evaluation.passed === false) {
      return "听辨内容未匹配，请重听后完整输入听到的句子";
    }
    return "最低音量硬校准未通过，请按卡片提示重试";
  }

  function formatSaveError(err, kind) {
    const message = String(err?.message || err || "");
    const label = KIND_LABELS[kind] || "该项检测";
    return `${label}结果未能保存：${message}`;
  }

  global.TTA_QUALITY_GATE = Object.freeze({
    NETWORK,
    MICROPHONE,
    PHONE_MICROPHONE,
    REACTION,
    evaluateNetwork,
    evaluateMicrophone,
    evaluateReaction,
    formatIssues,
    formatNetworkSummary,
    formatMicrophoneSummary,
    formatReactionSummary,
    formatPhoneBrowserSummary,
    formatSaveError,
  });
})(window);
