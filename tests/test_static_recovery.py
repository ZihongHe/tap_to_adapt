"""Regression tests for static recovery.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "static"


def test_network_probe_is_bounded_and_timeout_samples_can_be_saved() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "const sampleTimeoutMs = 2000" in script
    assert "controller.abort()" in script
    assert "rtts.push(sampleTimeoutMs)" in script
    assert "successful_samples_ms: successfulRtts" in script
    assert "timeout_ms: sampleTimeoutMs" in script


def test_pair_status_only_reports_day_mismatch_when_both_roles_are_online() -> None:
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")
    companion_script = (STATIC / "companion.js").read_text(encoding="utf-8")

    expected = 'message.roles.includes("study") && message.roles.includes("companion")'
    assert expected in study_script
    assert expected in companion_script
    assert 'const dayMismatch = bothRoles && message.same_day_online !== true' in study_script


def test_open_session_restores_quality_without_repeating_passed_browser_probe() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "已通过（从本次记录恢复）" in script
    assert 'if (state.quality.get("browser") !== true) checkBrowser();' in script


def test_ready_step_requires_quality_and_exposes_machine_readable_blockers() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'if (!passed) markStep("network")' in script
    assert 'else if (!state.pairReady || !state.companionReady) markStep("phone")' in script
    assert 'if (state.pairReady) markStep("ready")' not in script
    assert "function startEligibilityBlockers()" in script
    assert "status.dataset.blockers" in script
    assert 'id="start-gate-status"' in page
    assert 'data-blocker-code="session_missing"' in page


def test_blocked_start_points_participant_to_the_phone_link_without_starting() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'id="companion-link-instruction"' in page
    assert "请现在用手机打开此专属链接" in page
    start_button = page.split('id="start-study-btn"', 1)[1].split("</button>", 1)[0]
    assert 'aria-disabled="true"' in start_button
    assert " disabled" not in start_button

    assert "function emphasizeCompanionLink()" in script
    assert "function explainBlockedStart(blockers)" in script
    assert 'code === "pair_not_connected" || code === "companion_not_ready"' in script
    assert 'instruction.classList.add("attention-highlight")' in script
    assert 'instruction.scrollIntoView({ behavior: "smooth", block: "center" })' in script
    assert 'if (blockers.length > 0) {' in script
    assert "explainBlockedStart(blockers);\n    return;" in script

    click_body = script.split(
        '$("#start-study-btn").addEventListener("click", async () => {', 1
    )[1].split("async function startExperimentRuntime", 1)[0]
    assert click_body.index("explainBlockedStart(blockers);") < click_body.index(
        "await startExperimentRuntime();"
    )
    assert ".companion-link-instruction.attention-highlight" in styles
    assert "@keyframes companion-link-attention" in styles


def test_mandatory_tap_practice_corrects_errors_without_entering_formal_learning() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert 'data-step="training"' in page
    assert 'id="tap-training-overlay"' in page
    assert 'id="training-tap-btn"' in page
    assert 'id="training-wait-btn"' in page
    assert "约 2 分钟 · 不进入正式计时与模型学习" in page
    assert "答错会立即解释原因，并要求在同一题重新选择" in page
    assert "说完且期待回答：激活" in page
    assert "还没开口/没说完，或正和手机对话：打断" in page
    assert "其他情况：不点击" in page
    assert "你还没开口、还没说完" in page
    assert "界面中的“激活 / 打断”按钮用于纠正响应时机" in page
    assert '"说完且期待回答：激活": "Finished and expecting an answer: activate"' in (
        STATIC / "i18n.js"
    ).read_text(encoding="utf-8")

    for scenario_id in (
        "manual_activate_after_wait",
        "appropriate_response",
        "premature_interruption",
        "companion_intrusion",
        "no_reply_needed",
    ):
        assert f'id: "{scenario_id}"' in script
    assert 'if (!correct) return;' in script
    assert 'if (!state.trainingPassed) {\n    openTapTraining();\n    return;\n  }' in script
    assert 'sessionStorage.setItem(trainingStorageKey(), "1")' in script
    assert 'sendEvent("training.answer"' in script
    assert 'sendEvent("training.completed"' in script
    assert 'id="training-agent-emotion"' in page
    assert 'static/emotions/wait.png' in page
    assert 'static/emotions/think.png' in script
    assert 'static/emotions/speak.png' in script
    assert '$("#training-tap-btn").textContent = I18N.t(presentation.action);' in script
    assert "!training && state.analysisEligible && !state.scheduledRest" in script
    assert ".tap-training-overlay" in styles
    assert ".training-feedback.incorrect" in styles


def test_tap_training_dialog_uses_readable_type_sizes() -> None:
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert "width: min(900px, 100%);" in styles
    assert "font-size: clamp(30px, 4vw, 40px);" in styles
    assert ".tap-training-dialog .helper-text" in styles
    assert ".training-rules strong" in styles
    assert "font-size: 18px;" in styles
    assert ".training-rules span" in styles
    assert "font-size: 16px;" in styles


def test_formal_interaction_uses_one_explicit_state_dependent_correction_button() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert '<section id="tap-zone"' in page
    assert '<button id="tap-action-btn"' in page
    assert 'id="agent-emotion"' in page
    assert 'id="agent-state-description"' in page
    assert '$("#tap-action-btn").addEventListener("pointerdown"' in script
    assert '$("#tap-zone").addEventListener("pointerdown"' not in script
    assert 'waiting: {' in script and 'intent: "manual_activate"' in script
    assert 'thinking: {' in script and 'intent: "interrupt"' in script
    assert 'speaking: {' in script
    assert '.tap-action-button' in styles
    assert '.emotion-frame' in styles


def test_stale_decision_context_is_acknowledged_without_pausing_the_study() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    server = (STATIC.parents[0] / "app" / "main.py").read_text(encoding="utf-8")

    assert 'if str(exc) == "decision context requires the waiting state":' in server
    assert '"type": "decision_context_ack"' in server
    assert '"accepted": False' in server
    assert '"stale": True' in server
    assert '"rejection_reason": "agent_not_waiting"' in server
    assert '"model.decision_context_stale"' in server
    assert "transition_id=payload.get(\"transition_id\")" in server

    assert "if (acknowledgement.accepted === false)" in script
    assert 'queueEvent("decision_context.stale_ack"' in script
    assert 'pauseSchedule("timing_model_error")' in script
    stale_branch = script.split("if (acknowledgement.accepted === false)", 1)[1].split(
        "state.decisionContextTarget", 1
    )[0]
    assert "pauseSchedule" not in stale_branch


def test_pair_socket_stops_when_the_interaction_window_finishes() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "state.pairReconnectEnabled = true;" in script
    assert "schedulePairReconnect();" in script
    assert "function stopPairConnection()" in script
    assert "function disablePairReconnect()" in script
    assert "state.pairReconnectEnabled = false;" in script

    finish_body = script.split("async function finishInteraction", 1)[1].split(
        "async function showQuestionnaire", 1
    )[0]
    assert 'pairSend({ type: "background_state", action: "stop" });' in finish_body
    assert "disablePairReconnect();" in finish_body
    assert "stopPairConnection();" not in finish_body

    closeout_body = script.split("async function completeDay", 1)[1].split(
        "function showCloseoutFailure", 1
    )[0]
    end_request = 'await api(`/api/sessions/${state.sessionId}/${endpoint}`, { method: "POST" });'
    assert end_request in closeout_body
    assert "throw error;" in closeout_body
    assert "stopPairConnection();" in closeout_body
    assert closeout_body.index(end_request) < closeout_body.index("stopPairConnection();")
    assert closeout_body.index("throw error;") < closeout_body.index("stopPairConnection();")


def test_closeout_failure_is_retryable_and_machine_readable() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")

    assert "error.closeoutFailed = true" in script
    assert "error.closeoutErrorCode" in script
    assert "status.dataset.errorCode = code" in script
    assert '$("#retry-closeout-btn").addEventListener("click"' in script
    assert 'id="closeout-status"' in page
    assert 'data-error-code=""' in page
    assert 'id="retry-closeout-btn"' in page
    assert "问卷及已提交的互动数据已经保存，但本次实验完成确认失败" in page


def test_login_submission_waits_for_application_ready() -> None:
    study_page = (STATIC / "index.html").read_text(encoding="utf-8")
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")
    companion_page = (STATIC / "companion.html").read_text(encoding="utf-8")
    companion_script = (STATIC / "companion.js").read_text(encoding="utf-8")

    assert '<html lang="zh-CN" data-app-ready="false">' in study_page
    assert 'id="enroll-submit" class="btn btn-primary" type="button" disabled' in study_page
    assert 'i18n.js?v=20260803-guarded-voice-v24' in study_page
    assert 'playback-test.js?v=20260731-playback-sentence-v9' in study_page
    assert 'app.js?v=20260803-guarded-voice-v24' in study_page
    assert "async function initializeApplication()" in study_script
    assert "await withTimeout(loadBootstrap(), 10000" in study_script
    assert 'enrollSubmit.disabled = false;\n  document.documentElement.dataset.appReady = "true";' in study_script
    assert 'document.documentElement.dataset.appReady = "true";' in study_script

    assert '<html lang="zh-CN" data-app-ready="false">' in companion_page
    assert 'id="phone-login-submit" class="btn btn-primary" type="submit" disabled' in companion_page
    assert 'i18n.js?v=20260801-dual-self-enrollment-v14' in companion_page
    assert 'playback-test.js?v=20260731-playback-sentence-v9' in companion_page
    assert 'companion.js?v=20260801-device-role-gate-v15' in companion_page
    assert companion_page.index('id="phone-reaction-check"') < companion_page.index(
        'id="phone-reaction-stage"'
    ) < companion_page.index('id="phone-browser-check"')
    assert 'id="phone-reaction-target" class="reaction-target hidden" type="button">点我（还差 6 次）' in companion_page
    assert 'target.textContent = I18N.t(`点我（还差 ${6 - trial} 次）`);' in companion_script
    assert '$("#reaction-target").textContent = I18N.t(`点我（还差 ${6 - trial} 次）`);' in study_script
    assert "phoneLoginSubmit.disabled = false;" in companion_script
    assert 'document.documentElement.dataset.appReady = "true";' in companion_script


def test_baseline_questionnaire_is_shown_only_after_study_session_creation() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    bootstrap_body = script.split("async function loadBootstrap()", 1)[1].split(
        "function responseExists", 1
    )[0]
    create_body = script.split(
        '$("#create-session-btn").addEventListener("click", async () => {', 1
    )[1].split("function prepareDailySetup", 1)[0]

    assert 'showQuestionnaire("baseline"' not in bootstrap_body
    assert "enterSetup();" in bootstrap_body
    assert "state.sessionId = session.id;" in create_body
    assert create_body.index("state.sessionId = session.id;") < create_body.index(
        'showQuestionnaire("baseline"'
    )
    assert "if (!state.sessionId)" in script
    assert "问卷尚未绑定电脑端实验会话" in script


def test_questionnaire_order_is_stably_randomized_and_audited() -> None:
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")
    admin_page = (STATIC / "admin.html").read_text(encoding="utf-8")
    admin_script = (STATIC / "admin.js").read_text(encoding="utf-8")

    assert "function questionnairePresentationOrder" in study_script
    assert "seededQuestionnaireRandom" in study_script
    assert "participant?.neutral_id" in study_script
    assert "question_order: presentationOrder" in study_script
    assert "locale: I18N.locale" in study_script
    assert 'id="participant-responses-table"' in admin_page
    assert "response.selections" in admin_script
    assert "所选选项内容" in admin_script


def test_emotion_asset_is_centered_by_its_crop_frame_in_training_and_study() -> None:
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert "position: relative;" in styles
    assert "top: 50%;" in styles
    assert "left: 50%;" in styles
    assert "transform: translate(-50%, -50%);" in styles
    assert ".training-emotion-frame .agent-emotion" not in styles


def test_compensation_identity_can_be_declined_without_blocking_the_study() -> None:
    study_page = (STATIC / "index.html").read_text(encoding="utf-8")
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'name="compensation_route" type="radio" value="provide"' in study_page
    assert 'name="compensation_route" type="radio" value="decline"' in study_page
    assert 'id="compensation-decline-submit"' in study_page
    assert 'id="compensation-decline-explanation"' in study_page
    assert 'id="compensation-debug-fill"' not in study_page
    assert "!state.bootstrap.compensation_registration_complete" in study_script
    assert 'await api("/api/compensation-decline"' in study_script
    assert 'decision: "decline"' in study_script
    assert "renderCompensationRouteChoice" in study_script


def test_companion_link_carries_the_code_and_both_microphones_check_normal_volume() -> None:
    study_page = (STATIC / "index.html").read_text(encoding="utf-8")
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")
    companion_page = (STATIC / "companion.html").read_text(encoding="utf-8")
    companion_script = (STATIC / "companion.js").read_text(encoding="utf-8")

    assert 'id="companion-url"' in study_page
    assert "`${companionBase}${encodeURIComponent(state.code)}${I18N.isEnglish ? \"?lang=en\" : \"\"}`" in study_script
    assert 'id="mic-btn"' in study_page
    assert "正常说话音量与环境噪声" in study_page
    assert "instructed_distance_cm: [30, 90]" in study_script

    assert "function directCodeFromLocation()" in companion_script
    assert "requestSubmit()" in companion_script
    assert 'id="phone-mic-test"' in companion_page
    assert "state.phoneMicPassed" in companion_script
    assert "instructed_distance_cm: [30, 90]" in companion_script
    assert "30–90 厘米" in companion_page
    for check_id in (
        "phone-network-check",
        "phone-mic-check",
        "phone-reaction-check",
        "phone-browser-check",
    ):
        assert f'id="{check_id}"' in companion_page
    for kind in (
        "phone_network",
        "phone_microphone",
        "phone_reaction_latency",
        "phone_browser",
    ):
        assert kind in companion_script
    assert 'id="phone-audio-confirm"' not in companion_page
    assert 'id="phone-audio-language"' in companion_page
    assert '<option value="zh-CN" selected>中文</option>' in companion_page
    assert '<option value="en-US">英文</option>' in companion_page
    assert 'id="phone-audio-answer"' in companion_page
    assert 'id="phone-audio-verify"' in companion_page
    assert "phone_quality: Object.fromEntries(state.phoneQuality)" in companion_script


def test_both_devices_require_typed_sentence_recall_for_audio_playback() -> None:
    study_page = (STATIC / "index.html").read_text(encoding="utf-8")
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")
    companion_page = (STATIC / "companion.html").read_text(encoding="utf-8")
    companion_script = (STATIC / "companion.js").read_text(encoding="utf-8")
    playback_script = (STATIC / "playback-test.js").read_text(encoding="utf-8")

    for page, prefix in ((study_page, "browser"), (companion_page, "phone")):
        assert f'id="{prefix}-audio-language"' in page
        assert f'id="{prefix}-audio-answer"' in page
        assert f'id="{prefix}-audio-verify"' in page
        assert '<option value="zh-CN" selected>中文</option>' in page
        assert '<option value="en-US">英文</option>' in page

    assert "window.TTA_PLAYBACK_TEST.speak(language)" in study_script
    assert "window.TTA_PLAYBACK_TEST.evaluate(language" in study_script
    assert "verification_method: \"typed_keyword_recall\"" in study_script
    assert "typed_answer_stored: false" in study_script
    assert "window.TTA_PLAYBACK_TEST.speak(language)" in companion_script
    assert "window.TTA_PLAYBACK_TEST.evaluate(language" in companion_script
    assert "verification_method: \"typed_keyword_recall\"" in companion_script
    assert "typed_answer_stored: false" in companion_script
    assert 'id: "zh-blue-kite-v1"' in playback_script
    assert 'id: "en-blue-kite-v1"' in playback_script
    assert 'keywords: Object.freeze(["蓝色", "风筝", "安静", "湖面"])' in playback_script
    assert 'keywords: Object.freeze(["blue", "kite", "quiet", "lake"])' in playback_script


def test_physical_device_role_gates_are_present_on_both_clients() -> None:
    study_page = (STATIC / "index.html").read_text(encoding="utf-8")
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")
    companion_page = (STATIC / "companion.html").read_text(encoding="utf-8")
    companion_script = (STATIC / "companion.js").read_text(encoding="utf-8")

    assert 'id="study-device-gate"' in study_page
    assert "无法开始，需要在电脑端打开实验系统" in study_script
    assert 'headers.set("X-TTA-Device-Class", CLIENT_DEVICE_CLASS)' in study_script
    assert 'id="companion-device-gate"' in companion_page
    assert "无法开始，需要在手机端打开陪伴系统" in companion_script
    assert 'headers.set("X-TTA-Device-Class", CLIENT_DEVICE_CLASS)' in companion_script


def test_topic_book_is_presented_after_admission_not_in_the_start_gate() -> None:
    study_page = (STATIC / "index.html").read_text(encoding="utf-8")
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'id="setup-topic-select"' not in study_page
    assert "topic_not_saved" not in study_page
    assert "topic_not_saved" not in study_script
    assert 'id="topic-drawer"' in study_page
    assert 'showView("study-view");\n    if (!state.topicSaved) await saveTopicSelection("free_topic", "free");' in study_script
    assert '$("#topic-drawer").classList.remove("hidden");' in study_script


def test_model_runtime_failure_has_a_specific_pause_reason() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'pauseSchedule("timing_model_error")' in script
    assert "计时模型发生运行错误" in script


def test_participants_can_exit_a_saved_number_without_deleting_server_data() -> None:
    study_page = (STATIC / "index.html").read_text(encoding="utf-8")
    study_script = (STATIC / "app.js").read_text(encoding="utf-8")
    companion_page = (STATIC / "companion.html").read_text(encoding="utf-8")
    companion_script = (STATIC / "companion.js").read_text(encoding="utf-8")

    assert 'id="logout-btn"' in study_page
    assert ">退出编号</button>" in study_page
    assert 'id="phone-logout-btn"' in companion_page
    assert "服务器上的知情同意、实验进度和已上传数据不会删除" in study_script
    assert 'localStorage.removeItem(storageKey("study_token"))' in study_script
    assert 'localStorage.removeItem(storageKey("study_code"))' in study_script
    assert "await withTimeout(flushEvents(true), 3000" in study_script
    assert "window.location.reload();" in study_script
    assert "服务器上的实验进度和已上传数据不会删除" in companion_script
    assert "sessionStorage.removeItem(COMPANION_TOKEN_KEY)" in companion_script
    assert "state.socket?.close();" in companion_script
    assert "/api/sessions/" not in study_script.split(
        "async function logoutParticipant()", 1
    )[1].split('$("#logout-btn").addEventListener', 1)[0]


def test_conditional_taps_wait_for_server_result_and_demo_pcm_has_one_scheduler() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "tapResultWaiters: new Map()" in script
    assert "window.tapStudyConditionalTap = (intent) => sendTapRequest(intent || {});" in script
    assert 'for (const key of ["intended_action", "expected_state", "expected_activation_id", "expected_state_version", "feedback_attribution"])' in script
    assert 'cancel_reason: conditionalSnapshotComplete ? "client_not_ready" : "conditional_snapshot_incomplete"' in script
    assert 'const intendedAction = expectedState === "waiting" ? "manual_activate" : "interrupt";' in script
    assert 'expected_activation_id: state.currentActivationId || ""' in script
    assert "expected_state_version: state.timingStateVersion" in script
    assert "sendTapRequest().catch" not in script
    assert "waiter.resolve(message);" in script
    assert 'message.type === "timing_error" && message.tap_id && state.tapResultWaiters.has(message.tap_id)' in script
    assert 'waiter.reject(new Error(message.code || message.message || "tap rejected"));' in script
    assert "window.tapStudyQueueVoicedPcm" in script
    assert 'state.demoPcmMode = "controlled";' in script
    assert "socket.send(samples.buffer.slice(0));" in script


def test_interrupt_cancels_pending_chat_tts_and_playback_without_resurrection() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "currentTurnAbort: null" in script
    assert "agentTurnGeneration: 0" in script
    assert "const controller = new AbortController();" in script
    assert "signal: controller.signal" in script
    assert "generation !== state.agentTurnGeneration" in script
    assert 'playback.settle("cancelled")' in script
    assert "state.currentTurnAbort.abort();" in script
    assert 'state.currentAudio = playback;' in script
    assert 'setAgentState("speaking");' in script
    assert script.index("state.currentAudio = playback;") < script.index('setAgentState("speaking");')


def test_network_gate_treats_one_timeout_as_loss_not_latency_p95() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "successfulRtts.sort((a, b) => a - b);" in script
    assert "const median = percentile(successfulRtts, 0.5);" in script
    assert "const p95 = percentile(successfulRtts, 0.95);" in script
    assert "const localPassed = successfulRtts.length >= 11" in script
    assert "failures <= 1" in script


def test_decision_context_uses_the_server_activation_gate_field() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert (
        "acknowledgement.context_allows_activation === true"
        in script
    )
    assert "acknowledgement.context_allows_reply === true" not in script

    capture = script.split("worklet.port.onmessage = (event) => {", 1)[1].split(
        "state.workletNode = worklet;", 1
    )[0]
    assert "preparePrimaryDecisionContext().catch" in capture
    assert capture.index("preparePrimaryDecisionContext().catch") < capture.index(
        "socket.send(event.data);"
    )


def test_standardized_rest_is_a_real_capture_and_learning_boundary() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")

    assert 'kind: "rest"' in script
    assert 'await setTimingPhase("rest", "A"' in script
    assert 'state.recorder?.state === "recording"' in script
    assert "state.recorder.pause()" in script
    assert "state.audioContext.suspend()" in script
    assert "state.recorder.resume()" in script
    assert "state.audioContext.resume()" in script
    assert "analysisEligible: false" in script
    assert "excluded_from_analysis: true" in script
    assert "&& !state.scheduledRest" in script
    assert 'id="rest-overlay"' in page
    assert 'id="rest-countdown"' in page
