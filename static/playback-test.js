/**
 * @file Browser playback-check utilities for confirming audible test signals and reporting the resulting quality measurement.
 */

"use strict";

(() => {
  const SENTENCES = Object.freeze({
    "zh-CN": Object.freeze({
      id: "zh-blue-kite-v1",
      text: "蓝色的风筝飞过安静的湖面。",
      keywords: Object.freeze(["蓝色", "风筝", "安静", "湖面"]),
    }),
    "en-US": Object.freeze({
      id: "en-blue-kite-v1",
      text: "The blue kite is flying over the quiet lake.",
      keywords: Object.freeze(["blue", "kite", "quiet", "lake"]),
    }),
  });

  function selected(language) {
    return SENTENCES[language] || SENTENCES["zh-CN"];
  }

  function normalize(language, value) {
    const compatible = String(value || "").normalize("NFKC").toLocaleLowerCase(
      language === "en-US" ? "en-US" : "zh-CN",
    );
    return compatible.replace(/[^\p{L}\p{N}]+/gu, "");
  }

  function evaluate(language, answer) {
    const test = selected(language);
    const normalized = normalize(language, answer);
    const matchedKeywords = test.keywords.filter((keyword) =>
      normalized.includes(normalize(language, keyword)));
    return {
      passed: normalized.length >= 6 && matchedKeywords.length === test.keywords.length,
      answer_chars: normalized.length,
      matched_keyword_count: matchedKeywords.length,
      required_keyword_count: test.keywords.length,
      phrase_id: test.id,
    };
  }

  function syncVerifyButton(input, button) {
    const hasInput = String(input?.value || "").length > 0;
    button.disabled = !hasInput;
    button.classList.toggle("btn-verify-ready", hasInput);
  }

  function resetVerifyButton(button) {
    button.disabled = true;
    button.classList.remove("btn-verify-ready");
  }

  async function pickVoice(availableVoices, language) {
    const prefix = language.toLowerCase().split("-")[0];
    const candidates = availableVoices.filter((candidate) =>
      String(candidate.lang || "").toLowerCase().startsWith(prefix));
    const preferredPatterns = [
      /xiaoxiao|xiaoyi|yunxi|yunjian|neural|natural|premium|ting-ting|sin-ji/i,
      /google.*中文|microsoft.*中文|microsoft.*chinese/i,
    ];
    for (const pattern of preferredPatterns) {
      const match = candidates.find((candidate) => pattern.test(candidate.name || ""));
      if (match) return match;
    }
    return candidates.find((candidate) => candidate.localService) || candidates[0] || null;
  }

  async function speakText(text, language, timeoutMs = 60000) {
    const synthesis = window.speechSynthesis;
    if (!synthesis || !window.SpeechSynthesisUtterance) {
      throw new Error("speech_synthesis_unavailable");
    }
    const spoken = String(text || "").trim();
    if (!spoken) throw new Error("speech_synthesis_empty_text");
    let availableVoices = synthesis.getVoices?.() || [];
    if (!availableVoices.length) {
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, 300);
        synthesis.onvoiceschanged = () => {
          clearTimeout(timer);
          resolve();
        };
      });
      availableVoices = synthesis.getVoices?.() || [];
    }
    const voice = await pickVoice(availableVoices, language);
    synthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(spoken);
    utterance.lang = language;
    utterance.rate = language === "en-US" ? 0.96 : 0.94;
    utterance.pitch = language === "en-US" ? 1.02 : 1.0;
    utterance.volume = 0.9;
    if (voice) utterance.voice = voice;
    await new Promise((resolve, reject) => {
      let settled = false;
      const finish = (error = null) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (error) reject(error);
        else resolve();
      };
      utterance.onend = () => finish();
      utterance.onerror = (event) => finish(new Error(event.error || "speech_synthesis_failed"));
      const timer = setTimeout(() => {
        synthesis.cancel();
        finish(new Error("speech_synthesis_timeout"));
      }, timeoutMs);
      synthesis.speak(utterance);
    });
    return {
      language,
      playback_api: "speechSynthesis",
      provider: "speechSynthesis",
      voice_name: voice?.name || "",
      voice_lang: voice?.lang || language,
      matching_voice_available: Boolean(voice),
      text_chars: spoken.length,
    };
  }

  async function speak(language, timeoutMs = 15000) {
    const test = selected(language);
    const result = await speakText(test.text, language, timeoutMs);
    return {
      ...result,
      phrase_id: test.id,
    };
  }

  window.TTA_PLAYBACK_TEST = Object.freeze({
    evaluate,
    resetVerifyButton,
    selected,
    speak,
    speakText,
    syncVerifyButton,
  });
})();
