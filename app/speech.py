"""Speech synthesis and role-specific voice configuration.

SpeechService supports the configured synthesis provider and voice assignment,
converts responses to playable audio, and reports synthesis failures. It
produces response content without updating the response-initiation model.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import subprocess
import threading
import time
import wave
from email.utils import formatdate
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import websocket

from .config import Settings


class SpeechSynthesisError(RuntimeError):
    pass


class SpeechService:
    MINIMUM_STREAMING_PCM_SAMPLES = 6_400
    VOICE_COUNTERBALANCE_NAMESPACE = b"tap-to-adapt-voice-counterbalance-v1"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0
        self._offline_tts = None

    @property
    def provider(self) -> str:
        return (
            "sherpa_onnx_vits_aishell3"
            if self.settings.tts_mode in {"sherpa_onnx", "hybrid"}
            else "iflytek"
        )

    @property
    def voice(self) -> str:
        if self.settings.tts_mode in {"sherpa_onnx", "hybrid"}:
            return f"aishell3-sid{self.settings.sherpa_tts_primary_speaker_id}"
        return self.settings.xunfei_primary_voice

    def voice_assignment_for(self, participant_id: str) -> str:
        """Return a participant-stable Bernoulli(0.5) A/B role assignment."""
        if not participant_id:
            raise SpeechSynthesisError("Participant ID is required for voice assignment")
        digest = hmac.new(
            self.settings.code_pepper.encode("utf-8"),
            self.VOICE_COUNTERBALANCE_NAMESPACE + b":" + participant_id.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return "AB" if digest[0] < 128 else "BA"

    def voice_for(self, client_role: str, *, participant_id: str | None = None) -> str:
        if client_role not in {"study", "companion"}:
            raise SpeechSynthesisError("Unknown TTS client role")
        if self.settings.tts_mode in {"sherpa_onnx", "hybrid"}:
            if client_role == "companion":
                speaker_id = self.settings.sherpa_tts_companion_speaker_id
            else:
                speaker_id = self.settings.sherpa_tts_primary_speaker_id
            return f"aishell3-sid{speaker_id}"
        voice_a = self.settings.xunfei_primary_voice
        voice_b = self.settings.xunfei_companion_voice
        if not participant_id:
            return voice_a if client_role == "study" else voice_b
        assignment = self.voice_assignment_for(participant_id)
        if assignment == "AB":
            return voice_a if client_role == "study" else voice_b
        return voice_b if client_role == "study" else voice_a

    def voice_counterbalance(self, participant_id: str) -> dict[str, str]:
        assignment = self.voice_assignment_for(participant_id)
        return {
            "scheme": "participant_stable_bernoulli_50_50_v1",
            "assignment": assignment,
            "study_voice": self.voice_for("study", participant_id=participant_id),
            "companion_voice": self.voice_for("companion", participant_id=participant_id),
            "voice_a": self.settings.xunfei_primary_voice
            if self.settings.tts_mode == "xunfei"
            else f"aishell3-sid{self.settings.sherpa_tts_primary_speaker_id}",
            "voice_b": self.settings.xunfei_companion_voice
            if self.settings.tts_mode == "xunfei"
            else f"aishell3-sid{self.settings.sherpa_tts_companion_speaker_id}",
        }

    @property
    def speed(self) -> float:
        return (
            self.settings.sherpa_tts_speed
            if self.settings.tts_mode == "sherpa_onnx"
            else 50.0
        )

    @property
    def sample_rate_hz(self) -> int:
        return 16000

    @property
    def output_format(self) -> str:
        return "mp3"

    @property
    def available(self) -> bool:
        if self.settings.tts_mode in {"sherpa_onnx", "hybrid"}:
            model_dir = Path(self.settings.sherpa_tts_model_dir)
            local_ready = all(
                (model_dir / filename).is_file()
                for filename in (
                    "model.onnx",
                    "lexicon.txt",
                    "tokens.txt",
                    "phone.fst",
                    "date.fst",
                    "number.fst",
                )
            )
            return local_ready
        return self.settings.tts_mode == "xunfei" and all(
            (
                self.settings.xunfei_app_id,
                self.settings.xunfei_api_key,
                self.settings.xunfei_api_secret,
            )
        )

    def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        if not self.available:
            raise SpeechSynthesisError("Server TTS is not configured")
        encoded = text.encode("utf-8")
        if not encoded or len(encoded) >= 8000:
            raise SpeechSynthesisError("TTS text length is invalid")

        last_error: SpeechSynthesisError | None = None
        with self._request_lock:
            for attempt, retry_delay in enumerate((0.0, 0.4, 1.0), start=1):
                if retry_delay:
                    time.sleep(retry_delay)
                since_last = time.monotonic() - self._last_request_at
                if since_last < 0.15:
                    time.sleep(0.15 - since_last)
                self._last_request_at = time.monotonic()
                try:
                    return self._synthesize_once(text, encoded, voice=voice)
                except SpeechSynthesisError as exc:
                    last_error = exc
                    if attempt == 3:
                        break
        raise SpeechSynthesisError(
            f"Server TTS failed after 3 attempts: {last_error}"
        ) from last_error

    def _synthesize_once(
        self, text: str, encoded: bytes, *, voice: str | None = None
    ) -> bytes:
        if self.settings.tts_mode in {"sherpa_onnx", "hybrid"}:
            return self._synthesize_sherpa(
                text, speaker_id=self._speaker_id_from_voice(voice)
            )
        return self._synthesize_xunfei(encoded, voice=voice)

    def _speaker_id_from_voice(self, voice: str | None) -> int:
        if not voice:
            return self.settings.sherpa_tts_primary_speaker_id
        prefix = "aishell3-sid"
        if not voice.startswith(prefix):
            raise SpeechSynthesisError("Local TTS voice must use aishell3-sid<N>")
        try:
            return int(voice[len(prefix) :])
        except ValueError as exc:
            raise SpeechSynthesisError("Local TTS speaker ID is invalid") from exc

    def _sherpa_samples(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        speed: float | None = None,
    ) -> tuple[np.ndarray, int]:
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise SpeechSynthesisError("sherpa-onnx is not installed") from exc

        model_dir = Path(self.settings.sherpa_tts_model_dir)
        selected_speaker = (
            self.settings.sherpa_tts_primary_speaker_id
            if speaker_id is None
            else int(speaker_id)
        )
        selected_speed = (
            self.settings.sherpa_tts_speed if speed is None else float(speed)
        )
        speakers_path = model_dir / "speakers.txt"
        if speakers_path.is_file():
            speaker_count = sum(
                bool(line.strip())
                for line in speakers_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            )
            if not 0 <= selected_speaker < speaker_count:
                raise SpeechSynthesisError(
                    f"AISHELL3 speaker id must be in [0, {speaker_count - 1}]"
                )
        elif selected_speaker != self.settings.sherpa_tts_primary_speaker_id:
            raise SpeechSynthesisError(
                "Multi-speaker synthesis requires the locked speakers.txt manifest"
            )
        if not 0.75 <= selected_speed <= 1.25:
            raise SpeechSynthesisError("Simulation TTS speed must be in [0.75, 1.25]")
        if self._offline_tts is None:
            vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_dir / "model.onnx"),
                lexicon=str(model_dir / "lexicon.txt"),
                tokens=str(model_dir / "tokens.txt"),
            )
            model = sherpa_onnx.OfflineTtsModelConfig(
                vits=vits,
                provider="cpu",
                debug=False,
                num_threads=max(1, self.settings.sherpa_tts_num_threads),
            )
            config = sherpa_onnx.OfflineTtsConfig(
                model=model,
                rule_fsts=",".join(
                    str(model_dir / filename)
                    for filename in ("phone.fst", "date.fst", "number.fst")
                ),
                max_num_sentences=1,
            )
            if not config.validate():
                raise SpeechSynthesisError("The local sherpa-onnx TTS config is invalid")
            self._offline_tts = sherpa_onnx.OfflineTts(config)

        generation = sherpa_onnx.GenerationConfig()
        generation.sid = selected_speaker
        generation.speed = selected_speed
        generation.silence_scale = 0.2
        try:
            audio = self._offline_tts.generate(text, generation)
        except (RuntimeError, ValueError) as exc:
            raise SpeechSynthesisError(f"Local TTS generation failed: {exc}") from exc
        samples = np.asarray(audio.samples, dtype=np.float32)
        if not samples.size or int(audio.sample_rate) <= 0:
            raise SpeechSynthesisError("Local TTS returned no audio")
        return samples, int(audio.sample_rate)

    def synthesize_pcm(
        self,
        text: str,
        *,
        speaker_id: int,
        speed: float = 1.0,
    ) -> bytes:
        """Return simulation-only mono signed 16-bit PCM at 16 kHz."""
        if self.settings.tts_mode not in {"sherpa_onnx", "hybrid"} or not self.available:
            raise SpeechSynthesisError(
                "Multi-voice simulation requires the locked local sherpa-onnx TTS"
            )
        encoded = text.encode("utf-8")
        if not encoded or len(encoded) >= 8000:
            raise SpeechSynthesisError("TTS text length is invalid")
        with self._request_lock:
            samples, sample_rate = self._sherpa_samples(
                text, speaker_id=speaker_id, speed=speed
            )
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2", copy=False)
        if sample_rate == self.sample_rate_hz:
            pcm_16k = pcm
        else:
            duration = len(pcm) / max(sample_rate, 1)
            target_size = max(1, int(round(duration * self.sample_rate_hz)))
            positions = np.linspace(0, max(len(pcm) - 1, 0), target_size)
            resampled = np.interp(
                positions,
                np.arange(len(pcm), dtype=np.float64),
                pcm.astype(np.float64),
            )
            pcm_16k = np.clip(resampled, -32768, 32767).astype("<i2")
        return self._pad_streaming_pcm(pcm_16k)

    @classmethod
    def _pad_streaming_pcm(cls, pcm: np.ndarray) -> bytes:
        """Right-pad sub-400 ms utterances; never stretch or repeat speech."""
        pcm = np.asarray(pcm, dtype="<i2")
        missing = cls.MINIMUM_STREAMING_PCM_SAMPLES - int(pcm.size)
        if missing > 0:
            pcm = np.pad(pcm, (0, missing), mode="constant")
        return pcm.astype("<i2", copy=False).tobytes()

    def synthesize_agent_pcm(
        self,
        text: str,
        *,
        client_role: str,
        participant_id: str | None = None,
    ) -> bytes:
        """Return the role-locked agent voice as mono 16 kHz PCM without playback."""
        voice = self.voice_for(client_role, participant_id=participant_id)
        if self.settings.tts_mode in {"sherpa_onnx", "hybrid"}:
            return self.synthesize_pcm(
                text,
                speaker_id=self._speaker_id_from_voice(voice),
                speed=self.settings.sherpa_tts_speed,
            )
        mp3 = self.synthesize(text, voice=voice)
        try:
            converted = subprocess.run(
                [
                    self.settings.ffmpeg_binary,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(self.sample_rate_hz),
                    "-f",
                    "s16le",
                    "pipe:1",
                ],
                input=mp3,
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SpeechSynthesisError(
                f"Agent TTS decoding failed: {exc}"
            ) from exc
        if converted.returncode != 0 or not converted.stdout:
            raise SpeechSynthesisError("Agent TTS decoding returned no PCM")
        return converted.stdout

    def _synthesize_sherpa(
        self, text: str, *, speaker_id: int | None = None
    ) -> bytes:
        samples, sample_rate = self._sherpa_samples(text, speaker_id=speaker_id)

        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2", copy=False)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        try:
            converted = subprocess.run(
                [
                    self.settings.ffmpeg_binary,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "wav",
                    "-i",
                    "pipe:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(self.sample_rate_hz),
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    "128k",
                    "-f",
                    "mp3",
                    "pipe:1",
                ],
                input=wav_buffer.getvalue(),
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SpeechSynthesisError(f"Local TTS encoding failed: {exc}") from exc
        if converted.returncode != 0 or not converted.stdout:
            detail = converted.stderr.decode("utf-8", errors="replace")[:500]
            raise SpeechSynthesisError(f"Local TTS encoding failed: {detail}")
        return converted.stdout

    def _synthesize_xunfei(
        self, encoded: bytes, *, voice: str | None = None
    ) -> bytes:

        host = "tts-api.xfyun.cn"
        path = "/v2/tts"
        date = formatdate(usegmt=True)
        origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        signature = base64.b64encode(
            hmac.new(
                self.settings.xunfei_api_secret.encode("utf-8"),
                origin.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("ascii")
        authorization_origin = (
            f'api_key="{self.settings.xunfei_api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature}"'
        )
        query = urlencode(
            {
                "authorization": base64.b64encode(
                    authorization_origin.encode("utf-8")
                ).decode("ascii"),
                "date": date,
                "host": host,
            }
        )
        socket = None
        chunks: list[bytes] = []
        try:
            socket = websocket.create_connection(
                f"wss://{host}{path}?{query}", timeout=15
            )
            socket.send(
                json.dumps(
                    {
                        "common": {"app_id": self.settings.xunfei_app_id},
                        "business": {
                            "aue": "lame",
                            "sfl": 1,
                            "auf": "audio/L16;rate=16000",
                            "vcn": voice or self.settings.xunfei_primary_voice,
                            "speed": 50,
                            "volume": 50,
                            "pitch": 50,
                            "bgs": 0,
                            "tte": "UTF8",
                        },
                        "data": {
                            "status": 2,
                            "text": base64.b64encode(encoded).decode("ascii"),
                        },
                    },
                    ensure_ascii=True,
                )
            )
            while True:
                message = json.loads(socket.recv())
                if int(message.get("code", -1)) != 0:
                    raise SpeechSynthesisError(
                        f"iFLYTEK TTS error {message.get('code')}: {message.get('message')}"
                    )
                data = message.get("data") or {}
                if data.get("audio"):
                    chunks.append(base64.b64decode(data["audio"]))
                if data.get("status") == 2:
                    break
        except SpeechSynthesisError:
            raise
        except (OSError, ValueError, websocket.WebSocketException) as exc:
            raise SpeechSynthesisError(f"iFLYTEK TTS transport error: {exc}") from exc
        finally:
            if socket is not None:
                socket.close()
        if not chunks:
            raise SpeechSynthesisError("iFLYTEK returned no audio")
        return b"".join(chunks)
