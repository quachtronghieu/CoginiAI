"""Offline Piper TTS adapter."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import re
import threading
import wave
from collections import OrderedDict
from pathlib import Path
import logging
from time import monotonic

from .ai import AiUnavailableError, PcmAudio
from .config import Settings


logger = logging.getLogger(__name__)


class PiperTtsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._voices: dict[str, object] = {}
        self._lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._audio_cache: OrderedDict[tuple[str, str], PcmAudio] = OrderedDict()
        self._cache_bytes = 0

    def available(self) -> bool:
        return self.health()["tts"]

    def health(self) -> dict[str, bool]:
        piper_available = importlib.util.find_spec("piper") is not None
        vi_ready = bool(
            piper_available
            and self._model_files_ready(self.settings.tts_model_vi)
        )
        en_ready = bool(
            piper_available
            and self._model_files_ready(self.settings.tts_model_en)
        )
        bilingual_configured = bool(
            self.settings.tts_model_vi or self.settings.tts_model_en
        )
        legacy_ready = bool(
            piper_available
            and self._model_files_ready(
                self.settings.tts_model, self.settings.tts_voice
            )
        )
        return {
            "tts": vi_ready and en_ready if bilingual_configured else legacy_ready,
            "tts_vi": vi_ready,
            "tts_en": en_ready,
        }

    @staticmethod
    def _model_files_ready(model_path: str, config_path: str = "") -> bool:
        if not model_path or not Path(model_path).is_file():
            return False
        expected_config = Path(config_path) if config_path else Path(f"{model_path}.json")
        return expected_config.is_file()

    async def synthesize(
        self,
        text: str,
        language: str,
        language_probability: float,
        *, voice_language: str | None = None,
    ) -> PcmAudio:
        if not self.available():
            raise AiUnavailableError(
                "TTS unavailable. Configure both TTS_MODEL_VI and TTS_MODEL_EN, or legacy TTS_MODEL."
            )
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")
        selected_language = voice_language or self.select_language(
            text, language, language_probability
        )
        started = monotonic()
        try:
            return await asyncio.to_thread(
                self._cached_synthesize, text.strip(), selected_language
            )
        finally:
            logger.info(
                "TTS completed: latency_ms=%.1f language=%s",
                (monotonic() - started) * 1000,
                selected_language,
            )

    def _cached_synthesize(self, text: str, language: str) -> PcmAudio:
        # Serialize Piper access, including a prefetch finishing after cancellation.
        # Cache exact audio only; never reuse a previous dialogue/RAG decision.
        key = (language, text)
        with self._synthesis_lock:
            if key in self._audio_cache:
                self._audio_cache.move_to_end(key)
                return self._audio_cache[key]
            audio = self._synthesize_sync(text, language)
            if len(text) <= 260 and len(audio.data) <= 2 * 1024 * 1024:
                self._audio_cache[key] = audio
                self._cache_bytes += len(audio.data)
                while len(self._audio_cache) > 32 or self._cache_bytes > 16 * 1024 * 1024:
                    _, removed = self._audio_cache.popitem(last=False)
                    self._cache_bytes -= len(removed.data)
            return audio

    async def warmup(self) -> None:
        """Load configured voices and execute synthesis before accepting clients."""
        languages = ("vi", "en") if self.settings.tts_model_vi or self.settings.tts_model_en else (self.settings.tts_default_language,)
        for language in languages:
            text = "Xin chào!" if language == "vi" else "Hello!"
            await self.synthesize(text, language, 1.0)
        logger.info("Piper warmup ready: languages=%s cache_entries=%d", languages, len(self._audio_cache))

    def select_language(
        self,
        text: str,
        detected_language: str,
        language_probability: float = 1.0,
    ) -> str:
        answer_language = self._detect_answer_language(text)
        if answer_language:
            return answer_language
        normalized = detected_language.lower().strip()
        if language_probability >= self.settings.language_hint_min_probability:
            if normalized.startswith("vi"):
                return "vi"
            if normalized.startswith("en"):
                return "en"
        return self.settings.tts_default_language

    @staticmethod
    def _detect_answer_language(text: str) -> str:
        lowered = text.lower()
        if re.search(r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]", lowered):
            return "vi"
        words = set(re.findall(r"[a-z]+", lowered))
        vi_words = {"xin", "chao", "cam", "on", "khong", "toi", "ban", "la", "va", "cua", "mot", "duoc"}
        en_words = {"hello", "hi", "ok", "okay", "thanks", "thank", "yes", "no", "good", "morning", "weather", "time", "help", "sure", "the", "is", "are", "you", "your", "and", "please"}
        vi_score = len(words & vi_words)
        en_score = len(words & en_words)
        if vi_score > en_score:
            return "vi"
        if en_score > vi_score:
            return "en"
        if words and not vi_score:
            return "en"
        return ""

    def _synthesize_sync(self, text: str, language: str) -> PcmAudio:
        voice = self._get_voice(language)
        output = io.BytesIO()
        if hasattr(voice, "synthesize_wav"):
            with wave.open(output, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            output.seek(0)
            with wave.open(output, "rb") as wav_file:
                channels = wav_file.getnchannels()
                width = wav_file.getsampwidth()
                rate = wav_file.getframerate()
                if channels != 1 or width != 2:
                    raise RuntimeError("Piper output must be signed 16-bit mono")
                return PcmAudio(wav_file.readframes(wav_file.getnframes()), rate, channels)

        chunks = list(voice.synthesize(text))
        pcm = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        return PcmAudio(pcm, voice.config.sample_rate, 1)

    def _get_voice(self, language: str):
        cache_key, model_path, config_path = self._voice_config(language)
        if cache_key in self._voices:
            return self._voices[cache_key]
        with self._lock:
            if cache_key not in self._voices:
                self._voices[cache_key] = self._load_voice(model_path, config_path)
            return self._voices[cache_key]

    def _voice_config(self, language: str) -> tuple[str, str, str]:
        bilingual_configured = bool(
            self.settings.tts_model_vi or self.settings.tts_model_en
        )
        if bilingual_configured:
            selected = "vi" if language.startswith("vi") else "en"
            model_path = (
                self.settings.tts_model_vi
                if selected == "vi"
                else self.settings.tts_model_en
            )
            if not model_path:
                fallback = self.settings.tts_default_language
                model_path = (
                    self.settings.tts_model_vi
                    if fallback == "vi"
                    else self.settings.tts_model_en
                )
                selected = fallback
            if not model_path:
                raise AiUnavailableError("No Piper model is configured for selected language")
            return selected, model_path, ""
        return "legacy", self.settings.tts_model, self.settings.tts_voice

    @staticmethod
    def _load_voice(model_path: str, config_path: str):
        from piper import PiperVoice

        kwargs = {"config_path": config_path} if config_path else {}
        return PiperVoice.load(model_path, **kwargs)


def _mp3_to_pcm(mp3_bytes: bytes, sample_rate: int = 24000) -> PcmAudio:
    import av

    container = av.open(io.BytesIO(mp3_bytes))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    pcm_bytes = bytearray()
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            pcm_bytes.extend(resampled.to_ndarray().tobytes())
    return PcmAudio(bytes(pcm_bytes), sample_rate, 1)


class EdgeTtsService:
    """Microsoft Edge-TTS service with streaming Neural voices."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.voice_vi = settings.tts_voice_vi or "vi-VN-HoaiMyNeural"
        self.voice_en = settings.tts_voice_en or "en-US-JennyNeural"

    def available(self) -> bool:
        return self.health()["tts"]

    def health(self) -> dict[str, bool]:
        edge_available = importlib.util.find_spec("edge_tts") is not None
        av_available = importlib.util.find_spec("av") is not None
        ready = edge_available and av_available
        return {
            "tts": ready,
            "tts_vi": ready,
            "tts_en": ready,
        }

    async def synthesize(
        self,
        text: str,
        language: str,
        language_probability: float = 1.0,
        *, voice_language: str | None = None,
    ) -> PcmAudio:
        if not self.available():
            raise AiUnavailableError("Edge-TTS is unavailable. Install edge-tts and av.")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Cannot synthesize empty text")

        selected_lang = voice_language or self.select_language(cleaned, language, language_probability)
        voice = self.voice_vi if selected_lang.startswith("vi") else self.voice_en
        started = monotonic()
        try:
            import edge_tts

            communicate = edge_tts.Communicate(cleaned, voice)
            mp3_buf = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buf.extend(chunk["data"])

            if not mp3_buf:
                raise RuntimeError(f"Edge-TTS returned no audio data for voice {voice}")

            target_rate = self.settings.server_sample_rate
            return await asyncio.to_thread(_mp3_to_pcm, bytes(mp3_buf), target_rate)
        finally:
            logger.info(
                "Edge-TTS completed: latency_ms=%.1f voice=%s",
                (monotonic() - started) * 1000,
                voice,
            )

    def select_language(
        self,
        text: str,
        detected_language: str,
        language_probability: float = 1.0,
    ) -> str:
        answer_language = PiperTtsService._detect_answer_language(text)
        if answer_language:
            return answer_language
        normalized = detected_language.lower().strip()
        if language_probability >= self.settings.language_hint_min_probability:
            if normalized.startswith("vi"):
                return "vi"
            if normalized.startswith("en"):
                return "en"
        return self.settings.tts_default_language


def create_tts_service(settings: Settings):
    if getattr(settings, "tts_engine", "edge") == "edge":
        edge = EdgeTtsService(settings)
        if edge.available():
            return edge
    return PiperTtsService(settings)
