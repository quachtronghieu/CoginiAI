"""Offline Piper TTS adapter."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import re
import threading
import wave
from pathlib import Path
import logging
from time import monotonic

from .ai import AiUnavailableError, PcmAudio
from .config import Settings


logger = logging.getLogger(__name__)


def clean_speech_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    # Clean dashes, bullet points and non-breaking spaces
    cleaned = re.sub(r"^[–—•\-\*]\s*", "", cleaned)
    cleaned = re.sub(r"[–—]", "-", cleaned)
    cleaned = cleaned.replace("\xa0", " ").strip()

    # Normalize price/currency for speech (e.g. 200.000 VNĐ -> 200 nghìn đồng)
    cleaned = re.sub(r"(\d+)\.000\s*(?:VNĐ|VND|đồng)", r"\1 nghìn đồng", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:VNĐ|VND)\b", "đồng", cleaned)

    # Convert semicolons into full stops so clauses stay short, digestible, and well-paused
    cleaned = re.sub(r"\s*;\s*", ". ", cleaned)
    # Convert colons into comma
    cleaned = re.sub(r"\s*:\s*", ", ", cleaned)
    # Remove enclosing parentheses while preserving contents
    cleaned = re.sub(r"\(([^)]+)\)", r", \1, ", cleaned)
    # Tidy punctuation spacing
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"\s*\.\s*", ". ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class PiperTtsService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._voices: dict[str, object] = {}
        self._lock = threading.Lock()

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
    ) -> PcmAudio:
        if not self.available():
            raise AiUnavailableError(
                "TTS unavailable. Configure both TTS_MODEL_VI and TTS_MODEL_EN, or legacy TTS_MODEL."
            )
        cleaned = clean_speech_text(text)
        if not cleaned:
            raise ValueError("Cannot synthesize empty text")
        selected_language = self.select_language(
            cleaned, language, language_probability
        )
        started = monotonic()
        try:
            return await asyncio.to_thread(
                self._synthesize_sync, cleaned, selected_language
            )
        finally:
            logger.info(
                "TTS completed: latency_ms=%.1f language=%s",
                (monotonic() - started) * 1000,
                selected_language,
            )

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
        vi_words = {"xin", "chao", "cam", "on", "khong", "toi", "ban", "la", "va", "cua", "mot", "duoc", "co", "gi", "mon", "an", "ngon", "pho", "bun", "com", "banh", "nuoc", "di", "ung", "nay", "do", "o", "tai", "nhe", "a", "da", "vui", "long", "minh"}
        en_words = {"hello", "hi", "ok", "okay", "thanks", "thank", "yes", "no", "good", "morning", "weather", "time", "help", "sure", "the", "is", "are", "you", "your", "and", "please", "certainly"}
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

        # If real PiperVoice with sentence streaming is available
        if hasattr(voice, "synthesize") and hasattr(voice, "config"):
            syn_config = None
            try:
                from piper.config import SynthesisConfig
                syn_config = SynthesisConfig(length_scale=1.18)
            except Exception:
                pass

            chunks = list(
                voice.synthesize(text, syn_config=syn_config)
                if syn_config
                else voice.synthesize(text)
            )
            if chunks:
                silence_samples = int(voice.config.sample_rate * 0.25)
                silence_bytes = b"\x00\x00" * silence_samples
                pcm_parts = []
                for i, chunk in enumerate(chunks):
                    pcm_parts.append(chunk.audio_int16_bytes)
                    if i < len(chunks) - 1:
                        pcm_parts.append(silence_bytes)
                return PcmAudio(b"".join(pcm_parts), voice.config.sample_rate, 1)

        # Fallback for voice objects or mocks defining synthesize_wav
        output = io.BytesIO()
        if hasattr(voice, "synthesize_wav"):
            with wave.open(output, "wb") as wav_file:
                try:
                    from piper.config import SynthesisConfig
                    voice.synthesize_wav(text, wav_file, syn_config=SynthesisConfig(length_scale=1.18))
                except (TypeError, Exception):
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
        return PcmAudio(pcm, getattr(voice, "config", None).sample_rate if hasattr(voice, "config") else 22050, 1)

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
        self._piper_fallback = PiperTtsService(settings)

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
    ) -> PcmAudio:
        if not self.available() and not self._piper_fallback.available():
            raise AiUnavailableError("TTS is unavailable. Install edge-tts or configure Piper models.")
        cleaned = clean_speech_text(text)
        if not cleaned:
            raise ValueError("Cannot synthesize empty text")

        selected_lang = self.select_language(cleaned, language, language_probability)
        voice = self.voice_vi if selected_lang.startswith("vi") else self.voice_en
        started = monotonic()
        target_rate = self.settings.server_sample_rate
        try:
            import edge_tts

            # For multi-sentence or long text (> 160 chars), synthesize sentence-by-sentence with natural pauses
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
            if not sentences or len(cleaned) < 160:
                sentences = [cleaned]

            pcm_parts = bytearray()
            silence_between = b"\x00\x00" * int(target_rate * 0.2)  # 200ms natural breathing pause

            for i, sentence in enumerate(sentences):
                communicate = edge_tts.Communicate(sentence, voice)
                mp3_buf = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_buf.extend(chunk["data"])

                if not mp3_buf:
                    raise RuntimeError(f"Edge-TTS returned no audio data for voice {voice}")

                pcm = await asyncio.to_thread(_mp3_to_pcm, bytes(mp3_buf), target_rate)
                pcm_parts.extend(pcm.data)
                if i < len(sentences) - 1:
                    pcm_parts.extend(silence_between)

            return PcmAudio(bytes(pcm_parts), target_rate, 1)
        except Exception as exc:
            logger.warning(
                "Edge-TTS error (%s: %s). Falling back to offline Piper TTS.",
                type(exc).__name__,
                exc,
            )
            if self._piper_fallback.available():
                return await self._piper_fallback.synthesize(
                    cleaned, selected_lang, language_probability
                )
            raise
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
