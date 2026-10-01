"""Offline faster-whisper ASR adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import importlib.util
import logging
import os
import re
import site
import sys
import threading
from collections import Counter
from time import monotonic

from .ai import AiUnavailableError, AsrResult, PcmAudio
from .config import Settings
from .allergy import CustomerProfile, ALLERGEN_ALIASES, ingredient_list, is_confirmation, is_rejection
from .dialogue import courtesy_or_domain_reply, is_food_question


logger = logging.getLogger(__name__)


def _ensure_cuda_dlls() -> None:
    """Ensure NVIDIA CUDA and cuBLAS DLLs are discoverable on Windows."""
    search_dirs: list[str] = [
        
    ]
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if torch_lib not in search_dirs:
            search_dirs.append(torch_lib)
    except Exception:
        pass

    for base in sys.path + getattr(site, "getsitepackages", lambda: [])():
        for candidate in [
            os.path.join(base, "torch", "lib"),
            os.path.join(base, "nvidia", "cublas", "bin"),
        ]:
            if os.path.isdir(candidate) and candidate not in search_dirs:
                search_dirs.append(candidate)

    for path in search_dirs:
        if os.path.isdir(path):
            if path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(path)
                except Exception:
                    pass


_ensure_cuda_dlls()
SUPPORTED_LANGUAGES = {"en", "vi"}
CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04ff]")
VIETNAMESE_PATTERN = re.compile(
    r"[\u0103\u00e2\u0111\u00ea\u00f4\u01a1\u01b0"
    r"\u00e0\u00e1\u1ea1\u1ea3\u00e3\u00e8\u00e9\u1eb9\u1ebb\u1ebd"
    r"\u00ec\u00ed\u1ecb\u1ec9\u0129\u00f2\u00f3\u1ecd\u1ecf\u00f5"
    r"\u00f9\u00fa\u1ee5\u1ee7\u0169\u1ef3\u00fd\u1ef5\u1ef7\u1ef9]"
)


def repetitive_transcript(text: str) -> bool:
    tokens = re.findall(r"\w+", text.casefold())
    if len(tokens) < 12:
        return False
    if max(Counter(tokens).values()) / len(tokens) >= 0.55:
        return True
    for size in range(2, min(12, len(tokens) // 4) + 1):
        count = max(Counter(tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)).values())
        if count >= 4 and count * size >= len(tokens) * .6:
            return True
    return False


@dataclass(frozen=True)
class _TranscriptCandidate:
    text: str
    language: str
    language_probability: float
    avg_logprob: float
    no_speech_probability: float
    audio_source: str = "processed"

    @property
    def valid(self) -> bool:
        return bool(
            self.text
            and self.language in SUPPORTED_LANGUAGES
            and not CYRILLIC_PATTERN.search(self.text)
            and not repetitive_transcript(self.text)
            and self.no_speech_probability <= 0.75
            and self.avg_logprob >= -1.2
        )

    @property
    def score(self) -> float:
        text_bonus = 0.0
        if self.language == "vi" and VIETNAMESE_PATTERN.search(self.text.lower()):
            text_bonus = 0.35
        return (
            self.language_probability * 0.25
            + self.avg_logprob
            - self.no_speech_probability
            + text_bonus
        )


class FasterWhisperAsrService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return bool(
            self.settings.asr_model
            and importlib.util.find_spec("faster_whisper")
            and importlib.util.find_spec("numpy")
        )

    async def transcribe(self, audio: PcmAudio, customer_profile: CustomerProfile | None = None, raw_audio: PcmAudio | None = None) -> AsrResult:
        if not self.available():
            raise AiUnavailableError(
                "ASR unavailable. Install faster-whisper and configure ASR_MODEL."
            )
        return await asyncio.to_thread(self._transcribe_sync, audio, customer_profile, raw_audio)

    def _transcribe_sync(self, audio: PcmAudio, customer_profile: CustomerProfile | None = None, raw_audio: PcmAudio | None = None) -> AsrResult:
        started = monotonic()
        import numpy as np

        raw_samples = np.frombuffer(audio.data, dtype="<i2")
        samples = raw_samples.astype(np.float32) / 32768.0
        audio_max = int(np.max(np.abs(raw_samples.astype(np.int32)))) if len(raw_samples) > 0 else 0
        audio_rms = float(np.sqrt(np.mean(samples**2))) if len(samples) > 0 else 0.0

        if audio.sample_rate != 16000 or audio.channels != 1:
            raise AiUnavailableError("Whisper requires 16 kHz mono PCM input")
        if len(samples) / audio.sample_rate < self.settings.asr_min_duration_seconds or audio_rms < self.settings.asr_min_rms:
            logger.info("ASR rejected near-silence: rms=%.5f peak=%d samples=%d", audio_rms, audio_max, len(samples))
            return AsrResult("", "und", 0.0, -2.0)
        model = self._get_model()

        initial_language = self.settings.asr_language or None
        context_hotwords = self._context_hotwords(customer_profile)
        initial = self._transcribe_candidate(model, samples, initial_language, hotwords=context_hotwords)
        candidates = [initial]
        retry_languages: list[str] = []
        if not self.settings.asr_language and self._needs_fallback(initial):
            retry_languages = ["en", "vi"]
            candidates.extend(
                self._transcribe_candidate(model, samples, language, hotwords=context_hotwords)
                for language in retry_languages
            )

        valid_candidates = [candidate for candidate in candidates if candidate.valid]
        if not valid_candidates:
            # Do not keep repeating a hallucinated hotword list on every retry.
            retry_language = self.settings.asr_language or (initial.language if initial.language in SUPPORTED_LANGUAGES else "vi")
            retry = self._transcribe_candidate(model, samples, retry_language, use_hotwords=False)
            if retry.valid:
                valid_candidates.append(retry)
        if valid_candidates:
            selected = max(valid_candidates, key=lambda candidate: candidate.score)
        else:
            selected = _TranscriptCandidate("", "und", 0.0, 0.0, 1.0)

        needs_clarification, reason, alternatives = False, "", ()
        safety_context = customer_profile is not None and bool(customer_profile.allergy_question_asked or customer_profile.pending_allergy_status or customer_profile.pending_allergy_note)
        # Short greetings are also vulnerable to losing their first consonant
        # during suppression. Compare them without ingredient-biased hotwords.
        short_uncertain = selected.valid and len(selected.text.split()) <= 4 and selected.avg_logprob < -0.65
        if (safety_context or short_uncertain) and raw_audio is not None and raw_audio.data != audio.data:
            if raw_audio.sample_rate != audio.sample_rate or raw_audio.channels != 1:
                raise AiUnavailableError("Raw/filtered ASR comparison requires matching 16 kHz mono PCM")
            raw_signal = np.frombuffer(raw_audio.data, dtype="<i2").astype(np.float32) / 32768.0
            raw_candidate = replace(self._transcribe_candidate(model, raw_signal, selected.language, hotwords=context_hotwords), audio_source="raw")
            if raw_candidate.valid:
                alternatives = (selected.text, raw_candidate.text)
                if safety_context:
                    selected, needs_clarification, reason = self._select_safety_candidate(selected, raw_candidate, customer_profile)
                else:
                    selected = self._select_short_candidate(selected, raw_candidate)
                logger.info("ASR raw/filtered comparison: processed=%r raw=%r selected_source=%s clarification=%s", alternatives[0], alternatives[1], selected.audio_source, needs_clarification)

        logger.info(
            "ASR completed: latency_ms=%.1f signal_rms=%.4f signal_max=%d initial_language=%s initial_probability=%.3f "
            "selected_language=%s selected_probability=%.3f selected_avg_logprob=%.3f "
            "selected_no_speech=%.3f retries=%s samples=%d",
            (monotonic() - started) * 1000,
            audio_rms,
            audio_max,
            initial.language,
            initial.language_probability,
            selected.language,
            selected.language_probability,
            selected.avg_logprob,
            selected.no_speech_probability,
            ",".join(retry_languages) or "none",
            len(samples),
        )
        return AsrResult(
            selected.text,
            selected.language,
            selected.language_probability,
            selected.avg_logprob,
            needs_clarification,
            alternatives,
            reason,
            selected.no_speech_probability,
        )

    @staticmethod
    def _select_short_candidate(processed: _TranscriptCandidate, raw: _TranscriptCandidate) -> _TranscriptCandidate:
        """Rescue a clear courtesy phrase; never guess an ingredient or dish."""
        processed_courtesy = courtesy_or_domain_reply(processed.text, processed.language)
        raw_courtesy = courtesy_or_domain_reply(raw.text, raw.language)
        if raw_courtesy and not processed_courtesy and not is_food_question(processed.text):
            return raw
        if processed_courtesy and not raw_courtesy and not is_food_question(raw.text):
            return processed
        return max((processed, raw), key=lambda candidate: candidate.score)

    @staticmethod
    def _context_hotwords(profile: CustomerProfile | None) -> str:
        if profile is not None and profile.pending_allergy_status:
            return "chính xác, đúng rồi, không đúng, yes, correct, no, incorrect"
        if profile is not None and (profile.allergy_question_asked or profile.pending_allergy_note):
            return ", ".join(ALLERGEN_ALIASES)
        return ""

    @staticmethod
    def _select_safety_candidate(processed: _TranscriptCandidate, raw: _TranscriptCandidate, profile: CustomerProfile) -> tuple[_TranscriptCandidate, bool, str]:
        if profile.pending_allergy_status:
            def confirmation_kind(text):
                return "yes" if is_confirmation(text) else "no" if is_rejection(text) else "unknown"
            left, right = confirmation_kind(processed.text), confirmation_kind(raw.text)
            if left != right and "yes" in {left, right}:
                return processed, True, "confirmation_disagreement"
            if left == "no" or right == "no":
                return processed if left == "no" else raw, False, ""
            if left == right == "yes":
                return max((processed, raw), key=lambda candidate: candidate.score), False, ""
        left_labels, left_unknown = ingredient_list(processed.text)
        right_labels, right_unknown = ingredient_list(raw.text)
        left_complete = bool(left_labels and not left_unknown)
        right_complete = bool(right_labels and not right_unknown)
        if left_complete and right_complete and set(left_labels) != set(right_labels):
            return processed, True, "ingredient_disagreement"
        if left_complete != right_complete and set(left_labels) == set(right_labels):
            # The same recognised keywords plus an unexplained extra word in
            # either recording can indicate a LOST second ingredient. Do not
            # treat the shorter, apparently complete list as a clean rescue.
            return processed, True, "ingredient_disagreement"
        if right_complete and not left_complete:
            return raw, False, ""
        if left_complete and not right_complete:
            return processed, False, ""
        return max((processed, raw), key=lambda candidate: candidate.score), False, ""

    def _transcribe_candidate(
        self,
        model,
        samples,
        language: str | None,
        *, hotwords: str = "", use_hotwords: bool = True,
    ) -> _TranscriptCandidate:
        segments, info = model.transcribe(
            samples,
            language=language,
            beam_size=3,
            best_of=3,
            temperature=0.0,
            condition_on_previous_text=False,
            hotwords=(hotwords or self.settings.asr_hotwords) if language != "en" and use_hotwords else None,
            no_speech_threshold=0.45,
            log_prob_threshold=-1.0,
            hallucination_silence_threshold=1.0,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 100,
            },
        )
        segment_list = list(segments)
        transcript = " ".join(
            segment.text.strip() for segment in segment_list
        ).strip()
        detected_language = str(
            language or getattr(info, "language", "") or "und"
        ).lower()
        probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        avg_logprob = self._average_metric(segment_list, "avg_logprob", 0.0)
        no_speech_probability = self._average_metric(
            segment_list, "no_speech_prob", 0.0
        )
        return _TranscriptCandidate(
            transcript,
            detected_language,
            probability,
            avg_logprob,
            no_speech_probability,
        )

    def _needs_fallback(self, candidate: _TranscriptCandidate) -> bool:
        return bool(
            not candidate.text
            or not candidate.valid
            or candidate.language not in SUPPORTED_LANGUAGES
            or candidate.language_probability
            < self.settings.language_hint_min_probability
            or CYRILLIC_PATTERN.search(candidate.text)
        )

    @staticmethod
    def _average_metric(segments: list[object], name: str, default: float) -> float:
        values = [
            float(value)
            for segment in segments
            if (value := getattr(segment, name, None)) is not None
        ]
        return sum(values) / len(values) if values else default

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            
            from faster_whisper import WhisperModel

            device = self._resolve_device()
            compute_type = self.settings.asr_compute_type or (
                "float16" if device == "cuda" else "int8"
            )
            kwargs = {
                "device": device,
                "compute_type": compute_type,
                "local_files_only": self.settings.asr_local_files_only,
            }
            try:
                self._model = WhisperModel(self.settings.asr_model, **kwargs)
            except (TypeError, ValueError, OSError) as exc:
                if self.settings.asr_local_files_only:
                    logger.warning(
                        "ASR model '%s' is not available locally: %s",
                        self.settings.asr_model,
                        exc,
                    )
                    raise AiUnavailableError(
                        f"ASR model '{self.settings.asr_model}' is not available locally"
                    ) from exc
                raise
            return self._model

    def _resolve_device(self) -> str:
        if self.settings.model_device != "auto":
            return self.settings.model_device
        try:
            import ctranslate2

            if hasattr(ctranslate2, "get_cuda_device_count") and ctranslate2.get_cuda_device_count() > 0:
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"
