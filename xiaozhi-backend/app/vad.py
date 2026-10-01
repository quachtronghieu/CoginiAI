"""Per-session voice activity detection with timeout fallback."""

from __future__ import annotations

import logging
from typing import Protocol

from .config import Settings


logger = logging.getLogger(__name__)


class VadService(Protocol):
    def should_finalize(self, frame: bytes, elapsed_seconds: float) -> bool: ...

    @property
    def speech_detected(self) -> bool: ...

    @property
    def evidence(self) -> dict: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class VadFactory(Protocol):
    def create(self) -> VadService: ...

    def available(self) -> bool: ...


class TimeoutVadService:
    """Safety fallback only. This is not speech or silence detection."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    def should_finalize(self, frame: bytes, elapsed_seconds: float) -> bool:
        del frame
        return elapsed_seconds >= self.timeout_seconds

    @property
    def speech_detected(self) -> bool:
        # The timeout fallback has no classifier and therefore cannot safely
        # reject audio as non-speech.
        return True

    @property
    def evidence(self) -> dict:
        return {"available": False, "speech_detected": True}

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class TimeoutVadFactory:
    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds

    def create(self) -> VadService:
        return TimeoutVadService(self.timeout_seconds)

    def available(self) -> bool:
        return False


class SileroVadFactory:
    """Load one Silero model; create isolated iterator/decoder per device."""

    def __init__(self, settings: Settings) -> None:
        from silero_vad import VADIterator, load_silero_vad

        from .codec import PyAvOpusCodec

        self.settings = settings
        self.codec = PyAvOpusCodec(settings)
        if not self.codec.available():
            raise RuntimeError("PyAV and NumPy are required for Silero VAD")
        self.model = load_silero_vad(onnx=False)
        self.iterator_class = VADIterator

    def create(self) -> VadService:
        return SileroVadSession(
            model=self.model,
            iterator_class=self.iterator_class,
            codec=self.codec,
            settings=self.settings,
        )

    def available(self) -> bool:
        return True


class SileroVadSession:
    chunk_samples = 512

    def __init__(self, model, iterator_class, codec, settings: Settings) -> None:
        import numpy as np
        import torch

        self.model = model
        self.iterator_class = iterator_class
        self.codec = codec
        self.settings = settings
        self.np = np
        self.torch = torch
        self.decoder = None
        self.iterator = None
        self.pending_pcm = bytearray()
        self.speech_started = False
        self.closed = False
        self.noise_filter = None
        self.probabilities: list[float] = []
        # In raw/shadow comparison modes RNNoise must not discard speech before
        # the complete raw utterance reaches diagnostics and Whisper.
        if settings.rnnoise_enabled and settings.rnnoise_asr_mode == "filtered":
            from .noise import RnnoiseStreamFilter

            self.noise_filter = RnnoiseStreamFilter(
                settings.xiaozhi_input_sample_rate
            )
        self.reset()

    def should_finalize(self, frame: bytes, elapsed_seconds: float) -> bool:
        if self.closed:
            return True
        pcm = self.decoder.decode(frame)
        if self.noise_filter is not None:
            pcm = self.noise_filter.process(pcm)
        self.pending_pcm.extend(pcm)
        chunk_bytes = self.chunk_samples * 2
        while len(self.pending_pcm) >= chunk_bytes:
            raw = bytes(self.pending_pcm[:chunk_bytes])
            del self.pending_pcm[:chunk_bytes]
            samples = self.np.frombuffer(raw, dtype="<i2").copy()
            tensor = self.torch.from_numpy(samples).float().div_(32768.0)
            event = self.iterator(tensor, return_seconds=False)
            if event and "start" in event:
                self.speech_started = True
            if event and "end" in event and self.speech_started:
                return True
        # The idle timeout must not cut a sentence while speech is in progress.
        # Once speech starts, wait for VAD's silence/end event; buffer limits
        # and MAX_AUDIO_SECONDS remain the hard safety cap.
        return not self.speech_started and elapsed_seconds >= self.settings.auto_stop_timeout_seconds

    @property
    def speech_detected(self) -> bool:
        return self.speech_started

    @property
    def evidence(self) -> dict:
        threshold = self.settings.vad_threshold
        longest = current = 0
        for probability in self.probabilities:
            if probability >= threshold:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        frame_ms = self.chunk_samples / self.settings.xiaozhi_input_sample_rate * 1000
        return {
            "available": True,
            "speech_detected": self.speech_started,
            "threshold": threshold,
            "frame_ms": frame_ms,
            "frame_probabilities": [round(value, 4) for value in self.probabilities],
            "max_probability": max(self.probabilities, default=0.0),
            "frames_above_threshold": sum(
                value >= threshold for value in self.probabilities
            ),
            "max_contiguous_speech_ms": longest * frame_ms,
        }

    def reset(self) -> None:
        if self.closed:
            return
        if self.decoder is not None:
            self.decoder.close()
        if self.noise_filter is not None:
            self.noise_filter.reset()
        self.decoder = self.codec.create_stream_decoder()
        self.probabilities.clear()
        recording_model = _ProbabilityRecordingModel(
            self.model, self.probabilities
        )
        self.iterator = self.iterator_class(
            recording_model,
            threshold=self.settings.vad_threshold,
            sampling_rate=self.settings.xiaozhi_input_sample_rate,
            min_silence_duration_ms=self.settings.vad_min_silence_ms,
            speech_pad_ms=self.settings.vad_speech_pad_ms,
        )
        self.pending_pcm.clear()
        self.speech_started = False

    def close(self) -> None:
        if self.decoder is not None:
            self.decoder.close()
        if self.noise_filter is not None:
            self.noise_filter.close()
        self.pending_pcm.clear()
        self.closed = True


class _ProbabilityRecordingModel:
    """Record Silero probabilities without running its stateful model twice."""

    def __init__(self, model, probabilities: list[float]) -> None:
        self.model = model
        self.probabilities = probabilities

    def __call__(self, samples, sample_rate):
        result = self.model(samples, sample_rate)
        self.probabilities.append(float(result.item()))
        return result

    def reset_states(self) -> None:
        self.model.reset_states()


def build_vad_factory(settings: Settings) -> VadFactory:
    fallback = TimeoutVadFactory(settings.auto_stop_timeout_seconds)
    if not settings.vad_enabled:
        return fallback
    try:
        return SileroVadFactory(settings)
    except Exception:
        logger.exception("Silero VAD unavailable; using timeout safety fallback")
        return fallback
