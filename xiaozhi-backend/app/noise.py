"""RNNoise-based microphone noise suppression."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic

from .ai import AiPipelineError, PcmAudio
from .config import Settings


logger = logging.getLogger(__name__)


class RnnoiseNoiseSuppressor:
    """Denoise a complete utterance while preserving its original sample rate."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.rnnoise_enabled
        self._rnnoise_class = None
        self._np = None
        if self.enabled:
            try:
                import numpy as np
                from pyrnnoise import RNNoise
                from pyrnnoise.rnnoise import FRAME_SIZE, SAMPLE_RATE

                self._np = np
                self._rnnoise_class = RNNoise
                if SAMPLE_RATE != 48_000 or FRAME_SIZE != 480:
                    raise RuntimeError(
                        f"Unsupported RNNoise runtime: rate={SAMPLE_RATE}, frame={FRAME_SIZE}"
                    )
                logger.info(
                    "RNNoise runtime verified: native_rate=%d frame_samples=%d input_dtype=int16 state_scope=utterance",
                    SAMPLE_RATE,
                    FRAME_SIZE,
                )
            except (ImportError, OSError):
                logger.exception("RNNoise could not be imported")

    def available(self) -> bool:
        return bool(self.enabled and self._rnnoise_class and self._np is not None)

    async def process(self, audio: PcmAudio) -> PcmAudio:
        if not self.enabled:
            return audio
        if not self.available():
            raise AiPipelineError(
                "RNNoise is enabled but unavailable. Install pyrnnoise in the backend environment."
            )
        return await asyncio.to_thread(self._process_sync, audio)

    def _process_sync(self, audio: PcmAudio) -> PcmAudio:
        if audio.channels != 1 or len(audio.data) % 2:
            raise AiPipelineError("RNNoise requires signed 16-bit mono PCM audio")
        if not audio.data:
            raise AiPipelineError("RNNoise cannot process empty PCM audio")

        np = self._np
        assert np is not None and self._rnnoise_class is not None
        started = monotonic()
        samples = np.frombuffer(audio.data, dtype="<i2").copy()
        denoiser = self._rnnoise_class(sample_rate=audio.sample_rate)
        parts = [
            np.asarray(denoised, dtype=np.int16)
            for _, denoised in denoiser.denoise_chunk(
                samples.reshape(1, -1),
                partial=True,
            )
        ]
        if not parts:
            raise AiPipelineError("RNNoise produced no audio")
        filtered = np.concatenate(parts, axis=1).reshape(-1)

        # The wrapper's resampler normally preserves length. Keep the pipeline
        # duration stable even for unusual, non-frame-aligned inputs.
        if filtered.size < samples.size:
            filtered = np.pad(filtered, (0, samples.size - filtered.size))
        elif filtered.size > samples.size:
            filtered = filtered[: samples.size]

        input_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        output_rms = float(np.sqrt(np.mean(filtered.astype(np.float64) ** 2)))
        input_peak = int(np.max(np.abs(samples.astype(np.int32))))
        output_peak = int(np.max(np.abs(filtered.astype(np.int32))))
        clipped_fraction = float(np.mean(np.abs(samples.astype(np.int32)) >= 32760))
        logger.info(
            "RNNoise completed: latency_ms=%.1f samples=%d rate=%d input_rms=%.1f output_rms=%.1f input_peak=%d output_peak=%d input_clipped_fraction=%.5f",
            (monotonic() - started) * 1000,
            samples.size,
            audio.sample_rate,
            input_rms,
            output_rms,
            input_peak,
            output_peak,
            clipped_fraction,
        )
        return PcmAudio(filtered.astype("<i2", copy=False).tobytes(), audio.sample_rate, 1)


class RnnoiseStreamFilter:
    """Per-session streaming filter used before Silero VAD."""

    def __init__(self, sample_rate: int) -> None:
        import numpy as np
        from pyrnnoise import RNNoise

        self.np = np
        self.rnnoise_class = RNNoise
        self.sample_rate = sample_rate
        self.denoiser = RNNoise(sample_rate=sample_rate)

    def process(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""
        if len(pcm) % 2:
            raise AiPipelineError("RNNoise streaming input must be signed 16-bit PCM")
        samples = self.np.frombuffer(pcm, dtype="<i2").copy().reshape(1, -1)
        parts = [
            self.np.asarray(denoised, dtype=self.np.int16)
            for _, denoised in self.denoiser.denoise_chunk(samples, partial=False)
        ]
        if not parts:
            return b""
        return self.np.concatenate(parts, axis=1).astype("<i2", copy=False).tobytes()

    def reset(self) -> None:
        self.denoiser.reset()
        self.denoiser = self.rnnoise_class(sample_rate=self.sample_rate)

    def close(self) -> None:
        self.denoiser.reset()

