from dataclasses import dataclass, field
import asyncio
import importlib.util
import logging
from time import monotonic
from typing import Callable, Protocol
from collections.abc import Awaitable
from contextlib import suppress

from .allergy import CustomerProfile
from .config import Settings
from .diagnostics import VoiceDiagnostics, supported_kwargs
from .dialogue import response_language


logger = logging.getLogger(__name__)


class AiPipelineError(RuntimeError):
    pass


class AiUnavailableError(AiPipelineError):
    pass


class AudioCodecError(AiPipelineError):
    pass


class EmptyUtteranceError(AiPipelineError):
    pass


class NonSpeechUtteranceError(AiPipelineError):
    pass


@dataclass
class PcmAudio:
    data: bytes
    sample_rate: int
    channels: int = 1


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str
    language_probability: float = 0.0
    confidence: float = 0.0  # average log probability from whisper segments
    needs_clarification: bool = False
    alternatives: tuple[str, ...] = ()
    clarification_reason: str = ""
    no_speech_probability: float = 0.0
    candidates: tuple[dict, ...] = field(default=(), compare=False)


@dataclass
class AiResponse:
    transcript: str
    answer: str
    audio_frames: list[bytes] = field(default_factory=list)
    emotion: str = "neutral"
    language: str = ""
    language_probability: float = 0.0
    diagnostic_key: str | None = field(default=None, compare=False, repr=False)


class AudioCodecService(Protocol):
    async def decode_opus(self, frames: list[bytes]) -> PcmAudio: ...

    async def encode_opus(self, audio: PcmAudio) -> list[bytes]: ...

    def available(self) -> bool: ...


class AsrService(Protocol):
    async def transcribe(self, audio: PcmAudio) -> AsrResult: ...

    def available(self) -> bool: ...


class NoiseSuppressorService(Protocol):
    async def process(self, audio: PcmAudio) -> PcmAudio: ...

    def available(self) -> bool: ...


class LlmService(Protocol):
    async def generate(
        self,
        transcript: str,
        language: str,
        language_probability: float,
        confidence: float = 0.0,
        history: list[dict[str, str]] | None = None,
        persona: str = "default",
        customer_profile: CustomerProfile | None = None,
    ) -> str: ...

    def available(self) -> bool: ...


class TtsService(Protocol):
    async def synthesize(
        self, text: str, language: str, language_probability: float
    ) -> PcmAudio: ...

    def available(self) -> bool: ...

    def health(self) -> dict[str, bool]: ...


class AiPipeline(Protocol):
    async def run(
        self,
        opus_frames: list[bytes],
        history: list[dict[str, str]] | None = None,
        persona: str = "default",
        on_transcript: Callable[[str], str] | None = None,
        customer_profile: CustomerProfile | None = None,
        vad_evidence: dict | None = None,
        on_audio_chunk: Callable[[AiResponse], Awaitable[None]] | None = None,
    ) -> AiResponse: ...

    def health(self) -> dict[str, bool]: ...

    async def record_rejected_utterance(
        self,
        opus_frames: list[bytes],
        rejection_reason: str,
        vad_evidence: dict | None = None,
    ) -> None: ...


class LocalAiPipeline:
    """Offline pipeline assembled from local service adapters."""

    def __init__(
        self,
        codec: AudioCodecService,
        asr: AsrService,
        llm: LlmService,
        tts: TtsService,
        noise_suppressor: NoiseSuppressorService | None = None,
        diagnostics: VoiceDiagnostics | None = None,
        rnnoise_asr_mode: str = "raw",
        vad_strong_speech_ms: int = 250,
        asr_no_speech_gate_probability: float = 0.60,
    ) -> None:
        self.codec = codec
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.noise_suppressor = noise_suppressor
        self.diagnostics = diagnostics
        self.rnnoise_asr_mode = rnnoise_asr_mode
        self.vad_strong_speech_ms = vad_strong_speech_ms
        self.asr_no_speech_gate_probability = asr_no_speech_gate_probability

    @staticmethod
    def _asr_metadata(result: AsrResult) -> dict:
        return {
            "text": result.text.strip(),
            "language": result.language,
            "language_probability": result.language_probability,
            "confidence": result.confidence,
            "needs_clarification": result.needs_clarification,
            "clarification_reason": result.clarification_reason,
            "no_speech_probability": result.no_speech_probability,
            "candidates": list(result.candidates),
        }

    def _weak_vad_and_asr(
        self, vad_evidence: dict | None, result: AsrResult
    ) -> bool:
        if not vad_evidence or not vad_evidence.get("available"):
            return False
        weak_vad = (
            float(vad_evidence.get("max_contiguous_speech_ms", 0.0))
            < self.vad_strong_speech_ms
        )
        weak_asr = (
            result.no_speech_probability
            >= self.asr_no_speech_gate_probability
        )
        return weak_vad and weak_asr

    async def _transcribe(self, audio: PcmAudio, customer_profile, raw_audio=None) -> AsrResult:
        kwargs = supported_kwargs(
            self.asr.transcribe,
            {"customer_profile": customer_profile, "raw_audio": raw_audio},
        )
        return await self.asr.transcribe(audio, **kwargs)

    async def run(
        self,
        opus_frames: list[bytes],
        history: list[dict[str, str]] | None = None,
        persona: str = "default",
        on_transcript: Callable[[str], str] | None = None,
        customer_profile: CustomerProfile | None = None,
        vad_evidence: dict | None = None,
        on_audio_chunk: Callable[[AiResponse], Awaitable[None]] | None = None,
    ) -> AiResponse:
        started = monotonic()
        if not opus_frames:
            raise AiPipelineError("Cannot process an empty utterance")
        diagnostic_key = None
        timings = {}
        try:
            raw_audio = await self.codec.decode_opus(opus_frames)
            if self.diagnostics is not None:
                diagnostic_key = self.diagnostics.start_sample(
                    raw_audio,
                    {
                        "input_opus_frames": len(opus_frames),
                        "vad": vad_evidence or {},
                        "rnnoise_asr_mode": self.rnnoise_asr_mode,
                    },
                )

            filtered_audio = None
            if self.rnnoise_asr_mode != "raw" and self.noise_suppressor is not None:
                filtered_audio = await self.noise_suppressor.process(raw_audio)
                if diagnostic_key is not None:
                    self.diagnostics.save_filtered(diagnostic_key, filtered_audio)

            timings["audio_preparation"] = (monotonic() - started) * 1000
            stage_started = monotonic()
            raw_result = None
            filtered_result = None
            if self.rnnoise_asr_mode == "shadow" and filtered_audio is not None:
                raw_result = await self._transcribe(raw_audio, customer_profile)
                filtered_result = await self._transcribe(filtered_audio, customer_profile)
                asr_result = raw_result
                logger.info(
                    "RNNoise shadow comparison sample=%s raw=%r filtered=%r",
                    diagnostic_key,
                    raw_result.text.strip(),
                    filtered_result.text.strip(),
                )
            elif filtered_audio is None:
                raw_result = await self._transcribe(raw_audio, customer_profile)
                asr_result = raw_result
            else:
                filtered_result = await self._transcribe(
                    filtered_audio,
                    customer_profile,
                    raw_audio if self.noise_suppressor is not None else None,
                )
                asr_result = filtered_result

            timings["asr"] = (monotonic() - stage_started) * 1000
            if diagnostic_key is not None:
                self.diagnostics.update_result(
                    diagnostic_key,
                    status="asr_complete" if asr_result.text.strip() else "asr_rejected",
                    rejection_reason=None if asr_result.text.strip() else "empty_transcript",
                    asr={
                        "primary_source": "raw" if raw_result is not None else "rnnoise",
                        "raw": self._asr_metadata(raw_result) if raw_result else None,
                        "rnnoise": self._asr_metadata(filtered_result) if filtered_result else None,
                    },
                )
            transcript = asr_result.text.strip()
            if transcript and self._weak_vad_and_asr(vad_evidence, asr_result):
                logger.info(
                    "Combined VAD/ASR gate rejected sample=%s transcript=%r max_contiguous_speech_ms=%.1f no_speech_probability=%.3f",
                    diagnostic_key,
                    transcript,
                    float((vad_evidence or {}).get("max_contiguous_speech_ms", 0.0)),
                    asr_result.no_speech_probability,
                )
                if diagnostic_key is not None:
                    self.diagnostics.update_result(
                        diagnostic_key,
                        status="asr_vad_rejected",
                        rejection_reason="weak_vad_and_high_asr_no_speech",
                    )
                raise NonSpeechUtteranceError(
                    "VAD and ASR both indicate uncertain non-speech"
                )
            if not transcript:
                if vad_evidence and vad_evidence.get("available"):
                    if diagnostic_key is not None:
                        self.diagnostics.update_result(
                            diagnostic_key,
                            status="asr_vad_rejected",
                            rejection_reason="empty_asr_with_vad_evidence",
                        )
                duration_ms = (
                    len(raw_audio.data)
                    / (2 * raw_audio.channels * raw_audio.sample_rate)
                    * 1000
                )
                logger.warning(
                    "ASR empty transcript ignored: input_frames=%d pcm_duration_ms=%.1f",
                    len(opus_frames),
                    duration_ms,
                )
                raise EmptyUtteranceError("ASR returned an empty transcript")

            if diagnostic_key is not None:
                logger.info("Voice comparison sample=%s transcript=%r", diagnostic_key, transcript)
            language = response_language(customer_profile, transcript, asr_result.language) if customer_profile is not None else asr_result.language

            if on_transcript is not None and not asr_result.needs_clarification:
                new_persona = on_transcript(transcript)
                if new_persona:
                    persona = new_persona

            if asr_result.needs_clarification:
                if customer_profile is not None:
                    customer_profile.last_route = "asr_clarification"
                    customer_profile.dialogue_action = "asr_clarification"
                    customer_profile.resolved_request = ""
                    customer_profile.evidence_sources.clear()
                logger.info("ASR safety clarification: reason=%s alternatives=%r", asr_result.clarification_reason, asr_result.alternatives)
                if asr_result.clarification_reason == "confirmation_disagreement":
                    raw_answer = "I could not reliably hear your confirmation. Please say correct or incorrect." if language == "en" else "Mình chưa nghe rõ phần xác nhận. Bạn hãy nói chính xác, hoặc không đúng."
                else:
                    raw_answer = "I heard different ingredient names. Please repeat your complete allergy list slowly." if language == "en" else "Mình nghe chưa thống nhất tên thành phần dị ứng. Bạn hãy nói chậm từng thành phần; mình sẽ đọc lại để bạn xác nhận."
            else:
                stage_started = monotonic()
                kwargs = supported_kwargs(self.llm.generate, {"history": history, "persona": persona, "customer_profile": customer_profile})
                raw_answer = await self.llm.generate(
                    transcript, language, asr_result.language_probability,
                    asr_result.confidence, **kwargs,
                )
                timings["dialogue_rag_llm"] = (monotonic() - stage_started) * 1000
            answer = raw_answer.strip()
            if not answer:
                raise AiPipelineError("LLM returned an empty response")
            if diagnostic_key is not None:
                self.diagnostics.update_result(
                    diagnostic_key, answer=answer, delivery_status="rendering",
                    region=customer_profile.region if customer_profile else "",
                    route=customer_profile.last_route if customer_profile else "",
                    sources=list(customer_profile.evidence_sources) if customer_profile else [],
                )

            from .speech import speech_chunks
            chunks = speech_chunks(answer) if on_audio_chunk is not None else [answer]
            timings["tts"] = timings["encode"] = 0.0
            select_voice = getattr(self.tts, "select_language", None)
            voice_language = select_voice(answer, language, asr_result.language_probability) if select_voice is not None else language

            async def render(chunk):
                stage_started = monotonic()
                output_audio = await self.tts.synthesize(chunk, language, asr_result.language_probability,
                    **supported_kwargs(self.tts.synthesize, {"voice_language": voice_language}))
                tts_ms = (monotonic() - stage_started) * 1000
                stage_started = monotonic()
                frames = await self.codec.encode_opus(output_audio)
                return frames, tts_ms, (monotonic() - stage_started) * 1000

            audio_frames = []
            pending = asyncio.create_task(render(chunks[0]))
            try:
                for index, chunk in enumerate(chunks):
                    frames, tts_ms, encode_ms = await pending
                    timings["tts"] += tts_ms
                    timings["encode"] += encode_ms
                    # At most one sentence is prefetched while the current
                    # sentence is paced to the device. Abort cancels the task.
                    pending = asyncio.create_task(render(chunks[index + 1])) if index + 1 < len(chunks) else None
                    if index == 0:
                        timings["first_audio_ready"] = (monotonic() - started) * 1000
                    if on_audio_chunk is not None:
                        await on_audio_chunk(AiResponse(
                            transcript=transcript, answer=chunk, audio_frames=frames,
                            language=language, language_probability=asr_result.language_probability,
                            diagnostic_key=diagnostic_key,
                        ))
                    else:
                        audio_frames.extend(frames)
            finally:
                if pending is not None:
                    pending.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await pending
            timings["total"] = (monotonic() - started) * 1000
            if diagnostic_key is not None:
                self.diagnostics.update_result(
                    diagnostic_key,
                    timings_ms={name: round(value, 2) for name, value in timings.items()},
                    answer=answer,
                    region=customer_profile.region if customer_profile else "",
                    route=customer_profile.last_route if customer_profile else "",
                    sources=list(customer_profile.evidence_sources) if customer_profile else [],
                    sentence_streaming=on_audio_chunk is not None,
                    delivery_status="audio_sent" if on_audio_chunk is not None else "audio_ready",
                )
            return AiResponse(
                transcript=transcript,
                answer=answer,
                audio_frames=audio_frames,
                language=language,
                language_probability=asr_result.language_probability,
                diagnostic_key=diagnostic_key,
            )
        except asyncio.CancelledError:
            if diagnostic_key is not None:
                self.diagnostics.update_result(diagnostic_key, delivery_status="cancelled")
            raise
        except Exception as exc:
            if diagnostic_key is not None and not isinstance(
                exc, (EmptyUtteranceError, NonSpeechUtteranceError)
            ):
                self.diagnostics.update_result(
                    diagnostic_key,
                    status="pipeline_error",
                    rejection_reason=f"{type(exc).__name__}: {exc}",
                )
            raise
        finally:
            logger.info("AI pipeline total latency_ms=%.1f", (monotonic() - started) * 1000)

    async def record_rejected_utterance(
        self,
        opus_frames: list[bytes],
        rejection_reason: str,
        vad_evidence: dict | None = None,
    ) -> None:
        """Retain diagnostics for VAD-rejected audio without invoking ASR/LLM."""
        if self.diagnostics is None or not opus_frames:
            return
        key = None
        try:
            raw_audio = await self.codec.decode_opus(opus_frames)
            key = self.diagnostics.start_sample(
                raw_audio,
                {
                    "input_opus_frames": len(opus_frames),
                    "vad": vad_evidence or {},
                    "rnnoise_asr_mode": self.rnnoise_asr_mode,
                },
            )
            if self.rnnoise_asr_mode != "raw" and self.noise_suppressor is not None:
                filtered_audio = await self.noise_suppressor.process(raw_audio)
                self.diagnostics.save_filtered(key, filtered_audio)
            self.diagnostics.update_result(
                key,
                status="vad_rejected",
                rejection_reason=rejection_reason,
                asr={
                    "primary_source": None,
                    "raw": None,
                    "rnnoise": None,
                },
            )
            logger.info(
                "VAD diagnostic sample=%s rejected_before_asr reason=%s",
                key,
                rejection_reason,
            )
        except asyncio.CancelledError:
            # A device may close the listening session immediately after the
            # VAD timeout. Raw audio is already durable; finish its rejection
            # record even if the optional RNNoise shadow write is interrupted.
            if key is not None:
                self.diagnostics.update_result(
                    key,
                    status="vad_rejected",
                    rejection_reason=rejection_reason,
                    diagnostic_note="RNNoise shadow recording was cancelled by session close.",
                    asr={
                        "primary_source": None,
                        "raw": None,
                        "rnnoise": None,
                    },
                )
            raise
        except Exception as exc:
            if key is not None:
                self.diagnostics.update_result(
                    key,
                    status="diagnostic_error",
                    rejection_reason=f"{type(exc).__name__}: {exc}",
                )
            logger.exception("Could not retain VAD-rejected diagnostic audio")

    def health(self) -> dict[str, bool]:
        status = {
            "codec": self.codec.available(),
            "asr": self.asr.available(),
            "llm": self.llm.available(),
            "rnnoise": bool(
                self.noise_suppressor is not None
                and self.noise_suppressor.available()
            ),
        }
        rag_available = getattr(self.llm, "rag_available", None)
        if callable(rag_available):
            status["rag"] = bool(rag_available())
        status.update(self.tts.health())
        return status


class UnavailableAiPipeline:
    def __init__(self, reason: str, component_health: dict[str, bool] | None = None) -> None:
        self.reason = reason
        self.component_health = component_health or {
            "codec": False,
            "asr": False,
            "llm": False,
            "rnnoise": False,
            "rag": False,
            "tts": False,
            "tts_vi": False,
            "tts_en": False,
        }

    async def run(
        self,
        opus_frames: list[bytes],
        history: list[dict[str, str]] | None = None,
        persona: str = "default",
        on_transcript: Callable[[str], str] | None = None,
        customer_profile: CustomerProfile | None = None,
        vad_evidence: dict | None = None,
    ) -> AiResponse:
        del opus_frames, history, persona, on_transcript, customer_profile, vad_evidence
        raise AiUnavailableError(self.reason)

    def health(self) -> dict[str, bool]:
        return dict(self.component_health)

    async def record_rejected_utterance(
        self,
        opus_frames: list[bytes],
        rejection_reason: str,
        vad_evidence: dict | None = None,
    ) -> None:
        del opus_frames, rejection_reason, vad_evidence


def build_ai_pipeline(settings: Settings) -> AiPipeline:
    from .noise import RnnoiseNoiseSuppressor
    from .rag import CuisineRagService
    from .tts import PiperTtsService, create_tts_service

    tts = create_tts_service(settings)
    noise_suppressor = (
        RnnoiseNoiseSuppressor(settings)
        if settings.rnnoise_enabled and settings.rnnoise_asr_mode != "raw"
        else None
    )
    tts_health = tts.health()
    component_health = {
        "codec": bool(
            importlib.util.find_spec("av") and importlib.util.find_spec("numpy")
        ),
        "asr": bool(settings.asr_model and importlib.util.find_spec("faster_whisper")),
        "llm": bool(settings.llm_base_url and settings.llm_model),
        "rnnoise": bool(noise_suppressor and noise_suppressor.available()),
        "rag": CuisineRagService(settings).available(),
        **tts_health,
    }
    tts_configured = tts.available()
    configured_models = all(
        [
            settings.asr_model,
            settings.llm_base_url,
            settings.llm_model,
            tts_configured,
            not settings.rnnoise_enabled
            or settings.rnnoise_asr_mode == "raw"
            or component_health["rnnoise"],
        ]
    )
    if not configured_models:
        return UnavailableAiPipeline(
            "Local AI is not configured. Set ASR_MODEL, LLM_BASE_URL, LLM_MODEL, and TTS.",
            component_health,
        )

    from .asr import FasterWhisperAsrService
    from .codec import PyAvOpusCodec
    from .llm import OllamaLlmService
    return LocalAiPipeline(
        codec=PyAvOpusCodec(settings),
        asr=FasterWhisperAsrService(settings),
        llm=OllamaLlmService(settings),
        tts=tts,
        noise_suppressor=noise_suppressor,
        diagnostics=VoiceDiagnostics(slots=settings.diagnostics_slots) if settings.diagnostics_enabled else None,
        rnnoise_asr_mode=settings.rnnoise_asr_mode,
        vad_strong_speech_ms=settings.vad_strong_speech_ms,
        asr_no_speech_gate_probability=settings.asr_no_speech_gate_probability,
    )
