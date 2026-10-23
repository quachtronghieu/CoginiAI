"""Re-run a saved raw/RNNoise pair through the current Whisper configuration.

This is an offline diagnostic utility.  It does not alter runtime settings or
the original WAV/JSON files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import wave

import numpy as np

from app.ai import PcmAudio
from app.asr import FasterWhisperAsrService, _TranscriptCandidate
from app.config import get_settings
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps


ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = ROOT / "logs" / "voice"
SAMPLE_ID = "sample-00"
REPORT_PATH = VOICE_DIR / f"{SAMPLE_ID}-asr-diagnostic.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_wav(path: Path) -> tuple[PcmAudio, np.ndarray]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate != 16_000:
        raise RuntimeError(
            f"Unexpected WAV format for {path.name}: "
            f"channels={channels}, width={sample_width}, rate={sample_rate}"
        )
    audio = PcmAudio(frames, sample_rate, channels)
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio, samples


def metric_average(segments: list[object], name: str, default: float = 0.0) -> float:
    values = [
        float(value)
        for segment in segments
        if (value := getattr(segment, name, None)) is not None
    ]
    return sum(values) / len(values) if values else default


def run_whisper_once(
    model,
    samples: np.ndarray,
    settings,
    *,
    language: str | None,
    use_hotwords: bool,
    vad_filter: bool,
) -> dict:
    actual_hotwords = (
        settings.asr_hotwords if language != "en" and use_hotwords else None
    )
    segments, info = model.transcribe(
        samples,
        language=language,
        beam_size=3,
        best_of=3,
        temperature=0.0,
        condition_on_previous_text=False,
        hotwords=actual_hotwords,
        no_speech_threshold=0.45,
        log_prob_threshold=-1.0,
        hallucination_silence_threshold=1.0,
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 100}
        if vad_filter
        else None,
    )
    segment_list = list(segments)
    text = " ".join(segment.text.strip() for segment in segment_list).strip()
    detected_language = str(
        language or getattr(info, "language", "") or "und"
    ).lower()
    language_probability = float(
        getattr(info, "language_probability", 0.0) or 0.0
    )
    avg_logprob = metric_average(segment_list, "avg_logprob")
    no_speech_probability = metric_average(
        segment_list, "no_speech_prob"
    )
    candidate = _TranscriptCandidate(
        text,
        detected_language,
        language_probability,
        avg_logprob,
        no_speech_probability,
    )
    return {
        "text": text,
        "language_mode": "forced" if language else "auto",
        "requested_language": language,
        "detected_language": detected_language,
        "language_probability": language_probability,
        "hotwords": actual_hotwords,
        "vad_filter": vad_filter,
        "avg_logprob": avg_logprob,
        "no_speech_probability": no_speech_probability,
        "valid": candidate.valid,
        "selection_score": candidate.score,
        "segments": [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_probability": float(segment.no_speech_prob),
            }
            for segment in segment_list
        ],
    }


class TracingAsrService(FasterWhisperAsrService):
    def __init__(self, settings, model) -> None:
        super().__init__(settings)
        self._model = model
        self.trace: list[dict] = []

    def _transcribe_candidate(
        self,
        model,
        samples,
        language: str | None,
        *,
        hotwords: str = "",
        use_hotwords: bool = True,
    ) -> _TranscriptCandidate:
        actual_hotwords = (
            hotwords
            if language != "en" and use_hotwords and hotwords
            else None
        )
        candidate = super()._transcribe_candidate(
            model,
            samples,
            language,
            hotwords=hotwords,
            use_hotwords=use_hotwords,
        )
        self.trace.append(
            {
                "text": candidate.text,
                "language_mode": "forced" if language else "auto",
                "requested_language": language,
                "returned_language": candidate.language,
                "language_probability": candidate.language_probability,
                "hotwords": actual_hotwords,
                "vad_filter": True,
                "avg_logprob": candidate.avg_logprob,
                "no_speech_probability": candidate.no_speech_probability,
                "valid": candidate.valid,
                "selection_score": candidate.score,
                "audio_source": candidate.audio_source,
            }
        )
        return candidate


def save_internal_vad_audio(
    source_name: str, samples: np.ndarray
) -> dict:
    options = VadOptions(
        min_silence_duration_ms=300,
        speech_pad_ms=100,
        max_speech_duration_s=30,
    )
    timestamps = get_speech_timestamps(samples, options)
    chunks, metadata = collect_chunks(samples, timestamps, max_duration=30)
    kept = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    pcm = (np.clip(kept, -1.0, 1.0) * 32767.0).astype("<i2")
    output_path = VOICE_DIR / f"{SAMPLE_ID}-{source_name}-whisper-vad.wav"
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(pcm.tobytes())
    return {
        "output": output_path.name,
        "sha256": sha256(output_path),
        "input_duration_seconds": len(samples) / 16_000,
        "kept_duration_seconds": len(kept) / 16_000,
        "speech_timestamps_samples": timestamps,
        "speech_timestamps_seconds": [
            {"start": item["start"] / 16_000, "end": item["end"] / 16_000}
            for item in timestamps
        ],
        "chunk_metadata": metadata,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    metadata_path = VOICE_DIR / f"{SAMPLE_ID}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    paths = {
        "raw": VOICE_DIR / metadata["audio"]["raw"],
        "rnnoise": VOICE_DIR / metadata["audio"]["rnnoise"],
    }
    loaded = {name: read_wav(path) for name, path in paths.items()}

    service = FasterWhisperAsrService(settings)
    model = service._get_model()

    variants = {
        "current": {"language": None, "use_hotwords": True, "vad_filter": True},
        "hotwords_off": {
            "language": None,
            "use_hotwords": False,
            "vad_filter": True,
        },
        "forced_vi": {"language": "vi", "use_hotwords": True, "vad_filter": True},
        "internal_vad_off": {
            "language": None,
            "use_hotwords": True,
            "vad_filter": False,
        },
    }

    standalone: dict[str, dict[str, dict]] = {}
    for source_name, (_, samples) in loaded.items():
        standalone[source_name] = {}
        for variant_name, variant in variants.items():
            standalone[source_name][variant_name] = run_whisper_once(
                model, samples, settings, **variant
            )

    application_flow = {}
    for source_name, (audio, _) in loaded.items():
        tracing = TracingAsrService(settings, model)
        selected = tracing._transcribe_sync(audio)
        application_flow[source_name] = {
            "candidates_before_selection": tracing.trace,
            "selected": {
                "text": selected.text,
                "language": selected.language,
                "language_probability": selected.language_probability,
                "avg_logprob": selected.confidence,
                "no_speech_probability": selected.no_speech_probability,
            },
        }

    internal_vad = {
        source_name: save_internal_vad_audio(source_name, samples)
        for source_name, (_, samples) in loaded.items()
    }

    report = {
        "sample_identity": {
            "sample_id": metadata.get("sample_id"),
            "created_at": metadata.get("created_at"),
            "websocket_session_id": metadata.get("session_id"),
            "note": "The current diagnostics schema does not persist the WebSocket session UUID.",
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in [paths["raw"], paths["rnnoise"], metadata_path]
            },
        },
        "current_settings": {
            "model": settings.asr_model,
            "device": settings.model_device,
            "compute_type": settings.asr_compute_type or "float16 (resolved for CUDA)",
            "configured_language": settings.asr_language or None,
            "hotwords": settings.asr_hotwords,
            "rnnoise_asr_mode": settings.rnnoise_asr_mode,
            "application_primary_source": metadata["asr"]["primary_source"],
        },
        "json_asr": metadata["asr"],
        "standalone_whisper": standalone,
        "application_asr_flow": application_flow,
        "application_selected": application_flow["raw"]["selected"],
        "internal_whisper_vad": internal_vad,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"REPORT={REPORT_PATH}")


if __name__ == "__main__":
    main()
