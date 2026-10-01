"""Replay every retained raw diagnostic WAV through the VAD/ASR confidence gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import unicodedata
import wave

import numpy as np
import torch
from silero_vad import load_silero_vad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai import PcmAudio
from app.asr import FasterWhisperAsrService
from app.config import get_settings


ROOT = Path(__file__).resolve().parent.parent
VOICE = ROOT / "logs" / "voice"
REPORT = ROOT / "logs" / "voice-gate-evaluation.json"


def normalized(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().strip(" .!?,")


def vad_evidence(model, samples: np.ndarray, threshold: float) -> dict:
    model.reset_states()
    probabilities = []
    for offset in range(0, len(samples) - 511, 512):
        chunk = torch.from_numpy(samples[offset:offset + 512])
        probabilities.append(float(model(chunk, 16_000).item()))
    longest = current = 0
    for probability in probabilities:
        if probability >= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "max_probability": max(probabilities, default=0.0),
        "frames_above_threshold": sum(value >= threshold for value in probabilities),
        "max_contiguous_speech_ms": longest * 32.0,
    }


async def main() -> None:
    settings = get_settings()
    vad_model = load_silero_vad(onnx=False)
    asr = FasterWhisperAsrService(settings)
    rows = []
    for metadata_path in sorted(VOICE.glob("sample-*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_name = (metadata.get("audio") or {}).get("raw")
        if not raw_name or not (VOICE / raw_name).exists():
            continue
        with wave.open(str(VOICE / raw_name), "rb") as wav:
            pcm = wav.readframes(wav.getnframes())
            rate = wav.getframerate()
            channels = wav.getnchannels()
        if rate != 16_000 or channels != 1:
            continue
        float_samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        evidence = vad_evidence(vad_model, float_samples, settings.vad_threshold)
        speech_started = evidence["frames_above_threshold"] > 0
        # A weak VAD is evidence, not a hard drop: ASR may rescue genuine
        # short/quiet speech. The utterance is rejected only when ASR agrees.
        result = await asr.transcribe(PcmAudio(pcm, rate, channels))
        weak_combined = bool(
            result
            and evidence["max_contiguous_speech_ms"] < settings.vad_strong_speech_ms
            and result.no_speech_probability >= settings.asr_no_speech_gate_probability
        )
        accepted = bool(result.text.strip() and not weak_combined)
        truth = metadata.get("ground_truth")
        expected = None if truth is None else bool(str(truth).strip())
        critical = metadata.get("critical_terms") or []
        transcript = result.text.strip() if result else ""
        row = {
            "sample_id": metadata.get("sample_id"),
            "ground_truth": truth,
            "condition": metadata.get("condition"),
            "vad": evidence,
            "asr": None if result is None else {
                "text": transcript,
                "language": result.language,
                "confidence": result.confidence,
                "no_speech_probability": result.no_speech_probability,
            },
            "decision": "accept" if accepted else "reject",
            "expected_accept": expected,
            "decision_correct": None if expected is None else accepted == expected,
            "critical_terms_ok": None if not critical else all(
                normalized(term) in normalized(transcript) for term in critical
            ),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {
        "total": len(rows),
        "labeled": sum(row["expected_accept"] is not None for row in rows),
        "decision_correct": sum(row["decision_correct"] is True for row in rows),
        "decision_wrong": sum(row["decision_correct"] is False for row in rows),
        "rows": rows,
    }
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
