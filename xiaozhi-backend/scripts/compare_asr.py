r"""Compare local Whisper models / paired RNNoise WAVs; no recording or downloads.

Run from backend: .venv\Scripts\python.exe scripts\compare_asr.py --model small
    --audio logs\voice\sample-00-raw.wav logs\voice\sample-00-rnnoise.wav
Supply --reference with words checked by the speaker to measure character error.
Without a reference this reports latency/transcripts, NOT recognition accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import monotonic
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai import PcmAudio
from app.asr import FasterWhisperAsrService
from app.config import get_settings
from app.noise import RnnoiseNoiseSuppressor


def character_error(reference: str, actual: str) -> float:
    reference, actual = reference.casefold().strip(), actual.casefold().strip()
    previous = list(range(len(actual) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, found in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != found)))
        previous = current
    return previous[-1] / max(1, len(reference))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", nargs="+", required=True, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--language", default="auto", choices=("auto", "en", "vi"))
    parser.add_argument("--reference", default="")
    parser.add_argument("--with-rnnoise", action="store_true")
    args = parser.parse_args()
    configured = get_settings()
    settings = replace(
        configured,
        asr_model=args.model or configured.asr_model,
        model_device=args.device or configured.model_device,
        asr_compute_type=args.compute_type if args.compute_type is not None else configured.asr_compute_type,
        asr_language="" if args.language == "auto" else args.language,
        asr_local_files_only=True,
    )
    service = FasterWhisperAsrService(settings)
    for path in args.audio:
        with wave.open(str(path), "rb") as wav:
            if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() != 16000:
                raise ValueError("Use a 16 kHz mono PCM16 WAV (paired diagnostics use this format)")
            audio = PcmAudio(wav.readframes(wav.getnframes()), 16000)
        variants = [("raw", audio)]
        if args.with_rnnoise:
            suppressor = RnnoiseNoiseSuppressor(replace(settings, rnnoise_enabled=True))
            variants.append(("rnnoise", await suppressor.process(audio)))
        for label, candidate in variants:
            started = monotonic()
            result = await service.transcribe(candidate)
            resolved_device = service._resolve_device()
            report = {"model": settings.asr_model, "device": resolved_device, "compute_type": settings.asr_compute_type or ("float16" if resolved_device == "cuda" else "int8"), "language_mode": args.language, "audio": str(path), "variant": label, "seconds": round(monotonic() - started, 3), "transcript": result.text, "language": result.language, "language_probability": result.language_probability, "avg_logprob": result.confidence}
            if args.reference:
                report["character_error_rate"] = round(character_error(args.reference, result.text), 4)
            else:
                report["accuracy"] = "unmeasured: no speaker-verified reference"
            print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
