"""Small local AI smoke-test CLI; never downloads models."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.ai import build_ai_pipeline
from app.asr import FasterWhisperAsrService
from app.config import get_settings
from app.llm import OllamaLlmService
from app.tts import PiperTtsService
from app.ai import PcmAudio


def read_wav(path: Path) -> PcmAudio:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Input WAV must be signed 16-bit mono")
        return PcmAudio(source.readframes(source.getnframes()), source.getframerate(), 1)


def write_wav(path: Path, audio: PcmAudio) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(audio.channels)
        target.setsampwidth(2)
        target.setframerate(audio.sample_rate)
        target.writeframes(audio.data)


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    if args.command == "check":
        pipeline = build_ai_pipeline(settings)
        status = pipeline.health()
        llm = getattr(pipeline, "llm", None)
        if llm is not None and hasattr(llm, "ready"):
            status["llm"] = await llm.ready()
        print(json.dumps(status, indent=2))
        return
    if args.command == "asr":
        result = await FasterWhisperAsrService(settings).transcribe(read_wav(Path(args.wav)))
        print(json.dumps({"text": result.text, "language": result.language, "language_probability": result.language_probability}, ensure_ascii=False))
        return
    if args.command == "llm":
        print(
            await OllamaLlmService(settings).generate(
                args.text, args.language, args.language_probability
            )
        )
        return
    if args.command == "tts":
        audio = await PiperTtsService(settings).synthesize(
            args.text, args.language, args.language_probability
        )
        write_wav(Path(args.output), audio)
        print(f"wrote {args.output} ({audio.sample_rate} Hz, {len(audio.data)} bytes)")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    asr = sub.add_parser("asr")
    asr.add_argument("wav")
    llm = sub.add_parser("llm")
    llm.add_argument("text")
    llm.add_argument("--language", default="vi")
    llm.add_argument("--language-probability", type=float, default=1.0)
    tts = sub.add_parser("tts")
    tts.add_argument("text")
    tts.add_argument("output")
    tts.add_argument("--language", default="vi")
    tts.add_argument("--language-probability", type=float, default=1.0)
    return command


if __name__ == "__main__":
    asyncio.run(run(parser().parse_args()))
