"""Compare audio readiness for identical saved answers using real Piper/Opus.

ASR, RAG and Gemma are intentionally excluded: the answer text is held fixed.
Voices are warmed, exact-text audio caches are cleared before each condition.
No physical speaker or network playback is measured.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter


async def benchmark(replay_dir, output):
    from app.ai import AsrResult, LocalAiPipeline, PcmAudio
    from app.codec import PyAvOpusCodec
    from app.config import get_settings
    from app.tts import PiperTtsService

    tts = PiperTtsService(get_settings())
    await tts.warmup()
    codec = PyAvOpusCodec(get_settings())

    class SavedCodec:
        async def decode_opus(self, frames):
            return PcmAudio(b"\0\0" * 160, 16000)
        async def encode_opus(self, audio):
            return await codec.encode_opus(audio)

    class SavedAsr:
        async def transcribe(self, audio):
            return AsrResult("xin chào", "vi", 1)

    rows = []
    originals = json.loads((replay_dir / "small" / "results.json").read_text(encoding="utf-8"))["rows"]
    for index, original in enumerate(originals):
        if not original.get("answer"):
            continue
        answer = original["answer"]

        class SavedAnswer:
            async def generate(self, *args, **kwargs):
                return answer

        pipe = LocalAiPipeline(SavedCodec(), SavedAsr(), SavedAnswer(), tts)
        row = {"id": original["id"], "answer": answer}
        # Alternate order to reduce systematic first-run bias. Cache is empty
        # for both modes; models themselves stay loaded in the same service.
        for streaming in ((False, True) if index % 2 == 0 else (True, False)):
            tts._audio_cache.clear()
            tts._cache_bytes = 0
            first = None
            count = 0
            frames = 0
            started = perf_counter()
            async def capture(chunk):
                nonlocal first, count, frames
                if first is None:
                    first = perf_counter() - started
                count += 1
                frames += len(chunk.audio_frames)
            result = await pipe.run([b"saved"], on_audio_chunk=capture if streaming else None)
            elapsed = perf_counter() - started
            row["sentences" if streaming else "full"] = {
                "first_audio_seconds": first if streaming else elapsed,
                "all_audio_seconds": elapsed,
                "chunks": count if streaming else 1,
                "output_frames": frames if streaming else len(result.audio_frames)}
        rows.append(row)
        print(json.dumps({"id": row["id"], "full": row["full"]["first_audio_seconds"],
            "sentences": row["sentences"]["first_audio_seconds"]}), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Fixed-answer Piper/Opus benchmark", "", __doc__, "",
        "| Answer | Full first audio s | Sentence first audio s | Sentence all audio s |",
        "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['full']['first_audio_seconds']:.3f} | {row['sentences']['first_audio_seconds']:.3f} | {row['sentences']['all_audio_seconds']:.3f} |")
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT_DIR={output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "logs" / ("audio-readiness-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    asyncio.run(benchmark(args.replay_dir, output))


if __name__ == "__main__":
    main()
