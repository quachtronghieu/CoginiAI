"""Read-only paired-audio checks; not a general word-accuracy measurement."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.ai import PcmAudio
from app.allergy import CustomerProfile
from app.asr import FasterWhisperAsrService, repetitive_transcript
from app.config import get_settings
from app.dialogue import prepare_turn


def read(path):
    with wave.open(str(path), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        return PcmAudio(wav.readframes(wav.getnframes()), 16000)


def stats(audio):
    samples = np.frombuffer(audio.data, dtype='<i2').astype(np.float64)
    return {"duration_seconds": len(samples)/16000, "rms": round(float(np.sqrt(np.mean(samples*samples))), 2), "peak": int(np.max(np.abs(samples))), "clipped_fraction": round(float(np.mean(np.abs(samples) >= 32760)), 5)}


async def main():
    root = Path("logs/regression-voice-20260928-0851")
    service = FasterWhisperAsrService(replace(get_settings(), asr_local_files_only=True))
    for slot in ("02", "11", "08", "00"):
        raw, filtered = read(root / f"sample-{slot}-raw.wav"), read(root / f"sample-{slot}-rnnoise.wav")
        assert len(raw.data) == len(filtered.data)
        profile = CustomerProfile(allergy_question_asked=True)
        if slot == "08":
            profile.pending_allergy_status = "declared"
            profile.pending_allergens = ["sữa"]
        result = await service.transcribe(filtered, customer_profile=profile, raw_audio=raw)
        reply = ""
        if result.text and not result.needs_clarification:
            reply, _ = prepare_turn(profile, result.text, "vi")
        if slot == "02":
            # The user reported these intended allergens for this exact turn.
            assert not result.needs_clarification and set(profile.pending_allergens) == {"hải sản", "trứng"}, (result, reply)
            assert profile.allergy_status == "unknown", "Never commit without readback confirmation"
        assert not repetitive_transcript(result.text)
        print(json.dumps({"slot": slot, "raw_signal": stats(raw), "rnnoise_signal": stats(filtered), "transcript": result.text, "alternatives": result.alternatives, "clarification": result.needs_clarification, "reason": result.clarification_reason, "reply": reply, "pending_allergens": profile.pending_allergens, "reference_status": "user-reported ingredient set" if slot == "02" else "no speaker-verified verbatim reference; diagnostic only"}, ensure_ascii=False), flush=True)
    print(json.dumps({"result": "PASS", "paired_recordings": 4, "user_reported_allergen_case": 1, "general_word_accuracy_measured": False}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
