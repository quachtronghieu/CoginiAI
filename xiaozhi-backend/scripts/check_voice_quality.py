"""Synthetic clean/noisy/RNNoise ASR contract tests with known input references."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.ai import PcmAudio
from app.allergy import CustomerProfile, is_rejection
from app.asr import FasterWhisperAsrService
from app.codec import PyAvOpusCodec
from app.config import get_settings
from app.dialogue import prepare_turn
from app.noise import RnnoiseNoiseSuppressor
from app.tts import create_tts_service


async def main():
    settings = get_settings()
    tts = create_tts_service(settings)
    codec = PyAvOpusCodec(replace(settings, server_sample_rate=16000))
    asr = FasterWhisperAsrService(settings)
    noise = RnnoiseNoiseSuppressor(replace(settings, rnnoise_enabled=True))
    rng = np.random.default_rng(230928)
    references = [("Xin chào", "greeting", set()), ("Tôi bị dị ứng với hải sản và trứng", "allergen", {"hải sản", "trứng"}), ("Đậu phộng, sữa", "allergen", {"đậu phộng", "sữa"}), ("Không đúng", "negative", set())]
    passed = 0
    clarifications = 0
    for reference, kind, expected in references:
        clean = await codec.decode_opus(await codec.encode_opus(await tts.synthesize(reference, "vi", 1.0)))
        samples = np.frombuffer(clean.data, dtype='<i2').astype(np.float64)
        signal_rms = np.sqrt(np.mean(samples*samples))
        noise_rms = max(400.0, signal_rms / (10**(12/20)))
        time = np.arange(len(samples)) / 16000
        interference = rng.normal(0, noise_rms * .8, len(samples)) + noise_rms * .6 * np.sin(2*np.pi*180*time)
        noisy = PcmAudio(np.clip(samples + interference, -32768, 32767).astype('<i2').tobytes(), 16000)
        for variant, raw in (("clean", clean), ("seeded_noise", noisy)):
            filtered = await noise.process(raw)
            assert len(filtered.data) == len(raw.data) and filtered.sample_rate == raw.sample_rate
            profile = CustomerProfile()
            if kind == "allergen":
                profile.allergy_question_asked = True
            elif kind == "negative":
                profile.pending_allergy_status, profile.pending_allergens = "declared", ["sữa"]
            result = await asr.transcribe(filtered, customer_profile=profile, raw_audio=raw)
            assert result.text, (reference, variant, result)
            reply = "ASR requires clarification" if result.needs_clarification else prepare_turn(profile, result.text, "vi")[0]
            handled_unclear = False
            if kind == "greeting":
                correct = "tư vấn món" in reply and "dị ứng" not in reply
                handled_unclear = "chưa nghe rõ" in reply and not profile.allergy_question_asked
            elif kind == "allergen":
                correct = set(profile.pending_allergens) == expected and not profile.pending_allergy_note
                handled_unclear = result.needs_clarification or bool(profile.pending_allergy_note and not profile.pending_allergy_status)
                assert profile.allergy_status == "unknown", "ASR must not commit allergens without confirmation"
            else:
                correct = is_rejection(result.text) and not profile.pending_allergy_status
                handled_unclear = result.needs_clarification or bool(profile.pending_allergy_status and profile.allergy_confirmation_unclear)
            # Recognition and safe handling are different metrics. Even a clean
            # short TTS clip can be misheard: it MUST trigger clarification, not
            # silently drop an unknown ingredient or commit the wrong list.
            assert correct or handled_unclear, (reference, variant, result, reply)
            clarifications += int(not correct and handled_unclear)
            passed += 1
            print(json.dumps({"reference": reference, "variant": variant, "transcript": result.text, "raw_filtered_alternatives": result.alternatives, "reply": reply, "status": "CORRECT" if correct else "SAFE_CLARIFICATION"}, ensure_ascii=False), flush=True)
    # Pure interference must not invent a greeting/allergen utterance.
    noise_only = PcmAudio(np.clip(rng.normal(0, 1500, 32000), -32768, 32767).astype('<i2').tobytes(), 16000)
    filtered = await noise.process(noise_only)
    result = await asr.transcribe(filtered, customer_profile=CustomerProfile(allergy_question_asked=True), raw_audio=noise_only)
    assert result.text == "", result
    print(json.dumps({"result": "PASS", "synthetic_clean_noisy_cases": passed, "recognized_correctly": passed - clarifications, "clarifications_required": clarifications, "noise_only_rejected": True, "physical_microphone_accuracy_measured": False}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
