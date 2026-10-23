import asyncio
from dataclasses import replace

import pytest

from app.ai import PcmAudio
from app.allergy import CustomerProfile
from app.asr import FasterWhisperAsrService, _TranscriptCandidate
from app.config import get_settings


class Model:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(kwargs)
        segment = type("Segment", (), {"text": "Không đúng", "avg_logprob": -.2, "no_speech_prob": .01})()
        info = type("Info", (), {"language": kwargs["language"] or "vi", "language_probability": 1.0})()
        return iter([segment]), info


@pytest.mark.parametrize("field", ["language_preference", "response_language"])
@pytest.mark.parametrize("language", ["vi", "en"])
def test_session_language_is_passed_to_whisper_without_auto_retry(field, language):
    model = Model()
    service = FasterWhisperAsrService(replace(get_settings(), asr_model="local", asr_language=""))
    service._model = model
    profile = CustomerProfile(**{field: language})
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 16000, 16000), customer_profile=profile))
    assert [call["language"] for call in model.calls] == [language]
    assert result.text == "Không đúng"
    assert result.language_probability == 0.0  # A forced language is not detection confidence.
    assert result.candidates[0]["reported_language_probability"] == 1.0
    assert result.candidates[0]["language_mode"] == "forced"


def test_explicit_language_preference_overrides_previous_response_language():
    model = Model()
    service = FasterWhisperAsrService(replace(get_settings(), asr_model="local", asr_language=""))
    service._model = model
    profile = CustomerProfile(language_preference="en", response_language="vi")
    asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 16000, 16000), customer_profile=profile))
    assert model.calls[0]["language"] == "en"


@pytest.mark.parametrize("profile", [CustomerProfile(), CustomerProfile(response_language="vi"), None])
def test_empty_context_hotwords_do_not_reactivate_global_allergen_list(profile):
    model = Model()
    service = FasterWhisperAsrService(replace(get_settings(), asr_model="local", asr_hotwords="tôm, trứng, Nha Trang"))
    service._model = model
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 16000, 16000), customer_profile=profile))
    assert all(call["hotwords"] is None for call in model.calls)
    assert all(candidate["hotwords"] is None for candidate in result.candidates)


def test_allergy_confirmation_uses_only_confirmation_hotwords():
    model = Model()
    service = FasterWhisperAsrService(replace(get_settings(), asr_model="local"))
    service._model = model
    profile = CustomerProfile(response_language="vi", pending_allergy_status="declared", pending_allergens=["trứng"])
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 16000, 16000), customer_profile=profile))
    assert "không đúng" in model.calls[0]["hotwords"]
    assert "trứng" not in model.calls[0]["hotwords"]
    assert result.candidates[0]["hotwords"] == model.calls[0]["hotwords"]


def test_accents_and_language_probability_cannot_override_better_acoustic_evidence():
    better = _TranscriptCandidate("Good morning", "en", .1, -.2, .01)
    worse = _TranscriptCandidate("Xin chào", "vi", 1.0, -.4, .01, language_forced=True)
    assert better.score > worse.score
    assert replace(better, language_probability=1.0).score == better.score
