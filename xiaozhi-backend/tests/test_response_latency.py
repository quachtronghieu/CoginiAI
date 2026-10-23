import asyncio
import json
from dataclasses import replace
from time import monotonic

import pytest
import httpx

from app.ai import AiPipelineError, AiResponse, AsrResult, LocalAiPipeline, PcmAudio
from app.allergy import CustomerProfile
from app.config import get_settings
from app.diagnostics import VoiceDiagnostics
from app.dialogue import prepare_turn
from app.speech import speech_chunks
from app.tts import PiperTtsService
from app.llm import OllamaLlmService
from app.rag import RagMatch
from scripts.audit_saved_vad import endpoint
import app.websocket as ws


@pytest.mark.parametrize("language", ["vi", "en"])
@pytest.mark.parametrize("prior_status", ["unknown", "declared"])
def test_repeated_unclear_allergies_never_commit_partial_list(language, prior_status):
    profile = CustomerProfile(allergy_status=prior_status,
        allergens=["sữa"] if prior_status == "declared" else [], allergy_question_asked=True)
    original = list(profile.allergens)
    for _ in range(3):
        reply, _ = prepare_turn(profile, "Tôi bị dị ứng với hải sản và abcxyz", language)
    assert profile.allergy_status == prior_status
    assert profile.allergens == original
    assert profile.allergy_review_deferred and profile.pending_allergens == ["hải sản"]
    assert profile.pending_allergy_note and not profile.allergy_updated_this_turn
    assert profile.dialogue_action == "allergy_unconfirmed"
    reply, _ = prepare_turn(profile, "Gợi ý món ăn ở Đà Lạt", language)
    assert reply is None  # Can describe dishes without repeating onboarding.
    assert profile.allergy_review_deferred
    prepare_turn(profile, "Tôi bị dị ứng với hải sản và trứng", language)
    assert profile.pending_allergy_status == "declared"
    assert profile.allergens == original
    prepare_turn(profile, "chính xác", language)
    assert not profile.allergy_review_deferred
    assert {"hải sản", "trứng"} <= set(profile.allergens)


def test_complete_speech_chunks_preserve_prices_place_names_and_negation():
    text = "Món ở TP. Hồ Chí Minh giá 50.000 đồng. Chưa xác nhận món không chứa trứng. Hãy hỏi nhân viên!"
    chunks = speech_chunks(text)
    assert chunks[0] == "Món ở TP. Hồ Chí Minh giá 50.000 đồng."
    assert " ".join(chunks) == text


def test_shorter_endpoint_can_cut_speech_after_a_pause():
    probabilities = [.9] * 30 + [.1] * 28 + [.9] * 30 + [.1] * 40
    original = endpoint(probabilities, 1000)
    shorter = endpoint(probabilities, 700)
    assert original["later_speech_frames"] == 0
    assert shorter["later_speech_frames"] == 30
    assert shorter["decision_audio_ms"] < original["decision_audio_ms"]


def test_tts_cache_is_exact_bounded_and_warmed_in_both_languages(monkeypatch):
    service = PiperTtsService(replace(get_settings(), tts_model_vi="vi", tts_model_en="en"))
    monkeypatch.setattr(service, "available", lambda: True)
    calls = []
    def render(text, language):
        calls.append((text, language))
        return PcmAudio(b"\0\0" * 160, 16000)
    monkeypatch.setattr(service, "_synthesize_sync", render)

    async def exercise():
        await service.warmup()
        await service.synthesize("Xin chào!", "vi", 1)
        await service.synthesize("Không dị ứng với trứng.", "vi", 1)
        await service.synthesize("Dị ứng với trứng.", "vi", 1)
        for index in range(40):
            await service.synthesize(f"Món số {index}.", "vi", 1)
    asyncio.run(exercise())
    assert calls[:2] == [("Xin chào!", "vi"), ("Hello!", "en")]
    assert calls.count(("Xin chào!", "vi")) == 1
    assert ("Không dị ứng với trứng.", "vi") in calls and ("Dị ứng với trứng.", "vi") in calls
    assert len(service._audio_cache) == 32


def test_streamed_sentence_uses_voice_selected_for_complete_answer(monkeypatch):
    service = PiperTtsService(get_settings())
    monkeypatch.setattr(service, "available", lambda: True)
    languages = []
    def synthesize(text, language):
        languages.append(language)
        return PcmAudio(b"audio", 16000)
    monkeypatch.setattr(service, "_synthesize_sync", synthesize)
    asyncio.run(service.synthesize("Nem.", "vi", 0, voice_language="vi"))
    assert languages == ["vi"]


def test_unconfirmed_allergies_cannot_reach_verified_menu_or_unqualified_answer(monkeypatch):
    settings = replace(get_settings(), rag_enabled=True)
    def handler(request):
        payload = json.loads(request.content)
        assert payload["options"]["num_predict"] <= 128
        return httpx.Response(200, json={"message": {"content": '{"evidence_ids":[1]}'}})
    service = OllamaLlmService(settings, httpx.MockTransport(handler))
    monkeypatch.setattr(service.rag, "available", lambda: True)
    monkeypatch.setattr(service.rag, "retrieve", lambda *args, **kwargs: [RagMatch(
        "Món ăn Đà Lạt", "source", "Nem nướng Đà Lạt là món ăn được làm từ thịt heo và gia vị.", 1)])
    monkeypatch.setattr(service.menu, "candidates", lambda *args: pytest.fail("Unconfirmed list used to select menu"))
    profile = CustomerProfile(allergy_status="declared", allergens=["sữa"],
        allergy_review_deferred=True, pending_allergy_note="abcxyz", pending_allergens=["hải sản"])
    answer = asyncio.run(service.generate("Gợi ý món ăn ở Đà Lạt", "vi", .9, customer_profile=profile))
    assert "chưa được xác nhận đầy đủ" in answer
    assert "nhiễm chéo" in answer
    assert profile.allergens == ["sữa"] and profile.allergy_review_deferred


class Codec:
    async def decode_opus(self, frames):
        return PcmAudio(b"\0\0" * 160, 16000)
    async def encode_opus(self, audio):
        return [audio.data]


class Asr:
    async def transcribe(self, audio):
        return AsrResult("xin chào", "vi", .9)


class Llm:
    async def generate(self, *args, **kwargs):
        return "Đây là câu đầu. Đây là câu thứ hai. Chưa xác minh thành phần dị ứng."


def test_stream_prefetch_overlaps_playback_and_retains_full_answer(tmp_path):
    async def exercise():
        next_started = asyncio.Event()
        class Tts:
            async def synthesize(self, text, *args):
                if text.startswith("Đây là câu thứ"):
                    next_started.set()
                return PcmAudio(text.encode(), 16000)
        pipeline = LocalAiPipeline(Codec(), Asr(), Llm(), Tts(), diagnostics=VoiceDiagnostics(tmp_path))
        chunks = []
        async def send(chunk):
            chunks.append(chunk)
            if len(chunks) == 1:
                await asyncio.wait_for(next_started.wait(), .5)
        result = await pipeline.run([b"input"], on_audio_chunk=send)
        assert " ".join(chunk.answer for chunk in chunks) == result.answer
        assert len(chunks) == 2 and not result.audio_frames
        assert chunks[0].diagnostic_key == result.diagnostic_key
    asyncio.run(exercise())
    meta = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert meta["sentence_streaming"]
    assert meta["timings_ms"]["first_audio_ready"] <= meta["timings_ms"]["total"]


def test_cancelling_playback_cancels_prefetched_sentence():
    async def exercise():
        started, cancelled = asyncio.Event(), asyncio.Event()
        class Tts:
            async def synthesize(self, text, *args):
                if text.startswith("Đây là câu thứ"):
                    started.set()
                    try:
                        await asyncio.Future()
                    finally:
                        cancelled.set()
                return PcmAudio(b"audio", 16000)
        async def send(chunk):
            await started.wait()
            await asyncio.Future()
        pipeline = LocalAiPipeline(Codec(), Asr(), Llm(), Tts())
        task = asyncio.create_task(pipeline.run([b"input"], on_audio_chunk=send))
        await asyncio.wait_for(started.wait(), .5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
    asyncio.run(exercise())


class Socket:
    def __init__(self):
        self.events = []
        self.close_code = None
    async def send_json(self, value):
        self.events.append(value)
    async def send_bytes(self, value):
        self.events.append(value)
    async def close(self, code, **kwargs):
        self.close_code = code


@pytest.mark.parametrize("fail_after_first", [False, True])
def test_stream_protocol_has_single_start_and_stop_or_normal_close(monkeypatch, fail_after_first, tmp_path):
    class Tts:
        async def synthesize(self, text, *args):
            if fail_after_first and text.startswith("Đây là câu thứ"):
                raise AiPipelineError("TTS failed")
            return PcmAudio(b"audio", 16000)
    pipe = LocalAiPipeline(Codec(), Asr(), Llm(), Tts(), diagnostics=VoiceDiagnostics(tmp_path))
    monkeypatch.setattr(ws, "ai_pipeline", pipe)
    monkeypatch.setattr(ws, "get_settings", lambda: replace(get_settings(), frame_duration_ms=1))
    socket = Socket()
    now = monotonic()
    asyncio.run(ws._process_utterance(socket, "test-session", [b"input"], ws.FinalizeReason.MANUAL_STOP,
        vad_evidence={"finalized_monotonic": now, "last_speech_received_monotonic": now - 1}))
    assert sum(isinstance(event, dict) and event.get("state") == "start" for event in socket.events) == 1
    assert sum(isinstance(event, dict) and event.get("type") == "stt" for event in socket.events) == 1
    if fail_after_first:
        assert socket.close_code == 1000
    else:
        assert socket.events[-1]["state"] == "stop"
        assert sum(isinstance(event, dict) and event.get("state") == "stop" for event in socket.events) == 1
    meta = json.loads((tmp_path / "sample-00.json").read_text(encoding="utf-8"))
    assert meta["session_id"] == "test-session"
    timings = meta["network_timings_ms"]
    assert timings["estimated_speech_end_to_first_audio_sent_ms"] >= timings["finalize_to_first_audio_sent_ms"] + 999
