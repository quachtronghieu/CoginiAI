import asyncio
import json
import sqlite3
from dataclasses import replace

import httpx
import pytest

from app.ai import AsrResult, LocalAiPipeline, PcmAudio
from app.allergy import CustomerProfile
from app.asr import FasterWhisperAsrService, _TranscriptCandidate
from app.config import get_settings
from app.dialogue import prepare_turn
from app.llm import OllamaLlmService
from app.rag import CuisineRagService


def _index(tmp_path, rows):
    path = tmp_path / "safety.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE VIRTUAL TABLE cuisine_fts USING fts5(title,source UNINDEXED,content,tokenize='unicode61 remove_diacritics 2')")
        db.executemany("INSERT INTO cuisine_fts VALUES(?,?,?)", rows)
    return replace(get_settings(), rag_enabled=True, rag_database=str(path), llm_base_url="http://localhost:11434")


def test_request_survives_allergy_onboarding_and_resume():
    profile = CustomerProfile()
    reply, _ = prepare_turn(profile, "Gợi ý món ăn sáng ở Cần Thơ", "vi")
    assert "bạn có dị ứng" in reply
    assert profile.region == "Cần Thơ"
    reply, _ = prepare_turn(profile, "Trứng và sữa", "vi")
    assert "trứng" in reply and "sữa" in reply and "đúng không" in reply
    assert profile.allergy_status == "unknown"
    reply, resolved = prepare_turn(profile, "Đúng rồi", "vi")
    assert reply is None
    assert "sáng" in resolved and "Cần Thơ" in resolved
    assert set(profile.allergens) == {"trứng", "sữa"}
    reply, _ = prepare_turn(profile, "Phở chua có đặc điểm gì?", "vi")
    assert reply is None
    assert set(profile.allergens) == {"trứng", "sữa"}


def test_short_no_rejects_confirmation_without_clearing_constraints():
    profile = CustomerProfile(allergy_status="declared", allergens=["đậu phộng"])
    prepare_turn(profile, "Tôi dị ứng sữa", "vi")
    reply, _ = prepare_turn(profile, "Không", "vi")
    assert "chưa xác định" in reply
    assert profile.allergens == ["đậu phộng"]
    assert profile.allergy_status == "declared"


@pytest.mark.parametrize("transcript", ["Đạo phỏng và chứng", "Tô bị dự ứng với chứng và sở", "Trứng và sở", "Trứng với sở"])
def test_bad_allergy_transcripts_never_silently_commit(transcript):
    profile = CustomerProfile(allergy_question_asked=True)
    reply, _ = prepare_turn(profile, transcript, "vi")
    assert "chưa xác định" in reply
    assert profile.allergens == []
    assert profile.pending_allergy_status == ""
    prepare_turn(profile, "Đúng", "vi")
    assert profile.allergy_status == "unknown"


def test_location_hypothesis_requires_separate_confirmation():
    profile = CustomerProfile()
    prepare_turn(profile, "Gợi ý món ăn sáng ở Cà Mèo", "vi")
    assert profile.region == ""
    prepare_turn(profile, "Không", "vi")
    reply, _ = prepare_turn(profile, "Đúng", "vi")
    assert "Cà Mau" in reply and "đúng không" in reply
    assert profile.region == ""
    reply, request = prepare_turn(profile, "Đúng", "vi")
    assert reply is None
    assert profile.region == "Cà Mau"
    assert "Cà Mèo" not in request and "sáng" in request and "Cà Mau" in request


def test_unrecognised_location_clarification_preserves_breakfast():
    profile = CustomerProfile(allergy_status="none")
    reply, _ = prepare_turn(profile, "Gợi ý món ăn sáng ở khu vực thành phố", "vi")
    assert "tỉnh hoặc thành phố" in reply
    reply, request = prepare_turn(profile, "Cần Thơ", "vi")
    assert reply is None
    assert "sáng" in request and "Cần Thơ" in request


def test_food_question_is_not_an_allergy_disclosure():
    profile = CustomerProfile(allergy_question_asked=True)
    prepare_turn(profile, "Món này có tôm không?", "vi")
    assert profile.pending_allergy_status == ""


@pytest.mark.parametrize("text", ["Đừng", "Dừng", "Đứng", "Chứng"])
def test_tonal_lookalikes_are_not_confirmation(text):
    profile = CustomerProfile(allergy_question_asked=True)
    prepare_turn(profile, "Trứng", "vi")
    prepare_turn(profile, text, "vi")
    assert profile.allergy_status == "unknown" and profile.allergens == []
    location_profile = CustomerProfile(allergy_status="none")
    prepare_turn(location_profile, "Gợi ý món ở Cà Mèo", "vi")
    prepare_turn(location_profile, text, "vi")
    assert location_profile.region == ""


def test_unclear_confirmation_keeps_pending_list_for_next_reply():
    profile = CustomerProfile(allergy_question_asked=True)
    prepare_turn(profile, "Đậu phộng và trứng", "vi")
    reply, _ = prepare_turn(profile, "Tùng", "vi")
    assert "phần xác nhận" in reply
    assert profile.pending_allergens == ["đậu phộng", "trứng"]
    assert profile.allergy_status == "unknown"
    prepare_turn(profile, "Đúng rồi", "vi")
    assert profile.allergens == ["đậu phộng", "trứng"]


def test_whole_source_excludes_allergen_in_an_unretrieved_chunk(tmp_path):
    settings = _index(tmp_path, [
        ("Phở chua Lạng Sơn", "unsafe", "Phở chua Lạng Sơn có sốt chua ngọt."),
        ("Phở chua Lạng Sơn", "unsafe", "Rắc thêm lạc rang trước khi ăn."),
        ("Phở chua Lạng Sơn", "other", "Phở chua Lạng Sơn có rau thơm."),
    ])
    rag = CuisineRagService(settings)
    profile = CustomerProfile(allergy_status="declared", allergens=["đậu phộng"])
    matches = rag.retrieve("Phở chua Lạng Sơn", customer_profile=profile)
    assert matches and {match.source for match in matches} == {"other"}


def test_region_recommendations_reject_incidental_other_city(tmp_path):
    settings = _index(tmp_path, [
        ("Ẩm thực Hà Nội", "hanoi", "Khách Cần Thơ cũng thích bún chả Hà Nội."),
        ("Ẩm thực Cần Thơ", "cantho", "Bánh cống Cần Thơ được nhắc tới trong bài."),
    ])
    matches = CuisineRagService(settings).retrieve("Gợi ý món ăn ở Cần Thơ", region="Cần Thơ", recommendation=True)
    assert {match.source for match in matches} == {"cantho"}


@pytest.mark.parametrize("model_content", [
    {"quotes": [{"source_id": 1, "text": "Cà Mèo là một quận của thành phố Hồ Chí Minh."}]},
    {"quotes": [{"source_id": True, "text": "Phở chua Lạng Sơn có sốt chua ngọt."}]},
    {"quotes": [{"source_id": 999, "text": "Phở chua Lạng Sơn có sốt chua ngọt."}]},
    {"quotes": [], "answer": "Món hoàn toàn an toàn."},
    {"evidence_ids": [True]},
    {"evidence_ids": [999]},
    {"evidence_ids": [1, 1]},
    {"evidence_ids": [1], "answer": "Cà Mèo ở TP.HCM"},
])
def test_ungrounded_model_output_is_never_spoken(tmp_path, model_content):
    settings = _index(tmp_path, [("Phở chua Lạng Sơn", "source", "Phở chua Lạng Sơn có sốt chua ngọt.")])
    def handler(request):
        payload = json.loads(request.content)
        assert len(payload["messages"]) == 2
        assert "assistant" not in [message["role"] for message in payload["messages"]]
        assert payload["format"]["type"] == "object"
        return httpx.Response(200, json={"message": {"content": json.dumps(model_content)}})
    service = OllamaLlmService(settings, httpx.MockTransport(handler))
    answer = asyncio.run(service.generate("Phở chua Lạng Sơn", "vi", .99, history=[{"role": "assistant", "content": "Cà Mèo ở TP.HCM"}]))
    assert "chưa có đủ thông tin" in answer
    assert "Cà Mèo" not in answer


def test_no_verified_ingredients_means_no_allergy_safe_recommendations(tmp_path):
    settings = _index(tmp_path, [("Món ăn Cần Thơ", "source", "Một món không nhắc đậu phộng trong đoạn này.")])
    def handler(request):
        payload = json.loads(request.content)
        assert "infer or add facts" in payload["messages"][0]["content"]
        return httpx.Response(200, json={"message": {"content": json.dumps({"evidence_ids": [1]})}})
    profile = CustomerProfile(allergy_status="declared", allergens=["đậu phộng"])
    service = OllamaLlmService(settings, httpx.MockTransport(handler))
    answer = asyncio.run(service.generate("Gợi ý món ăn ở Cần Thơ", "vi", .99, customer_profile=profile))
    assert "không phải gợi ý món phù hợp" in answer and "nhiễm chéo" in answer
    assert "không nhắc đậu phộng" in answer


def test_explicit_allergy_safety_request_still_requires_verified_menu(tmp_path):
    settings = _index(tmp_path, [("Món ăn Cần Thơ", "source", "Một món ngon không nhắc đậu phộng trong đoạn này.")])
    def handler(request):
        raise AssertionError("A source cannot prove allergy safety")
    profile = CustomerProfile(allergy_status="declared", allergens=["đậu phộng"])
    service = OllamaLlmService(settings, httpx.MockTransport(handler))
    answer = asyncio.run(service.generate("Gợi ý món an toàn phù hợp dị ứng ở Cần Thơ", "vi", .99, customer_profile=profile))
    assert "Chưa có bảng thành phần" in answer and "nhiễm chéo" in answer
    assert profile.last_route == "verified_menu_missing"


def test_silence_does_not_load_whisper_or_generate_fake_subtitle():
    service = FasterWhisperAsrService(replace(get_settings(), asr_model="small"))
    def forbidden():
        raise AssertionError("Silence should be rejected before loading model")
    service._get_model = forbidden
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x00" * 16000, 16000)))
    assert result.text == ""


def test_high_language_probability_does_not_rescue_invalid_transcription():
    service = FasterWhisperAsrService(replace(get_settings(), asr_model="small", asr_language="vi"))
    class Model:
        def transcribe(self, samples, **kwargs):
            segment = type("Segment", (), {"text": "Hãy subscribe cho kênh", "avg_logprob": -1.7, "no_speech_prob": .99})()
            info = type("Info", (), {"language": "vi", "language_probability": .999})()
            assert kwargs["hotwords"] is None or isinstance(kwargs["hotwords"], str)
            return iter([segment]), info
    service._model = Model()
    result = asyncio.run(service.transcribe(PcmAudio(b"\x00\x08" * 16000, 16000)))
    assert result.text == ""


@pytest.mark.parametrize("transcript", ["Lạc, " + "Ấy, " * 44, "Ăn ngon " * 12])
def test_repetitive_high_confidence_transcript_is_rejected(transcript):
    assert not _TranscriptCandidate(transcript, "vi", 1.0, -.07, .1).valid


def test_normal_long_food_request_is_not_repetition():
    assert _TranscriptCandidate("Gợi ý cho tôi một số món ăn sáng ở Hải Phòng cho bốn người không ăn cay và có một khách ăn chay", "vi", .99, -.2, .01).valid
