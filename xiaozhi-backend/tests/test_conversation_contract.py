import asyncio
from dataclasses import replace

import httpx
import pytest

from app.allergy import CustomerProfile, allergen_keywords, ingredient_list
from app.asr import FasterWhisperAsrService, _TranscriptCandidate, repetitive_transcript
from app.config import get_settings
from app.dialogue import prepare_turn
from app.llm import OllamaLlmService


@pytest.mark.parametrize("hello", ["Xin chào", "Chào bạn", "Xin giao!"])
def test_greeting_asks_for_food_not_allergies(hello):
    profile = CustomerProfile()
    reply, _ = prepare_turn(profile, hello, "vi")
    assert "Xin chào" in reply and "tư vấn món" in reply
    assert "dị ứng" not in reply and not profile.allergy_question_asked
    assert profile.food_request == "" and profile.dialogue_action == "greeting"


def test_full_requested_order():
    profile = CustomerProfile()
    assert "tư vấn món" in prepare_turn(profile, "Xin chào", "vi")[0]
    reply, _ = prepare_turn(profile, "Gợi ý món ăn sáng ở Nha Trang", "vi")
    assert "bạn có dị ứng" in reply
    reply, _ = prepare_turn(profile, "Tôi bị dứng ứng với hải sản và trứng", "vi")
    assert "hải sản" in reply and "trứng" in reply and "đúng không" in reply
    assert profile.pending_allergy_note == "" and profile.allergy_status == "unknown"
    reply, request = prepare_turn(profile, "Chính xác", "vi")
    assert reply is None and "ăn sáng" in request and "Nha Trang" in request
    assert set(profile.allergens) == {"hải sản", "trứng"}
    reply, request = prepare_turn(profile, "Hải Phòng", "vi")
    assert reply is None and "ăn sáng" in request and "Hải Phòng" in request
    assert "Nha Trang" not in request and profile.allergy_status == "declared"


def test_short_dish_response_after_greeting_starts_allergy_question():
    profile = CustomerProfile()
    prepare_turn(profile, "Xin chào", "vi")
    reply, _ = prepare_turn(profile, "Phở", "vi")
    assert "bạn có dị ứng" in reply and profile.food_request == "Tư vấn món Phở"


def test_location_without_food_request_does_not_claim_allergies_confirmed():
    profile = CustomerProfile()
    prepare_turn(profile, "Xin chào", "vi")
    reply, _ = prepare_turn(profile, "Nha Trang", "vi")
    assert "tư vấn món" in reply and "xác nhận" not in reply
    assert profile.allergy_status == "unknown"


@pytest.mark.parametrize("text,expected", [
    ("Tôi bị dị ứng với hải sản và trứng", {"hải sản", "trứng"}),
    ("Tôi bị dứng ứng với hải sản và trứng", {"hải sản", "trứng"}),
    ("Tôi bị giữ với hải sản và trứng", {"hải sản", "trứng"}),
    ("Tôi bị chỉ ứng với sữa và trứng", {"sữa", "trứng"}),
    ("Mình bị dưỡng với hải sản và trứng", {"hải sản", "trứng"}),
    ("Dị ứng với hải sản và trứng", {"hải sản", "trứng"}),
    ("Đậu phộng, sữa", {"đậu phộng", "sữa"}),
    ("Sữa", {"sữa"}),
    ("Tôm cua trứng", {"tôm", "cua", "trứng"}),
    ("Tôi bị dị ứng với trứng nhé", {"trứng"}),
])
def test_keyword_payloads_stage_complete_readback(text, expected):
    profile = CustomerProfile(allergy_question_asked=True)
    reply, _ = prepare_turn(profile, text, "vi")
    assert "đúng không" in reply
    assert set(profile.pending_allergens) == expected and not profile.pending_allergy_note
    assert profile.allergens == [] and profile.allergy_status == "unknown"
    prepare_turn(profile, "Chính xác", "vi")
    assert set(profile.allergens) == expected and profile.allergy_status == "declared"


@pytest.mark.parametrize("text", ["Sở và trứng", "Tửa và trứng", "Dị ứng hải sản và chứng", "Tôi bị dị ứng hoa đào và trứng", "Tôi bị giữ với hoa đào và trứng"])
def test_unknown_ingredient_is_named_not_silently_corrected(text):
    profile = CustomerProfile(allergy_question_asked=True)
    reply, _ = prepare_turn(profile, text, "vi")
    assert "chưa xác định" in reply and profile.pending_allergy_note
    assert profile.pending_allergy_status == ""
    prepare_turn(profile, "Chính xác", "vi")
    assert profile.allergy_status == "unknown" and profile.allergens == []


def test_allergy_intent_without_food_name_asks_for_name_without_storing_asr_residue():
    profile = CustomerProfile(allergy_question_asked=True)

    reply, _ = prepare_turn(profile, "Tôi bị dị ứng với phạm", "vi")

    assert reply == "Bạn dị ứng với thực phẩm nào? Bạn nói lại tên giúp mình nhé."
    assert profile.dialogue_action == "allergy_name_missing"
    assert profile.allergens == []
    assert profile.pending_allergens == []
    assert profile.pending_allergy_note == ""
    assert profile.allergy_status == "unknown"

    reply, _ = prepare_turn(profile, "Hải sản", "vi")
    assert "hải sản" in reply.lower() and "đúng không" in reply.lower()
    assert profile.allergy_status == "unknown"


@pytest.mark.parametrize("text", ["Tôi không dị ứng với sữa", "Nếu tôi dị ứng hải sản thì sao", "Ví dụ đậu phộng và sữa"])
def test_negations_and_examples_do_not_become_positive_disclosures(text):
    profile = CustomerProfile(allergy_question_asked=True)
    prepare_turn(profile, text, "vi")
    assert profile.pending_allergy_status != "declared" and profile.allergens == []


def test_intro_never_discards_known_allergen_word():
    known, unknown = ingredient_list("Tôi bị tôm với sữa")
    assert set(known) == {"tôm", "sữa"} and unknown


def test_vietnamese_tones_are_not_allergen_synonyms():
    assert allergen_keywords("thành phần của tôi là sữa") == ["sữa"]
    assert allergen_keywords("cả trùng lắc") == []
    assert set(allergen_keywords("cá cua trứng lạc")) == {"cá", "cua", "trứng", "đậu phộng"}
    known, unknown = ingredient_list("hải sản và cà")
    assert known == ["hải sản"] and unknown == ["cà"]


@pytest.mark.parametrize("text", ["Hôm nay thời tiết thế nào?", "Giá vàng hôm nay bao nhiêu?", "Giải phương trình bậc hai", "Viết code Python cho thực đơn", "Ai là cầu thủ bóng đá giỏi nhất?", "What is the capital of France?", "Nguồn gốc vũ trụ là gì?", "Máy tính của tôi bị hỏng", "What is the origin of the universe?"])
def test_off_topic_is_refused_before_allergy_mutation(text):
    profile = CustomerProfile(allergy_question_asked=True, pending_allergy_status="declared", pending_allergens=["sữa"])
    reply, _ = prepare_turn(profile, text, "vi")
    assert "chỉ trả lời" in reply or "only answer" in reply
    assert profile.pending_allergens == ["sữa"] and profile.allergy_status == "unknown"
    assert profile.dialogue_action == "off_topic"


@pytest.mark.parametrize("text", ["Món này có sữa không?", "Sữa có tốt không?", "Thành phần món này là gì?"])
def test_ingredient_questions_are_not_disclosures(text):
    profile = CustomerProfile(allergy_question_asked=True)
    prepare_turn(profile, text, "vi")
    assert profile.pending_allergy_status == "" and profile.pending_allergy_note == ""


def test_greeting_does_not_reset_confirmed_allergies():
    profile = CustomerProfile(allergy_status="declared", allergens=["trứng"])
    reply, _ = prepare_turn(profile, "Xin chào", "vi")
    assert "tư vấn món" in reply and profile.allergens == ["trứng"]


def test_domain_refusal_does_not_call_ollama():
    service = OllamaLlmService(replace(get_settings(), rag_enabled=False), httpx.MockTransport(lambda request: pytest.fail("Out-of-domain question reached Ollama")))
    assert "chỉ trả lời" in asyncio.run(service.generate("Thời tiết hôm nay thế nào?", "vi", .99))


def test_culinary_question_with_weather_context_is_still_in_domain():
    profile = CustomerProfile()
    reply, _ = prepare_turn(profile, "Thời tiết nóng nên ăn món gì?", "vi")
    assert "chỉ trả lời" not in reply and "bạn có dị ứng" in reply


def test_english_greeting_sets_matching_tts_language():
    profile = CustomerProfile()
    reply, _ = prepare_turn(profile, "Hello", "vi")
    assert reply.startswith("Hello") and profile.response_language == "en"


def _candidate(text, source="processed"):
    return _TranscriptCandidate(text, "vi", .99, -.3, .01, source)


def test_raw_complete_allergen_list_can_rescue_denoised_unknown_word():
    profile = CustomerProfile(allergy_question_asked=True)
    chosen, unclear, reason = FasterWhisperAsrService._select_safety_candidate(_candidate("tửa và trứng"), _candidate("sữa và trứng", "raw"), profile)
    assert chosen.text == "sữa và trứng" and not unclear
    profile.update_from_text(chosen.text)
    assert profile.allergy_status == "unknown" and set(profile.pending_allergens) == {"sữa", "trứng"}


def test_clear_raw_greeting_can_rescue_short_unclear_filtered_phrase():
    selected = FasterWhisperAsrService._select_short_candidate(_candidate("Có chào!"), _candidate("Xin chào!", "raw"))
    assert selected.text == "Xin chào!"


def test_short_candidate_never_changes_porridge_to_greeting_by_semantic_bias():
    processed = _candidate("Có cháo!")
    raw = _TranscriptCandidate("Xin chào!", "vi", .99, -.9, .01, "raw")
    assert FasterWhisperAsrService._select_short_candidate(processed, raw).text == "Có cháo!"


@pytest.mark.parametrize("raw", ["Không đúng", "Ông đúng", "Tổng đu"])
def test_positive_confirmation_is_not_accepted_when_audio_variants_disagree(raw):
    profile = CustomerProfile(pending_allergy_status="declared", pending_allergens=["sữa"])
    chosen, unclear, reason = FasterWhisperAsrService._select_safety_candidate(_candidate("Đúng rồi"), _candidate(raw, "raw"), profile)
    assert unclear and reason == "confirmation_disagreement"
    assert profile.allergy_status == "unknown"


def test_two_clear_but_different_allergen_lists_require_clarification():
    profile = CustomerProfile(allergy_question_asked=True)
    _, unclear, reason = FasterWhisperAsrService._select_safety_candidate(_candidate("Sữa và trứng"), _candidate("Hải sản và trứng", "raw"), profile)
    assert unclear and reason == "ingredient_disagreement"


def test_extra_unknown_word_is_not_dropped_when_other_recording_only_has_same_known_keyword():
    profile = CustomerProfile(allergy_question_asked=True)
    _, unclear, reason = FasterWhisperAsrService._select_safety_candidate(_candidate("Đậu phộng"), _candidate("Đậu phộng thứ", "raw"), profile)
    assert unclear and reason == "ingredient_disagreement"


def test_clear_negative_can_reject_bad_readback_without_forgetting_confirmed_allergies():
    profile = CustomerProfile(allergy_status="declared", allergens=["đậu phộng"], pending_allergy_status="declared", pending_allergens=["sữa"])
    chosen, unclear, _ = FasterWhisperAsrService._select_safety_candidate(_candidate("Ông đúng"), _candidate("Không đúng", "raw"), profile)
    assert not unclear and chosen.text == "Không đúng"
    profile.update_from_text(chosen.text)
    assert profile.allergens == ["đậu phộng"] and not profile.pending_allergy_status


def test_hotword_sequence_with_nonrepetitive_prefix_is_rejected():
    text = "Cà Mau Cà Mau " + "Hà Nội Huế Đà Nẵng Khánh Hòa " * 8
    assert repetitive_transcript(text)
    assert not _candidate(text).valid
