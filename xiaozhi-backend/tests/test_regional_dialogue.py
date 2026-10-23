import asyncio
import json
import sqlite3
from dataclasses import replace

import httpx
import pytest

from app.allergy import CustomerProfile
from app.config import get_settings
from app.dialogue import is_recommendation, prepare_turn, regions_in_text, response_language
from app.llm import OllamaLlmService
from app.grounding import evidence_passages
from app.rag import RagMatch


@pytest.mark.parametrize("status", ["unknown", "none"])
@pytest.mark.parametrize("text", [
    "Gể cho tôi những món ăn ở Đà Lạc",
    "Vậy ý cho tôi những món ăn ở Đà Lạc",
    "Những món ăn ở Xyzville",
    "Món phở ở Xyzville có thành phần gì?",
])
def test_damaged_location_blocks_unscoped_retrieval(text, status):
    profile = CustomerProfile(allergy_status=status, region="Hải Phòng")
    reply, _ = prepare_turn(profile, text, "vi")
    assert "chưa nghe rõ địa điểm" in reply
    assert profile.region_needs_clarification
    assert profile.region == "Hải Phòng"  # Do not silently map to another place.
    assert profile.dialogue_action == "location_clarification"
    reply, resolved = prepare_turn(profile, "Đà Lạt", "vi")
    if status == "none":
        assert reply is None
        assert regions_in_text(resolved) == ["Đà Lạt"]
        assert "Đà Lạc" not in resolved and "Xyzville" not in resolved
    assert profile.region == "Đà Lạt"


def test_unknown_place_never_calls_rag_or_ollama_without_profile(monkeypatch):
    service = OllamaLlmService(replace(get_settings(), rag_enabled=True))
    monkeypatch.setattr(service.rag, "available", lambda: True)
    monkeypatch.setattr(service.rag, "retrieve", lambda *args, **kwargs: pytest.fail("Unresolved location reached RAG"))
    answer = asyncio.run(service.generate("Vậy ý cho tôi những món ăn ở Đà Lạc", "vi", .9))
    assert "chưa nghe rõ địa điểm" in answer


def test_unknown_english_location_requires_repeat_before_retrieval():
    profile = CustomerProfile(allergy_status="none", response_language="en")
    reply, _ = prepare_turn(profile, "Suggest food in Xyzville", "en")
    assert "repeat the location" in reply
    reply, resolved = prepare_turn(profile, "Nha Trang", "en")
    assert reply is None and "Xyzville" not in resolved
    assert profile.region == "Nha Trang"


@pytest.mark.parametrize("text", [
    "Món này có nguồn gốc ở đâu?", "Gợi ý món ăn nấu tại nhà",
    "Món ăn có tinh bột không?", "Tại sao phở ngon?",
])
def test_non_place_phrases_do_not_trigger_location_clarification(text):
    profile = CustomerProfile(allergy_status="none")
    prepare_turn(profile, text, "vi")
    assert not profile.region_needs_clarification


@pytest.mark.parametrize("noise", ["Xin chào tạm biệt, tạm biệt.", "Thật đẹp", "Cây hít cho tòi mộng sâu máu"])
def test_noise_cannot_replace_pending_breakfast_request(noise):
    profile = CustomerProfile(allergy_status="none")
    prepare_turn(profile, "Gợi ý món ăn sáng ở", "vi")
    prepare_turn(profile, noise, "vi")
    assert "ăn sáng" in profile.food_request
    reply, resolved = prepare_turn(profile, "Hải Phòng", "vi")
    assert reply is None and "ăn sáng" in resolved and "Hải Phòng" in resolved
    assert "tạm biệt" not in resolved


def test_successive_locations_preserve_meal_budget_and_diet():
    profile = CustomerProfile(allergy_status="declared", allergens=["trứng"])
    prepare_turn(profile, "Gợi ý món ăn sáng ở Hải Phòng cho 4 người ngân sách dưới 500.000 đồng không cay có khách ăn chay", "vi")
    for city in ["Nha Trang", "Huế", "Hà Nội", "Nam Định", "Hải Phòng"]:
        reply, resolved = prepare_turn(profile, city, "vi")
        assert reply is None
        assert "ăn sáng" in resolved and "4 người" in resolved and "500.000" in resolved
        assert "không cay" in resolved and "ăn chay" in resolved
        assert regions_in_text(resolved) == [city]
        assert profile.allergens == ["trứng"]


def test_new_fact_question_exits_location_clarification():
    profile = CustomerProfile(allergy_status="none")
    prepare_turn(profile, "Gợi ý món ăn sáng ở nhà chiền", "vi")
    reply, resolved = prepare_turn(profile, "Phở chua Lạng Sơn có đặc điểm gì?", "vi")
    assert reply is None and "Phở chua" in resolved
    assert not profile.region_needs_clarification and not profile.pending_region


def test_facts_do_not_require_allergy_onboarding():
    profile = CustomerProfile()
    reply, resolved = prepare_turn(profile, "Phở chua Lạng Sơn có đặc điểm gì?", "vi")
    assert reply is None and "Phở chua" in resolved
    assert profile.allergy_status == "unknown"


@pytest.mark.parametrize("heard", ["nhà chiền", "nha chèm", "nhà chiêm"])
def test_nha_trang_hypothesis_requires_confirmation(heard):
    profile = CustomerProfile(allergy_status="none")
    reply, _ = prepare_turn(profile, "Gợi ý món ăn sáng ở " + heard, "vi")
    assert "Nha Trang" in reply and profile.region == ""
    reply, resolved = prepare_turn(profile, "Đúng rồi", "vi")
    assert reply is None and "ăn sáng" in resolved and "Nha Trang" in resolved
    assert heard not in resolved


def test_rejected_hypothesis_does_not_use_old_location():
    profile = CustomerProfile(allergy_status="none", region="Hải Phòng")
    prepare_turn(profile, "Gợi ý món ăn sáng ở nhà chiền", "vi")
    reply, _ = prepare_turn(profile, "Không đúng", "vi")
    assert "tỉnh hoặc thành phố" in reply
    assert profile.region == "Hải Phòng" and profile.region_needs_clarification
    reply, resolved = prepare_turn(profile, "Cần Thơ", "vi")
    assert reply is None and "Cần Thơ" in resolved and "Hải Phòng" not in resolved


def test_new_request_changes_meal_instead_of_reusing_breakfast():
    profile = CustomerProfile(allergy_status="none")
    prepare_turn(profile, "Gợi ý món ăn sáng ở Nha Trang", "vi")
    reply, resolved = prepare_turn(profile, "Gợi ý món ăn tối ở Hải Phòng", "vi")
    assert reply is None and "ăn tối" in resolved and "ăn sáng" not in resolved
    _, resolved = prepare_turn(profile, "Huế", "vi")
    assert "ăn tối" in resolved and "Huế" in resolved


def test_short_location_and_confirmation_keep_language():
    profile = CustomerProfile()
    assert response_language(profile, "Gợi ý món ăn sáng", "en") == "vi"
    assert response_language(profile, "Cà Mau", "en") == "vi"
    assert response_language(profile, "Yes", "en") == "vi"
    assert response_language(profile, "Please recommend breakfast in Nha Trang", "en") == "en"
    assert response_language(profile, "Hải Phòng", "vi") == "en"
    assert response_language(profile, "Trả lời bằng tiếng Việt", "en") == "vi"
    assert response_language(profile, "Nha Trang", "en") == "vi"


@pytest.mark.parametrize("word", ["Yeah", "Yes", "OK", "Until", "You", "Very same"])
def test_ambiguous_one_word_asr_does_not_start_or_switch_to_english(word):
    fresh = CustomerProfile()
    assert response_language(fresh, word, "en") == "vi"
    established = CustomerProfile(response_language="vi")
    assert response_language(established, word, "en") == "vi"


def test_selected_session_language_survives_short_english_phrase():
    profile = CustomerProfile(response_language="vi")
    assert response_language(profile, "Hello", "en") == "vi"
    assert response_language(CustomerProfile(), "Hello", "en") == "en"


@pytest.mark.parametrize("city", ["Nha Trang", "Hải Phòng", "Nam Định", "Ninh Bình", "Đắk Lắk", "Hà Giang", "Khánh Hòa"])
def test_more_corpus_search_labels(city):
    assert regions_in_text(city) == [city]


def test_informational_allergy_path_reaches_rag_and_model(tmp_path):
    path = tmp_path / "regional.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE VIRTUAL TABLE cuisine_fts USING fts5(title,source UNINDEXED,content,tokenize='unicode61 remove_diacritics 2')")
        db.executemany("INSERT INTO cuisine_fts VALUES(?,?,?)", [
            ("Bún cá Hải Phòng", "haiphong", "Bún cá Hải Phòng là món ăn sáng có cá và rau thơm."),
            ("Bún cá Nha Trang", "nhatrang", "Bún cá Nha Trang là món ăn sáng có cá và sợi bún."),
            ("Bánh Hà Nội", "hanoi", "Bánh Hà Nội là món ăn sáng."),
        ])
    settings = replace(get_settings(), rag_database=str(path), rag_enabled=True, verified_menu_path=str(tmp_path / "absent.json"))
    calls = []
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"message": {"content": json.dumps({"evidence_ids": [1]})}})
    profile = CustomerProfile(allergy_status="declared", allergens=["hải sản", "trứng"])
    service = OllamaLlmService(settings, httpx.MockTransport(handler))
    for city in ["Hải Phòng", "Nha Trang"]:
        text = "Gợi ý món ăn sáng ở " + city if not calls else city
        answer = asyncio.run(service.generate(text, "vi", .99, customer_profile=profile))
        assert f"Bún cá {city}" in answer
        assert "không phải gợi ý món phù hợp" in answer
        assert "nhiễm chéo" in answer and "hải sản" in answer
        assert profile.region == city and profile.last_route == "rag_information"
        assert "Bánh Hà Nội" not in answer
    assert len(calls) == 2


def test_descriptions_of_allergen_dishes_are_not_filtered_out_for_fact_questions(tmp_path):
    path = tmp_path / "facts.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE VIRTUAL TABLE cuisine_fts USING fts5(title,source UNINDEXED,content,tokenize='unicode61 remove_diacritics 2')")
        db.execute("INSERT INTO cuisine_fts VALUES(?,?,?)", ("Phở chua Lạng Sơn", "pho", "Phở chua Lạng Sơn có thành phần gồm bánh phở và lạc rang."))
    settings = replace(get_settings(), rag_database=str(path), rag_enabled=True)
    service = OllamaLlmService(settings, httpx.MockTransport(lambda request: httpx.Response(200, json={"message": {"content": '{"evidence_ids":[1]}'}})))
    profile = CustomerProfile(allergy_status="declared", allergens=["đậu phộng"])
    answer = asyncio.run(service.generate("Phở chua Lạng Sơn có thành phần gì?", "vi", .99, customer_profile=profile))
    assert "lạc rang" in answer and "nhiễm chéo" in answer


def test_informational_allergy_mode_rejects_source_safety_claims_and_personal_advice():
    match = RagMatch("Món Nha Trang", "source", "Món này là món ăn sáng an toàn không gây dị ứng.\n\nBạn nên ăn món này vào bữa sáng.\n\nMón này được phục vụ vào bữa sáng với sợi bún.", 1)
    evidence = evidence_passages([match], "Nha Trang", True, "Gợi ý món ăn sáng ở Nha Trang", information_only=True)
    assert [item.text for item in evidence] == ["Món này được phục vụ vào bữa sáng với sợi bún."]


def test_ingredient_question_drops_taste_only_evidence():
    match = RagMatch("Bún cá Hải Phòng", "source", "Món ăn có vị chua thanh và hương thơm đậm đà.", 1)
    assert evidence_passages([match], "Hải Phòng", False, "Món này có thành phần gì?") == []


def test_price_question_keeps_only_price_evidence():
    match = RagMatch(
        "Bánh cuốn Lạng Sơn",
        "source",
        "Bánh mềm, thơm và ăn nóng rất ngon. Giá khoảng 30.000 đồng một suất.",
        1,
    )
    evidence = evidence_passages([match], "Lạng Sơn", False, "Món này giá bao nhiêu?")
    assert len(evidence) == 1 and "30.000 đồng" in evidence[0].text


def test_breakfast_related_article_link_does_not_qualify_dish():
    match = RagMatch("Bún cá Hải Phòng", "source", "Bún cá có cá chiên và rau thơm.\n\n>>> Gợi ý: 11 món ăn sáng Hải Phòng.", 1)
    assert evidence_passages([match], "Hải Phòng", True, "Gợi ý món ăn sáng ở Hải Phòng") == []


def test_breakfast_list_does_not_qualify_unrelated_dessert():
    match = RagMatch("Tổng hợp đặc sản Hải Phòng", "source", "Bánh đa cua được ăn vào buổi sáng.\n\nChè dừa dầm ăn kèm đá lạnh.", 1)
    evidence = evidence_passages([match], "Hải Phòng", True, "Gợi ý món ăn sáng ở Hải Phòng")
    assert [item.text for item in evidence] == ["Bánh đa cua được ăn vào buổi sáng."]


@pytest.mark.parametrize("text", ["Nguồn gốc món ăn sáng này?", "What is the origin of this breakfast dish?", "Món ăn sáng này có thành phần gì?"])
def test_factual_meal_questions_are_not_recommendations(text):
    assert not is_recommendation(text)


def test_food_query_captions_and_guest_quotes_are_not_answers():
    match = RagMatch("Bánh Nam Định", "source", "Bánh Nam Định là món ăn sáng có thịt và rau.\n\nẢnh: món ăn sáng chụp bởi tôi.\n\nÔng cho biết món bánh này rất ngon.\n\nHiệu bánh đóng gói 10 chiếc bánh một túi.", 1)
    evidence = evidence_passages([match], "Nam Định", True, "Gợi ý món ăn sáng ở Nam Định")
    assert not any("Ông cho biết" in item.text or "Ảnh:" in item.text for item in evidence)
