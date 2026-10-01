from dataclasses import replace
from datetime import date, timedelta
import json

import pytest

from app.allergy import CustomerProfile
from app.config import get_settings
from app.menu import VerifiedMenuService


def _item(**updates):
    item = {
        "id": "test-only", "name": "Món kiểm thử, không phải thực đơn thật",
        "served_in": "Cần Thơ", "ingredients": ["gạo", "thịt heo", "rau"],
        "allergens": [], "may_contain": [], "ingredients_verified": True,
        "allergens_verified": True, "cross_contact_assessed": True,
        "verified_by": "Synthetic test fixture", "verified_on": date.today().isoformat(),
        "availability": "available", "price_vnd": 40000,
        "meal_types": ["breakfast"], "dietary_tags": [], "spicy": False,
    }
    return item | updates


def _service(tmp_path, items):
    path = tmp_path / "menu.json"
    path.write_text(json.dumps({"schema_version": 1, "items": items}, ensure_ascii=False), encoding="utf-8")
    return VerifiedMenuService(replace(get_settings(), verified_menu_path=str(path)))


def test_verified_menu_excludes_ingredients_and_cross_contact(tmp_path):
    profile = CustomerProfile(allergy_status="declared", allergens=["đậu phộng"], region="Cần Thơ")
    service = _service(tmp_path, [
        _item(id="ingredient", ingredients=["gạo", "lạc rang"]),
        _item(id="cross-contact", may_contain=["đậu phộng"]),
        _item(id="filtered"),
    ])
    matches = service.candidates(profile, "Gợi ý món ăn sáng ở Cần Thơ không cay")
    assert [item["id"] for item in matches] == ["filtered"]
    answer = service.spoken_answer(matches, profile, "vi")
    assert "40.000 VNĐ" in answer
    assert "an toàn tuyệt đối" in answer and "xác nhận lại" in answer


@pytest.mark.parametrize("updates", [
    {"ingredients_verified": False}, {"ingredients_verified": "true"},
    {"allergens_verified": False}, {"cross_contact_assessed": False},
    {"verified_on": (date.today() - timedelta(days=31)).isoformat()},
    {"verified_on": (date.today() + timedelta(days=1)).isoformat()},
    {"verified_on": "not-a-date"}, {"availability": "unknown"},
    {"availability": "unavailable"}, {"ingredients": []},
    {"allergens": None}, {"may_contain": ["unknown"]},
    {"served_in": "Hà Nội"}, {"verified_by": ""}, {"price_vnd": True},
])
def test_unverified_incomplete_stale_or_wrong_location_menu_is_not_recommended(tmp_path, updates):
    profile = CustomerProfile(allergy_status="declared", allergens=["sữa"], region="Cần Thơ")
    assert _service(tmp_path, [_item(**updates)]).candidates(profile, "Gợi ý món ăn sáng ở Cần Thơ") == []


def test_verified_menu_respects_diet_and_meal_without_guessing(tmp_path):
    profile = CustomerProfile(allergy_status="declared", allergens=["sữa"], region="Cần Thơ")
    service = _service(tmp_path, [_item()])
    assert service.candidates(profile, "Gợi ý món ăn sáng ăn chay ở Cần Thơ") == []
    assert service.candidates(profile, "Gợi ý món cho 4 người ngân sách dưới 500000") == []


def test_malformed_menu_fails_closed(tmp_path):
    profile = CustomerProfile(allergy_status="declared", allergens=["sữa"])
    service = _service(tmp_path, [])
    service.path.write_text("not json", encoding="utf-8")
    assert service.candidates(profile, "Gợi ý món") == []
