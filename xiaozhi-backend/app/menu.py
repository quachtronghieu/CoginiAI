"""Allergy recommendations from explicit, recently staff-verified menu records only."""

from __future__ import annotations

from datetime import date
import json
import logging
from pathlib import Path
import re

from .allergy import ALLERGEN_ALIASES, CustomerProfile, _normalized
from .config import PROJECT_ROOT, Settings
from .dialogue import regions_in_text

logger = logging.getLogger(__name__)


class VerifiedMenuService:
    def __init__(self, settings: Settings):
        self.path = Path(settings.verified_menu_path) if settings.verified_menu_path else None
        if self.path is not None and not self.path.is_absolute():
            self.path = PROJECT_ROOT / self.path
        self.max_age_days = settings.verified_menu_max_age_days

    def candidates(self, profile: CustomerProfile, question: str) -> list[dict]:
        if self.path is None or profile.allergy_status != "declared" or profile.allergy_note:
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("items"), list):
                return []
        except (OSError, ValueError):
            logger.warning("Verified menu could not be read: %s", self.path)
            return []
        candidates = []
        for item in data["items"]:
            if not isinstance(item, dict):
                continue
            if not all(item.get(flag) is True for flag in ("ingredients_verified", "allergens_verified", "cross_contact_assessed")):
                continue
            if item.get("availability") != "available" or not all(isinstance(item.get(field), str) and item[field].strip() for field in ("id", "name", "served_in", "verified_by", "verified_on")):
                continue
            try:
                age = (date.today() - date.fromisoformat(item["verified_on"])).days
                if not 0 <= age <= self.max_age_days:
                    continue
            except ValueError:
                continue
            if not all(isinstance(item.get(field), list) and all(isinstance(value, str) and value.strip() for value in item[field]) for field in ("ingredients", "allergens", "may_contain", "meal_types", "dietary_tags")):
                continue
            if not item["ingredients"] or any(label not in ALLERGEN_ALIASES for label in [*item["allergens"], *item["may_contain"]]):
                continue
            if profile.region and profile.region not in regions_in_text(item["served_in"]):
                continue
            ingredients = " ".join([*item["ingredients"], *item["allergens"], *item["may_contain"]])
            if profile.allergens_in_text(ingredients):
                continue
            normalized = _normalized(question)
            if any(phrase in normalized for phrase in ("an sang", "bua sang", "buoi sang", "breakfast")) and "breakfast" not in item["meal_types"]:
                continue
            if "an chay" in normalized and "vegetarian" not in item["dietary_tags"]:
                continue
            if "vegan" in normalized and "vegan" not in item["dietary_tags"]:
                continue
            if "khong cay" in normalized and item.get("spicy") is not False:
                continue
            price = item.get("price_vnd")
            if price is not None and (type(price) is not int or price <= 0):
                continue
            # Group budgets require structured portion/pricing data, not naive
            # per-dish comparisons. Do not pretend to satisfy them here.
            if re.search(r"\b(?:ngan sach|budget|\d+ nguoi|duoi \d+)\b", normalized):
                continue
            candidates.append(item)
        return candidates[:3]

    @staticmethod
    def spoken_answer(items: list[dict], profile: CustomerProfile, language: str) -> str:
        descriptions = []
        for item in items:
            ingredients = ", ".join(item["ingredients"][:4]) if item.get("ingredients") else ""
            ing_text = f", thành phần gồm {ingredients}" if ingredients else ""
            price = f"{item['price_vnd']:,}".replace(",", ".") + " VNĐ" if item.get("price_vnd") else ("ask staff for price" if language == "en" else "cần hỏi giá nhân viên")
            if language == "en":
                descriptions.append(f"{item['name']}: ingredients {ingredients}; recorded price {price}.")
            else:
                descriptions.append(f"{item['name']}{ing_text}; giá ghi trong thực đơn {price}.")
        if language == "en":
            return f"Based on the staff-verified menu, these dishes do not list your confirmed allergens ({profile.spoken_allergens()}): " + " ".join(descriptions) + " This is not a guarantee of safety; please confirm ingredients and current availability with staff before ordering."
        return f"Theo thực đơn được nhân viên xác thực, các món sau không ghi nhận thành phần dị ứng đã xác nhận của bạn ({profile.spoken_allergens()}): " + " ".join(descriptions) + " Đây không phải cam kết an toàn tuyệt đối; trước khi gọi món, hãy xác nhận lại thành phần và tình trạng còn món với nhân viên."
