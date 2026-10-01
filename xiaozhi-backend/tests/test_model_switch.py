"""Changing the LLM selector must not alter deterministic conversation safety."""

import asyncio
from dataclasses import asdict, replace

import httpx
import pytest

from app.allergy import CustomerProfile
from app.config import get_settings
from app.llm import OllamaLlmService


def _snapshots(model, turns):
    async def run():
        def reject_network(request):
            pytest.fail(f"Deterministic dialogue must not depend on {model}: {request.url}")
        service = OllamaLlmService(
            replace(get_settings(), llm_model=model, llm_base_url="http://127.0.0.1:11434", rag_enabled=False),
            httpx.MockTransport(reject_network),
        )
        profile = CustomerProfile()
        snapshots = []
        for text in turns:
            answer = await service.generate(text, "vi", .99, customer_profile=profile)
            snapshots.append((answer, asdict(profile)))
        return snapshots
    return asyncio.run(run())


@pytest.mark.parametrize("ingredient_list", ["Tôi bị dị ứng với hải sản và trứng", "Đậu phộng, sữa"])
def test_model_switch_preserves_dialogue_answers_and_entire_profile(ingredient_list):
    turns = ["Xin chào", "Thời tiết hôm nay thế nào?", "Gợi ý món ăn sáng ở Nha Trang", ingredient_list, "Chính xác", "Hải Phòng", "Gợi ý món an toàn phù hợp dị ứng"]
    qwen = _snapshots("qwen2.5:3b", turns)
    gemma = _snapshots("gemma3:4b", turns)
    assert qwen == gemma
    assert gemma[0][1]["dialogue_action"] == "greeting"
    assert gemma[1][1]["dialogue_action"] == "off_topic"
    assert gemma[3][1]["allergy_status"] == "unknown"
    assert gemma[4][1]["allergy_status"] == "declared"
    assert gemma[5][1]["allergens"] == gemma[4][1]["allergens"]
    assert gemma[-1][1]["last_route"] == "verified_menu_missing"


def test_model_switch_keeps_unclear_ingredients_unconfirmed():
    turns = ["Xin chào", "Gợi ý món ăn sáng ở Nha Trang", "Sở và trứng", "Chính xác"]
    qwen = _snapshots("qwen2.5:3b", turns)
    gemma = _snapshots("gemma3:4b", turns)
    assert qwen == gemma
    assert all(profile["allergy_status"] == "unknown" for _, profile in gemma)
    assert gemma[-1][1]["pending_allergy_note"]
