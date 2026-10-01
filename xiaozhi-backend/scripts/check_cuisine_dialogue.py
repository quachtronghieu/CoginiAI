"""Assert regional dialogue against local Ollama and real RAG, not physical-mic accuracy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.allergy import CustomerProfile
from app.config import get_settings
from app.llm import OllamaLlmService


async def main():
    service = OllamaLlmService(get_settings())
    passed = 0

    async def turn(profile, text, *, city="", dish="", route="", contains="", language="vi"):
        nonlocal passed
        started = monotonic()
        answer = await service.generate(text, language, .99, customer_profile=profile)
        assert answer, text
        if city:
            assert profile.region == city, (text, profile.region)
        if dish:
            assert dish.casefold() in answer.casefold(), (text, answer)
            assert profile.evidence_sources, (text, "No selected sources")
            assert "chưa có đủ thông tin" not in answer and "chưa có đủ bằng chứng" not in answer, answer
        if route:
            assert profile.last_route == route, (text, profile.last_route)
        if contains:
            assert contains in answer, (text, answer)
        passed += 1
        print(json.dumps({"test": passed, "input": text, "answer": answer, "seconds": round(monotonic()-started, 2), "status": "PASS", "allergy_status": profile.allergy_status, "allergens": profile.allergens, "region": profile.region, "resolved": profile.resolved_request, "route": profile.last_route, "response_language": profile.response_language, "sources": profile.evidence_sources}, ensure_ascii=False), flush=True)
        return answer

    facts = CustomerProfile()
    await turn(facts, "Phở chua Lạng Sơn có đặc điểm gì?", dish="nước sốt", route="rag_information")
    assert facts.allergy_status == "unknown"
    no_allergy = CustomerProfile()
    await turn(no_allergy, "Xin chào", route="dialogue", contains="tư vấn món")
    assert not no_allergy.allergy_question_asked
    await turn(no_allergy, "Thời tiết hôm nay thế nào?", route="dialogue", contains="chỉ trả lời")
    await turn(no_allergy, "Gợi ý món ăn sáng ở Nha Trang", city="Nha Trang", route="dialogue", contains="bạn có dị ứng")
    await turn(no_allergy, "Không", route="dialogue", contains="đúng không")
    await turn(no_allergy, "Đúng rồi", city="Nha Trang", dish="bún lá cá dầm", route="rag_information")
    assert no_allergy.allergy_status == "none"
    await turn(no_allergy, "Hải Phòng", city="Hải Phòng", dish="bánh đa cua", route="rag_information")
    await turn(no_allergy, "Nam Định", city="Nam Định", dish="xíu", route="rag_information")
    assert "ăn sáng" in no_allergy.resolved_request
    allergic = CustomerProfile()
    await turn(allergic, "Gợi ý món ăn sáng ở Hải Phòng", city="Hải Phòng", contains="bạn có dị ứng")
    await turn(allergic, "Tôi bị dứng ứng với hải sản và trứng", contains="đúng không")
    await turn(allergic, "Đúng rồi", city="Hải Phòng", dish="bánh đa cua", route="rag_information", contains="không phải gợi ý món phù hợp")
    assert set(allergic.allergens) == {"trứng", "hải sản"}
    await turn(allergic, "Nha Trang", city="Nha Trang", dish="bún lá cá dầm", route="rag_information", contains="nhiễm chéo")
    await turn(allergic, "Gợi ý món ăn sáng ở nhà chiền", contains="Nha Trang", route="dialogue")
    await turn(allergic, "Xin chào tạm biệt, tạm biệt.", route="dialogue")
    await turn(allergic, "Đúng rồi", city="Nha Trang", dish="bún lá cá dầm", route="rag_information")
    await turn(allergic, "Hải Phòng", city="Hải Phòng", dish="bánh đa cua", route="rag_information", language="en")
    assert allergic.response_language == "vi" and "ăn sáng" in allergic.resolved_request
    await turn(allergic, "Gợi ý món an toàn phù hợp dị ứng ở Hải Phòng", route="verified_menu_missing", contains="Chưa có bảng thành phần")
    await turn(allergic, "Phở chua Lạng Sơn có đặc điểm gì?", dish="nước sốt", route="rag_information", contains="nhiễm chéo")
    assert set(allergic.allergens) == {"trứng", "hải sản"}
    bare = CustomerProfile()
    await turn(bare, "Xin chào", contains="tư vấn món")
    await turn(bare, "Gợi ý món ăn sáng ở Nha Trang", contains="bạn có dị ứng")
    await turn(bare, "Trứng và sở", contains="chưa xác định")
    assert bare.pending_allergy_note and bare.allergy_status == "unknown"
    await turn(bare, "Đậu phộng, sữa", contains="đúng không")
    await turn(bare, "Chính xác", dish="bún lá cá dầm", route="rag_information")
    assert set(bare.allergens) == {"đậu phộng", "sữa"}
    print(json.dumps({"result": "PASS", "asserted_turns": passed, "physical_microphone_tested": False, "verified_restaurant_menu_available": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
