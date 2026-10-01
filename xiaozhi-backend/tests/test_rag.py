import asyncio
import json
import sqlite3
from dataclasses import replace

import httpx
import pytest

from app.allergy import CustomerProfile
from app.config import get_settings
from app.llm import OllamaLlmService
from app.rag import CuisineRagService
from app.grounding import evidence_passages
from app.dialogue import regions_in_text
from app.geography import search_regions


def _make_index(tmp_path):
    database = tmp_path / "cuisine.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE VIRTUAL TABLE cuisine_fts USING fts5("
            "title, source UNINDEXED, content, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        db.executemany(
            "INSERT INTO cuisine_fts (title, source, content) VALUES (?, ?, ?)",
            [
                (
                    "Độc đáo hương vị phở chua Lạng Sơn",
                    "https://example.test/pho-chua",
                    "Phở chua Lạng Sơn có nước sốt chua ngọt nguội mát và được trộn đều khi ăn.",
                ),
                (
                    "Quán vịt quay Lạng Sơn",
                    "https://example.test/vit-quay",
                    "Quán phục vụ vịt quay và các đặc sản Lạng Sơn.",
                ),
            ],
        )
        db.commit()
    return database


def _rag_settings(database):
    return replace(
        get_settings(),
        llm_base_url="http://127.0.0.1:11434",
        llm_model="qwen2.5:3b",
        rag_enabled=True,
        rag_database=str(database),
        rag_top_k=3,
        rag_strict=True,
    )


def test_rag_retrieves_only_relevant_source(tmp_path) -> None:
    service = CuisineRagService(_rag_settings(_make_index(tmp_path)))

    matches = service.retrieve("Phở chua Lạng Sơn có đặc điểm gì?")

    assert service.available() is True
    assert matches
    assert {match.source for match in matches} == {"https://example.test/pho-chua"}


def test_current_province_retrieves_destination_article(tmp_path) -> None:
    database = tmp_path / "regional.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE VIRTUAL TABLE cuisine_fts USING fts5("
            "title, source UNINDEXED, content, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        db.execute(
            "INSERT INTO cuisine_fts VALUES (?, ?, ?)",
            (
                "Bún lá cá dầm Nha Trang",
                "https://example.test/bun-la",
                "Bún lá cá dầm là món ăn sáng có nước dùng thanh và chả cá thơm ngon.",
            ),
        )
    service = CuisineRagService(_rag_settings(database))

    matches = service.retrieve(
        "Gợi ý món ăn sáng ở Khánh Hòa",
        region="Khánh Hòa",
        recommendation=True,
    )
    evidence = evidence_passages(
        matches,
        "Khánh Hòa",
        True,
        "Gợi ý món ăn sáng ở Khánh Hòa",
    )

    assert [match.source for match in matches] == ["https://example.test/bun-la"]
    assert evidence


def test_exact_destination_title_is_prioritized_over_parent_province(tmp_path) -> None:
    database = tmp_path / "geo-priority.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE VIRTUAL TABLE cuisine_fts USING fts5(title, source UNINDEXED, content, tokenize='unicode61 remove_diacritics 2')"
        )
        db.executemany(
            "INSERT INTO cuisine_fts VALUES (?, ?, ?)",
            [
                ("Đặc sản Khánh Hòa", "province", "Món ăn địa phương có hương vị đậm đà."),
                ("Bún chả cá Nha Trang", "destination", "Món ăn sáng có chả cá và nước dùng thanh."),
            ],
        )
    service = CuisineRagService(_rag_settings(database))
    matches = service.retrieve(
        "Gợi ý món ăn sáng ở Nha Trang", region="Nha Trang", recommendation=True
    )
    assert matches[0].source == "destination"


def test_current_province_retrieves_former_province_article(tmp_path) -> None:
    database = tmp_path / "merged.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute(
            "CREATE VIRTUAL TABLE cuisine_fts USING fts5("
            "title, source UNINDEXED, content, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        db.execute(
            "INSERT INTO cuisine_fts VALUES (?, ?, ?)",
            (
                "Cháo ấu tẩu Hà Giang",
                "https://example.test/chao-au-tau",
                "Cháo ấu tẩu Hà Giang có vị đắng nhẹ và thường được ăn nóng.",
            ),
        )
    service = CuisineRagService(_rag_settings(database))

    matches = service.retrieve(
        "Gợi ý món ăn ở Tuyên Quang",
        region="Tuyên Quang",
        recommendation=True,
    )

    assert [match.source for match in matches] == ["https://example.test/chao-au-tau"]


@pytest.mark.parametrize(
    ("current_province", "destination", "parent_label"),
    [
        ("Quảng Ninh", "Hạ Long", "Quảng Ninh"),
        ("Hải Phòng", "Cát Bà", "Hải Phòng"),
        ("Sơn La", "Mộc Châu", "Sơn La"),
        ("Khánh Hòa", "Phan Rang", "Ninh Thuận"),
        ("Đắk Lắk", "Tuy Hòa", "Phú Yên"),
        ("An Giang", "Châu Đốc", "An Giang"),
        ("Đồng Nai", "Biên Hòa", "Đồng Nai"),
    ],
)
def test_tourist_destination_scopes(
    current_province: str, destination: str, parent_label: str
) -> None:
    assert destination in search_regions(current_province)
    assert parent_label in search_regions(destination)


def test_old_ba_ria_vung_tau_name_is_recognised() -> None:
    assert regions_in_text("Món ngon Bà Rịa - Vũng Tàu") == ["Vũng Tàu"]


def test_strict_rag_returns_no_data_without_calling_ollama(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Ollama should not be called: {request.url}")

    service = OllamaLlmService(
        _rag_settings(_make_index(tmp_path)),
        httpx.MockTransport(handler),
    )

    answer = asyncio.run(service.generate("Gyoza làm như thế nào?", "vi", 0.99))

    assert answer == "Kho dữ liệu hiện chưa có đủ thông tin để trả lời câu hỏi này."


def test_strict_rag_rejects_unknown_subject_with_common_intent_words(tmp_path) -> None:
    service = CuisineRagService(_rag_settings(_make_index(tmp_path)))

    matches = service.retrieve("Gyoza có nguồn gốc từ đâu?")

    assert matches == []


@pytest.mark.parametrize("model", ["qwen2.5:3b", "gemma3:4b"])
def test_rag_context_is_injected_into_ollama_request(tmp_path, model) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == model
        assert payload["options"]["temperature"] == 0
        assert "RAG SOURCE RULES" in payload["messages"][0]["content"]
        user_content = payload["messages"][-1]["content"]
        assert "BẰNG CHỨNG 1" in user_content
        assert "nước sốt chua ngọt" in user_content
        assert "Quán vịt quay" not in user_content
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": json.dumps({"evidence_ids": [1]})}},
        )

    service = OllamaLlmService(
        replace(_rag_settings(_make_index(tmp_path)), llm_model=model),
        httpx.MockTransport(handler),
    )

    answer = asyncio.run(
        service.generate("Phở chua Lạng Sơn có đặc điểm gì?", "vi", 0.99)
    )

    assert answer == "Phở chua Lạng Sơn có nước sốt chua ngọt nguội mát và được trộn đều khi ăn."


def test_declared_allergy_forces_post_generation_safety_warning(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": json.dumps({"evidence_ids": [1]})}},
        )

    profile = CustomerProfile(
        allergy_status="declared",
        allergens=["đậu phộng"],
        allergy_updated_this_turn=False,
    )
    service = OllamaLlmService(
        _rag_settings(_make_index(tmp_path)),
        httpx.MockTransport(handler),
    )

    answer = asyncio.run(
        service.generate(
            "Phở chua Lạng Sơn có đặc điểm gì?",
            "vi",
            0.99,
            customer_profile=profile,
        )
    )

    assert "chưa đủ để khẳng định" in answer
    assert "đậu phộng" in answer
    assert "nhiễm chéo" in answer
