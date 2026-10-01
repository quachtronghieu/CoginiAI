import asyncio
from dataclasses import replace

import httpx

from app.allergy import CustomerProfile
from app.config import get_settings
from app.llm import OllamaLlmService


def _settings():
    return replace(
        get_settings(),
        llm_base_url="http://127.0.0.1:11434",
        llm_model="qwen2.5:3b",
        rag_enabled=False,
    )


def test_profile_records_common_allergens() -> None:
    profile = CustomerProfile()

    assert profile.update_from_text("Tôi bị dị ứng đậu phộng và tôm") is False
    assert profile.allergy_status == "unknown"
    assert profile.pending_allergens == ["đậu phộng", "tôm"]
    assert profile.update_from_text("Đúng") is True
    assert profile.allergy_status == "declared"
    assert profile.allergens == ["đậu phộng", "tôm"]
    assert profile.allergens_in_text("Món này gồm thịt, lạc rang và rau") == {"đậu phộng"}


def test_profile_records_no_known_allergy() -> None:
    profile = CustomerProfile(allergy_status="declared", allergens=["sữa"])

    assert profile.update_from_text("Mình không có dị ứng") is False
    assert profile.allergens == ["sữa"]
    assert profile.update_from_text("Đúng") is True
    assert profile.allergy_status == "none"
    assert profile.allergens == []


def test_question_about_allergy_does_not_become_a_disclosure() -> None:
    profile = CustomerProfile()

    assert profile.update_from_text("Món này có gây dị ứng không?") is False
    assert profile.allergy_status == "unknown"


def test_short_answer_is_understood_after_allergy_question() -> None:
    profile = CustomerProfile(allergy_question_asked=True)

    assert profile.update_from_text("Đậu phộng") is False
    assert profile.update_from_text("Đúng rồi") is True
    assert profile.allergy_status == "declared"
    assert profile.allergens == ["đậu phộng"]


def test_short_no_is_understood_after_allergy_question() -> None:
    profile = CustomerProfile(allergy_question_asked=True)

    assert profile.update_from_text("Không") is False
    assert profile.allergy_status == "unknown"
    assert profile.update_from_text("Đúng") is True
    assert profile.allergy_status == "none"


def test_unknown_profile_is_asked_before_ollama() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Ollama should not be called: {request.url}")

    service = OllamaLlmService(_settings(), httpx.MockTransport(handler))
    answer = asyncio.run(
        service.generate(
            "Gợi ý món cho tôi",
            "vi",
            0.99,
            customer_profile=CustomerProfile(),
        )
    )

    assert "bạn có dị ứng" in answer.lower()
    assert "đậu phộng" in answer.lower()


def test_disclosure_is_read_back_for_confirmation_without_ollama() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Ollama should not be called: {request.url}")

    profile = CustomerProfile()
    profile.update_from_text("Tôi dị ứng đậu phộng")
    service = OllamaLlmService(_settings(), httpx.MockTransport(handler))

    answer = asyncio.run(
        service.generate(
            "Tôi dị ứng đậu phộng",
            "vi",
            0.99,
            customer_profile=profile,
        )
    )

    assert "đúng không" in answer.lower()
    assert "đậu phộng" in answer.lower()
    assert profile.allergy_status == "unknown"
    assert profile.pending_allergens == ["đậu phộng"]
