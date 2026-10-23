"""Evaluate the 30-sample Gold Q&A Dataset against the live Ollama (gemma3:4b) + RAG pipeline."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "xiaozhi-backend"
DATASET_PATH = Path(__file__).resolve().parent / "gold_qa_dataset_30.json"
sys.path.insert(0, str(BACKEND_DIR))

from app.allergy import CustomerProfile  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm import OllamaLlmService  # noqa: E402


async def run_evaluation():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    settings = get_settings()
    service = OllamaLlmService(settings)

    passed = 0
    results = []

    print(f"Running evaluation on {len(dataset)} samples using model={settings.llm_model}...\n")
    for item in dataset:
        profile = CustomerProfile()
        query = item["user_query"]
        answer = await service.generate(query, "vi", 0.99, customer_profile=profile)
        lowered = answer.casefold()

        inc_ok = all(kw.casefold() in lowered for kw in item["must_include_keywords"])
        exc_ok = all(kw.casefold() not in lowered for kw in item["must_not_include_keywords"])
        ok = inc_ok and exc_ok
        if ok:
            passed += 1

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {item['id']} ({item['category']}): {query}")
        print(f"   -> Answer: {answer}")
        if not ok:
            missing = [kw for kw in item["must_include_keywords"] if kw.casefold() not in lowered]
            forbidden = [kw for kw in item["must_not_include_keywords"] if kw.casefold() in lowered]
            print(f"   -> MISSING: {missing} | FORBIDDEN: {forbidden}")
        results.append({"id": item["id"], "query": query, "answer": answer, "passed": ok})

    print(f"\n==========================================")
    print(f"SUMMARY: {passed}/{len(dataset)} passed ({passed * 100 / len(dataset):.1f}%)")
    print(f"==========================================")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
