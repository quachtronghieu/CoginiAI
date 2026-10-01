"""Build complete, source-backed passages whose IDs can be selected by the LLM."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .dialogue import regions_in_text
from .geography import search_regions
from .allergy import _normalized
from .rag import RagMatch


INGREDIENT_EVIDENCE = re.compile(
    r"nguyên liệu|thành phần|được làm từ|chế biến từ|gồm|bao gồm|"
    r"phần (?:nhân|vỏ)|bột gạo|quả trứng|nước dùng|nước chấm|"
    r"\b(?:trứng|thịt|cá|tôm|cua|mực|gạo|bột|rau|đậu)\b",
    re.I,
)


@dataclass(frozen=True)
class Evidence:
    source: str
    title: str
    text: str


def _sentences(content: str) -> list[str]:
    """Return contiguous source spans, excluding captions, headings and links."""
    result = []
    for paragraph in re.split(r"\n\s*\n", content):
        for line in paragraph.splitlines():
            line = line.strip().lstrip("+ ")
            if re.match(r"^(?:Xem |Xem thêm|Xem nhanh|>>>|Ảnh:|Địa chỉ:|Giờ mở cửa:|Cận cảnh|\d+\S*[^.!?]*$)", line, re.I):
                continue
            for span in re.split(r"\((?:Ảnh|Nguồn):[^)]*\)|Ảnh:", line):
                for sentence in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ỸĐ])", span):
                    sentence = sentence.strip()
                    if 20 <= len(sentence) <= 650 and re.search(r"[.!]", sentence) and not sentence.endswith("?"):
                        result.append(sentence)
    return result


def evidence_passages(matches: list[RagMatch], region: str = "", recommendation: bool = False, question: str = "", *, information_only: bool = False) -> list[Evidence]:
    evidence = []
    seen = set()
    breakfast = any(phrase in _normalized(question) for phrase in ("an sang", "bua sang", "breakfast"))
    for match in matches:
        source_sentences = _sentences(match.content)
        breakfast_evidence = any(re.search(r"ăn sáng|buổi sáng|bữa sáng|breakfast", f"{match.title} {sentence}", re.I) for sentence in source_sentences)
        # A single-dish article can support its descriptive sentences with a
        # separate meal sentence. In a list, one breakfast dish cannot qualify
        # every other dish; each selected span must mention breakfast itself.
        single_dish = not re.search(r"tổng hợp|đặc sản|quán|địa chỉ|\btop\b|\d+\s*(?:món|địa điểm)|ẩm thực", match.title, re.I)
        for sentence in source_sentences:
            if sentence in seen:
                continue
            if re.search(r"bài viết này|bài viết sau|bên trên là|chúc bạn|cùng vào bếp|chúng tôi sẽ|bạn ơi|oanh tạc|hãy hỏi|bạn hãy|trên mạng|giàu năng lượng và hứng khởi", sentence, re.I):
                continue
            if recommendation and not re.search(r"\b(?:món|bún|phở|bánh|cháo|cơm|xôi|thịt|cá|cua|tôm|nước dùng|nước lèo|sợi|rau|nguyên liệu|thành phần|hương vị)\b", sentence, re.I):
                continue
            if recommendation and re.search(r"^[“\"']?Tôi\b|\b(?:ông|bà|anh|chị)\b.*(?:cho biết|ông nói|bà nói)", sentence, re.I):
                continue
            if information_only and re.search(r"bạn (?:nên|hãy|có thể|đừng|nhất định)|hãy (?:thử|ăn|thưởng thức)|an toàn|không (?:chứa|gây) dị ứng", sentence, re.I):
                # A source's advice or safety claim must not become advice
                # to a customer whose recipe/allergy data is unverified.
                continue
            if recommendation and breakfast:
                if not breakfast_evidence:
                    continue
                if not single_dish and not re.search(r"ăn sáng|buổi sáng|bữa sáng|breakfast", f"{match.title} {sentence}", re.I):
                    continue
            if recommendation and re.search(r"chuyên gia dinh dưỡng|mang thai|chăm sóc sức khỏe|chữa bệnh", sentence, re.I):
                continue
            if recommendation and region and any(
                place not in search_regions(region)
                for place in regions_in_text(sentence)
            ):
                continue
            seen.add(sentence)
            evidence.append(Evidence(match.source, match.title, sentence))
    normalized_question = _normalized(question)
    topic_question = question.casefold()
    if any(phrase in normalized_question for phrase in ("nguon goc", "xuat xu", "origin")):
        evidence = [item for item in evidence if re.search(r"xuất xứ|bắt nguồn|có từ đâu|ra đời|originat", item.text, re.I)]
    elif re.search(r"thành phần|nguyên liệu|ingredients?", topic_question, re.I):
        evidence = [
            item for item in evidence
            if INGREDIENT_EVIDENCE.search(item.text)
        ]
    elif re.search(r"cách ăn|ăn như thế nào|thưởng thức|how to eat", topic_question, re.I):
        evidence = [
            item for item in evidence
            if re.search(r"cách ăn|thưởng thức|ăn kèm|dùng kèm|chấm|khi ăn|ăn nóng|ăn nguội", item.text, re.I)
        ]
    elif re.search(r"giá|bao nhiêu tiền|price|cost", topic_question, re.I):
        evidence = [
            item for item in evidence
            if re.search(r"giá|đồng|vnđ|vnd|khoảng|từ \d|\d+[.,]?\d*\s*(?:nghìn|triệu)", item.text, re.I)
        ]
    elif re.search(r"dị ứng|allergen|allergy", topic_question, re.I):
        # Ingredient evidence is necessary, but never sufficient to claim
        # that a dish is safe for a person; the LLM safety contract adds the
        # cross-contact/staff-confirmation warning afterwards.
        evidence = [
            item for item in evidence
            if INGREDIENT_EVIDENCE.search(item.text)
        ]
    elif any(phrase in normalized_question for phrase in ("dac diem", "huong vi", "dac biet")):
        evidence = [item for item in evidence if not re.search(r"bắt nguồn|xuất xứ|truyền tai|ẩm thực Trung Quốc|phiên bản biến tấu", item.text, re.I)]
        features = [item for item in evidence if re.search(r"nước|sốt|chua ngọt|ngọt|sợi|trộn|nguyên liệu|nóng|nguội|tanh|hương vị|rau|thành phần", item.text, re.I)]
        if features:
            evidence = features
    elif "calo" in normalized_question:
        evidence.sort(key=lambda item: bool(re.search(r"\d+(?:[.,]\d+)?\s*(?:kcal|calo)", item.text, re.I)), reverse=True)
    return evidence[:40]
