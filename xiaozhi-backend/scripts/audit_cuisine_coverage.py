"""Audit real regional cuisine coverage in the active SQLite RAG index.

The report distinguishes current provinces, legacy place labels and tourist
destinations. A current province inherits titled sources from its former
provinces/destinations through app.geography.search_regions(). Dish names are
conservative candidates backed by a source title or a short heading followed
by food-description prose; they are not invented by an LLM.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from app.dialogue import REGION_ALIASES, regions_in_text
from app.geography import CURRENT_PROVINCE_MEMBERS, DESTINATION_SCOPES, search_regions


FOOD_NAME = re.compile(
    r"\b(?:bánh|bún|phở|cháo|cơm|xôi|mì|miến|hủ tiếu|cao lầu|nem|chả|gỏi|"
    r"lẩu|canh|súp|thịt|cá|tôm|cua|ốc|mực|gà|bò|heo|dê|vịt|ếch|rau|chè|"
    r"kẹo|mứt|hạt|trái cây|hoa quả|cà phê|nước|rượu)\b",
    re.I,
)
FOOD_PROSE = re.compile(
    r"\b(?:món|đặc sản|nguyên liệu|thành phần|được làm|ăn kèm|nước dùng|"
    r"hương vị|thơm|ngọt|mặn|chua|cay|giòn|béo|thịt|bột|gạo)\b",
    re.I,
)
GENERIC_HEADING = re.compile(
    r"\b(?:địa chỉ|giá tham khảo|nguyên liệu|cách làm|cách nấu|cách chọn|"
    r"sơ chế|thực hiện|thưởng thức|lưu ý|kết luận|most popular|hot news|"
    r"xem thêm|bước\s+\d+|cận cảnh|ảnh|quán|bảng giá|nguồn gốc|khám phá|"
    r"cảm nhận|mách bạn|bỏ túi|top\s+\d+|ẩm thực|đặc sản|món ngon)\b",
    re.I,
)
BREAKFAST = re.compile(r"\b(?:ăn sáng|bữa sáng|buổi sáng|điểm tâm|breakfast)\b", re.I)
TOPIC_PATTERNS = {
    "breakfast": BREAKFAST,
    "origin": re.compile(r"\b(?:nguồn gốc|xuất xứ|bắt nguồn|ra đời|lịch sử)\b", re.I),
    "ingredients": re.compile(r"\b(?:nguyên liệu|thành phần|được làm từ|chế biến từ)\b", re.I),
    "taste": re.compile(r"\b(?:hương vị|vị ngọt|vị mặn|vị chua|vị cay|thơm|béo|giòn|đậm đà)\b", re.I),
    "how_to_eat": re.compile(r"\b(?:cách ăn|thưởng thức|ăn kèm|dùng kèm|chấm cùng)\b", re.I),
    "price": re.compile(r"\b(?:giá tham khảo|vnđ|đồng(?:/|\s)|\d+[.,]\d{3})\b", re.I),
    "allergen": re.compile(r"\b(?:dị ứng|chất gây dị ứng|nhiễm chéo|đậu phộng|lạc|sữa|trứng|hải sản)\b", re.I),
}
VERIFIED_DISHES = re.compile(r"^Món được kiểm chứng:\s*(.+)$", re.I | re.M)


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _clean_candidate(value: str) -> str:
    value = re.sub(r"^\s*(?:\d+[.)-]?|[-+*])\s*", "", value).strip(" :–—-*#\t")
    value = re.sub(r"\s+", " ", value)
    return value


def _title_candidate(title: str) -> str | None:
    value = re.sub(r"^(?:cách (?:làm|nấu)|bí quyết (?:làm|nấu))\s+", "", title, flags=re.I)
    normalized_value = _normalized(value)
    region_ends = [
        normalized_value.find(alias) + len(alias)
        for aliases in REGION_ALIASES.values()
        for alias in aliases
        if normalized_value.find(alias) >= 0
    ]
    if region_ends:
        # A title such as "bánh tai Phú Thọ vỏ mềm..." contains a reliable
        # dish/place boundary; discard the promotional suffix after it.
        value = value[: min(region_ends)]
    value = re.split(r"\s+(?:ở|tại|đặc sản|chuẩn vị|thơm ngon|ngon|đơn giản|và top|–|—|\|)", value, maxsplit=1, flags=re.I)[0]
    value = _clean_candidate(value)
    if not FOOD_NAME.match(value) or GENERIC_HEADING.search(value):
        return None
    words = value.split()
    return value if 1 < len(words) <= 10 else None


def extract_dishes(title: str, content: str) -> list[str]:
    """Return conservative, source-backed dish-name candidates."""
    candidates: list[str] = []
    title_candidate = _title_candidate(title)
    if title_candidate:
        candidates.append(title_candidate)

    verified = VERIFIED_DISHES.search(content)
    if verified:
        candidates.extend(
            candidate.strip() for candidate in verified.group(1).split(",") if candidate.strip()
        )

    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    for index, block in enumerate(blocks[:-1]):
        if "\n" in block or len(block) > 90 or re.search(r"[.!?]", block):
            continue
        candidate = _clean_candidate(block)
        if not 2 <= len(candidate.split()) <= 10:
            continue
        if (
            GENERIC_HEADING.search(candidate)
            or not FOOD_NAME.match(candidate)
            or re.search(r"[@\\/]", candidate)
        ):
            continue
        following = blocks[index + 1][:500]
        if len(following) < 60 or not FOOD_PROSE.search(following):
            continue
        candidates.append(candidate)

    unique: dict[str, str] = {}
    for candidate in candidates:
        unique.setdefault(_normalized(candidate), candidate)
    return list(unique.values())


def dish_belongs_to_scope(dish: str, record_regions: list[str], labels: list[str]) -> bool:
    """Avoid assigning a place-qualified dish to every place in a mixed title."""
    dish_regions = regions_in_text(dish)
    allowed = set(labels)
    if dish_regions:
        return bool(allowed.intersection(dish_regions))
    return bool(allowed.intersection(record_regions))


@dataclass
class PlaceCoverage:
    place: str
    kind: str
    search_labels: list[str]
    source_count: int
    dish_count: int
    dishes: list[str]
    topic_sources: dict[str, int]
    gaps: list[str]


def _read_sources(database: Path) -> list[tuple[str, str, str]]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        return db.execute(
            "SELECT title, source, group_concat(content, '\n\n') "
            "FROM cuisine_fts GROUP BY title, source ORDER BY title, source"
        ).fetchall()


def build_report(database: Path) -> dict[str, object]:
    sources = _read_sources(database)
    records = []
    by_exact_region: dict[str, list[int]] = defaultdict(list)
    for index, (title, source, content) in enumerate(sources):
        exact_regions = regions_in_text(title)
        dishes = extract_dishes(title, content)
        topics = {name for name, pattern in TOPIC_PATTERNS.items() if pattern.search(f"{title}\n{content}")}
        records.append({
            "title": title,
            "source": source,
            "content": content,
            "regions": exact_regions,
            "dishes": dishes,
            "topics": topics,
        })
        for region in exact_regions:
            by_exact_region[region].append(index)

    def summarize(place: str, kind: str) -> PlaceCoverage:
        labels = list(search_regions(place))
        record_ids = sorted({item for label in labels for item in by_exact_region.get(label, [])})
        dish_map: dict[str, str] = {}
        topic_sources = {topic: 0 for topic in TOPIC_PATTERNS}
        for record_id in record_ids:
            record = records[record_id]
            for dish in record["dishes"]:
                if dish_belongs_to_scope(dish, record["regions"], labels):
                    dish_map.setdefault(_normalized(dish), dish)
            for topic in record["topics"]:
                topic_sources[topic] += 1
        gaps = []
        if len(record_ids) < 5:
            gaps.append("few_sources")
        if len(dish_map) < 5:
            gaps.append("few_dishes")
        if topic_sources["breakfast"] < 2:
            gaps.append("breakfast")
        for topic in ("origin", "ingredients", "how_to_eat", "price", "allergen"):
            if topic_sources[topic] == 0:
                gaps.append(topic)
        return PlaceCoverage(
            place=place,
            kind=kind,
            search_labels=labels,
            source_count=len(record_ids),
            dish_count=len(dish_map),
            dishes=sorted(dish_map.values(), key=_normalized),
            topic_sources=topic_sources,
            gaps=gaps,
        )

    current = [summarize(place, "current_province") for place in CURRENT_PROVINCE_MEMBERS]
    all_labels = [summarize(place, "legacy_or_destination") for place in REGION_ALIASES]
    metadata = {}
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        try:
            metadata = dict(db.execute("SELECT key, value FROM rag_meta"))
        except sqlite3.Error:
            metadata = {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(database.resolve()),
        "metadata": metadata,
        "source_count": len(sources),
        "current_provinces": [asdict(item) for item in current],
        "legacy_and_destinations": [asdict(item) for item in all_labels],
    }


def markdown_report(report: dict[str, object]) -> str:
    rows = report["current_provinces"]
    lines = [
        "# Cuisine coverage audit",
        "",
        f"Database: `{report['database']}`",
        f"Sources: {report['source_count']}; indexed articles: {report['metadata'].get('article_count', '?')}; chunks: {report['metadata'].get('chunk_count', '?')}.",
        "",
        "Dish counts are conservative source-backed candidates from article titles/headings. Current provinces include former province and destination labels listed in `search labels`.",
        "",
        "| Current province/city | Search labels | Sources | Dishes | Breakfast sources | Gaps |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['place']} | {', '.join(row['search_labels'])} | {row['source_count']} | "
            f"{row['dish_count']} | {row['topic_sources']['breakfast']} | {', '.join(row['gaps']) or '-'} |"
        )
    lines.extend(["", "## Dish inventory", ""])
    for row in rows:
        dishes = ", ".join(row["dishes"]) or "No source-backed dish candidate detected"
        lines.append(f"- **{row['place']}** ({row['dish_count']}): {dishes}")
    lines.extend(["", "## Legacy names and tourist destinations", ""])
    for row in report["legacy_and_destinations"]:
        lines.append(
            f"- **{row['place']}**: {row['source_count']} sources, {row['dish_count']} dishes; "
            f"scope: {', '.join(row['search_labels'])}; gaps: {', '.join(row['gaps']) or '-'}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.database)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(f"Wrote {args.json} and {args.markdown}")


if __name__ == "__main__":
    main()
