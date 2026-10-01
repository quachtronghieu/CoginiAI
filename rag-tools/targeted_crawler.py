"""Fetch an explicit, audited list of public cuisine sources.

Each URL is tied to a known coverage gap and expected place/topic. robots.txt
is checked per host, requests are serialized, and failed validation never
enters the RAG source directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx


USER_AGENT = "CogniAICuisineResearch/1.1 (+personal study project)"
FOOD = re.compile(
    r"\b(?:món|ẩm thực|đặc sản|bánh|bún|phở|cháo|cơm|xôi|mì|miến|nem|chả|"
    r"gỏi|lẩu|canh|thịt|cá|tôm|cua|mực|gà|bò|heo|rau|chè|kẹo|mứt|rượu)\b",
    re.I,
)
TOPICS = {
    "breakfast": re.compile(
        r"\b(?:ăn sáng|bữa sáng|buổi sáng|quà sáng|điểm tâm|sáng sớm|\d{1,2}h(?:\d{2})?\s*sáng)\b",
        re.I,
    ),
    "origin": re.compile(r"\b(?:nguồn gốc|xuất xứ|bắt nguồn|du nhập|ra đời|lịch sử)\b", re.I),
    "ingredients": re.compile(r"\b(?:nguyên liệu|thành phần|được làm từ|gồm)\b", re.I),
    "taste": re.compile(r"\b(?:hương vị|thơm|ngọt|mặn|chua|cay|giòn|béo|đậm đà)\b", re.I),
    "how_to_eat": re.compile(r"\b(?:cách ăn|thưởng thức|ăn kèm|dùng kèm|chấm)\b", re.I),
    "price": re.compile(r"\b(?:giá|vnđ|đồng|\d+[.,]\d{3})\b", re.I),
    "allergen": re.compile(r"\b(?:dị ứng|chất gây dị ứng|nhiễm chéo)\b", re.I),
}


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def decode_html(response: httpx.Response) -> str:
    """Prefer real UTF-8 bytes over a server's occasionally wrong charset."""
    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError:
        return response.text


class ArticleParser(HTMLParser):
    CONTENT_HINT = re.compile(r"(?:article|post|entry|detail|news)[-_ ]?(?:body|content|detail)?", re.I)
    BLOCKS = {"h1", "h2", "h3", "h4", "p", "li"}
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "form", "aside"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.content_depth = 0
        self.content_markers: list[bool] = []
        self.current_tag = ""
        self.current_parts: list[str] = []
        self.current_scoped = False
        self.all_blocks: list[tuple[str, str]] = []
        self.scoped_blocks: list[tuple[str, str]] = []
        self.title_parts: list[str] = []
        self.in_title = False
        self.visible_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP:
            self.skip_depth += 1
        attributes = " ".join(value or "" for key, value in attrs if key in {"id", "class", "role"})
        marker = tag in {"article", "main"} or bool(self.CONTENT_HINT.search(attributes))
        self.content_markers.append(marker)
        if marker:
            self.content_depth += 1
        if tag == "title":
            self.in_title = True
        if not self.skip_depth and tag in self.BLOCKS:
            self.current_tag = tag
            self.current_parts = []
            self.current_scoped = self.content_depth > 0

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.current_tag == tag:
            text = re.sub(r"\s+", " ", " ".join(self.current_parts)).strip()
            if text:
                block = (tag, text)
                self.all_blocks.append(block)
                if self.current_scoped:
                    self.scoped_blocks.append(block)
            self.current_tag = ""
            self.current_parts = []
        if tag == "title":
            self.in_title = False
        marker = self.content_markers.pop() if self.content_markers else False
        if marker:
            self.content_depth = max(0, self.content_depth - 1)
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        visible = re.sub(r"\s+", " ", data).strip()
        if len(visible) >= 20:
            self.visible_chunks.append(visible)
        if self.in_title:
            self.title_parts.append(data)
        if self.current_tag:
            self.current_parts.append(data)

    def article(self) -> tuple[str, list[tuple[str, str]]]:
        blocks = self.scoped_blocks
        if sum(len(text) for _, text in blocks) < 500:
            blocks = self.all_blocks
        if sum(len(text) for _, text in blocks) < 500:
            # Some government portals render prose directly in nested divs.
            # Keep visible chunks as a last resort; validation below still
            # requires the expected place, food domain and declared topics.
            blocks = [("p", text) for text in self.visible_chunks]
        title = next((text for tag, text in blocks if tag == "h1"), "")
        if not title:
            title = re.sub(r"\s+", " ", " ".join(self.title_parts)).strip()
        return title, blocks


def markdown_from_blocks(blocks: list[tuple[str, str]]) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for tag, text in blocks:
        fingerprint = normalized(text)
        if fingerprint in seen or len(text) < 20:
            continue
        if re.search(r"^(?:trang chủ|đăng nhập|tìm kiếm|tin mới|tin xem nhiều|cùng chuyên mục)$", text, re.I):
            continue
        seen.add(fingerprint)
        prefix = "## " if tag in {"h1", "h2"} else "### " if tag in {"h3", "h4"} else ""
        result.append(prefix + text)
    return "\n\n".join(result)


def load_existing_sources(database: Path | None) -> set[str]:
    if database is None or not database.is_file():
        return set()
    import sqlite3
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        return {row[0] for row in db.execute("SELECT DISTINCT source FROM cuisine_fts")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-database", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    targets = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output / "articles_vietnam"
    output.mkdir(parents=True, exist_ok=True)
    existing_sources = load_existing_sources(args.existing_database)
    robots: dict[str, RobotFileParser | None] = {}
    last_request: defaultdict[str, float] = defaultdict(float)
    records = []

    with httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": USER_AGENT}) as client:
        for target in targets:
            url = target["url"]
            place = target["place"]
            host = urlsplit(url).netloc.lower()
            if url in existing_sources:
                records.append({**target, "status": "already_indexed"})
                continue
            if host not in robots:
                robots_url = f"{urlsplit(url).scheme}://{host}/robots.txt"
                try:
                    response = client.get(robots_url)
                    robot = RobotFileParser(robots_url)
                    if response.status_code == 404:
                        # RFC-style convention: no robots file means no rules.
                        robot.allow_all = True
                    elif response.status_code in {401, 403}:
                        robot.disallow_all = True
                    else:
                        response.raise_for_status()
                        robot.parse(response.text.splitlines())
                    robots[host] = robot
                except Exception as exc:
                    records.append({**target, "status": "robots_unavailable", "error": str(exc)})
                    robots[host] = None
                    continue
            robot = robots[host]
            if robot is None:
                continue
            if not robot.can_fetch(USER_AGENT, url):
                records.append({**target, "status": "robots_denied"})
                continue
            crawl_delay = robot.crawl_delay(USER_AGENT) or args.delay
            wait = max(args.delay, float(crawl_delay)) - (time.monotonic() - last_request[host])
            if wait > 0:
                time.sleep(wait)
            try:
                response = client.get(url)
                last_request[host] = time.monotonic()
                response.raise_for_status()
                if "html" not in response.headers.get("content-type", "").lower():
                    raise ValueError("target is not HTML")
                extractor = ArticleParser()
                extractor.feed(decode_html(response))
                title, blocks = extractor.article()
                body = markdown_from_blocks(blocks)
                combined = f"{title}\n{body}"
                if len(body) < 500:
                    raise ValueError("article body is too short")
                if normalized(place) not in normalized(combined):
                    raise ValueError(f"place {place!r} is not supported by the page")
                if not FOOD.search(combined):
                    raise ValueError("no cuisine evidence detected")
                missing_topics = [topic for topic in target.get("topics", []) if not TOPICS[topic].search(combined)]
                if missing_topics:
                    raise ValueError("missing expected topics: " + ", ".join(missing_topics))
                verified_dishes = target.get("dishes", [])
                missing_dishes = [dish for dish in verified_dishes if normalized(dish) not in normalized(combined)]
                if missing_dishes:
                    raise ValueError("missing expected dishes: " + ", ".join(missing_dishes))
                display_title = title if normalized(place) in normalized(title) else f"{title} — {place}"
                identifier = sha256(url.encode("utf-8")).hexdigest()[:12]
                slug = re.sub(r"[^a-z0-9]+", "-", normalized(title)).strip("-")[:70] or "article"
                destination = output / f"{slug}-{identifier}.md"
                document = (
                    f"# {display_title}\n\nNguồn: {url}\n"
                    f"Địa phương kiểm chứng: {place}\n"
                    f"Chủ đề kiểm chứng: {', '.join(target.get('topics', []))}\n"
                    f"Ngày thu thập: {datetime.now(timezone.utc).isoformat()}\n\n"
                    f"## Món được kiểm chứng: {', '.join(verified_dishes)}\n\n{body}\n"
                )
                if "breakfast" in target.get("topics", []):
                    document = document.replace(
                        f"## Món được kiểm chứng: {', '.join(verified_dishes)}\n\n",
                        f"## Món được kiểm chứng: {', '.join(verified_dishes)}\n\n"
                        "## Ngữ cảnh được kiểm chứng: món ăn sáng\n\n",
                    )
                destination.write_text(document, encoding="utf-8")
                records.append({**target, "status": "saved", "file": str(destination), "title": title})
            except Exception as exc:
                records.append({**target, "status": "rejected", "error": str(exc)})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = sum(record["status"] == "saved" for record in records)
    print(f"Saved {saved}/{len(records)} targeted sources; report={args.report}")


if __name__ == "__main__":
    main()
