"""Build and query a source-grounded Vietnamese cuisine knowledge base.

Examples (PowerShell):

  py cuisine_rag.py build --data-dir C:\\path\\to\\tasteofvietnam_data
  py cuisine_rag.py ask "Phở chua Lạng Sơn có đặc điểm gì?" \
      --database C:\\path\\to\\tasteofvietnam_data\\cuisine.sqlite3

The index is SQLite FTS5. Ollama must be running for the ``ask`` command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_DATA_DIR = Path(os.getenv("CUISINE_DATA_DIR", "tasteofvietnam_data"))
DEFAULT_OLLAMA_API = os.getenv(
    "CUISINE_OLLAMA_API", "http://127.0.0.1:11434/api/chat"
)
DEFAULT_MODEL = os.getenv("CUISINE_MODEL", "gemma3:4b")
DEFAULT_TOP_K = 3
CHUNK_SIZE = 1100

STOP_WORDS = {
    "ai", "bao", "bị", "bởi", "các", "cái", "cho", "chưa", "chuyện", "có",
    "của", "đâu", "đây", "được", "gì", "hay", "hơn", "không", "là", "lại",
    "làm", "một", "món", "nào", "nhiêu", "nhiều", "như", "những", "này", "nên", "ra",
    "sao", "thế", "thì", "từ", "tại", "trong", "và", "về", "với", "ăn",
    "gợi", "ý", "giới", "thiệu", "tôi", "mình", "số", "khu", "vực", "thành", "phố",
    "muốn", "thèm", "thích", "thử", "hỏi", "biết", "xin", "cách", "nấu",
}

TERMINAL_MARKERS = {
    "advertisement", "hot news", "latest articles", "load more", "most popular",
    "related articles", "tin mới", "tin mới nhất",
}

QUESTION_PHRASES = (
    "ăn nhiều", "bao nhiêu calo", "cách làm", "có tốt không", "đặc điểm",
    "nguồn gốc", "tốt không", "xuất xứ",
)

BOILERPLATE_PATTERNS = (
    r"^home$",
    r"^các nội dung chính(?:\s*ẩn)?$",
    r"^mục lục$",
    r"^bài viết liên quan$",
    r"^tin liên quan$",
    r"^xem thêm:?$",
    r"^quảng cáo$",
    r"^nguồn:\s*$",
)

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
RAW_URL_RE = re.compile(r"https?://\S+")
TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _normalized(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _tokens(text: str) -> list[str]:
    return [_normalized(token) for token in TOKEN_RE.findall(text)]


def _question_terms(question: str) -> list[str]:
    question = re.sub(r"\b(?:sả|xả)\s+tất\b", "sả tắc", question, flags=re.I)
    question = re.sub(r"\bxả\s+tắc\b", "sả tắc", question, flags=re.I)
    question = re.sub(r"\bhủ\s+tíu\b", "hủ tiếu", question, flags=re.I)
    question = re.sub(r"\bmắm\s+tôn\b", "mắm tôm", question, flags=re.I)
    terms: list[str] = []
    for token in TOKEN_RE.findall(question.casefold()):
        normalized = _normalized(token)
        if len(normalized) <= 1 or token in STOP_WORDS or normalized in STOP_WORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms[:12]


def _extract_metadata(markdown: str, fallback: str = "") -> tuple[str, str]:
    lines = markdown.splitlines()
    title = lines[0].lstrip("# ").strip() if lines else Path(fallback).stem
    match = re.search(r"^Nguồn:\s*(\S+)", markdown, flags=re.MULTILINE)
    source = match.group(1) if match else fallback
    return title, source


def _strip_markdown(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = RAW_URL_RE.sub("", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`~]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_paragraphs(markdown: str, title: str) -> list[str]:
    """Remove metadata, navigation, link-heavy TOCs, and scraper noise."""
    blocks = re.split(r"\n\s*\n", markdown.replace("\r\n", "\n"))
    cleaned: list[str] = []
    body_started = False
    normalized_title = _normalized(title)

    for block in blocks:
        raw = block.strip()
        if not raw:
            continue
        if re.match(r"^(Nguồn|Danh mục|Ngày đăng):", raw, flags=re.IGNORECASE):
            continue

        raw_lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not raw_lines:
            continue
        first = raw_lines[0]
        is_heading = first.startswith("#")
        visible = _strip_markdown(raw)
        visible = re.sub(r"^#{1,6}\s*", "", visible).strip()
        normalized_visible = _normalized(visible)
        marker = re.sub(r"[^a-z0-9 ]", "", normalized_visible).strip()

        if marker in TERMINAL_MARKERS:
            break

        if normalized_visible == normalized_title:
            continue
        if re.fullmatch(r"\d{1,2}\s+tháng\s+\d{1,2},?\s+\d{4}", visible, re.I):
            continue
        if re.fullmatch(r"\d+", visible):
            continue
        if any(re.fullmatch(pattern, normalized_visible) for pattern in BOILERPLATE_PATTERNS):
            continue

        link_count = len(MARKDOWN_LINK_RE.findall(raw))
        list_lines = sum(
            1 for line in raw_lines if re.match(r"^[*+-]\s*\[", line)
        )
        if link_count >= 3 or (list_lines and list_lines >= max(1, len(raw_lines) // 2)):
            continue

        if not body_started:
            substantial_text = len(visible) >= 80 and link_count <= 1
            meaningful_heading = is_heading and len(visible) >= 12
            if not substantial_text and not meaningful_heading:
                continue
            body_started = True

        lines: list[str] = []
        for line in raw_lines:
            stripped = _strip_markdown(line)
            stripped = re.sub(r"^#{1,6}\s*", "", stripped).strip()
            if not stripped:
                continue
            normalized_line = _normalized(stripped)
            if normalized_line == normalized_title:
                continue
            if any(re.fullmatch(pattern, normalized_line) for pattern in BOILERPLATE_PATTERNS):
                continue
            lines.append(stripped)
        paragraph = "\n".join(lines).strip()
        if paragraph and paragraph not in cleaned:
            cleaned.append(paragraph)

    return cleaned


def _split_long_paragraph(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > size:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(
                sentence[start : start + size].strip()
                for start in range(0, len(sentence), size)
                if sentence[start : start + size].strip()
            )
        elif len(current) + len(sentence) + 1 > size and current:
            pieces.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current.strip())
    return pieces


def chunks(markdown: str, size: int = CHUNK_SIZE) -> list[str]:
    title, _ = _extract_metadata(markdown)
    paragraphs = clean_paragraphs(markdown, title)
    result: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for piece in _split_long_paragraph(paragraph, size):
            if current and len(current) + len(piece) + 2 > size:
                if len(current) >= 80:
                    result.append(current.strip())
                current = ""
            current = f"{current}\n\n{piece}".strip()
    if len(current) >= 80:
        result.append(current.strip())
    return result


def _resolve_database(directory: Path, database: str | Path | None) -> Path:
    return Path(database) if database else directory / "cuisine.sqlite3"


def build(
    directory: str | Path = DEFAULT_DATA_DIR,
    database: str | Path | None = None,
    extra_directories: list[str | Path] | tuple[str | Path, ...] = (),
) -> tuple[int, int]:
    directory = Path(directory)
    database_path = _resolve_database(directory, database)
    roots = [directory, *(Path(item) for item in extra_directories)]
    files = sorted(
        (path for root in roots for path in (root / "articles_vietnam").glob("*.md")),
        key=lambda path: str(path).casefold(),
    )
    if not files:
        raise RuntimeError(
            f"Không thấy file .md trong {directory / 'articles_vietnam'}; chạy crawler trước."
        )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=database_path.stem + "-", suffix=".tmp", dir=database_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    total = 0
    indexed_files = 0
    indexed_sources: set[str] = set()
    try:
        with closing(sqlite3.connect(temporary_path)) as db:
            db.execute(
                "CREATE VIRTUAL TABLE cuisine_fts USING fts5("
                "title, source UNINDEXED, content, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
            db.execute("CREATE TABLE rag_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            for path in files:
                raw = path.read_text(encoding="utf-8", errors="replace")
                title, source = _extract_metadata(raw, str(path))
                if source in indexed_sources:
                    continue
                parts = chunks(raw)
                if not parts:
                    continue
                db.executemany(
                    "INSERT INTO cuisine_fts (title, source, content) VALUES (?, ?, ?)",
                    ((title, source, part) for part in parts),
                )
                indexed_files += 1
                indexed_sources.add(source)
                total += len(parts)
            db.executemany(
                "INSERT INTO rag_meta (key, value) VALUES (?, ?)",
                (("schema_version", "2"), ("article_count", str(indexed_files)), ("chunk_count", str(total))),
            )
            db.commit()
        os.replace(temporary_path, database_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return indexed_files, total


def _question_phrases(question: str) -> list[str]:
    normalized_question = _normalized(question)
    return [phrase for phrase in QUESTION_PHRASES if _normalized(phrase) in normalized_question]


def _subject_terms(question: str, terms: list[str]) -> list[str]:
    """Keep entity/topic terms while removing recognised question-intent phrases."""
    intent_terms = {
        token
        for phrase in _question_phrases(question)
        for token in _tokens(phrase)
    }
    return [term for term in terms if term not in intent_terms]


def _has_quantitative_answer(question: str, content: str) -> bool:
    normalized_question = _normalized(question)
    if "calo" in normalized_question:
        return bool(
            re.search(
                r"(?:calo\s*[:=]?\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*(?:kcal|calo))",
                content,
                re.I,
            )
        )
    return bool(re.search(r"\d", content))


def _row_score(
    question: str,
    question_terms: list[str],
    title: str,
    content: str,
    bm25: float,
) -> float:
    wanted = set(question_terms)
    title_tokens = set(_tokens(title))
    content_tokens = set(_tokens(content))
    title_matches = len(wanted & title_tokens)
    content_matches = len(wanted & content_tokens)
    coverage = len(wanted & (title_tokens | content_tokens)) / max(1, len(wanted))
    content_coverage = content_matches / max(1, len(wanted))
    ordered = " ".join(question_terms)
    title_text = _normalized(title)
    content_text = _normalized(content)
    phrase_bonus = 0.0
    if ordered and ordered in title_text:
        phrase_bonus += 3.0
    for left, right in zip(question_terms, question_terms[1:]):
        phrase = f"{left} {right}"
        if phrase in title_text:
            phrase_bonus += 1.2
        elif phrase in content_text:
            phrase_bonus += 0.5
    lexical_bonus = min(max(-bm25, 0.0), 30.0) / 30.0
    normalized_question = _normalized(question)
    if "bao nhieu" in normalized_question and _has_quantitative_answer(question, content):
        phrase_bonus += 4.0
    if any(phrase in normalized_question for phrase in ("tot khong", "an nhieu")):
        if re.search(r"không nên|quá nhiều|nguy cơ|hạn chế|vừa phải", content, re.I):
            phrase_bonus += 3.0
    if re.search(r"Món:.*Danh mục:|Giới thiệu.*Nguyên liệu.*Hương vị", content, re.I | re.S):
        phrase_bonus += 2.5
    return coverage * 8.0 + content_coverage * 6.0 + title_matches * 0.8 + phrase_bonus + lexical_bonus


def retrieve(
    question: str,
    database: str | Path,
    limit: int = DEFAULT_TOP_K,
) -> list[dict[str, str | float]]:
    terms = _question_terms(question)
    phrases = [_normalized(phrase) for phrase in _question_phrases(question)]
    if not terms:
        return []
    query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
    candidate_limit = max(80, limit * 30)
    with closing(sqlite3.connect(database)) as db:
        rows = db.execute(
            "SELECT title, source, content, bm25(cuisine_fts, 3.0, 0.0, 1.0) AS rank "
            "FROM cuisine_fts WHERE cuisine_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, candidate_limit),
        ).fetchall()

    minimum_coverage = 1.0 if len(terms) <= 2 else 0.65
    subject_terms = _subject_terms(question, terms)
    minimum_subject_coverage = 1.0 if len(subject_terms) <= 3 else 0.67
    scored: list[dict[str, str | float]] = []
    wanted_subject = set(subject_terms)
    for title, source, content, rank in rows:
        combined_tokens = set(_tokens(f"{title} {content}"))
        coverage = len(set(terms) & combined_tokens) / len(terms)
        if coverage < minimum_coverage:
            continue
        if wanted_subject:
            subject_coverage = len(wanted_subject & combined_tokens) / len(subject_terms)
            if subject_coverage < minimum_subject_coverage:
                continue
        score = _row_score(question, terms, title, content, rank)
        scored.append({"title": title, "source": source, "content": content, "score": score})

    if not scored:
        return []
    scored.sort(key=lambda row: float(row["score"]), reverse=True)

    source_best: dict[str, float] = defaultdict(float)
    for row in scored:
        source = str(row["source"])
        source_best[source] = max(source_best[source], float(row["score"]))
    best_source_score = max(source_best.values())
    allowed_sources = {
        source
        for source, score in source_best.items()
        if score >= max(best_source_score * 0.82, best_source_score - 2.5)
    }

    selected: list[dict[str, str | float]] = []
    seen_content: set[str] = set()
    for phrase in phrases:
        phrase_rows = [
            row
            for row in scored
            if str(row["source"]) in allowed_sources
            and phrase in _normalized(str(row["content"]))
            and _normalized(str(row["content"]))[:240] not in seen_content
        ]
        if not phrase_rows:
            continue
        if "bao nhieu" in phrase:
            phrase_rows.sort(
                key=lambda row: (
                    _has_quantitative_answer(question, str(row["content"])),
                    float(row["score"]),
                ),
                reverse=True,
            )
        chosen = phrase_rows[0]
        fingerprint = _normalized(str(chosen["content"]))[:240]
        seen_content.add(fingerprint)
        selected.append(chosen)
        if len(selected) >= limit:
            return selected

    for row in scored:
        if str(row["source"]) not in allowed_sources:
            continue
        fingerprint = _normalized(str(row["content"]))[:240]
        if fingerprint in seen_content:
            continue
        seen_content.add(fingerprint)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _post_json(url: str, payload: dict[str, object], timeout: int = 180) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama trả lỗi HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError("Không kết nối được Ollama tại " + url) from exc


def _clean_answer(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"^[#>*-]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[`_]", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    meta_phrases = (
        "chào bạn", "tuyệt vời", "tôi đã hiểu", "mình đã hiểu", "tôi sẽ",
        "dựa trên ngữ cảnh", "câu trả lời:", "trả lời trực tiếp:",
    )
    while lines and any(phrase in lines[0].casefold() for phrase in meta_phrases):
        lines.pop(0)
    return " ".join(lines).strip()


def _valid_answer(answer_text: str) -> bool:
    lowered = answer_text.casefold()
    if len(answer_text) < 20:
        return False
    forbidden = ("tôi đã hiểu", "mình đã hiểu", "tôi sẽ chỉ trả lời", "trích đoạn:")
    return not any(phrase in lowered for phrase in forbidden)


def answer(
    question: str,
    database: str | Path,
    model: str = DEFAULT_MODEL,
    api: str = DEFAULT_OLLAMA_API,
    limit: int = DEFAULT_TOP_K,
) -> dict[str, object]:
    database_path = Path(database)
    if not database_path.exists():
        raise RuntimeError("Chưa có chỉ mục. Hãy chạy lệnh build trước.")
    matches = retrieve(question, database_path, limit=limit)
    if not matches:
        return {"answer": "Kho dữ liệu hiện chưa có đủ thông tin để trả lời câu hỏi này.", "sources": []}

    context = "\n\n".join(
        f"TÀI LIỆU {index}: {row['title']}\nNỘI DUNG: {row['content']}"
        for index, row in enumerate(matches, 1)
    )
    system_prompt = (
        "Bạn là trợ lý tra cứu ẩm thực Việt Nam. "
        "Chỉ sử dụng sự kiện xuất hiện rõ ràng trong TÀI LIỆU; không dùng kiến thức bên ngoài. "
        "Không chào hỏi, không nói đã hiểu yêu cầu, không nhắc lại chỉ dẫn và không chép nguyên văn tài liệu. "
        "Trả lời trực tiếp bằng 1 đến 3 câu tiếng Việt tự nhiên, không Markdown. "
        "Không tự thêm con số, nguyên liệu, nguồn gốc, địa chỉ hoặc nhận định. "
        "Phải trả lời đủ từng vế của câu hỏi. Khi người dùng nói muốn ăn hoặc hỏi về một món cụ thể, hãy tóm tắt đầy đủ nguyên liệu chính, hương vị đặc trưng (độ ngon) và nơi bán hoặc cách thưởng thức có trong tài liệu. Với câu hỏi dinh dưỡng hoặc sức khỏe, "
        "không được suy diễn rằng có chất dinh dưỡng nghĩa là ăn nhiều sẽ tốt; hãy nêu đúng giới hạn và nguy cơ trong tài liệu. "
        "Nếu các đoạn mâu thuẫn, hãy nói rõ sự mâu thuẫn và chọn cách diễn đạt thận trọng. "
        "Nếu tài liệu không chứa câu trả lời, nói đúng câu: Kho dữ liệu hiện chưa có đủ thông tin để trả lời câu hỏi này."
    )
    payload: dict[str, object] = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 320, "repeat_penalty": 1.1},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nCÂU HỎI: {question}\nTRẢ LỜI NGAY:"},
        ],
    }
    response = _post_json(api, payload)
    message = response.get("message", {})
    raw_answer = message.get("content", "") if isinstance(message, dict) else ""
    result_text = _clean_answer(str(raw_answer))
    if not _valid_answer(result_text):
        result_text = "Kho dữ liệu hiện chưa có đủ thông tin để trả lời câu hỏi này."

    sources = list(dict.fromkeys(str(row["source"]) for row in matches))
    return {"answer": result_text, "sources": sources, "matches": matches}


def _add_database_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Thư mục tasteofvietnam_data")
    command.add_argument("--database", type=Path, help="File SQLite; mặc định là <data-dir>/cuisine.sqlite3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lập chỉ mục và hỏi đáp từ kho bài ẩm thực Việt Nam")
    commands = parser.add_subparsers(dest="command", required=True)

    build_command = commands.add_parser("build", help="Làm sạch và tạo lại chỉ mục")
    _add_database_arguments(build_command)
    build_command.add_argument(
        "--extra-data-dir", action="append", type=Path, default=[],
        help="Thư mục bổ sung có articles_vietnam; có thể dùng nhiều lần",
    )

    ask_command = commands.add_parser("ask", help="Hỏi model dựa trên kho nguồn")
    ask_command.add_argument("question")
    _add_database_arguments(ask_command)
    ask_command.add_argument("--model", default=DEFAULT_MODEL)
    ask_command.add_argument("--api", default=DEFAULT_OLLAMA_API)
    ask_command.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    inspect_command = commands.add_parser("inspect", help="Xem đoạn được truy xuất mà chưa gọi model")
    inspect_command.add_argument("question")
    _add_database_arguments(inspect_command)
    inspect_command.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    args = parser.parse_args()
    database = _resolve_database(args.data_dir, args.database)
    if args.command == "build":
        files, pieces = build(args.data_dir, database, args.extra_data_dir)
        print(f"Indexed {files} clean articles and {pieces} chunks at {database}.")
        return

    if args.command == "inspect":
        matches = retrieve(args.question, database, limit=max(1, args.top_k))
        if not matches:
            print("Không tìm thấy đoạn đủ liên quan.")
            return
        for index, row in enumerate(matches, 1):
            print(f"\n[{index}] {row['title']}")
            print(row["source"])
            print(str(row["content"])[:500])
        return

    result = answer(args.question, database, model=args.model, api=args.api, limit=max(1, args.top_k))
    print(result["answer"])
    if result["sources"]:
        print("\nNguồn:")
        for source in result["sources"]:
            print("- " + str(source))


if __name__ == "__main__":
    main()
