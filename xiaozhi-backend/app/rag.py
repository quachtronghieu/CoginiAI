"""Read-only retrieval from the curated cuisine SQLite FTS5 index."""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .allergy import CustomerProfile
from .dialogue import REGION_ALIASES, regions_in_text
from .geography import search_regions


logger = logging.getLogger(__name__)

STOP_WORDS = {
    "ai", "bao", "bị", "bởi", "các", "cái", "cho", "chưa", "chuyện", "có",
    "của", "đâu", "đây", "được", "gì", "hay", "hơn", "không", "là", "lại",
    "làm", "một", "món", "nào", "nhiêu", "nhiều", "như", "những", "này",
    "nên", "ra", "sao", "thế", "thì", "từ", "tại", "trong", "và", "về",
    "với", "ăn",
    "gợi", "ý", "giới", "thiệu", "tôi", "mình", "số", "khu", "vực", "thành", "phố",
}
QUESTION_PHRASES = (
    "ăn nhiều", "bao nhiêu calo", "cách làm", "có tốt không", "đặc điểm",
    "nguồn gốc", "tốt không", "xuất xứ",
)
TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class RagMatch:
    title: str
    source: str
    content: str
    score: float
    geo_priority: int = 0


def _normalized(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def _tokens(text: str) -> list[str]:
    return [_normalized(token) for token in TOKEN_RE.findall(text)]


def _question_terms(question: str) -> list[str]:
    terms: list[str] = []
    for token in TOKEN_RE.findall(question.casefold()):
        normalized = _normalized(token)
        if len(normalized) <= 1 or token in STOP_WORDS or normalized in STOP_WORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms[:12]


def _question_phrases(question: str) -> list[str]:
    normalized_question = _normalized(question)
    return [
        _normalized(phrase)
        for phrase in QUESTION_PHRASES
        if _normalized(phrase) in normalized_question
    ]


def _subject_terms(question: str, terms: list[str]) -> list[str]:
    """Keep entity/topic terms while removing recognised question-intent phrases."""
    intent_terms = {
        token
        for phrase in _question_phrases(question)
        for token in _tokens(phrase)
    }
    return [term for term in terms if term not in intent_terms]


def _has_quantitative_answer(question: str, content: str) -> bool:
    if "calo" in _normalized(question):
        return bool(
            re.search(
                r"(?:calo\s*[:=]?\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*(?:kcal|calo))",
                content,
                re.I,
            )
        )
    return bool(re.search(r"\d", content))


def _score(question: str, terms: list[str], title: str, content: str, bm25: float) -> float:
    wanted = set(terms)
    title_tokens = set(_tokens(title))
    content_tokens = set(_tokens(content))
    title_matches = len(wanted & title_tokens)
    content_matches = len(wanted & content_tokens)
    coverage = len(wanted & (title_tokens | content_tokens)) / max(1, len(wanted))
    content_coverage = content_matches / max(1, len(wanted))
    title_text = _normalized(title)
    content_text = _normalized(content)
    phrase_bonus = 0.0
    ordered = " ".join(terms)
    if ordered and ordered in title_text:
        phrase_bonus += 3.0
    for left, right in zip(terms, terms[1:]):
        phrase = f"{left} {right}"
        if phrase in title_text:
            phrase_bonus += 1.2
        elif phrase in content_text:
            phrase_bonus += 0.5
    normalized_question = _normalized(question)
    if "bao nhieu" in normalized_question and _has_quantitative_answer(question, content):
        phrase_bonus += 4.0
    if any(phrase in normalized_question for phrase in ("tot khong", "an nhieu")):
        if re.search(r"không nên|quá nhiều|nguy cơ|hạn chế|vừa phải", content, re.I):
            phrase_bonus += 3.0
    lexical_bonus = min(max(-bm25, 0.0), 30.0) / 30.0
    return (
        coverage * 8.0
        + content_coverage * 6.0
        + title_matches * 0.8
        + phrase_bonus
        + lexical_bonus
    )


def _geo_priority(region: str, title: str) -> int:
    """Prefer an exact requested place before its wider administrative scope."""
    if not region:
        return 0
    found = set(regions_in_text(title))
    if not found:
        return 0
    scope = search_regions(region)
    if region in found:
        return 100
    for index, label in enumerate(scope):
        if label in found:
            return max(1, 90 - index * 10)
    return 0


class CuisineRagService:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.rag_enabled
        self.database = Path(settings.rag_database) if settings.rag_database else None
        self.top_k = settings.rag_top_k
        self.strict = settings.rag_strict
        self._source_cache: dict[str, str] = {}
        self._cache_version: tuple[int, int] | None = None

    def available(self) -> bool:
        if not self.enabled:
            return True
        if self.database is None or not self.database.is_file():
            return False
        try:
            with closing(sqlite3.connect(self.database)) as db:
                db.execute("SELECT 1 FROM cuisine_fts LIMIT 1").fetchone()
            return True
        except sqlite3.Error as exc:
            logger.warning("Cuisine RAG index is unavailable: %s", exc)
            return False

    def retrieve(
        self, question: str, *, region: str = "", recommendation: bool = False,
        customer_profile: CustomerProfile | None = None,
        exclude_allergens: bool = True,
    ) -> list[RagMatch]:
        if not self.enabled or self.database is None:
            return []
        terms = _question_terms(question)
        if not terms:
            return []
        query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        if region:
            scoped_regions = search_regions(region)
            aliases = tuple(dict.fromkeys(
                alias
                for scoped_region in scoped_regions
                for alias in (scoped_region, *REGION_ALIASES.get(scoped_region, ()))
            ))
            region_query = " OR ".join(f'"{alias.replace(chr(34), "")}"' for alias in aliases)
            query = f"title:({region_query})" if recommendation else f"({query}) AND ({region_query})"
        if recommendation and any(phrase in _normalized(question) for phrase in ("an sang", "bua sang", "breakfast")):
            query += ' AND ("ăn sáng" OR "buổi sáng" OR "bữa sáng" OR "breakfast")'
        candidate_limit = max(250, self.top_k * 60)
        try:
            with closing(sqlite3.connect(self.database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                rows = db.execute(
                    "SELECT title, source, content, "
                    "bm25(cuisine_fts, 3.0, 0.0, 1.0) AS rank "
                    "FROM cuisine_fts WHERE cuisine_fts MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (query, candidate_limit),
                ).fetchall()
                if exclude_allergens and customer_profile is not None and customer_profile.allergy_status == "declared":
                    # A safe-looking chunk must not hide an allergen elsewhere in the article.
                    stat = self.database.stat()
                    version = (stat.st_mtime_ns, stat.st_size)
                    if version != self._cache_version:
                        self._source_cache.clear()
                        self._cache_version = version
                    excluded = set()
                    for source in dict.fromkeys(row[1] for row in rows):
                        if source not in self._source_cache:
                            chunks = db.execute("SELECT title, content FROM cuisine_fts WHERE source = ?", (source,)).fetchall()
                            self._source_cache[source] = "\n".join(f"{title}\n{content}" for title, content in chunks)
                        if customer_profile.allergens_in_text(self._source_cache[source]):
                            excluded.add(source)
                    if excluded:
                        logger.info("Whole-article allergy filter excluded sources=%s", sorted(excluded))
                    rows = [row for row in rows if row[1] not in excluded]
        except sqlite3.Error as exc:
            raise RuntimeError(f"Không đọc được chỉ mục RAG: {exc}") from exc

        minimum_coverage = 1.0 if len(terms) <= 2 else 0.65
        subject_terms = _subject_terms(question, terms)
        minimum_subject_coverage = 1.0 if len(subject_terms) <= 3 else 0.67
        scored: list[RagMatch] = []
        wanted = set(terms)
        wanted_subject = set(subject_terms)
        for title, source, content, rank in rows:
            # Geographic recommendations need a title explicitly about that place;
            # an incidental mention in a nationwide article is not enough.
            if region and not set(regions_in_text(title)).intersection(search_regions(region)):
                continue
            if recommendation and any(phrase in _normalized(question) for phrase in ("an sang", "bua sang", "breakfast")):
                # Related-article links and photo captions are not evidence
                # that the dish described in this chunk is eaten for breakfast.
                substantive = "\n".join(line for line in content.splitlines() if not re.match(r"\s*(?:>>>|Xem |Ảnh:)", line, re.I))
                if not re.search(r"ăn sáng|buổi sáng|bữa sáng|breakfast", f"{title} {substantive}", re.I):
                    continue
            combined_tokens = set(_tokens(f"{title} {content}"))
            coverage = len(wanted & combined_tokens) / len(terms)
            if not recommendation and coverage < minimum_coverage:
                continue
            if wanted_subject and not recommendation:
                subject_coverage = len(wanted_subject & combined_tokens) / len(subject_terms)
                if subject_coverage < minimum_subject_coverage:
                    continue
            scored.append(
                RagMatch(
                    title=title,
                    source=source,
                    content=content,
                    score=_score(question, terms, title, content, rank),
                    geo_priority=_geo_priority(region, title),
                )
            )
        if not scored:
            return []
        scored.sort(key=lambda match: (match.geo_priority, match.score), reverse=True)

        source_best: dict[str, float] = defaultdict(float)
        for match in scored:
            source_best[match.source] = max(source_best[match.source], match.score)
        best_source_score = max(source_best.values())
        allowed_sources = {
            source
            for source, score in source_best.items()
            if score >= max(best_source_score * 0.82, best_source_score - 2.5)
        }

        selected: list[RagMatch] = []
        seen: set[str] = set()
        if recommendation:
            for match in scored:
                if match.source in {item.source for item in selected}:
                    continue
                selected.append(match)
                if len(selected) >= self.top_k:
                    break
            return selected
        for phrase in _question_phrases(question):
            phrase_matches = [
                match
                for match in scored
                if match.source in allowed_sources
                and phrase in _normalized(match.content)
                and _normalized(match.content)[:240] not in seen
            ]
            if not phrase_matches:
                continue
            if "bao nhieu" in phrase:
                phrase_matches.sort(
                    key=lambda match: (
                        _has_quantitative_answer(question, match.content),
                        match.score,
                    ),
                    reverse=True,
                )
            chosen = phrase_matches[0]
            seen.add(_normalized(chosen.content)[:240])
            selected.append(chosen)
            if len(selected) >= self.top_k:
                return selected

        for match in scored:
            fingerprint = _normalized(match.content)[:240]
            if match.source not in allowed_sources or fingerprint in seen:
                continue
            seen.add(fingerprint)
            selected.append(match)
            if len(selected) >= self.top_k:
                break
        return selected

    @staticmethod
    def format_context(matches: list[RagMatch]) -> str:
        return "\n\n".join(
            f"TÀI LIỆU {index}: {match.title}\nNỘI DUNG: {match.content}"
            for index, match in enumerate(matches, 1)
        )
