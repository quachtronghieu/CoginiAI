"""Conservative per-session food-allergy profile handling."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def _normalized(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d")


# Canonical labels are spoken back to Vietnamese users. Aliases are deliberately
# conservative: a match excludes a dish; it never proves that another dish is safe.
ALLERGEN_ALIASES: dict[str, tuple[str, ...]] = {
    "đậu phộng": ("đậu phộng", "đậu phộng rang", "hạt lạc", "lạc", "peanut", "peanuts"),
    "hạt cây": ("hạt cây", "hạt điều", "hạnh nhân", "óc chó", "tree nut", "tree nuts", "cashew", "almond", "walnut"),
    "sữa": ("sữa", "bơ sữa", "whey", "casein", "milk", "dairy"),
    "trứng": ("trứng", "egg", "eggs"),
    "gluten/lúa mì": ("gluten", "lúa mì", "bột mì", "wheat"),
    "đậu nành": ("đậu nành", "đậu tương", "soy", "soya", "soybean"),
    "mè/vừng": ("mè", "vừng", "sesame"),
    "cá": ("cá", "fish"),
    "tôm": ("tôm", "shrimp", "prawn"),
    "cua": ("cua", "crab"),
    "động vật có vỏ": ("hải sản có vỏ", "động vật có vỏ", "shellfish", "mollusk", "mollusc"),
    "hải sản": ("hải sản", "seafood", "tôm", "cua", "cá", "mực", "nghêu", "sò", "ốc", "hàu"),
}

_NORMALIZED_ALIASES = {
    canonical: tuple(_normalized(alias) for alias in aliases)
    for canonical, aliases in ALLERGEN_ALIASES.items()
}

_NO_ALLERGY_PATTERNS = (
    r"^(?:(?:toi|minh|tui|em|chau) )?khong (?:co |bi )?di ung(?: thuc pham)?(?: nao| gi)?[.!?]*$",
    r"^no food allerg(?:y|ies)[.!?]*$",
    r"^i have no allerg(?:y|ies)[.!?]*$",
)

_DISCLOSURE_PATTERNS = (
    r"\b(?:toi|minh|tui|em|chau|khach) (?:bi )?di ung\b",
    r"\b(?:toi|minh|tui|em|chau|khach) khong an duoc\b",
    r"\bi(?: am|'m) allergic\b",
    r"\ballergic to\b",
    r"\bmy allerg(?:y|ies)\b",
    r"^(?:bi )?di ung\b",
)


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def _alias_spans(text: str, aliases: tuple[str, ...]) -> list[tuple[int, int]]:
    """Match exact Vietnamese tones when available; 'của' is not crab 'cua'."""
    normalized = _normalized(text)
    accented = text.casefold() != normalized
    spans = []
    for alias in aliases:
        pattern = unicodedata.normalize("NFC", alias.casefold()) if accented and alias in {"cá", "cua", "sữa", "lạc", "trứng", "mè", "vừng"} else _normalized(alias)
        haystack = text.casefold() if accented and alias in {"cá", "cua", "sữa", "lạc", "trứng", "mè", "vừng"} else normalized
        spans.extend((match.start(), match.end()) for match in re.finditer(rf"(?<!\w){re.escape(pattern)}(?!\w)", haystack))
    return spans


def allergen_keywords(text: str, *, disclosure: bool = True) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    return [canonical for canonical, aliases in ALLERGEN_ALIASES.items()
            if _alias_spans(text, ("hải sản", "seafood") if disclosure and canonical == "hải sản" else aliases)]


def _ingredient_payload(text: str, *, onboarding: bool) -> str:
    """Discard a disclosure intro, never an unrecognised ingredient name."""
    normalized = _normalized(text)
    marker = re.search(r"(?:di ung|allergic to|khong an duoc)\s*(?:(?:voi|la)\s+)?", normalized)
    if marker:
        return text[marker.end():].strip(" .!?,")
    if onboarding:
        damaged_marker = re.search(r"\bung\s+voi\s+", normalized)
        if damaged_marker and not allergen_keywords(text[:damaged_marker.end()]):
            return text[damaged_marker.end():].strip(" .!?,")
        # An ASR-damaged self-reference is allowed only while answering the
        # allergy question, and only if its prefix contains NO allergen name.
        intro = re.match(r"^(?:toi|minh|tui|em|chau)(?:\s+bi)?(?:\s+\w+){0,3}\s+voi\s+", normalized)
        if intro and not allergen_keywords(text[:intro.end()]):
            return text[intro.end():].strip(" .!?,")
    return text


def ingredient_list(text: str, *, onboarding: bool = True) -> tuple[list[str], list[str]]:
    """Known keywords plus every unparsed list part, without mutating a profile."""
    text = unicodedata.normalize("NFC", text).strip()
    payload = _ingredient_payload(text, onboarding=onboarding)
    detected = allergen_keywords(payload)
    unknown = []
    for part in re.split(r"\s+(?:và|va|and|hoặc|hoac)\s+|[,;]", payload, flags=re.I):
        if not part.strip(" .!?,"):
            continue
        characters = list(part)
        for canonical in detected:
            aliases = ("hải sản", "seafood") if canonical == "hải sản" else ALLERGEN_ALIASES[canonical]
            for start, end in _alias_spans(part, aliases):
                characters[start:end] = [" "] * (end - start)
        remaining = _normalized("".join(characters)).strip(" .!?,")
        remaining = re.sub(r"\b(?:va|voi|and|hoac|to|nhe|nha|a|thoi|la|cua toi)\b", " ", remaining)
        if re.search(r"\w", remaining):
            unknown.append(part.strip())
    return detected, unknown


def is_confirmation(text: str) -> bool:
    # Do not strip Vietnamese tones here: "đừng" / "dừng" are NOT "đúng".
    value = unicodedata.normalize("NFC", text.casefold()).strip(" .!?,")
    return value in {"đúng", "đúng rồi", "đúng vậy", "chính xác", "vâng", "có", "ừ", "ừ đúng", "yes", "correct", "that's right"}


def is_rejection(text: str) -> bool:
    value = unicodedata.normalize("NFC", text.casefold()).strip(" .!?,")
    return value in {"không", "không đúng", "sai", "sai rồi", "no", "incorrect"}


@dataclass
class CustomerProfile:
    """Information that must survive beyond the short LLM chat history."""

    allergy_status: str = "unknown"  # unknown | none | declared
    allergens: list[str] = field(default_factory=list)
    allergy_note: str = ""
    allergy_updated_this_turn: bool = False
    allergy_question_asked: bool = False
    pending_allergens: list[str] = field(default_factory=list)
    pending_allergy_status: str = ""
    pending_allergy_note: str = ""
    allergy_confirmation_rejected: bool = False
    allergy_confirmation_unclear: bool = False
    allergy_name_missing: bool = False
    turn_consumed: bool = False
    region: str = ""
    pending_region: str = ""
    region_needs_clarification: bool = False
    food_request: str = ""
    response_language: str = ""
    language_preference: str = ""
    resolved_request: str = ""
    last_route: str = ""
    evidence_sources: list[str] = field(default_factory=list)
    dialogue_action: str = ""
    allergy_confirmation_attempts: int = 0
    allergy_unclear_attempts: int = 0
    awaiting_food_request: bool = False

    def reset(self) -> None:
        self.allergy_status = "unknown"
        self.allergens.clear()
        self.allergy_note = ""
        self.allergy_updated_this_turn = False
        self.allergy_question_asked = False
        self.pending_allergens.clear()
        self.pending_allergy_status = ""
        self.pending_allergy_note = ""
        self.allergy_confirmation_rejected = False
        self.allergy_confirmation_unclear = False
        self.allergy_name_missing = False
        self.turn_consumed = False
        self.region = ""
        self.pending_region = ""
        self.region_needs_clarification = False
        self.food_request = ""
        self.response_language = ""
        self.language_preference = ""
        self.resolved_request = ""
        self.last_route = ""
        self.evidence_sources.clear()
        self.dialogue_action = ""
        self.allergy_confirmation_attempts = 0
        self.allergy_unclear_attempts = 0
        self.awaiting_food_request = False

    def update_from_text(self, text: str) -> bool:
        """Stage disclosures; only a subsequent explicit confirmation commits them."""
        self.allergy_updated_this_turn = False
        self.allergy_confirmation_rejected = False
        self.allergy_confirmation_unclear = False
        self.turn_consumed = False
        text = unicodedata.normalize("NFC", text).strip()
        normalized = _normalized(text).strip(" .!?,")

        if self.pending_allergy_status:
            if is_confirmation(text):
                self.allergy_status = self.pending_allergy_status
                self.allergens = list(self.pending_allergens)
                self.allergy_note = self.pending_allergy_note
                self.pending_allergy_status = ""
                self.pending_allergens.clear()
                self.pending_allergy_note = ""
                self.allergy_question_asked = False
                self.allergy_updated_this_turn = True
                self.turn_consumed = True
                self.allergy_confirmation_attempts = 0
                self.allergy_name_missing = False
                return True
            if is_rejection(text):
                self.pending_allergy_status = ""
                self.pending_allergens.clear()
                self.pending_allergy_note = ""
                self.allergy_question_asked = True
                self.allergy_confirmation_rejected = True
                self.turn_consumed = True
                self.allergy_confirmation_attempts = 0
                self.allergy_name_missing = False
                return False

        short_no = self.allergy_question_asked and normalized.strip(" .!?") in {
            "khong",
            "khong co",
            "no",
            "none",
        }
        if short_no or any(re.search(pattern, normalized) for pattern in _NO_ALLERGY_PATTERNS):
            self.pending_allergy_status = "none"
            self.pending_allergens.clear()
            self.pending_allergy_note = ""
            self.allergy_name_missing = False
            self.turn_consumed = True
            return False

        explicit_disclosure = any(
            re.search(pattern, normalized) for pattern in _DISCLOSURE_PATTERNS
        )
        onboarding = self.allergy_question_asked or self.allergy_name_missing or bool(self.pending_allergy_status or self.pending_allergy_note)
        detected, unknown_parts = ingredient_list(text, onboarding=onboarding)
        if re.search(r"\b(?:khong (?:bi |co )?di ung voi|not allergic|neu .*di ung|if .*allergic|vi du|chang han)\b", normalized):
            # Negations/hypothetical examples are not positive disclosures.
            if onboarding:
                self.pending_allergy_note = text
                self.pending_allergy_status = ""
                self.allergy_question_asked = True
                self.turn_consumed = True
            return False
        if self.pending_allergy_status and not explicit_disclosure and not detected:
            # An unclear yes/no must not erase the list already read back.
            self.allergy_confirmation_unclear = True
            self.allergy_confirmation_attempts += 1
            self.turn_consumed = True
            return False
        if not explicit_disclosure and not self.allergy_question_asked and not self.pending_allergy_status:
            return False
        # While onboarding, don't interpret a new food question as an allergen.
        if not explicit_disclosure and re.search(r"\b(?:goi y|mon an|mon nay|o dau|bao nhieu|co gi|recommend|co .*khong|cach lam|thanh phan|tot khong|la gi)\b", normalized):
            return False
        if explicit_disclosure and not detected:
            # The allergy intent is clear, but an arbitrary ASR residue such as
            # "phạm" is not an ingredient and must never be stored/read back.
            self.pending_allergy_status = ""
            self.pending_allergens.clear()
            self.pending_allergy_note = ""
            self.allergy_name_missing = True
            self.allergy_question_asked = True
            self.turn_consumed = True
            return False
        if unknown_parts or not detected:
            # Never discard an unrecognised ingredient, even when another one is known.
            self.pending_allergy_status = ""
            self.pending_allergens = detected
            self.pending_allergy_note = ", ".join(unknown_parts) or text.strip()
            self.allergy_name_missing = False
            self.allergy_question_asked = True
            self.turn_consumed = True
            self.allergy_unclear_attempts += 1  # Track consecutive unclear turns
            return False
        self.pending_allergy_status = "declared"
        # Additional disclosures conservatively retain existing exclusions.
        self.pending_allergens = list(dict.fromkeys([*self.allergens, *detected]))
        self.pending_allergy_note = ""
        self.allergy_name_missing = False
        self.allergy_confirmation_attempts = 0
        self.turn_consumed = True
        return False

    def allergens_in_text(self, text: str) -> set[str]:
        """Return declared allergens explicitly mentioned in source text."""
        text = unicodedata.normalize("NFC", text)
        normalized = _normalized(text)
        matches: set[str] = set()
        for canonical in self.allergens:
            aliases = ALLERGEN_ALIASES.get(canonical, (canonical,))
            if _alias_spans(text, aliases):
                matches.add(canonical)
        if self.allergy_note:
            note = _normalized(self.allergy_note)
            if note and note in normalized:
                matches.add(self.allergy_note)
        return matches

    def spoken_allergens(self) -> str:
        if self.allergens:
            return ", ".join(self.allergens)
        return self.allergy_note or "thành phần bạn vừa khai báo"
