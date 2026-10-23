"""Deterministic safety onboarding and location constraints, independent of chat history."""

from __future__ import annotations

import re
import unicodedata

from .allergy import CustomerProfile, _contains_phrase, _normalized, is_confirmation, is_rejection, allergen_keywords


# Maximum consecutive times the system asks to repeat an unrecognised allergen
# before accepting the partial list and moving on (safety-conservative fallback).
_MAX_ALLERGY_UNCLEAR_RETRIES = 3


# These are search labels, not claims about current administrative boundaries.
REGION_ALIASES = {
    "Cần Thơ": ("can tho",),
    "Cà Mau": ("ca mau", "ga mau", "ga meo"),
    "TP.HCM": ("tp hcm", "tp.hcm", "ho chi minh", "sai gon", "saigon", "hcm", "ho chi minh city"),
    "Hà Nội": ("ha noi", "hanoi"),
    "Lạng Sơn": ("lang son",),
    "Đà Nẵng": ("da nang", "danang"),
    "Huế": ("hue",),
    "Đà Lạt": ("da lat", "dalat", "da lac", "dalac"),
    "Nha Trang": ("nha trang",),
    "Hội An": ("hoi an",),
    "Phú Quốc": ("phu quoc",),
    "Bến Tre": ("ben tre",),
    "Sóc Trăng": ("soc trang",),
    "Bạc Liêu": ("bac lieu",),
    "An Giang": ("an giang",),
    "Tiền Giang": ("tien giang",),
    "Đồng Tháp": ("dong thap",),
    "Vĩnh Long": ("vinh long",),
    "Trà Vinh": ("tra vinh",),
    "Kiên Giang": ("kien giang",),
    "Hải Phòng": ("hai phong",),
    "Quảng Ninh": ("quang ninh",),
    "Nghệ An": ("nghe an",),
    "Thanh Hóa": ("thanh hoa",),
    "Bình Định": ("binh dinh",),
    "Phú Yên": ("phu yen",),
    "Quảng Nam": ("quang nam",),
    "Quảng Ngãi": ("quang ngai",),
    "Bình Thuận": ("binh thuan",),
    "Vũng Tàu": ("vung tau", "ba ria vung tau", "ba ria-vung tau", "ba ria"),
    "Khánh Hòa": ("khanh hoa",),
    "Ninh Thuận": ("ninh thuan",),
    "Quy Nhơn": ("quy nhon",),
    "Phan Thiết": ("phan thiet",),
    "Cam Ranh": ("cam ranh",),
    "Hà Giang": ("ha giang",),
    "Cao Bằng": ("cao bang",),
    "Bắc Kạn": ("bac kan",),
    "Tuyên Quang": ("tuyen quang",),
    "Lào Cai": ("lao cai",),
    "Sa Pa": ("sa pa", "sapa"),
    "Yên Bái": ("yen bai",),
    "Lai Châu": ("lai chau",),
    "Điện Biên": ("dien bien",),
    "Sơn La": ("son la",),
    "Hòa Bình": ("hoa binh",),
    "Thái Nguyên": ("thai nguyen",),
    "Phú Thọ": ("phu tho",),
    "Vĩnh Phúc": ("vinh phuc",),
    "Bắc Giang": ("bac giang",),
    "Bắc Ninh": ("bac ninh",),
    "Hải Dương": ("hai duong",),
    "Hưng Yên": ("hung yen",),
    "Thái Bình": ("thai binh",),
    "Hà Nam": ("ha nam",),
    "Nam Định": ("nam dinh",),
    "Ninh Bình": ("ninh binh",),
    "Hà Tĩnh": ("ha tinh",),
    "Quảng Bình": ("quang binh",),
    "Quảng Trị": ("quang tri",),
    "Kon Tum": ("kon tum",),
    "Gia Lai": ("gia lai",),
    "Đắk Lắk": ("dak lak", "dac lac"),
    "Đắk Nông": ("dak nong", "dac nong"),
    "Lâm Đồng": ("lam dong",),
    "Buôn Ma Thuột": ("buon ma thuot", "buon me thuot"),
    "Bình Phước": ("binh phuoc",),
    "Tây Ninh": ("tay ninh",),
    "Bình Dương": ("binh duong",),
    "Đồng Nai": ("dong nai",),
    "Long An": ("long an",),
    "Hậu Giang": ("hau giang",),
    "Hạ Long": ("ha long",),
    "Cô Tô": ("co to",),
    "Cát Bà": ("cat ba",),
    "Mộc Châu": ("moc chau",),
    "Đồng Văn": ("dong van",),
    "Phan Rang": ("phan rang",),
    "Ninh Hòa": ("ninh hoa",),
    "Tuy Hòa": ("tuy hoa",),
    "Côn Đảo": ("con dao",),
    "Mỹ Tho": ("my tho",),
    "Sa Đéc": ("sa dec",),
    "Châu Đốc": ("chau doc",),
    "Rạch Giá": ("rach gia",),
    "Hà Tiên": ("ha tien",),
    "Biên Hòa": ("bien hoa",),
}

# Observed ASR mistakes are hypotheses only; never commit them without confirmation.
LOCATION_HYPOTHESES = {
    "ca meo": "Cà Mau", "camel": "Cà Mau", "ga mau": "Cà Mau", "ga meo": "Cà Mau", "da lac": "Đà Lạt", "dalac": "Đà Lạt", "ho chi": "TP.HCM",
    "nha chiem": "Nha Trang", "nha chien": "Nha Trang", "nha chem": "Nha Trang",
}
LOCATION_MARKER = re.compile(r"\b(?:o|tai|khu vuc|thanh pho|tinh)\s+")


def regions_in_text(text: str) -> list[str]:
    normalized = _normalized(text)
    return [region for region, aliases in REGION_ALIASES.items() if any(_contains_phrase(normalized, alias) for alias in aliases)]


def is_recommendation(text: str) -> bool:
    normalized = _normalized(text)
    explicit = re.search(r"\b(?:goi y|gui y|de xuat|tu van mon|recommend\w*|suggest\w*)\b", normalized)
    want_open = re.search(r"\b(?:muon an|them an|want to eat)\b(?:\s+(?:mon\s+gi|gi|sang|trua|toi|chieu|dem|khuya|vat|ngon|o|tai|khu vuc|thanh pho|tinh)\b|\s*$)", normalized)
    if not (explicit or want_open) and re.search(r"\b(?:muon an|them an|thich an|want to eat)\s+\w+", normalized):
        return False
    explicit = explicit or want_open
    if not explicit and re.search(r"\b(?:nguon goc|xuat xu|thanh phan|dac diem|huong vi|origin|ingredients?)\b", normalized):
        return False
    return bool(explicit or re.search(r"\b(?:mon gi|mon an ngon|mon an sang|an gi|co mon gi|co gi ngon|mot so mon|breakfast|lunch|dinner)\b", normalized))


def is_food_question(text: str) -> bool:
    return is_recommendation(text) or bool(re.search(
        r"\b(?:muon an|them an|thich an|want to eat|mon|am thuc|thanh phan|nguyen lieu|nguon goc|xuat xu|calo|cach an|cach nau|cach lam|nau nhu the nao|lam nhu the nao|che bien|cong thuc|an nhu the nao|huong vi|pho|bun|banh|com|canh|sup|xoi|mi|mien|hu tieu|lau|goi|nom|nem|chan ga|cut lon|tokbokki|kimbap|bingsu|takoyaki|bac xiu|nuoc mia|sua chua|do uong|thuc don|nha hang|nau an|dinh duong|food|dish\w*|ingredient\w*|origin|cuisine|recipe|menu|restaurant|cooking|nutrition|gyoza|pizza|sushi|taco|hamburger)\b",
        _normalized(text),
    )) or bool(re.search(r"\b(?:cháo|chè|chả|giò|thịt|cá|gà|bò|heo|lợn|vịt|tôm|cua|ốc|hến|trà|sườn|mực|khổ qua)\b", text, re.I)) or bool(allergen_keywords(text) and re.search(r"\b(?:co .*khong|bao nhieu|lam .*nao|tot khong|la gi)\b", _normalized(text)))


def courtesy_or_domain_reply(text: str, language: str, profile: CustomerProfile | None = None) -> str | None:
    """Handle greetings/scope before touching allergy or location state."""
    normalized = _normalized(text).strip(" .!?,")
    english = language == "en"
    if re.fullmatch(r"(?:xin chao|xin giao|chao(?: ban| em| anh| chi)?|hello|hi|hey|good morning)(?:\s+(?:tro ly|ai|ban))?", normalized):
        if profile is not None:
            profile.dialogue_action = "greeting"
            profile.awaiting_food_request = True
        greeting_english = english or normalized in {"hello", "hi", "hey", "good morning"}
        return "Hello! What food or dish would you like advice about?" if greeting_english else "Xin chào! Bạn cần mình tư vấn món ăn gì?"
    if re.fullmatch(r"(?:cam on(?: ban| nhe)?|thanks|thank you)", normalized):
        return "You're welcome! What else would you like to know about food?" if english else "Không có gì! Bạn muốn tìm hiểu thêm món ăn nào?"
    if re.fullmatch(r"(?:tam biet|xin chao tam biet(?:,? tam biet)?|goodbye|bye)", normalized):
        return "Goodbye!" if english else "Chào bạn, hẹn gặp lại!"
    technical = re.search(r"\b(?:viet code|lap trinh|python|javascript|giai (?:bai|phuong trinh)|programming|solve .*equation)\b", normalized)
    foreign_topic = re.search(r"\b(?:thoi tiet|gia vang|bitcoin|chung khoan|chinh tri|bong da|thu do|vu tru|ngay may|thu may|toan hoc|may tinh|dien thoai|choi game|weather|football|soccer|stock price|politics|capital of|math|computer|smartphone|universe|astronomy)\b", normalized)
    # Generic words like "origin"/"ingredients" alone cannot make a question
    # about the universe, a computer, etc. a culinary request.
    concrete_food = is_recommendation(text) or bool(re.search(r"\b(?:mon|am thuc|pho|bun|banh|com|thuc don|an uong|nau an|do uong|food|dish\w*|cuisine|recipe|menu|restaurant|cooking|nutrition)\b", normalized))
    explicit_foreign = bool(technical or (foreign_topic and not concrete_food))
    clear_question = bool(re.search(r"\b(?:la gi|ai la|tai sao|giai thich|cho toi biet|what is|who is|tell me|why)\b", normalized) or text.rstrip().endswith("?"))
    allergy_context = profile is not None and (profile.allergy_question_asked or profile.pending_allergy_status or profile.pending_allergy_note)
    unknown_question = clear_question and not is_food_question(text) and not allergen_keywords(text) and "di ung" not in normalized and not allergy_context
    if explicit_foreign or unknown_question:
        if profile is not None:
            profile.dialogue_action = "off_topic"
        return "I only answer questions related to food, cuisine and dining. What dish would you like advice about?" if english else "Mình chỉ trả lời những nội dung liên quan đến ẩm thực, món ăn và ăn uống. Bạn cần tư vấn món gì?"
    return None


def response_language(profile: CustomerProfile, text: str, asr_language: str) -> str:
    """Keep a short place/confirmation from changing the conversation language."""
    normalized = _normalized(text)
    if re.search(r"\b(?:tieng anh|speak english|in english)\b", normalized):
        profile.language_preference = "en"
    elif re.search(r"\b(?:tieng viet|speak vietnamese|in vietnamese)\b", normalized):
        profile.language_preference = "vi"
    words = re.findall(r"\w+", normalized)
    explicit_english_short = bool(re.search(
        r"\b(?:hello|hi|hey|good morning|thank you|thanks|goodbye)\b",
        normalized,
    ))
    ambiguous_short_reply = normalized.strip(" .!?,") in {
        "yeah", "yes", "yep", "yup", "no", "ok", "okay", "until"
    } or (
        asr_language == "en"
        and len(words) <= 2
        and not explicit_english_short
    )
    if profile.language_preference:
        chosen = profile.language_preference
    elif profile.response_language and len(words) <= 2:
        chosen = profile.response_language
    elif ambiguous_short_reply:
        # A one-word ASR guess is not enough evidence to switch the session.
        chosen = profile.response_language or "vi"
    elif profile.response_language and (is_confirmation(text) or is_rejection(text) or (regions_in_text(text) and not is_food_question(text))):
        chosen = profile.response_language
    elif re.search(r"\b(?:hello|hi|hey|good morning|recommend\w*|suggest\w*|breakfast|ingredients?|where|what|how|origin)\b", normalized):
        chosen = "en"
    elif re.search(r"\b(?:xin chao|xin giao|chao|goi y|gui y|mon|toi|minh|di ung|dung roi|khong|nguon goc|thanh phan)\b", normalized):
        chosen = "vi"
    elif profile.response_language:
        chosen = profile.response_language
    elif regions_in_text(text):
        chosen = "vi"
    else:
        chosen = asr_language if asr_language in {"vi", "en"} else "vi"
    profile.response_language = chosen
    return chosen


def _resolved_request(profile: CustomerProfile) -> str:
    request = profile.food_request
    if not profile.region or not is_recommendation(request):
        return request
    normalized = _normalized(request)
    aliases = [alias for region_aliases in REGION_ALIASES.values() for alias in region_aliases]
    aliases += list(LOCATION_HYPOTHESES)
    pattern = r"(?<!\w)(?:(?:o|tai|khu vuc|thanh pho|tinh)\s+)*(?:" + "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)) + r")(?!\w)"
    spans = list(re.finditer(pattern, normalized))
    for match in reversed(spans):
        request = request[:match.start()] + request[match.end():]
    request = re.sub(r"\s+", " ", request).strip(" .,?!")
    return f"{request} tại {profile.region}"


def prepare_turn(profile: CustomerProfile, text: str, language: str) -> tuple[str | None, str]:
    """Return a deterministic reply or a resolved query. Never silently fix an allergen."""
    text = unicodedata.normalize("NFC", text).strip()
    normalized = _normalized(text).strip(" .!?,")
    language = response_language(profile, text, language)
    profile.dialogue_action = ""
    direct_reply = courtesy_or_domain_reply(text, language, profile)
    if direct_reply is not None:
        return direct_reply, text
    if profile.awaiting_food_request and is_food_question(text) and len(normalized.split()) <= 3 and not allergen_keywords(text):
        # A dish name answering "what food do you want?" is a consultation,
        # not an arbitrary article lookup. Preserve the raw STT in outer logs.
        text = f"Tư vấn món {text}" if language != "en" else f"Recommend {text}"
        normalized = _normalized(text).strip(" .!?,")
    if is_food_question(text):
        profile.awaiting_food_request = False
    profile.update_from_text(text)
    locations = regions_in_text(text)
    if not profile.turn_consumed:
        if len(locations) == 1:
            profile.region = locations[0]
            profile.pending_region = ""
            profile.region_needs_clarification = False
            if not is_food_question(text):
                profile.turn_consumed = True
        elif len(locations) > 1:
            profile.region_needs_clarification = True
            profile.pending_region = ""
        elif profile.pending_region and is_confirmation(text):
            profile.region = profile.pending_region
            profile.pending_region = ""
            profile.region_needs_clarification = False
            profile.turn_consumed = True
        elif profile.pending_region and is_rejection(text):
            profile.pending_region = ""
            profile.region_needs_clarification = True
            profile.turn_consumed = True
        elif hypothesis := next((place for alias, place in LOCATION_HYPOTHESES.items() if _contains_phrase(normalized, alias)), ""):
            profile.pending_region = hypothesis
            profile.region_needs_clarification = True
        elif re.search(r"\b(?:o kem|o ca|kem nay|khu vuc thanh pho)\b", normalized):
            profile.region_needs_clarification = True
        elif is_recommendation(text) and re.search(r"\b(?:o|tai|khu vuc|thanh pho|tinh)\b", normalized):
            if not profile.region or not re.search(r"\b(?:o day|tai day|thanh pho nay|khu vuc nay)\b", normalized):
                profile.region_needs_clarification = True

        if is_food_question(text) and not profile.turn_consumed:
            # Meaningful new requests replace old requests; subtitles/noise do not.
            if not is_recommendation(text):
                profile.pending_region = ""
                profile.region_needs_clarification = False
            request = text
            if profile.region_needs_clarification and not locations:
                markers = list(LOCATION_MARKER.finditer(normalized))
                if markers:
                    request = text[:markers[0].start()].rstrip()
            profile.food_request = request

    english = language == "en"
    if profile.pending_allergy_status:
        labels = ", ".join(profile.pending_allergens)
        if profile.allergy_confirmation_unclear:
            profile.dialogue_action = "allergy_confirmation_unclear"
            return ("I could not hear your confirmation clearly. Please say correct or incorrect." if english else "Mình chưa nghe rõ phần xác nhận. Bạn hãy nói chính xác, hoặc không đúng."), text
        if profile.pending_allergy_status == "none":
            return ("To confirm, you have no known food allergies, correct?" if english else "Mình nghe bạn không có dị ứng thực phẩm đã biết, đúng không? Bạn hãy nói đúng rồi hoặc không đúng."), text
        profile.dialogue_action = "allergy_readback"
        return (f"To confirm, your complete allergy list is {labels}, correct?" if english else f"Mình nghe toàn bộ danh sách thành phần bạn dị ứng là {labels}, đúng không? Bạn hãy nói chính xác hoặc không đúng."), text
    if profile.allergy_name_missing:
        profile.dialogue_action = "allergy_name_missing"
        return (
            "Which food are you allergic to? Please repeat its name."
            if english
            else "Bạn dị ứng với thực phẩm nào? Bạn nói lại tên giúp mình nhé."
        ), text
    if profile.allergy_confirmation_rejected or profile.pending_allergy_note:
        profile.dialogue_action = "allergy_list_unclear"
        known = ", ".join(profile.pending_allergens)
        unknown = profile.pending_allergy_note[:100]

        # --- Escape hatch: after too many failed retries, stop looping ---
        if profile.allergy_unclear_attempts >= _MAX_ALLERGY_UNCLEAR_RETRIES:
            if profile.pending_allergens:
                # Commit the partial list we could understand
                profile.allergy_status = "declared"
                profile.allergens = list(profile.pending_allergens)
                profile.allergy_note = unknown  # keep the unclear part as a note
                profile.pending_allergy_status = ""
                profile.pending_allergens.clear()
                profile.pending_allergy_note = ""
                profile.allergy_unclear_attempts = 0
                profile.allergy_question_asked = False
                profile.allergy_updated_this_turn = True
                labels = profile.spoken_allergens()
                profile.dialogue_action = "allergy_partial_accepted"
                return (
                    f"I had difficulty hearing part of your list, but I have noted your allergy to {labels}. "
                    "Please confirm with staff about any other ingredients. What food would you like?"
                    if english else
                    f"Mình không nghe rõ được một phần, nhưng đã ghi nhận bạn dị ứng với {labels}. "
                    "Các thành phần chưa rõ bạn cần xác nhận thêm với nhân viên khi gọi món. Bạn muốn tìm món gì?"
                ), text
            else:
                # Heard nothing at all — skip allergy check to unblock the user
                profile.allergy_status = "unknown"
                profile.pending_allergy_note = ""
                profile.allergy_unclear_attempts = 0
                profile.allergy_question_asked = False
                profile.dialogue_action = "allergy_skipped"
                return (
                    "I was unable to hear your allergy information clearly. "
                    "Please inform restaurant staff directly about any food allergies. What food would you like?"
                    if english else
                    "Mình không nghe rõ thông tin dị ứng của bạn. "
                    "Bạn hãy trực tiếp thông báo với nhân viên nhà hàng về dị ứng trước khi gọi món. Bạn muốn tư vấn món gì?"
                ), text
        # --- Normal retry path ---
        if unknown:
            return (f"I could not identify this ingredient clearly: {unknown}. Please repeat your complete allergy list." if english else f"Mình chưa xác định rõ thành phần trong cụm {unknown}." + (f" Mình đã nghe được {known}." if known else "") + " Bạn hãy nói lại toàn bộ danh sách thành phần dị ứng."), text
        return ("Please repeat your complete allergy list." if english else "Mình chưa xác định rõ danh sách đúng. Bạn hãy nói lại các thành phần dị ứng, ví dụ hải sản và trứng."), text
    if profile.allergy_status == "unknown" and is_recommendation(profile.food_request if profile.turn_consumed else text):
        profile.dialogue_action = "allergy_question"
        profile.allergy_question_asked = True
        return ("Before I recommend food, do you have any food allergies, such as peanuts, milk, eggs or seafood?" if english else "Trước khi gợi ý món, bạn có dị ứng với thành phần nào không, ví dụ đậu phộng, sữa, trứng hoặc hải sản?"), text
    if profile.region_needs_clarification:
        if profile.pending_region:
            return (f"Did you mean {profile.pending_region}? Please confirm or name the location again." if english else f"Bạn muốn hỏi món ở {profile.pending_region}, đúng không? Nếu không đúng, bạn hãy nói lại tên địa điểm."), text
        return ("Which city or region do you mean?" if english else "Bạn muốn tìm món ở tỉnh hoặc thành phố nào? Mình chưa nghe rõ địa điểm."), text
    if profile.turn_consumed:
        if profile.food_request:
            return None, _resolved_request(profile)
        if not profile.allergy_updated_this_turn:
            return ("What dish would you like advice about?" if english else "Bạn cần mình tư vấn món ăn gì?"), text
        labels = profile.spoken_allergens()
        acknowledgement = (f"Allergy information confirmed: {labels}. What food would you like?" if english else f"Mình đã xác nhận thông tin dị ứng: {labels}. Bạn muốn tìm món gì?")
        if profile.allergy_status == "none":
            acknowledgement = "No known food allergies confirmed. What food would you like?" if english else "Mình đã xác nhận bạn không có dị ứng thực phẩm đã biết. Bạn muốn tìm món gì?"
        return acknowledgement, text
    if not is_food_question(text):
        profile.dialogue_action = "unclear_food_request"
        return ("Please repeat your food question or name the location." if english else "Mình chưa nghe rõ yêu cầu về món ăn. Bạn hãy nói lại câu hỏi hoặc tên địa điểm."), text
    if profile.region and is_recommendation(profile.food_request or text):
        return None, _resolved_request(profile)
    # A location-only clarification resumes the request even after history rolls over.
    if locations and len(normalized.split()) < 4 and profile.food_request:
        return None, _resolved_request(profile)
    return None, text
