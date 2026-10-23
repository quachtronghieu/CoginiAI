import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SUPPORTED_PROTOCOL_VERSION = 1
VALID_MODEL_DEVICES = {"auto", "cpu", "cuda"}
VALID_TTS_DEFAULT_LANGUAGES = {"vi", "en"}
VALID_RNNOISE_ASR_MODES = {"filtered", "raw", "shadow"}


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _probability(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return value


@dataclass(frozen=True)
class Settings:
    app_name: str
    host: str
    port: int
    public_ws_url: str
    device_token: str
    server_sample_rate: int
    xiaozhi_input_sample_rate: int
    frame_duration_ms: int
    timezone_offset_minutes: int
    asr_model: str
    asr_language: str
    asr_compute_type: str
    asr_local_files_only: bool
    language_hint_min_probability: float
    llm_base_url: str
    llm_model: str
    llm_num_predict: int
    llm_num_ctx: int
    llm_timeout_seconds: float
    ai_pipeline_timeout_seconds: float
    rnnoise_enabled: bool
    rag_enabled: bool
    rag_database: str
    rag_top_k: int
    rag_strict: bool
    tts_model: str
    tts_voice: str
    tts_model_vi: str
    tts_model_en: str
    tts_default_language: str
    tts_engine: str
    tts_voice_vi: str
    tts_voice_en: str
    model_device: str
    vad_enabled: bool
    vad_threshold: float
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    max_audio_seconds: int
    max_audio_bytes: int
    auto_stop_timeout_seconds: float
    hello_timeout_seconds: float
    asr_hotwords: str = (
        "xin chào, chào bạn, cảm ơn, tạm biệt, đúng rồi, chính xác, không đúng, sai rồi, đồng ý, lặp lại, "
        "tư vấn món, gợi ý món, ăn sáng, ăn trưa, ăn tối, ăn đêm, ăn vặt, món ngon, đặc sản, hương vị, nguyên liệu, "
        "gia vị, cách nấu, công thức, thực đơn, nhà hàng, quán ăn, dị ứng, dị ứng thực phẩm, dị ứng hải sản, "
        "dị ứng đậu phộng, dị ứng sữa, dị ứng trứng, không ăn được, kiêng khem, không chứa, không có, hải sản có vỏ, "
        "hạt cây, đậu phộng, lạc, trứng gà, trứng vịt, trứng cút, sữa tươi, phô mai, bơ sữa, tôm sú, tôm hùm, cua đồng, "
        "cua biển, cua Cà Mau, ghẹ rang me, cá lóc, cá kèo, cá hồi, cá bớp, mực nướng sa tế, bạch tuộc, nghêu hấp sả, "
        "sò huyết, ốc móng tay, hàu nướng mỡ hành, thịt bò, thịt heo, thịt gà, thịt vịt, thịt dê, sườn nướng, "
        "chân gà sả tắc, sả tắc, phở bò, phở gà, phở tái, phở nạm, phở gầu, phở sốt vang, phở cuốn, phở chua, "
        "bún chả, bún thang, bún riêu cua, bún ốc, bún đậu mắm tôm, bún cá rô, bún mọc, bún bò Huế, bún mắm, "
        "bún kèn, bún chả cá, bún thịt nướng, bún bì, cơm tấm, cơm tấm sườn bì chả, cơm lam, cơm gà Tam Kỳ, "
        "cơm gà Hội An, cơm hến, bánh mì kẹp thịt, bánh mì chảo, bánh mì cay, bánh xèo, bánh xèo củ hũ dừa, "
        "bánh khọt, bánh cuốn, bánh bèo, bánh nậm, bánh bột lọc, bánh tét, bánh chưng, bánh pía, bánh tầm cay, "
        "bánh tầm bì, bánh đa cua, bánh canh ghẹ, bánh canh chả cá, bánh canh cá lóc, hủ tiếu Nam Vang, "
        "hủ tiếu Sa Đéc, hủ tiếu Mỹ Tho, mì Quảng, cao lầu, miến lươn, miến măng gà, cháo lòng, cháo sườn, "
        "cháo ếch, xôi xéo, xôi khúc, xôi vò, gỏi cuốn, nem rán, nem lụi, chả ram tôm đất, chả cá Lã Vọng, "
        "chả mực Hạ Long, súp cua, lẩu thái, lẩu cua đồng, lẩu mắm, lẩu cá kèo, lẩu gà lá é, bò bít tết, bò kho, "
        "bò né, bò sốt vang, chè bưởi, chè thái, cà phê sữa đá, bạc xỉu, trà tắc, trà sen, nước mắm, mắm tôm, "
        "mắm nêm, mắm ruốc, củ hũ dừa, hoa chuối, rau răm, ngò gai, tía tô, lá é, bông điên điển, măng chua, sả, "
        "gừng, tỏi, ớt hiểm, tiêu lốt, hạt dổi, mắc khén, Hà Nội, Hải Phòng, Quảng Ninh, Lạng Sơn, Sa Pa, "
        "Lào Cai, Hà Giang, Cao Bằng, Ninh Bình, Thanh Hóa, Nghệ An, Hà Tĩnh, Huế, Đà Nẵng, Hội An, Quy Nhơn, "
        "Quảng Nam, Bình Định, Phú Yên, Nha Trang, Khánh Hòa, Phan Thiết, Bình Thuận, Đà Lạt, Lâm Đồng, "
        "Buôn Ma Thuột, Đắk Lắk, TP Hồ Chí Minh, Sài Gòn, Vũng Tàu, Tây Ninh, Đồng Nai, Bình Dương, Long An, "
        "Tiền Giang, Mỹ Tho, Bến Tre, Đồng Tháp, Sa Đéc, Vĩnh Long, Trà Vinh, Cần Thơ, Hậu Giang, Sóc Trăng, "
        "Bạc Liêu, Cà Mau, An Giang, Châu Đốc, Long Xuyên, Kiên Giang, Rạch Giá, Phú Quốc, "
        "ở Đà Lạt, tại Đà Lạt, món ăn Đà Lạt, đặc sản Đà Lạt, ở Cà Mau, tại Cà Mau, món ăn Cà Mau, đặc sản Cà Mau"
    )
    asr_min_rms: float = 0.0015
    asr_min_duration_seconds: float = 0.15
    diagnostics_enabled: bool = False
    diagnostics_slots: int = 24
    rnnoise_asr_mode: str = "shadow"
    vad_strong_speech_ms: int = 180
    asr_no_speech_gate_probability: float = 0.70
    verified_menu_path: str = ""
    verified_menu_max_age_days: int = 30


@lru_cache
def get_settings() -> Settings:
    model_device = os.getenv("MODEL_DEVICE", "auto").lower()
    if model_device not in VALID_MODEL_DEVICES:
        allowed = ", ".join(sorted(VALID_MODEL_DEVICES))
        raise ValueError(f"MODEL_DEVICE must be one of: {allowed}")
    tts_default_language = os.getenv("TTS_DEFAULT_LANGUAGE", "vi").lower().strip()
    if tts_default_language not in VALID_TTS_DEFAULT_LANGUAGES:
        raise ValueError("TTS_DEFAULT_LANGUAGE must be one of: en, vi")
    rnnoise_asr_mode = os.getenv("RNNOISE_ASR_MODE", "shadow").lower().strip()
    if rnnoise_asr_mode not in VALID_RNNOISE_ASR_MODES:
        raise ValueError("RNNOISE_ASR_MODE must be one of: filtered, raw, shadow")

    return Settings(
        app_name=os.getenv("APP_NAME", "Xiaozhi Backend"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        public_ws_url=os.getenv("PUBLIC_WS_URL", "").strip(),
        device_token=os.getenv("XIAOZHI_DEVICE_TOKEN", ""),
        server_sample_rate=_positive_int("XIAOZHI_SERVER_SAMPLE_RATE", 24000),
        xiaozhi_input_sample_rate=_positive_int("XIAOZHI_INPUT_SAMPLE_RATE", 16000),
        frame_duration_ms=_positive_int("XIAOZHI_FRAME_DURATION_MS", 60),
        timezone_offset_minutes=int(os.getenv("TIMEZONE_OFFSET_MINUTES", "420")),
        asr_model=os.getenv("ASR_MODEL", "").strip(),
        asr_language=os.getenv("ASR_LANGUAGE", "vi").strip(),
        asr_compute_type=os.getenv("ASR_COMPUTE_TYPE", "").strip(),
        asr_local_files_only=os.getenv("ASR_LOCAL_FILES_ONLY", "true").lower()
        in {"1", "true", "yes", "on"},
        language_hint_min_probability=_probability(
            "LANGUAGE_HINT_MIN_PROBABILITY", 0.80
        ),
        llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
        llm_num_predict=_positive_int("LLM_NUM_PREDICT", 4096),
        llm_num_ctx=_positive_int("LLM_NUM_CTX", 8192),
        llm_timeout_seconds=_positive_float("LLM_TIMEOUT_SECONDS", 90.0),
        ai_pipeline_timeout_seconds=_positive_float(
            "AI_PIPELINE_TIMEOUT_SECONDS", 120.0
        ),
        rnnoise_enabled=os.getenv("RNNOISE_ENABLED", "false").lower()
        in {"1", "true", "yes", "on"},
        rag_enabled=os.getenv("RAG_ENABLED", "false").lower()
        in {"1", "true", "yes", "on"},
        rag_database=os.getenv("RAG_DB_PATH", "").strip(),
        rag_top_k=_positive_int("RAG_TOP_K", 3),
        rag_strict=os.getenv("RAG_STRICT", "true").lower()
        in {"1", "true", "yes", "on"},
        tts_model=os.getenv("TTS_MODEL", "").strip(),
        tts_voice=os.getenv("TTS_VOICE", "").strip(),
        tts_model_vi=os.getenv("TTS_MODEL_VI", "").strip(),
        tts_model_en=os.getenv("TTS_MODEL_EN", "").strip(),
        tts_default_language=tts_default_language,
        tts_engine=os.getenv("TTS_ENGINE", "edge").lower().strip(),
        tts_voice_vi=os.getenv("TTS_VOICE_VI", "vi-VN-HoaiMyNeural").strip(),
        tts_voice_en=os.getenv("TTS_VOICE_EN", "en-US-JennyNeural").strip(),
        model_device=model_device,
        vad_enabled=os.getenv("VAD_ENABLED", "false").lower()
        in {"1", "true", "yes", "on"},
        vad_threshold=float(os.getenv("VAD_THRESHOLD", "0.38")),
        vad_min_silence_ms=_positive_int("VAD_MIN_SILENCE_MS", 500),
        vad_speech_pad_ms=_positive_int("VAD_SPEECH_PAD_MS", 200),
        max_audio_seconds=_positive_int("MAX_AUDIO_SECONDS", 30),
        max_audio_bytes=_positive_int("MAX_AUDIO_BYTES", 2 * 1024 * 1024),
        auto_stop_timeout_seconds=_positive_float("AUTO_STOP_TIMEOUT_SECONDS", 4.0),
        hello_timeout_seconds=_positive_float("HELLO_TIMEOUT_SECONDS", 10.0),
        asr_hotwords=os.getenv("ASR_HOTWORDS", Settings.asr_hotwords).strip(),
        asr_min_rms=_positive_float("ASR_MIN_RMS", 0.0015),
        asr_min_duration_seconds=_positive_float("ASR_MIN_DURATION_SECONDS", 0.15),
        diagnostics_enabled=os.getenv("VOICE_DIAGNOSTICS_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
        diagnostics_slots=_positive_int("VOICE_DIAGNOSTICS_SLOTS", 24),
        rnnoise_asr_mode=rnnoise_asr_mode,
        vad_strong_speech_ms=_positive_int("VAD_STRONG_SPEECH_MS", 180),
        asr_no_speech_gate_probability=_probability("ASR_NO_SPEECH_GATE_PROBABILITY", 0.70),
        verified_menu_path=os.getenv("VERIFIED_MENU_PATH", "").strip(),
        verified_menu_max_age_days=_positive_int("VERIFIED_MENU_MAX_AGE_DAYS", 30),
    )
