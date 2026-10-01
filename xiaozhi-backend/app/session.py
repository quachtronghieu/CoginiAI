import logging
import math
from collections.abc import Callable
from enum import StrEnum
from time import monotonic

from .allergy import CustomerProfile
from .vad import VadService


logger = logging.getLogger(__name__)


class FinalizeReason(StrEnum):
    MANUAL_STOP = "manual_stop"
    AUTO_TIMEOUT = "auto_timeout"
    BUFFER_LIMIT = "buffer_limit"


class AudioSession:
    def __init__(
        self,
        session_id: str,
        device_id: str,
        client_id: str,
        frame_duration_ms: int,
        max_audio_seconds: int,
        max_audio_bytes: int,
        vad_service: VadService,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.session_id = session_id
        self.device_id = device_id
        self.client_id = client_id
        self.frame_duration_ms = frame_duration_ms
        self.max_frames = max(1, math.ceil(max_audio_seconds * 1000 / frame_duration_ms))
        self.max_audio_bytes = max_audio_bytes
        self.vad_service = vad_service
        self.clock = clock
        self.mode = "auto"
        self.active = False
        self._started_at = 0.0
        self._frames: list[bytes] = []
        self._total_bytes = 0
        self.persona: str = "default"
        self.history: list[dict[str, str]] = []
        self.customer_profile = CustomerProfile()
        self.last_utterance_had_speech = True
        self.last_vad_evidence: dict = {"available": False}

    def update_customer_profile(self, text: str) -> bool:
        updated = self.customer_profile.update_from_text(text)
        if updated:
            logger.info(
                "Session %s allergy profile updated: status=%s allergens=%s",
                self.session_id,
                self.customer_profile.allergy_status,
                self.customer_profile.allergens,
            )
        return updated

    def update_persona(self, text: str) -> bool:
        """Detect if the user wants to switch the AI persona/role. Returns True if changed."""
        import re

        lowered = text.lower().strip()

        # Reset to default culinary assistant
        if re.search(r"(trở lại|quay lại|về lại|như cũ|bình thường|hủy vai|trợ lý mặc định)", lowered):
            if self.persona != "default":
                self.persona = "default"
                logger.info("Session %s culinary persona reset to default", self.session_id)
                return True
            return False

        # Foodie friend (bạn thân sành ăn)
        if re.search(r"(bạn thân|người bạn thân|tri kỷ|cạ cứng)", lowered) and re.search(
            r"(là|thành|đóng vai|hóa thân|làm|trở thành)", lowered
        ):
            self.persona = "friend"
            logger.info("Session %s culinary persona switched to: friend", self.session_id)
            return True

        # Chef / Masterchef (đầu bếp / bếp trưởng)
        if re.search(r"(đầu bếp|bếp trưởng|chef|người nấu ăn)", lowered) and re.search(
            r"(là|thành|đóng vai|hóa thân|làm|trở thành)", lowered
        ):
            self.persona = "chef"
            logger.info("Session %s culinary persona switched to: chef", self.session_id)
            return True

        # Culinary nutritionist (chuyên gia dinh dưỡng ẩm thực)
        if re.search(r"(dinh dưỡng|chuyên gia dinh dưỡng|bác sĩ dinh dưỡng)", lowered) and re.search(
            r"(là|thành|đóng vai|hóa thân|làm|trở thành)", lowered
        ):
            self.persona = "nutritionist"
            logger.info("Session %s culinary persona switched to: nutritionist", self.session_id)
            return True

        # Clear history / change topic
        if re.search(r"(xóa lịch sử|đổi chủ đề|bắt đầu lại|quên đi|nói chuyện khác|món khác)", lowered):
            self.history.clear()
            # Changing dishes must never silently forget safety information.
            # Only explicit full-reset commands require allergy onboarding again.
            if re.search(r"(xóa lịch sử|bắt đầu lại|quên đi)", lowered):
                self.customer_profile.reset()
            logger.info("Session %s history cleared by user command", self.session_id)
            return True

        return False

    def add_history(self, user_text: str, assistant_text: str, max_turns: int = 2) -> None:
        """Append user and assistant utterances to conversation history with deduplication."""
        u = user_text.strip()
        a = assistant_text.strip()
        if not u or not a:
            return
        # If assistant repeated the exact same response as last turn, do not duplicate in history
        if self.history and self.history[-1].get("role") == "assistant" and self.history[-1].get("content") == a:
            return
        self.history.append({"role": "user", "content": u})
        self.history.append({"role": "assistant", "content": a})
        max_items = max_turns * 2
        if len(self.history) > max_items:
            self.history = self.history[-max_items:]

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def elapsed_seconds(self) -> float:
        if not self.active:
            return 0.0
        return max(0.0, self.clock() - self._started_at)

    def start(self, mode: str, preserve_audio: bool = False) -> None:
        if not preserve_audio:
            self._clear_buffer()
            self.vad_service.reset()
        self.mode = mode
        self.active = True
        self._started_at = self.clock()

    def append_frame(self, frame: bytes) -> FinalizeReason | None:
        if not self.active or not frame:
            return None

        if self.frame_count + 1 > self.max_frames or self.total_bytes + len(frame) > self.max_audio_bytes:
            logger.warning(
                "Audio buffer limit reached for session %s: frames=%d bytes=%d",
                self.session_id,
                self.frame_count,
                self.total_bytes,
            )
            return FinalizeReason.BUFFER_LIMIT

        self._frames.append(frame)
        self._total_bytes += len(frame)

        elapsed_seconds = self.clock() - self._started_at
        # Feed every listening mode to VAD so a later manual stop can still be
        # rejected as non-speech. Only auto mode lets VAD end the utterance.
        vad_should_finalize = self.vad_service.should_finalize(frame, elapsed_seconds)
        if self.mode == "auto" and vad_should_finalize:
            return FinalizeReason.AUTO_TIMEOUT
        return None

    def take_audio(self) -> list[bytes]:
        self.last_utterance_had_speech = bool(
            getattr(self.vad_service, "speech_detected", True)
        )
        self.last_vad_evidence = dict(
            getattr(self.vad_service, "evidence", {"available": False})
        )
        frames = self._frames
        self._frames = []
        self._total_bytes = 0
        self.active = False
        self._started_at = 0.0
        self.vad_service.reset()
        return frames

    def abort(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._clear_buffer()
        self.active = False
        self._started_at = 0.0
        self.vad_service.reset()

    def close(self) -> None:
        self._clear_buffer()
        self.active = False
        self._started_at = 0.0
        self.vad_service.close()

    def _clear_buffer(self) -> None:
        self._frames.clear()
        self._total_bytes = 0
