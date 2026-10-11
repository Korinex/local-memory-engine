import re
from dataclasses import dataclass

from memory_engine.models import MemoryType


@dataclass
class ExtractedMemory:
    text: str
    memory_type: MemoryType
    importance: float
    source_message: str


SMALL_TALK_MESSAGES = {
    "hello",
    "hi",
    "hey",
    "ok",
    "okay",
    "ok thanks",
    "thanks",
    "thank you",
    "cool",
    "got it",
    "sounds good",
    "great",
    "nice",
}


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def is_small_talk(text: str) -> bool:
    normalized = normalize_text(text)

    if normalized in SMALL_TALK_MESSAGES:
        return True

    return len(normalized.split()) <= 2 and normalized in {
        "sure",
        "alright",
        "understood",
        "yep",
        "yes",
        "no",
    }


def classify_memory(text: str) -> MemoryType:
    lowered = text.lower()

    if any(
        word in lowered
        for word in [
            "deadline",
            "goal",
            "target",
            "want to",
            "wants to",
            "plan to",
        ]
    ):
        return MemoryType.GOAL

    if any(
        word in lowered
        for word in ["prefer", "like", "favourite", "favorite", "enjoy"]
    ):
        return MemoryType.PREFERENCE

    if any(
        word in lowered
        for word in ["building", "project", "application", "app", "website"]
    ):
        return MemoryType.PROJECT

    if any(
        word in lowered
        for word in ["meeting", "moved", "happened", "yesterday", "today"]
    ):
        return MemoryType.EVENT

    return MemoryType.FACT


def calculate_importance(text: str) -> float:
    lowered = text.lower()

    if any(
        word in lowered
        for word in ["deadline", "goal", "target"]
    ):
        return 0.90

    if any(
        word in lowered
        for word in [
            "python",
            "javascript",
            "typescript",
            "linux",
            "postgresql",
            "sqlite",
            "next.js",
            "flutter",
        ]
    ):
        return 0.85

    if any(
        word in lowered
        for word in ["prefer", "like", "favourite", "favorite", "enjoy"]
    ):
        return 0.80

    return 0.70


def convert_to_third_person(message: str) -> str:
    text = message.strip()
    text = re.sub(
        r"^Actually,\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    replacements = [
        (
            r"^(?:the )?deadline (?:moved|changed) to (.+)$",
            r"User's project deadline is \1",
        ),
        (r"^I am building (.+)$", r"User is building \1"),
        (r"^I'm building (.+)$", r"User is building \1"),
        (r"^I want to learn (.+)$", r"User wants to learn \1"),
        (r"^I attended (.+)$", r"User attended \1"),
        (r"^I moved to (.+)$", r"User moved to \1"),
        (r"^I switched to (.+)$", r"User switched to \1"),
        (r"^I use (.+)$", r"User uses \1"),
        (r"^I prefer (.+)$", r"User prefers \1"),
        (r"^I like (.+)$", r"User likes \1"),
        (r"^I live in (.+)$", r"User lives in \1"),
        (r"^My (.+)$", r"User's \1"),
        (r"^I'm (.+)$", r"User is \1"),
        (r"^I am (.+)$", r"User is \1"),
    ]

    for pattern, replacement in replacements:
        converted = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

        if converted != text:
            return converted.rstrip(" .") + "."

    if text and text[-1] not in ".!?":
        return text + "."

    return text


def extract(message: str) -> ExtractedMemory | None:
    if is_small_talk(message):
        return None

    canonical_text = convert_to_third_person(message)
    memory_type = classify_memory(canonical_text)
    importance = calculate_importance(canonical_text)

    return ExtractedMemory(
        text=canonical_text,
        memory_type=memory_type,
        importance=importance,
        source_message=message,
    )
