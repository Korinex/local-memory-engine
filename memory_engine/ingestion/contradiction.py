from dataclasses import dataclass

from memory_engine.models import Memory


@dataclass
class UpdateMatch:
    old_memory: Memory | None
    action: str


TOPIC_WORDS = {
    "deadline": {"deadline", "due", "target date"},
    "operating_system": {"linux", "windows", "macos", "ubuntu", "arch"},
    "database": {"postgresql", "postgres", "mysql", "sqlite", "mongodb"},
    "editor": {"vscode", "visual studio code", "vim", "emacs"},
    "location": {"live", "lives", "located", "berlin", "amsterdam"},
    "framework": {"next.js", "nextjs", "flutter", "react", "django"},
}


def find_topic(text: str) -> str | None:
    lowered = text.lower()

    for topic, words in TOPIC_WORDS.items():
        if any(word in lowered for word in words):
            return topic

    return None


def is_duplicate(new_text: str, old_text: str) -> bool:
    new_words = set(new_text.lower().split())
    old_words = set(old_text.lower().split())

    if not new_words or not old_words:
        return False

    overlap = len(new_words & old_words) / max(
        len(new_words | old_words),
        1,
    )

    return overlap >= 0.80


def has_update_language(text: str) -> bool:
    lowered = text.lower()

    update_phrases = [
        "actually",
        "instead",
        "now",
        "changed to",
        "moved to",
        "switched to",
        "no longer",
        "not anymore",
    ]

    return any(phrase in lowered for phrase in update_phrases)


def find_update(
    new_memory: Memory,
    active_memories: list[Memory],
) -> UpdateMatch:
    new_topic = find_topic(new_memory.text)

    if new_topic is None:
        return UpdateMatch(old_memory=None, action="new")

    for old_memory in reversed(active_memories):
        old_topic = find_topic(old_memory.text)

        if old_topic != new_topic:
            continue

        if is_duplicate(new_memory.text, old_memory.text):
            return UpdateMatch(
                old_memory=old_memory,
                action="duplicate",
            )

        if has_update_language(new_memory.source_message):
            return UpdateMatch(
                old_memory=old_memory,
                action="replace",
            )

        return UpdateMatch(
            old_memory=old_memory,
            action="replace",
        )

    return UpdateMatch(old_memory=None, action="new")
