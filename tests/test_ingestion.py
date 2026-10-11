import pytest
from memory_engine.ingestion.extractor import (
    calculate_importance,
    classify_memory,
    convert_to_third_person,
    extract,
    is_small_talk,
)
from memory_engine.models import MemoryType
from memory_engine.engine import LocalMemoryEngine

def test_common_small_talk_is_rejected():
    assert is_small_talk("hello") is True
    assert is_small_talk("OK THANKS") is True
    assert is_small_talk("sounds good") is True


def test_real_message_is_not_small_talk():
    assert is_small_talk("I use Arch Linux.") is False


def test_first_person_linux_statement_becomes_third_person():
    result = extract("I use Arch Linux.")

    assert result is not None
    assert result.text == "User uses Arch Linux."
    assert result.memory_type == MemoryType.FACT


def test_preference_is_classified():
    result = extract("I prefer PostgreSQL.")

    assert result is not None
    assert result.text == "User prefers PostgreSQL."
    assert result.memory_type == MemoryType.PREFERENCE
    assert result.importance == 0.85


def test_project_is_classified():
    result = extract("I'm building a Next.js app.")

    assert result is not None
    assert result.memory_type == MemoryType.PROJECT


def test_deadline_is_classified_as_goal():
    result = extract("My project deadline is October 15.")

    assert result is not None
    assert result.memory_type == MemoryType.GOAL
    assert result.importance == 0.90


def test_small_talk_returns_no_memory():
    assert extract("cool") is None


def test_importance_stays_in_valid_range():
    messages = [
        "I use Arch Linux.",
        "I prefer PostgreSQL.",
        "My project deadline is October 15.",
        "I enjoy coffee.",
    ]

    for message in messages:
        result = extract(message)

        assert result is not None
        assert 0.0 <= result.importance <= 1.0

def test_new_information_replaces_old_information(tmp_path):
    engine = LocalMemoryEngine(
        database_path=str(tmp_path / "memory.db")
    )

    first = engine.ingest(
        "My project deadline is October 15."
    )
    second = engine.ingest(
        "Actually, the deadline moved to November 2."
    )

    assert first.status == "stored"
    assert second.status == "updated"

    active_memories = engine.list_memories(active_only=True)
    all_memories = engine.list_memories(active_only=False)

    assert len(active_memories) == 1
    assert len(all_memories) == 2
    assert "November 2" in active_memories[0].text


def test_small_talk_is_not_saved(tmp_path):
    engine = LocalMemoryEngine(
        database_path=str(tmp_path / "memory.db")
    )

    result = engine.ingest("ok thanks")

    assert result.status == "ignored"
    assert engine.list_memories() == []