import numpy as np

from memory_engine.models import Memory, MemoryType
from memory_engine.storage.db import Database
from memory_engine.storage.repository import MemoryRepository


def make_memory(
    memory_id: str = "memory-1",
    text: str = "User uses Arch Linux.",
    embedding: list[float] | None = None,
) -> Memory:
    return Memory(
        id=memory_id,
        text=text,
        type=MemoryType.FACT,
        importance=0.7,
        source_message="I use Arch Linux.",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        embedding=embedding,
    )


def test_database_uses_wal_mode(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()

    assert database.journal_mode() == "wal"


def test_memory_can_be_saved_and_loaded(tmp_path):
    database = Database(tmp_path / "test.db")
    repository = MemoryRepository(database)

    original = make_memory()
    repository.save(original)

    loaded = repository.get_by_id("memory-1")

    assert loaded is not None
    assert loaded.id == original.id
    assert loaded.text == original.text
    assert loaded.type == MemoryType.FACT
    assert loaded.is_active is True


def test_embedding_is_persisted_as_float32_data(tmp_path):
    database = Database(tmp_path / "test.db")
    repository = MemoryRepository(database)

    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    repository.save(make_memory(embedding=vector.tolist()))

    loaded = repository.get_by_id("memory-1")

    assert loaded is not None
    assert loaded.embedding is not None
    assert np.allclose(loaded.embedding, vector, atol=1e-6)


def test_memory_survives_repository_recreation(tmp_path):
    database_path = tmp_path / "test.db"

    first_repository = MemoryRepository(Database(database_path))
    first_repository.save(make_memory())

    second_repository = MemoryRepository(Database(database_path))
    loaded = second_repository.get_by_id("memory-1")

    assert loaded is not None
    assert loaded.text == "User uses Arch Linux."


def test_replacement_is_non_destructive(tmp_path):
    database = Database(tmp_path / "test.db")
    repository = MemoryRepository(database)

    old_memory = make_memory(
        memory_id="old-memory",
        text="User's project deadline is October 15.",
    )
    new_memory = make_memory(
        memory_id="new-memory",
        text="User's project deadline is November 2.",
    )

    repository.save(old_memory)
    repository.save(new_memory)
    repository.mark_as_replaced("old-memory", "new-memory")

    active_memories = repository.list_memories(active_only=True)
    all_memories = repository.list_memories(active_only=False)

    assert len(active_memories) == 1
    assert active_memories[0].id == "new-memory"

    assert len(all_memories) == 2

    old_loaded = repository.get_by_id("old-memory")
    assert old_loaded is not None
    assert old_loaded.is_active is False
    assert old_loaded.replaced_by == "new-memory"


def test_active_vector_matrix_contains_only_active_vectors(tmp_path):
    database = Database(tmp_path / "test.db")
    repository = MemoryRepository(database)

    repository.save(
        make_memory(
            memory_id="active-memory",
            embedding=[1.0, 0.0, 0.0],
        )
    )
    repository.save(
        make_memory(
            memory_id="inactive-memory",
            text="Old memory.",
            embedding=[0.0, 1.0, 0.0],
        )
    )
    repository.mark_as_replaced("inactive-memory", "active-memory")

    memories, matrix = repository.get_active_vector_matrix()

    assert [memory.id for memory in memories] == ["active-memory"]
    assert matrix.shape == (1, 3)
    assert matrix.dtype == np.float32
