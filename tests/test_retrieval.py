import numpy as np

from memory_engine.models import Memory, MemoryType
from memory_engine.retrieval.search import DenseMemorySearch


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.vectors[text], dtype=np.float32)


def make_memory(
    memory_id: str,
    text: str,
    embedding: list[float],
    is_active: bool = True,
) -> Memory:
    return Memory(
        id=memory_id,
        text=text,
        type=MemoryType.FACT,
        importance=0.7,
        source_message=text,
        is_active=is_active,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        embedding=embedding,
    )


def test_search_returns_most_similar_memory_first():
    memories = [
        make_memory(
            memory_id="linux",
            text="User uses Arch Linux.",
            embedding=[1.0, 0.0, 0.0],
        ),
        make_memory(
            memory_id="database",
            text="User prefers PostgreSQL.",
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    embedder = FakeEmbedder(
        {
            "What operating system does the user use?": [1.0, 0.0, 0.0],
        }
    )

    search = DenseMemorySearch(embedder)
    results = search.search(
        "What operating system does the user use?",
        memories,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].memory.id == "linux"
    assert results[0].score == 1.0


def test_search_respects_top_k():
    memories = [
        make_memory(
            memory_id="one",
            text="Memory one",
            embedding=[1.0, 0.0, 0.0],
        ),
        make_memory(
            memory_id="two",
            text="Memory two",
            embedding=[0.9, 0.1, 0.0],
        ),
        make_memory(
            memory_id="three",
            text="Memory three",
            embedding=[0.0, 1.0, 0.0],
        ),
    ]

    embedder = FakeEmbedder(
        {
            "query": [1.0, 0.0, 0.0],
        }
    )

    search = DenseMemorySearch(embedder)
    results = search.search("query", memories, top_k=1)

    assert len(results) == 1
    assert results[0].memory.id == "one"


def test_search_ignores_inactive_memories():
    memories = [
        make_memory(
            memory_id="old",
            text="Old memory",
            embedding=[1.0, 0.0, 0.0],
            is_active=False,
        ),
        make_memory(
            memory_id="active",
            text="Active memory",
            embedding=[0.0, 1.0, 0.0],
            is_active=True,
        ),
    ]

    embedder = FakeEmbedder(
        {
            "query": [1.0, 0.0, 0.0],
        }
    )

    search = DenseMemorySearch(embedder)
    results = search.search("query", memories)

    assert len(results) == 1
    assert results[0].memory.id == "active"


def test_search_returns_empty_when_no_embeddings_exist():
    memories = [
        make_memory(
            memory_id="without-vector",
            text="Memory without vector",
            embedding=None,
        )
    ]

    embedder = FakeEmbedder({"query": [1.0, 0.0, 0.0]})
    search = DenseMemorySearch(embedder)

    results = search.search("query", memories)

    assert results == []
