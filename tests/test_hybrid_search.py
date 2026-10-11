import numpy as np

from memory_engine.models import Memory, MemoryType
from memory_engine.retrieval.hybrid_search import HybridMemorySearch


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.vectors[text], dtype=np.float32)


def make_memory(
    memory_id: str,
    text: str,
    embedding: list[float],
    importance: float = 0.7,
) -> Memory:
    return Memory(
        id=memory_id,
        text=text,
        type=MemoryType.FACT,
        importance=importance,
        source_message=text,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        embedding=embedding,
    )


def test_keyword_match_can_return_a_memory():
    memories = [
        make_memory(
            "database",
            "User prefers PostgreSQL.",
            [0.2, 0.2, 0.2],
        ),
        make_memory(
            "os",
            "User uses Arch Linux.",
            [0.1, 0.1, 0.1],
        ),
    ]

    embedder = FakeEmbedder(
        {
            "Which database does the user prefer?": [0.2, 0.2, 0.2],
        }
    )

    search = HybridMemorySearch(
        embedder=embedder,
        threshold=0.1,
    )

    results = search.search(
        "Which database does the user prefer?",
        memories,
    )

    assert results
    assert results[0].memory.id == "database"


def test_low_scores_are_filtered_out():
    memories = [
        make_memory(
            "unrelated",
            "User enjoys hiking.",
            [0.1, 0.1, 0.1],
        )
    ]

    embedder = FakeEmbedder(
        {
            "What is the user's favourite database?": [0.0, 0.0, 0.0],
        }
    )

    search = HybridMemorySearch(
        embedder=embedder,
        threshold=0.45,
    )

    results = search.search(
        "What is the user's favourite database?",
        memories,
    )

    assert results == []


def test_low_score_relevant_project_is_returned():
    memory = make_memory(
        "project",
        "User is building a Next.js app.",
        [1.0, 0.0],
    )
    memory.type = MemoryType.PROJECT
    embedder = FakeEmbedder(
        {
            "What project is the user building?": [0.4, 0.0],
        }
    )
    search = HybridMemorySearch(embedder=embedder, threshold=0.45)

    results = search.search(
        "What project is the user building?",
        [memory],
    )

    assert [result.memory.id for result in results] == ["project"]


def test_low_score_unknown_food_question_returns_no_memory():
    memory = make_memory(
        "os",
        "User uses Arch Linux.",
        [1.0, 0.0],
    )
    embedder = FakeEmbedder(
        {
            "What is the user's favourite food?": [0.4, 0.0],
        }
    )
    search = HybridMemorySearch(embedder=embedder, threshold=0.45)

    results = search.search(
        "What is the user's favourite food?",
        [memory],
    )

    assert results == []


def test_high_score_unknown_car_question_returns_no_memory():
    memory = make_memory(
        "os",
        "User uses Arch Linux.",
        [1.0, 0.0],
    )
    embedder = FakeEmbedder(
        {
            "What car does the user drive?": [0.8, 0.0],
        }
    )
    search = HybridMemorySearch(embedder=embedder, threshold=0.45)

    results = search.search(
        "What car does the user drive?",
        [memory],
    )

    assert results == []


def test_high_score_known_food_matches_food_question():
    memory = make_memory(
        "food",
        "User likes pizza.",
        [1.0, 0.0],
    )
    embedder = FakeEmbedder(
        {
            "What is the user's favourite food?": [0.8, 0.0],
        }
    )
    search = HybridMemorySearch(embedder=embedder, threshold=0.45)

    results = search.search(
        "What is the user's favourite food?",
        [memory],
    )

    assert [result.memory.id for result in results] == ["food"]


def test_top_k_is_respected():
    memories = [
        make_memory("one", "User uses Linux.", [1.0, 0.0, 0.0]),
        make_memory("two", "User uses Python.", [0.9, 0.0, 0.0]),
        make_memory("three", "User uses SQLite.", [0.8, 0.0, 0.0]),
    ]

    embedder = FakeEmbedder(
        {
            "user technology": [1.0, 0.0, 0.0],
        }
    )

    search = HybridMemorySearch(
        embedder=embedder,
        threshold=0.1,
    )

    results = search.search(
        "user technology",
        memories,
        top_k=2,
    )

    assert len(results) == 2


def test_importance_can_break_a_close_tie():
    memories = [
        make_memory(
            "important",
            "User has an important project.",
            [1.0, 0.0, 0.0],
            importance=1.0,
        ),
        make_memory(
            "ordinary",
            "User has an ordinary project.",
            [1.0, 0.0, 0.0],
            importance=0.1,
        ),
    ]

    embedder = FakeEmbedder(
        {
            "project": [1.0, 0.0, 0.0],
        }
    )

    search = HybridMemorySearch(
        embedder=embedder,
        threshold=0.1,
    )

    results = search.search("project", memories)

    assert results[0].memory.id == "important"
