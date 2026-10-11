from dataclasses import dataclass

import numpy as np

from memory_engine.models import Memory
from memory_engine.retrieval.embedder import LocalEmbedder


@dataclass
class SearchResult:
    memory: Memory
    score: float


class DenseMemorySearch:
    def __init__(self, embedder: LocalEmbedder) -> None:
        self.embedder = embedder

    def search(
        self,
        query: str,
        memories: list[Memory],
        top_k: int = 5,
    ) -> list[SearchResult]:
        memories_with_embeddings = [
            memory
            for memory in memories
            if memory.is_active and memory.embedding is not None
        ]

        if not memories_with_embeddings:
            return []

        query_vector = self.embedder.encode(query)

        memory_matrix = np.ascontiguousarray(
            np.asarray(
                [
                    memory.embedding
                    for memory in memories_with_embeddings
                ],
                dtype=np.float32,
            )
        )

        scores = np.dot(memory_matrix, query_vector)

        ranked_indexes = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                memory=memories_with_embeddings[index],
                score=float(scores[index]),
            )
            for index in ranked_indexes
        ]
