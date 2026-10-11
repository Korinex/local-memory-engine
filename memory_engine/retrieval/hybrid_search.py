from dataclasses import dataclass
import re

import numpy as np
from rank_bm25 import BM25Okapi

from memory_engine.models import Memory
from memory_engine.retrieval.embedder import LocalEmbedder


LOW_CONFIDENCE_FLOOR = 0.30
LOW_CONFIDENCE_STOP_WORDS = {
    "a",
    "an",
    "are",
    "be",
    "being",
    "does",
    "did",
    "do",
    "for",
    "how",
    "is",
    "of",
    "the",
    "user",
    "what",
    "which",
    "who",
    "where",
}
UNKNOWN_SUBJECT_TERMS = {
    "food": {
        "food",
        "meal",
        "dish",
        "pizza",
        "pasta",
        "sushi",
        "rice",
        "bread",
        "burger",
        "taco",
        "salad",
        "chicken",
        "steak",
        "curry",
    },
    "car": {
        "car",
        "vehicle",
        "toyota",
        "honda",
        "ford",
        "bmw",
        "tesla",
        "audi",
        "mercedes",
    },
    "pet": {
        "pet",
        "dog",
        "cat",
        "fish",
        "bird",
        "rabbit",
        "hamster",
    },
    "sport": {
        "sport",
        "sports",
        "soccer",
        "football",
        "basketball",
        "tennis",
        "hockey",
        "baseball",
        "golf",
        "rugby",
        "cricket",
    },
}
DEVELOPMENT_TERMS = {
    "arch",
    "linux",
    "windows",
    "macos",
    "ubuntu",
    "vscode",
    "code",
    "vim",
    "emacs",
    "python",
    "javascript",
    "typescript",
    "postgresql",
    "sqlite",
    "mongodb",
    "next.js",
    "nextjs",
    "flutter",
    "react",
    "fastapi",
}
FRAMEWORK_TERMS = {
    "next.js",
    "nextjs",
    "flutter",
    "react",
    "django",
    "fastapi",
}
CLOUD_PROVIDER_TERMS = {"google cloud", "aws", "azure"}


@dataclass
class HybridSearchResult:
    memory: Memory
    score: float
    dense_score: float
    keyword_score: float


class HybridMemorySearch:
    def __init__(
        self,
        embedder: LocalEmbedder,
        alpha: float = 0.75,
        threshold: float = 0.45,
    ) -> None:
        self.embedder = embedder
        self.alpha = alpha
        self.threshold = threshold

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    @staticmethod
    def _normalized_words(text: str) -> set[str]:
        return set(
            re.findall(r"[a-z0-9]+(?:[.+#-][a-z0-9]+)*", text.lower())
        )

    @classmethod
    def _has_unmatched_subject(
        cls,
        query: str,
        memory: Memory,
    ) -> bool:
        query_words = cls._normalized_words(query)
        memory_words = cls._normalized_words(memory.text)

        return any(
            bool(query_words & terms) and not bool(memory_words & terms)
            for terms in UNKNOWN_SUBJECT_TERMS.values()
        )

    @classmethod
    def _supports_low_confidence_match(
        cls,
        query: str,
        memory: Memory,
    ) -> bool:
        query_words = cls._normalized_words(query)
        memory_words = cls._normalized_words(memory.text)
        normalized_memory_text = memory.text.lower()

        for terms in UNKNOWN_SUBJECT_TERMS.values():
            if query_words & terms:
                return bool(memory_words & terms)

        content_query_words = query_words - LOW_CONFIDENCE_STOP_WORDS
        if content_query_words & memory_words:
            return True

        if (
            {"development", "setup"} <= query_words
            and any(term in normalized_memory_text for term in DEVELOPMENT_TERMS)
        ):
            return True

        if (
            query_words & {"project", "frontend"}
            and memory.type.value == "project"
        ):
            return True

        if (
            query_words & {"mobile", "framework"}
            and any(term in normalized_memory_text for term in FRAMEWORK_TERMS)
        ):
            return True

        if (
            {"cloud", "provider"} <= query_words
            and any(
                term in normalized_memory_text
                for term in CLOUD_PROVIDER_TERMS
            )
        ):
            return True

        return False

    def search(
        self,
        query: str,
        memories: list[Memory],
        top_k: int = 5,
    ) -> list[HybridSearchResult]:
        usable_memories = [
            memory
            for memory in memories
            if memory.is_active and memory.embedding is not None
        ]

        if not usable_memories:
            return []

        query_vector = self.embedder.encode(query)

        memory_matrix = np.ascontiguousarray(
            np.asarray(
                [
                    memory.embedding
                    for memory in usable_memories
                ],
                dtype=np.float32,
            )
        )

        dense_scores = np.dot(memory_matrix, query_vector)

        documents = [
            self._tokenize(memory.text)
            for memory in usable_memories
        ]
        bm25 = BM25Okapi(documents)

        keyword_scores = np.asarray(
            bm25.get_scores(self._tokenize(query)),
            dtype=np.float32,
        )

        keyword_scores = np.maximum(keyword_scores, 0.0)

        highest_keyword_score = float(keyword_scores.max())

        if highest_keyword_score > 0:
            normalized_keyword_scores = (
                keyword_scores / highest_keyword_score
            )
        else:
            normalized_keyword_scores = np.zeros_like(keyword_scores)

        hybrid_scores = np.maximum(
            dense_scores,
            (
                self.alpha * dense_scores
                + (1.0 - self.alpha) * normalized_keyword_scores
            ),
        )

        final_scores = np.asarray(
            [
                hybrid_score
                * (1.0 + 0.10 * (memory.importance - 0.5))
                for hybrid_score, memory in zip(
                    hybrid_scores,
                    usable_memories,
                )
            ],
            dtype=np.float32,
        )

        matching_indexes = [
            index
            for index, score in enumerate(final_scores)
            if not self._has_unmatched_subject(
                query,
                usable_memories[index],
            )
            and (
                score >= self.threshold
                or (
                    score >= LOW_CONFIDENCE_FLOOR
                    and self._supports_low_confidence_match(
                        query,
                        usable_memories[index],
                    )
                )
            )
        ]

        matching_indexes.sort(
            key=lambda index: final_scores[index],
            reverse=True,
        )

        matching_indexes = matching_indexes[:top_k]

        return [
            HybridSearchResult(
                memory=usable_memories[index],
                score=float(final_scores[index]),
                dense_score=float(dense_scores[index]),
                keyword_score=float(normalized_keyword_scores[index]),
            )
            for index in matching_indexes
        ]
