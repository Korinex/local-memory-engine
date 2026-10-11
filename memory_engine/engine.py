from dataclasses import dataclass
from uuid import uuid4

from memory_engine.ingestion.contradiction import find_update
from memory_engine.ingestion.extractor import extract
from memory_engine.models import Memory
from memory_engine.retrieval.embedder import get_embedder
from memory_engine.retrieval.hybrid_search import (
    HybridMemorySearch,
)
from memory_engine.storage.db import Database
from memory_engine.storage.repository import (
    MemoryRepository,
    utc_now,
)


@dataclass
class IngestResponse:
    status: str
    message: str
    memory: Memory | None = None
    replaced_memory: Memory | None = None


class LocalMemoryEngine:
    def __init__(
        self,
        database_path: str = "memory_store.db",
    ) -> None:
        self.repository = MemoryRepository(
            Database(database_path)
        )
        self.embedder = get_embedder()
        self.searcher = HybridMemorySearch(self.embedder)

    def ingest(self, message: str) -> IngestResponse:
        extracted = extract(message)

        if extracted is None:
            return IngestResponse(
                status="ignored",
                message="Message was treated as small talk.",
            )

        memory = Memory(
            id=str(uuid4()),
            text=extracted.text,
            type=extracted.memory_type,
            importance=extracted.importance,
            source_message=extracted.source_message,
            created_at=utc_now(),
            updated_at=utc_now(),
            embedding=self.embedder.encode(
                extracted.text
            ).tolist(),
        )

        active_memories = self.repository.list_memories(
            active_only=True
        )

        update = find_update(memory, active_memories)

        if update.action == "duplicate":
            return IngestResponse(
                status="ignored",
                message="A very similar memory already exists.",
                memory=update.old_memory,
            )

        self.repository.save(memory)

        if update.action == "replace" and update.old_memory:
            self.repository.mark_as_replaced(
                old_id=update.old_memory.id,
                new_id=memory.id,
            )

            return IngestResponse(
                status="updated",
                message="New memory replaced the previous memory.",
                memory=memory,
                replaced_memory=update.old_memory,
            )

        return IngestResponse(
            status="stored",
            message="Memory stored successfully.",
            memory=memory,
        )

    def recall(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.45,
    ) -> list[dict]:
        searcher = HybridMemorySearch(
            embedder=self.embedder,
            threshold=threshold,
        )

        memories = self.repository.list_memories(
            active_only=True
        )

        results = searcher.search(
            query=query,
            memories=memories,
            top_k=top_k,
        )

        return [
            {
                "id": result.memory.id,
                "text": result.memory.text,
                "type": result.memory.type.value,
                "score": round(result.score, 4),
                "created_at": result.memory.created_at,
                "source_message": result.memory.source_message,
            }
            for result in results
        ]

    def list_memories(
        self,
        active_only: bool = True,
    ) -> list[Memory]:
        return self.repository.list_memories(
            active_only=active_only
        )

    def delete_memory(self, memory_id: str) -> bool:
        memory = self.repository.get_by_id(memory_id)

        if memory is None:
            return False

        self.repository.delete(memory_id)
        return True
