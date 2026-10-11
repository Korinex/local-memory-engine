import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np

from memory_engine.models import Memory, MemoryType
from memory_engine.storage.db import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryRepository:
    def __init__(self, database: Database) -> None:
        self.database = database
        self.database.initialize()

    @staticmethod
    def _embedding_to_blob(
        embedding: list[float] | np.ndarray | None,
    ) -> bytes | None:
        if embedding is None:
            return None

        vector = np.asarray(embedding, dtype=np.float32)
        return vector.tobytes()

    @staticmethod
    def _blob_to_embedding(blob: bytes | None) -> list[float] | None:
        if blob is None:
            return None

        return np.frombuffer(blob, dtype=np.float32).tolist()

    @classmethod
    def _row_to_memory(cls, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            text=row["text"],
            type=MemoryType(row["type"]),
            importance=row["importance"],
            source_message=row["source_message"],
            is_active=bool(row["is_active"]),
            replaced_by=row["replaced_by"],
            replaces_id=row["replaces_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
            embedding=cls._blob_to_embedding(row["embedding"]),
        )

    def save(self, memory: Memory) -> Memory:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id,
                    text,
                    type,
                    importance,
                    source_message,
                    is_active,
                    replaced_by,
                    replaces_id,
                    created_at,
                    updated_at,
                    metadata,
                    embedding
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.text,
                    memory.type.value,
                    memory.importance,
                    memory.source_message,
                    int(memory.is_active),
                    memory.replaced_by,
                    memory.replaces_id,
                    memory.created_at,
                    memory.updated_at,
                    json.dumps(memory.metadata),
                    self._embedding_to_blob(memory.embedding),
                ),
            )

            self._write_audit(
                connection=connection,
                memory_id=memory.id,
                action="created",
                details={"text": memory.text},
            )

        return memory

    def get_by_id(self, memory_id: str) -> Memory | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()

        return self._row_to_memory(row) if row else None

    def list_memories(self, active_only: bool = True) -> list[Memory]:
        query = "SELECT * FROM memories"
        parameters: tuple[Any, ...] = ()

        if active_only:
            query += " WHERE is_active = 1"

        query += " ORDER BY created_at ASC"

        with self.database.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def mark_as_replaced(self, old_id: str, new_id: str) -> None:
        timestamp = utc_now()

        with self.database.connect() as connection:
            connection.execute(
                """
                UPDATE memories
                SET is_active = 0,
                    replaced_by = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (new_id, timestamp, old_id),
            )

            self._write_audit(
                connection=connection,
                memory_id=old_id,
                action="replaced",
                details={"replaced_by": new_id},
            )

    def delete(self, memory_id: str) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            )

            self._write_audit(
                connection=connection,
                memory_id=memory_id,
                action="deleted",
                details={},
            )

    def get_active_vector_matrix(
        self,
    ) -> tuple[list[Memory], np.ndarray]:
        memories = [
            memory
            for memory in self.list_memories(active_only=True)
            if memory.embedding is not None
        ]

        if not memories:
            return memories, np.empty((0, 0), dtype=np.float32)

        matrix = np.ascontiguousarray(
            np.asarray(
                [memory.embedding for memory in memories],
                dtype=np.float32,
            )
        )

        return memories, matrix

    @staticmethod
    def _write_audit(
        connection: sqlite3.Connection,
        memory_id: str,
        action: str,
        details: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_audit_log (
                memory_id,
                action,
                details,
                timestamp
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                memory_id,
                action,
                json.dumps(details),
                utc_now(),
            ),
        )
