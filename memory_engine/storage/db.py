import sqlite3
from pathlib import Path


class Database:
    def __init__(self, database_path: str | Path = "memory_store.db") -> None:
        self.database_path = Path(database_path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    type TEXT NOT NULL,
                    importance REAL NOT NULL CHECK (
                        importance >= 0.0 AND importance <= 1.0
                    ),
                    source_message TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    replaced_by TEXT,
                    replaces_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    embedding BLOB
                );

                CREATE TABLE IF NOT EXISTS memory_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}',
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_active
                    ON memories(is_active);

                CREATE INDEX IF NOT EXISTS idx_memories_type
                    ON memories(type);

                CREATE INDEX IF NOT EXISTS idx_memories_replaces_id
                    ON memories(replaces_id);
                """
            )

    def journal_mode(self) -> str:
        with self.connect() as connection:
            row = connection.execute("PRAGMA journal_mode").fetchone()
            return str(row[0]).lower()
