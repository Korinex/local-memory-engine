from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    PROJECT = "project"
    EVENT = "event"
    GOAL = "goal"


class Memory(BaseModel):
    id: str
    text: str
    type: MemoryType
    importance: float = Field(ge=0.0, le=1.0)
    source_message: str
    is_active: bool = True
    replaced_by: str | None = None
    replaces_id: str | None = None
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
