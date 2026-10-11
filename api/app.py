from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from memory_engine.engine import LocalMemoryEngine


app = FastAPI(
    title="Local Memory Engine",
    description="A local-first memory engine for AI chatbots.",
    version="1.0.0",
)
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def home_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


engine = LocalMemoryEngine()


class MemoryRequest(BaseModel):
    message: str = Field(min_length=1)


class RecallRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    threshold: float = Field(default=0.45, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/memory")
def add_memory(request: MemoryRequest) -> dict[str, Any]:
    result = engine.ingest(request.message)

    return {
        "status": result.status,
        "message": result.message,
        "memory": (
            result.memory.model_dump()
            if result.memory
            else None
        ),
        "replaced_memory": (
            result.replaced_memory.model_dump()
            if result.replaced_memory
            else None
        ),
    }


@app.post("/recall")
def recall(request: RecallRequest) -> dict[str, list[dict]]:
    memories = engine.recall(
        query=request.query,
        top_k=request.top_k,
        threshold=request.threshold,
    )

    return {"memories": memories}


@app.get("/memories")
def list_memories(
    active_only: bool = Query(default=True),
) -> dict[str, list[dict]]:
    memories = engine.list_memories(
        active_only=active_only
    )

    return {
        "memories": [
            memory.model_dump()
            for memory in memories
        ]
    }


@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: str) -> dict[str, str]:
    deleted = engine.delete_memory(memory_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Memory not found.",
        )

    return {
        "status": "deleted",
        "id": memory_id,
    }
