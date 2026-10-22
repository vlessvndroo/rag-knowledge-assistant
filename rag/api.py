"""HTTP API for the RAG system.

Run: uvicorn rag.api:app --reload    then open http://localhost:8000/docs
"""

from fastapi import FastAPI, Query
from pydantic import BaseModel

from rag.search import search

app = FastAPI(
    title="RAG Knowledge Assistant",
    description="Semantic search (and soon cited answers) over the n8n documentation.",
)


class SearchHit(BaseModel):
    title: str
    url: str | None
    content: str
    similarity: float


@app.get("/search", response_model=list[SearchHit])
def search_endpoint(
    q: str = Query(..., min_length=3, description="Natural-language question"),
    k: int = Query(5, ge=1, le=20, description="Number of chunks to return"),
) -> list[SearchHit]:
    return [SearchHit(**vars(r)) for r in search(q, k)]
