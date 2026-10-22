"""Semantic search over the ingested chunks.

Usage: python -m rag.search "how do I loop over items?" [k]
"""

import sys
from dataclasses import dataclass

from rag.db import connect
from rag.embeddings import embed


@dataclass
class SearchResult:
    title: str
    url: str | None
    content: str
    similarity: float


def search(query: str, k: int = 5) -> list[SearchResult]:
    # Questions and documents use different task types: Gemini optimizes
    # each side of the asymmetric "short question vs. long passage" match.
    vector = embed([query], task_type="RETRIEVAL_QUERY")[0]
    with connect() as conn:
        rows = conn.execute(
            """
            select d.title, d.url, c.content, 1 - (c.embedding <=> %(v)s) as similarity
            from chunks c
            join documents d on d.id = c.document_id
            order by c.embedding <=> %(v)s
            limit %(k)s
            """,
            {"v": vector, "k": k},
        ).fetchall()
    return [SearchResult(*row) for row in rows]


def main(args: list[str]) -> None:
    if not args:
        sys.exit(__doc__)
    k = int(args[1]) if len(args) > 1 else 5
    for i, r in enumerate(search(args[0], k), 1):
        preview = " ".join(r.content.split())[:160]
        print(f"{i}. [{r.similarity:.3f}] {r.title}\n   {r.url}\n   {preview}...\n")


if __name__ == "__main__":
    main(sys.argv[1:])
