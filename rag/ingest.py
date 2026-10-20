"""Ingest markdown files: clean -> chunk -> embed -> store in pgvector.

Re-ingesting the same source replaces its chunks, so the command is idempotent.

Usage: python -m rag.ingest data/raw/n8n            (a directory, recursively)
       python -m rag.ingest data/raw/n8n/some-doc.md
"""

import sys
from pathlib import Path

from rag.chunking import chunk_text
from rag.db import connect
from rag.embeddings import embed
from rag.loaders import load_markdown


def ingest_file(conn, path: Path) -> int:
    doc = load_markdown(path)
    chunks = chunk_text(doc.text)
    if not chunks:
        return 0
    vectors = embed(chunks, task_type="RETRIEVAL_DOCUMENT")

    with conn.transaction():
        doc_id = conn.execute(
            """
            insert into documents (source, title, url) values (%s, %s, %s)
            on conflict (source) do update set title = excluded.title, url = excluded.url
            returning id
            """,
            (doc.source, doc.title, doc.url),
        ).fetchone()[0]
        conn.execute("delete from chunks where document_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "insert into chunks (document_id, chunk_index, content, embedding) values (%s, %s, %s, %s)",
                [(doc_id, i, c, v) for i, (c, v) in enumerate(zip(chunks, vectors))],
            )
    return len(chunks)


def _expand(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(sorted(p.rglob("*.md")) if p.is_dir() else [p])
    return files


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit(__doc__)
    files = _expand(paths)
    total = 0
    with connect() as conn:
        for f in files:
            n = ingest_file(conn, f)
            total += n
            print(f"{f.as_posix()}: {n} chunks")
    print(f"Done: {len(files)} files, {total} chunks")


if __name__ == "__main__":
    main(sys.argv[1:])
