"""Ingest a text/markdown file: chunk -> embed -> store in pgvector.

Re-ingesting the same source replaces its chunks, so the command is idempotent.

Usage: python -m rag.ingest data/raw/some-doc.md [more files...]
"""

import sys
from pathlib import Path

from rag.chunking import chunk_text
from rag.db import connect
from rag.embeddings import embed


def ingest_file(conn, path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vectors = embed(chunks, task_type="RETRIEVAL_DOCUMENT")

    with conn.transaction():
        doc_id = conn.execute(
            """
            insert into documents (source, title) values (%s, %s)
            on conflict (source) do update set title = excluded.title
            returning id
            """,
            (str(path.as_posix()), path.stem),
        ).fetchone()[0]
        conn.execute("delete from chunks where document_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "insert into chunks (document_id, chunk_index, content, embedding) values (%s, %s, %s, %s)",
                [(doc_id, i, c, v) for i, (c, v) in enumerate(zip(chunks, vectors))],
            )
    return len(chunks)


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit(__doc__)
    with connect() as conn:
        for p in paths:
            n = ingest_file(conn, Path(p))
            print(f"{p}: {n} chunks")


if __name__ == "__main__":
    main(sys.argv[1:])
