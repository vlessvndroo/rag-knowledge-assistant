"""Split documents into overlapping chunks.

Strategy: pack whole paragraphs into chunks of up to `size` characters.
A paragraph longer than `size` is cut with a sliding window. Each new chunk
starts with the last `overlap` characters of the previous one (when they fit), so a fact
that straddles a boundary is still retrievable from at least one chunk.
"""

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [text[i : i + size] for i in range(0, max(len(text) - overlap, 1), step)]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend(_split_long(p, size, overlap) if len(p) > size else [p])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n\n{piece}" if tail and len(tail) + len(piece) + 2 <= size else piece
    if current:
        chunks.append(current)
    return chunks
