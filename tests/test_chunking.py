import pytest

from rag.chunking import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 80 for i in range(50))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 80 for i in range(50))
    chunks = chunk_text(text, size=300, overlap=50)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.startswith(prev[-50:])


def test_long_paragraph_is_split():
    chunks = chunk_text("a" * 1000, size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(c[50:] if i else c for i, c in enumerate(chunks)).count("a") >= 1000


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("text", size=100, overlap=100)
