import numpy as np
from google.genai import types

from rag.config import EMBEDDING_DIM, EMBEDDING_MODEL
from rag.gemini import client, with_retry

# Gemini accepts up to 100 texts per embed request. On the free tier each
# text counts against a 100-per-minute quota, so 429s are expected on bulk ingests.
BATCH_SIZE = 100


def _normalize(vectors: np.ndarray) -> np.ndarray:
    # Only the full 3072-dim output comes normalized; truncated outputs
    # must be L2-normalized so cosine similarity behaves as expected.
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed(texts: list[str], task_type: str) -> list[np.ndarray]:
    """task_type: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    out: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = with_retry(lambda: client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIM),
        ))
        vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
        out.extend(_normalize(vectors))
    return out
