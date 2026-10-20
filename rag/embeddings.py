import numpy as np
from google import genai
from google.genai import types

from rag.config import EMBEDDING_DIM, EMBEDDING_MODEL, GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)

# Gemini accepts up to 100 texts per embed request.
BATCH_SIZE = 100


def _normalize(vectors: np.ndarray) -> np.ndarray:
    # Only the full 3072-dim output comes normalized; truncated outputs
    # must be L2-normalized so cosine similarity behaves as expected.
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed(texts: list[str], task_type: str) -> list[np.ndarray]:
    """task_type: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    out: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        result = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts[i : i + BATCH_SIZE],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
        out.extend(_normalize(vectors))
    return out
