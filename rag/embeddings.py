import re
import time

import numpy as np
from google import genai
from google.genai import errors, types

from rag.config import EMBEDDING_DIM, EMBEDDING_MODEL, GEMINI_API_KEY

_client = genai.Client(api_key=GEMINI_API_KEY)

# Gemini accepts up to 100 texts per embed request. On the free tier each
# text counts against a 100-per-minute quota, so 429s are expected on bulk ingests.
BATCH_SIZE = 100
MAX_RETRIES = 5


def _normalize(vectors: np.ndarray) -> np.ndarray:
    # Only the full 3072-dim output comes normalized; truncated outputs
    # must be L2-normalized so cosine similarity behaves as expected.
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _retry_delay(err: errors.ClientError, attempt: int) -> float:
    # The API says how long to wait ("retryDelay": "19s"); fall back to exponential backoff.
    if m := re.search(r"'retryDelay': '(\d+)s'", str(err)):
        return int(m.group(1)) + 1
    return 2 ** attempt * 5


def _embed_with_retry(texts: list[str], task_type: str):
    for attempt in range(MAX_RETRIES):
        try:
            return _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
        except errors.ClientError as err:
            if err.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            delay = _retry_delay(err, attempt)
            print(f"  rate limited, retrying in {delay:.0f}s")
            time.sleep(delay)


def embed(texts: list[str], task_type: str) -> list[np.ndarray]:
    """task_type: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for questions."""
    out: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        result = _embed_with_retry(texts[i : i + BATCH_SIZE], task_type)
        vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
        out.extend(_normalize(vectors))
    return out
