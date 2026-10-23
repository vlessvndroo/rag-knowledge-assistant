"""Shared Gemini client with retries. The only module that talks to the Gemini SDK."""

import re
import time
from collections.abc import Callable

from google import genai
from google.genai import errors

from rag.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

MAX_RETRIES = 5
# 429: free-tier quota (100 embedded texts/min). 503: model overloaded. Both are transient.
_RETRYABLE = {429, 503}


def _retry_delay(err: errors.APIError, attempt: int) -> float:
    # The API says how long to wait ("retryDelay": "19s"); fall back to exponential backoff.
    if m := re.search(r"'retryDelay': '(\d+)s'", str(err)):
        return int(m.group(1)) + 1
    return 2 ** attempt * 5


def with_retry[T](call: Callable[[], T]) -> T:
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except errors.APIError as err:
            if err.code not in _RETRYABLE or attempt == MAX_RETRIES - 1:
                raise
            delay = _retry_delay(err, attempt)
            print(f"  Gemini {err.code}, retrying in {delay:.0f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")
