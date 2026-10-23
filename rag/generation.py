"""Answer questions from retrieved chunks, citing the sources used.

Usage: python -m rag.generation "how do I merge data from two nodes?"
"""

import re
import sys
from dataclasses import dataclass

from google.genai import types

from rag.config import GENERATION_MODEL
from rag.gemini import client, with_retry
from rag.search import SearchResult, search

SYSTEM_PROMPT = """You answer questions about n8n using ONLY the numbered documentation excerpts provided.

Rules:
- Every factual sentence must end with the number of the excerpt that supports it, like [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: "I couldn't find this in the documentation."
  Do not use outside knowledge, even if you know the answer.
- Be concise: a short paragraph or a few steps."""

NOT_FOUND = "I couldn't find this in the documentation."
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    n: int
    title: str
    url: str | None


@dataclass
class Answer:
    answer: str
    sources: list[Source]


def build_prompt(question: str, chunks: list[SearchResult]) -> str:
    excerpts = "\n\n".join(f"[{i}] (from: {c.title})\n{c.content}" for i, c in enumerate(chunks, 1))
    return f"Documentation excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def cited_sources(answer: str, chunks: list[SearchResult]) -> list[Source]:
    # Keep only the excerpts the model actually cited, ignoring out-of-range numbers.
    cited = sorted({int(n) for n in _CITATION.findall(answer)})
    return [Source(n, chunks[n - 1].title, chunks[n - 1].url) for n in cited if 1 <= n <= len(chunks)]


def ask(question: str, k: int = 5) -> Answer:
    chunks = search(question, k)
    response = with_retry(lambda: client.models.generate_content(
        model=GENERATION_MODEL,
        contents=build_prompt(question, chunks),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
    ))
    text = (response.text or NOT_FOUND).strip()
    return Answer(answer=text, sources=cited_sources(text, chunks))


def main(args: list[str]) -> None:
    if not args:
        sys.exit(__doc__)
    result = ask(args[0])
    print(result.answer, "\n")
    for s in result.sources:
        print(f"[{s.n}] {s.title} - {s.url}")


if __name__ == "__main__":
    main(sys.argv[1:])
