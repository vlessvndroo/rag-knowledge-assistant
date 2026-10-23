from rag.generation import build_prompt, cited_sources
from rag.search import SearchResult

CHUNKS = [
    SearchResult("Merge data", "https://docs.n8n.io/merge", "Use the Merge node.", 0.78),
    SearchResult("Loop", "https://docs.n8n.io/loop", "Use Loop Over Items.", 0.70),
]


def test_prompt_numbers_excerpts_with_titles():
    prompt = build_prompt("how to merge?", CHUNKS)
    assert "[1] (from: Merge data)\nUse the Merge node." in prompt
    assert "[2] (from: Loop)\nUse Loop Over Items." in prompt
    assert prompt.endswith("Question: how to merge?")


def test_cited_sources_keeps_only_cited_in_order():
    sources = cited_sources("Use Merge [2]. Really [1][2].", CHUNKS)
    assert [(s.n, s.title) for s in sources] == [(1, "Merge data"), (2, "Loop")]


def test_cited_sources_ignores_out_of_range_numbers():
    assert cited_sources("Hallucinated citation [7].", CHUNKS) == []
