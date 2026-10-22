from fastapi.testclient import TestClient

from rag import api
from rag.search import SearchResult

client = TestClient(api.app)


def test_search_returns_hits(monkeypatch):
    fake = [SearchResult("Loop", "https://docs.n8n.io/loop", "Looping in n8n...", 0.73)]
    monkeypatch.setattr(api, "search", lambda q, k: fake)

    resp = client.get("/search", params={"q": "how do I loop?", "k": 1})

    assert resp.status_code == 200
    assert resp.json() == [
        {"title": "Loop", "url": "https://docs.n8n.io/loop", "content": "Looping in n8n...", "similarity": 0.73}
    ]


def test_search_validates_params():
    assert client.get("/search", params={"q": "hi"}).status_code == 422  # too short
    assert client.get("/search", params={"q": "loop items", "k": 50}).status_code == 422  # k > 20
