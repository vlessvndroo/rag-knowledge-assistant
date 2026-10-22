# rag-knowledge-assistant

A retrieval-augmented generation (RAG) system in Python: document ingestion, chunking,
semantic + full-text hybrid search over pgvector (Supabase/PostgreSQL), and cited answers
generated with Gemini, served via FastAPI.

> Portfolio project built to learn and demonstrate RAG end to end. Not a client production system.

**Corpus:** 53 pages of the public [n8n documentation](https://github.com/n8n-io/n8n-docs)
(getting started, workflows, flow logic, working with data), pinned to a commit for reproducible
evals. The docs are downloaded by a script rather than committed to this repo.

## Status

- [x] Ingestion: markdown cleaning, chunking, Gemini embeddings stored in pgvector (53 docs, 197 chunks)
- [x] Semantic search: CLI and `GET /search` endpoint (FastAPI)
- [ ] `POST /ask` with cited answers
- [ ] Hybrid search (vector + Postgres full-text, Reciprocal Rank Fusion)
- [ ] Evaluation harness (recall@k, answer accuracy)

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # fill in GEMINI_API_KEY and DATABASE_URL
```

Run the files in `sql/` in order in the Supabase SQL editor.

## Ingest the corpus

```bash
python scripts/download_corpus.py     # fetches the pinned n8n docs into data/raw/n8n
python -m rag.ingest data/raw/n8n     # clean -> chunk -> embed -> store (idempotent)
```

## Search

```bash
python -m rag.search "how do I loop over items?"     # CLI: top-5 chunks with similarity
uvicorn rag.api:app --reload                          # API: open http://localhost:8000/docs
```

`GET /search?q=<question>&k=5` returns the `k` most similar chunks with page title, public URL
and cosine similarity. Questions are embedded with `task_type=RETRIEVAL_QUERY` and chunks with
`RETRIEVAL_DOCUMENT`, Gemini's asymmetric setup for short-question vs. long-passage matching.

## Technical decisions

| Decision | Choice | Why |
|---|---|---|
| Corpus | n8n docs subset, pinned commit | Public and verifiable: anyone can check answers against the official docs. Function-reference tables excluded (they retrieve poorly) |
| Cleaning | Strip front matter, GitBook tags, HTML images/tags, link URLs | Removes tokens that carry no meaning; page title and public URL are kept as metadata for citations. Index pages without a URL get one derived from their path |
| Embeddings | `gemini-embedding-001`, 768 dims | Truncated output keeps storage small and fits pgvector's 2000-dim HNSW limit; vectors are L2-normalized after truncation |
| Chunking | Paragraph packing, 1200 chars, 200 overlap | Keeps paragraphs intact; overlap protects facts that cross boundaries. To be tuned with evals |
| Index | HNSW, cosine | Good recall/latency without training step (unlike IVFFlat) |
| Access | RLS enabled, no policies | Tables are unreachable via the public Supabase Data API; the app connects as table owner over Postgres directly |

| Rate limits | Retry on HTTP 429 using the API's `retryDelay` | Gemini's free tier allows 100 embedded texts per minute; bulk ingestion hits it by design |

## Limitations

- GitBook `{% include %}` blocks point to reusable snippets hosted outside the repo; they are dropped, so a few pages lose some shared content.
- Chunking is character-based, not token-based; parameters are untuned until the evaluation harness exists.
