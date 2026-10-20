# rag-knowledge-assistant

A retrieval-augmented generation (RAG) system in Python: document ingestion, chunking,
semantic + full-text hybrid search over pgvector (Supabase/PostgreSQL), and cited answers
generated with Gemini, served via FastAPI.

> Portfolio project built to learn and demonstrate RAG end to end. Not a client production system.

## Status

- [x] Ingestion: chunking + Gemini embeddings stored in pgvector
- [ ] Semantic search endpoint
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

Run `sql/001_init.sql` once in the Supabase SQL editor.

## Ingest documents

```bash
python -m rag.ingest data/raw/my-doc.md
```

## Technical decisions

| Decision | Choice | Why |
|---|---|---|
| Embeddings | `gemini-embedding-001`, 768 dims | Truncated output keeps storage small and fits pgvector's 2000-dim HNSW limit; vectors are L2-normalized after truncation |
| Chunking | Paragraph packing, 1200 chars, 200 overlap | Keeps paragraphs intact; overlap protects facts that cross boundaries. To be tuned with evals |
| Index | HNSW, cosine | Good recall/latency without training step (unlike IVFFlat) |

## Limitations

_To be written honestly as the project evolves._
