create extension if not exists vector;

create table if not exists documents (
    id          bigserial primary key,
    source      text not null unique,
    title       text,
    created_at  timestamptz not null default now()
);

create table if not exists chunks (
    id           bigserial primary key,
    document_id  bigint not null references documents(id) on delete cascade,
    chunk_index  int not null,
    content      text not null,
    -- 768 dims: Gemini embeddings truncated via output_dimensionality.
    -- pgvector HNSW indexes on `vector` support up to 2000 dims.
    embedding    vector(768) not null,
    -- Full-text column for hybrid search (week 1, day 5).
    -- 'simple' is language-agnostic; switch to 'spanish'/'english' to match the corpus.
    tsv          tsvector generated always as (to_tsvector('simple', content)) stored,
    unique (document_id, chunk_index)
);

create index if not exists chunks_embedding_hnsw
    on chunks using hnsw (embedding vector_cosine_ops);

create index if not exists chunks_tsv_gin
    on chunks using gin (tsv);
