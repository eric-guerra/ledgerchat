CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id        TEXT PRIMARY KEY,           -- "<doc_id>#<n>"
    doc_id    TEXT NOT NULL,
    title     TEXT NOT NULL,
    text      TEXT NOT NULL,
    entity    TEXT NOT NULL,              -- legal entity / company code: the ACL boundary
    kind      TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    tsv       tsvector NOT NULL           -- keyword side of hybrid search (account numbers, invoice ids)
);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX chunks_entity ON chunks (entity);

-- Audit: who asked what, which chunks were shown to the model, what came back.
CREATE TABLE ask_log (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    entities    TEXT[] NOT NULL,
    question    TEXT NOT NULL,
    chunk_ids   TEXT[] NOT NULL,
    grounded    BOOLEAN NOT NULL,
    stop_reason TEXT,
    model       TEXT,
    at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
