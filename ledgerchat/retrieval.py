"""Chunking, embeddings and the vector store for accounting documents.

Production stores vectors in Postgres with pgvector (`PgVectorStore`); tests and local dev use the
in-memory store with a deterministic hashing embedder so results are reproducible offline.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    entity: str  # legal entity / company code: the access-control boundary
    kind: str  # policy | close_memo | gl_balance | contract


@dataclass
class Hit:
    chunk: Chunk
    score: float


_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120, separators=["\n## ", "\n\n", "\n", ". ", " "])


def chunk_document(doc_id: str, title: str, text: str, entity: str, kind: str) -> list[Chunk]:
    return [Chunk(f"{doc_id}#{i}", doc_id, title, part.strip(), entity, kind)
            for i, part in enumerate(_splitter.split_text(text)) if part.strip()]


_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    dims: int
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Feature-hashed unigrams + bigrams, L2-normalised. A stand-in for a real embedding model
    (e.g. Voyage) with the same interface; good enough to make retrieval tests meaningful."""

    def __init__(self, dims: int = 1024):
        self.dims = dims

    def _idx(self, tok: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        return h % self.dims, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dims
            toks = tokens(t)
            for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                i, s = self._idx(tok)
                v[i] += s
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, entities: set[str], k: int) -> list[Hit]: ...


@dataclass
class MemoryStore:
    embedder: Embedder = field(default_factory=HashingEmbedder)
    _rows: list[tuple[Chunk, list[float]]] = field(default_factory=list)

    def add(self, chunks: list[Chunk]) -> None:
        ids = {c.id for c in chunks}
        self._rows = [r for r in self._rows if r[0].id not in ids]  # re-ingest replaces
        for c, v in zip(chunks, self.embedder.embed([f"{c.title}\n{c.text}" for c in chunks])):
            self._rows.append((c, v))

    def search(self, query: str, entities: set[str], k: int = 6) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        qtok = set(tokens(query))
        exact = {t for t in qtok if re.fullmatch(r"\d{4,}(?:-\d+)?", t)}  # account numbers, invoice ids
        hits = []
        for c, v in self._rows:
            if c.entity not in entities:  # ACL is applied *before* ranking, never after
                continue
            score = sum(a * b for a, b in zip(q, v))
            if exact and exact & set(tokens(c.text)):
                score += 0.25  # hybrid: exact identifiers beat semantic neighbours
            hits.append(Hit(c, score))
        return sorted(hits, key=lambda h: -h.score)[:k]


class PgVectorStore:
    """Postgres + pgvector. Same contract as MemoryStore; ACL and hybrid boost are done in SQL."""

    def __init__(self, conn, embedder: Embedder):
        self.conn, self.embedder = conn, embedder

    def add(self, chunks: list[Chunk]) -> None:
        vecs = self.embedder.embed([f"{c.title}\n{c.text}" for c in chunks])
        with self.conn.cursor() as cur:
            for c, v in zip(chunks, vecs):
                cur.execute(
                    """INSERT INTO chunks (id, doc_id, title, text, entity, kind, embedding, tsv)
                       VALUES (%s,%s,%s,%s,%s,%s,%s::vector, to_tsvector('english', %s))
                       ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding, tsv = EXCLUDED.tsv""",
                    (c.id, c.doc_id, c.title, c.text, c.entity, c.kind, str(v), c.text))
        self.conn.commit()

    def search(self, query: str, entities: set[str], k: int = 6) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id, doc_id, title, text, entity, kind,
                          1 - (embedding <=> %s::vector) + 0.25 * (tsv @@ plainto_tsquery('english', %s))::int AS score
                   FROM chunks WHERE entity = ANY(%s)
                   ORDER BY score DESC LIMIT %s""",
                (str(q), query, list(entities), k))
            return [Hit(Chunk(*row[:6]), float(row[6])) for row in cur.fetchall()]
