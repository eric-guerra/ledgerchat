"""HTTP API. The caller's entities come from the auth gateway (X-Entities header set after SSO);
documents outside them are never retrieved, so they can never reach the model or the answer."""
from __future__ import annotations

from dataclasses import asdict

import anthropic
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .answer import answer
from .retrieval import MemoryStore, VectorStore, chunk_document


class IngestDoc(BaseModel):
    doc_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    title: str
    text: str = Field(min_length=1)
    entity: str
    kind: str = "policy"


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def _entities(header: str | None) -> set[str]:
    ents = {e.strip() for e in (header or "").split(",") if e.strip()}
    if not ents:
        raise HTTPException(401, "no entity access")
    return ents


def create_app(client=None, store: VectorStore | None = None) -> FastAPI:
    client = client or anthropic.Anthropic()
    store = store or MemoryStore()
    app = FastAPI(title="LedgerChat")

    @app.post("/ingest", status_code=201)
    def ingest(docs: list[IngestDoc], x_entities: str | None = Header(default=None)):
        allowed = _entities(x_entities)
        n = 0
        for d in docs:
            if d.entity not in allowed:
                raise HTTPException(403, f"cannot ingest into entity {d.entity}")
            chunks = chunk_document(d.doc_id, d.title, d.text, d.entity, d.kind)
            store.add(chunks)
            n += len(chunks)
        return {"chunks": n}

    @app.post("/ask")
    def ask(body: Ask, x_entities: str | None = Header(default=None)):
        try:
            return asdict(answer(client, store, body.question, _entities(x_entities)))
        except anthropic.RateLimitError:
            raise HTTPException(429, "model is busy, retry shortly")
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"model error {e.status_code}")
        except anthropic.APIConnectionError:
            raise HTTPException(503, "model unreachable")

    return app
