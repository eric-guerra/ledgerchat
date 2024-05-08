"""Grounded answers with citations from Claude.

Retrieved chunks are sent as `document` blocks with citations enabled, so every claim in the answer
comes back linked to the exact source text. If retrieval finds nothing relevant, we answer
"not in the sources" without calling the model at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .retrieval import Hit, VectorStore

MODEL = "claude-opus-5-5"
MIN_SCORE = 0.12

SYSTEM = """You answer questions from finance and accounting staff using only the documents provided \
(accounting policies, month-end close memos, GL balance extracts, contracts).

- Base every statement on the documents and cite them. If the documents don't answer the question, \
say so plainly and name what is missing; do not fill gaps from general knowledge.
- When figures are involved, quote them exactly as they appear, with currency and period.
- If documents conflict (for example, an older policy and a newer memo), point out the conflict and \
say which document is more recent.
- Be concise: a direct answer first, then supporting detail."""


@dataclass
class Citation:
    document_title: str
    cited_text: str
    chunk_id: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    model: str | None = None
    stop_reason: str | None = None
    sources: list[str] = field(default_factory=list)


def build_request(question: str, hits: list[Hit]) -> dict:
    documents = [{
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": h.chunk.text},
        "title": h.chunk.title,
        "context": f"chunk={h.chunk.id} entity={h.chunk.entity} kind={h.chunk.kind}",
        "citations": {"enabled": True},
    } for h in hits]
    return {
        "model": MODEL,
        "max_tokens": 16000,
        "system": SYSTEM,
        # Opus 5.5 runs adaptive thinking by default; effort defaults to medium, set it explicitly.
        "output_config": {"effort": "medium"},
        # Server-side fallback on a policy decline, routed by refusal category.
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
        "messages": [{"role": "user", "content": [*documents, {"type": "text", "text": question}]}],
    }


def parse_response(response, hits: list[Hit]) -> Answer:
    if response.stop_reason == "refusal":
        return Answer("I can't help with that request.", grounded=False, model=response.model, stop_reason="refusal")
    text_parts, citations = [], []
    for block in response.content:
        if block.type != "text":
            continue  # thinking / fallback blocks
        text_parts.append(block.text)
        for c in getattr(block, "citations", None) or []:
            idx = getattr(c, "document_index", None)
            chunk_id = hits[idx].chunk.id if idx is not None and idx < len(hits) else ""
            citations.append(Citation(c.document_title or "", c.cited_text, chunk_id))
    return Answer("".join(text_parts).strip(), citations, grounded=bool(citations), model=response.model,
                  stop_reason=response.stop_reason, sources=sorted({h.chunk.doc_id for h in hits}))


def answer(client, store: VectorStore, question: str, entities: set[str], k: int = 6) -> Answer:
    hits = [h for h in store.search(question, entities, k) if h.score >= MIN_SCORE]
    if not hits:
        return Answer("I couldn't find this in the documents you have access to.", grounded=False)
    response = client.beta.messages.create(**build_request(question, hits))
    return parse_response(response, hits)
