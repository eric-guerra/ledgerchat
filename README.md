# LedgerChat

Portfolio reference implementation of a 2024 project, built to show a retrieval-augmented assistant for finance teams. Staff ask questions in plain English ("what was accrued freight at August close?", "what's our capitalisation threshold?") and get answers grounded in their own entity's policies, close memos and GL extracts, **with citations**.

**Stack:** FastAPI, Anthropic Python SDK (Claude Opus 5.5), LangChain text splitters, PostgreSQL + pgvector (HNSW + tsvector hybrid search), Next.js 13 (app router).

## Design
- **Ingest:**
  - Documents are split on section and paragraph boundaries (`RecursiveCharacterTextSplitter`, 900 characters with 120 overlap) and stored with their **entity**, which is the access-control boundary.
- **Retrieve:**
  - Hybrid search: vector cosine similarity plus a boost for exact identifiers such as account numbers or invoice ids, since "account 2150" must beat semantically similar text.
  - The **ACL is applied before ranking**, so another entity's documents never enter the candidate set, the prompt, or the answer.
  - If nothing clears the relevance floor, the API answers "not in your documents" **without calling the model**.
- **Answer** (`ledgerchat/answer.py`):
  - Retrieved chunks are sent as `document` blocks with `citations: {enabled: true}`, so every claim comes back linked to the exact source text. Responses without citations are flagged `grounded: false` in the UI.
  - Model `claude-opus-5-5` with adaptive thinking (its default) and `effort: "medium"` set explicitly.
  - The system prompt requires exact figures with period and currency, calls out conflicting documents and says which is newer, and asks for "say what's missing" rather than gap-filling.
  - **Refusal handling:** server-side fallbacks are enabled (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`), and `stop_reason == "refusal"` is checked before reading content.
  - SDK errors map to HTTP responses: 429 → 429, other API errors → 502, connection errors → 503.
- **Web** (`web/`):
  - A Next.js route handler proxies to the API. Entity access is resolved **server-side**; an early draft trusted a browser-supplied header, which would have let users widen their own access.
  - The chat UI shows numbered citations under each answer.

## Run
```bash
pip install -r requirements.txt
psql -f schema.sql                          # pgvector
ANTHROPIC_API_KEY=... uvicorn "ledgerchat.serve:app"
cd web && npm install && DEV_ENTITIES=US01 npm run dev
```

## Test
```bash
pytest -q
```
8 tests, using a fake Anthropic client that returns citation blocks, so no network is needed. They cover:
- Chunking, and retrieval ranking with the exact-identifier boost.
- Request shape: model, effort, fallbacks, cited documents placed before the question.
- Cross-entity isolation, and no model call when retrieval finds nothing.
- Refusal handling and the auth/ingest ACL.
