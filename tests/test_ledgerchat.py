from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from ledgerchat.answer import MODEL, build_request
from ledgerchat.api import create_app
from ledgerchat.retrieval import MemoryStore, chunk_document

POLICY = """# Revenue recognition policy (rev. 2023-04)

## Software subscriptions
Subscription revenue is recognised ratably over the contract term, starting on the date access is provisioned.

## Hardware
Hardware revenue is recognised at shipment when control transfers (FOB shipping point).

## Capitalisation threshold
Fixed assets under USD 5,000 are expensed. Items at or above USD 5,000 are capitalised to account 1510 and depreciated straight-line."""

CLOSE_MEMO = """# August 2023 close memo
Accrued freight (account 2150) was USD 412,880.15 at 31 Aug 2023, up from USD 298,004.00 in July due to the DC4 carrier change.
The bonus accrual (account 2310) was trued up by USD 1.2m after the H1 review."""

SECRET = """# EU entity payroll
Payroll for entity EU01 in August was EUR 3,402,117.00."""


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        docs = [b for b in kw["messages"][0]["content"] if b["type"] == "document"]
        cite_idx = next(i for i, d in enumerate(docs) if "412,880.15" in d["source"]["data"]) if any("412,880.15" in d["source"]["data"] for d in docs) else 0
        return NS(model=kw["model"], stop_reason="end_turn", content=[
            NS(type="thinking", thinking=""),
            NS(type="text", text="Accrued freight was USD 412,880.15 at 31 Aug 2023.", citations=[
                NS(type="char_location", document_index=cite_idx, document_title=docs[cite_idx]["title"],
                   cited_text="Accrued freight (account 2150) was USD 412,880.15 at 31 Aug 2023")]),
        ])


@pytest.fixture
def setup():
    msgs = FakeMessages()
    client = NS(beta=NS(messages=msgs))
    store = MemoryStore()
    app = TestClient(create_app(client=client, store=store))
    us = {"x-entities": "US01"}
    r = app.post("/ingest", headers=us, json=[
        {"doc_id": "rev-policy", "title": "Revenue recognition policy", "text": POLICY, "entity": "US01"},
        {"doc_id": "close-2023-08", "title": "August 2023 close memo", "text": CLOSE_MEMO, "entity": "US01", "kind": "close_memo"},
    ])
    assert r.status_code == 201
    app.post("/ingest", headers={"x-entities": "EU01"}, json=[{"doc_id": "eu-payroll", "title": "EU payroll", "text": SECRET, "entity": "EU01"}])
    return app, msgs, store


def test_chunking_respects_sections():
    chunks = chunk_document("p", "Policy", POLICY * 3, "US01", "policy")
    assert len(chunks) > 1 and all(len(c.text) <= 900 for c in chunks)


def test_retrieval_ranks_the_right_doc_and_boosts_exact_account_numbers(setup):
    _, _, store = setup
    top = store.search("what was the balance of account 2150 at end of August?", {"US01"}, 3)
    assert top[0].chunk.doc_id == "close-2023-08"
    top = store.search("when do we recognise subscription revenue", {"US01"}, 3)
    assert top[0].chunk.doc_id == "rev-policy"


def test_answer_is_cited_and_request_is_well_formed(setup):
    app, msgs, _ = setup
    r = app.post("/ask", headers={"x-entities": "US01"}, json={"question": "What was accrued freight (account 2150) at August month end?"}).json()
    assert r["grounded"] is True
    assert r["citations"][0]["chunk_id"].startswith("close-2023-08")
    assert "412,880.15" in r["citations"][0]["cited_text"]
    req = msgs.calls[0]
    assert req["model"] == MODEL == "claude-opus-5-5"
    assert req["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["betas"]
    assert req["output_config"] == {"effort": "medium"}
    docs = [b for b in req["messages"][0]["content"] if b["type"] == "document"]
    assert docs and all(d["citations"] == {"enabled": True} for d in docs)
    assert req["messages"][0]["content"][-1]["type"] == "text"  # question after the documents


def test_other_entities_documents_never_reach_the_model(setup):
    app, msgs, _ = setup
    app.post("/ask", headers={"x-entities": "US01"}, json={"question": "What was EU01 payroll in August?"})
    for call in msgs.calls:
        assert all("3,402,117" not in str(b) for b in call["messages"][0]["content"])


def test_nothing_relevant_means_no_model_call(setup):
    app, msgs, _ = setup
    r = app.post("/ask", headers={"x-entities": "APAC9"}, json={"question": "What is the capitalisation threshold?"}).json()
    assert r["grounded"] is False and msgs.calls == []


def test_refusal_is_handled(setup):
    app, msgs, _ = setup
    msgs.create = lambda **kw: NS(model=MODEL, stop_reason="refusal", content=[])
    r = app.post("/ask", headers={"x-entities": "US01"}, json={"question": "capitalisation threshold for fixed assets"}).json()
    assert r["stop_reason"] == "refusal" and r["grounded"] is False


def test_auth_and_ingest_acl(setup):
    app, _, _ = setup
    assert app.post("/ask", json={"question": "anything at all"}).status_code == 401
    r = app.post("/ingest", headers={"x-entities": "US01"}, json=[{"doc_id": "x", "title": "x", "text": "y", "entity": "EU01"}])
    assert r.status_code == 403


def test_build_request_shape():
    req = build_request("q", [])
    assert req["messages"][0]["content"] == [{"type": "text", "text": "q"}]
    assert "thinking" not in req  # adaptive by default on Opus 5.5
