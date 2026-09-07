"""Smoke tests: the app starts cleanly and routes behave correctly.
No real LLM or embedding calls are made — get_retriever/get_llm are stubbed
so these tests are fast and don't depend on a built vectorstore or API key.
"""
import pytest
from fastapi.testclient import TestClient

from backend import rag_chain
from backend.main import app


@pytest.fixture
def client(monkeypatch):
    # Stub out the expensive/real calls the lifespan startup would otherwise make.
    monkeypatch.setattr(rag_chain, "get_retriever", lambda: None)
    monkeypatch.setattr(rag_chain, "get_llm", lambda: None)

    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_rejects_empty_question(client):
    resp = client.post("/chat", json={"question": ""})
    assert resp.status_code == 422  # Pydantic min_length=1 validation error


def test_chat_requires_question_field(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 422  # missing required field