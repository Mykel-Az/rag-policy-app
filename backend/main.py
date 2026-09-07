"""
FastAPI backend for the RAG policy assistant.

Run locally with:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /        -> basic info
    POST /chat    -> {"question": "..."} -> answer + citations + latency
    GET  /health  -> {"status": "ok"}
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import rag_chain


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup: pay the ~15s embedding-model cold-start cost once, here,
    # instead of on the first user's request.
    print("Warming up retriever and LLM...")
    rag_chain.get_retriever()
    rag_chain.get_llm()
    print("Warm-up complete.")
    yield
    # Shutdown: nothing to clean up currently.


app = FastAPI(title="Policy RAG API", version="1.0.0", lifespan=lifespan)

# Allow the Streamlit frontend (running on a different port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend's exact origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class Citation(BaseModel):
    source: str
    section: str
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: float


@app.get("/")
def root():
    return {
        "message": "Policy RAG API is running. Use POST /chat to ask a question."
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = rag_chain.answer_question(req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result

