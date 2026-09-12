from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import rag_chain


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Warming up retriever and LLM...")
    rag_chain.get_retriever()
    rag_chain.get_llm()
    print("Warm-up complete.")
    yield


app = FastAPI(title="Policy RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
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

