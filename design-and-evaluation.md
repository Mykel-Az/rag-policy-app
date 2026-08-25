# Design and Evaluation

## 1. Design & Architecture Decisions

> Fill this in as you build. Each choice below should say *what* you picked and
> *why*, briefly.

- **Backend framework — FastAPI**: chosen over Flask for built-in request/response
  validation (Pydantic), automatic OpenAPI docs, and native async support for
  calling the LLM API without blocking. Also matches patterns used in production
  AI backends.
- **Frontend — Streamlit, decoupled from the backend**: Streamlit only calls the
  FastAPI `/chat` endpoint over HTTP; it holds no RAG logic. This keeps the
  backend reusable/replaceable independent of the UI.
- **Embedding model — `sentence-transformers/all-MiniLM-L6-v2`**: runs locally,
  no API key or rate limits, fast enough for a small corpus, adequate quality for
  short policy-document chunks. (Justify further if you swap embedding models.)
- **Vector store — Chroma (local, persistent)**: zero external setup, sufficient
  for a corpus of this size, simple Python API.
- **Chunking strategy — heading-based with sliding-window fallback**: splits on
  markdown `##` sections first (keeps each chunk topically coherent), then
  further splits any oversized section using an ~800-character window with 150
  characters of overlap to avoid losing context at chunk boundaries.
- **Retrieval — top-k, k=4**: (state why you kept/changed k after testing).
- **LLM — Groq (Llama 3.1 8B Instant)**: free tier, low latency, sufficient
  quality for extractive/grounded QA over short contexts.
- **Guardrails**: prompt instructs the model to answer only from retrieved
  context, return a fixed out-of-scope message when no relevant chunks are
  found, and always cite source documents. Output length capped via
  `max_tokens`.

## 2. Evaluation Approach

- **Question set**: 30 questions in `eval/eval_questions.json` spanning all 8
  policy documents, including one intentionally out-of-corpus question to test
  the refusal guardrail.
- **Metrics** (see `eval/run_eval.py`):
  - **Groundedness**: LLM-as-judge check of whether every claim in the answer is
    supported by the retrieved context.
  - **Citation accuracy**: whether the returned citation(s) match the expected
    source document for each question (or correctly trigger the out-of-scope
    message for the one question with no expected source).
  - **Latency (p50/p95)**: measured end-to-end from request to answer across all
    30 queries.

## 3. Results

> Run `python -m eval.run_eval` and paste the summary output here.

```
{
  "num_questions": 30,
  "groundedness_pct": ...,
  "citation_accuracy_pct": ...,
  "latency_p50_ms": ...,
  "latency_p95_ms": ...
}
```

### Observations

- (What worked well? Any systematic failure patterns — e.g., certain policy
  topics retrieving the wrong document, or long questions increasing latency?)

## 4. Ablations (optional)

- (If you tried different values of k, chunk size, or prompt wording, summarize
  what changed and why you kept your final configuration.)
