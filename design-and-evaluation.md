# Design and Evaluation

## 1. Design & Architecture Decisions

- **Backend framework — FastAPI**: chosen over Flask for built-in request/response
  validation via Pydantic, automatic OpenAPI docs at `/docs`, and a clean way to
  pre-load the embedding model and LLM client once at server startup (via a
  `lifespan` hook) rather than paying that cost on every request.

- **Frontend — Streamlit, fully decoupled from the backend**: Streamlit only
  calls the FastAPI `/chat` endpoint over HTTP (`requests`, 30s timeout); it
  holds no retrieval or generation logic. This keeps the backend independently
  reusable and testable, and means the frontend could be swapped for a
  different client without touching RAG logic.

- **Orchestration — LangChain (LCEL)**: used for document loading
  (`DirectoryLoader` + `TextLoader`), splitting (`MarkdownHeaderTextSplitter` +
  `RecursiveCharacterTextSplitter`), the vector store integration
  (`langchain-chroma`), and the generation chain itself, composed with LCEL
  (`prompt | llm | StrOutputParser()`).

- **Chunking — two-stage, heading-aware**: each policy document is first split
  on its `##` sub-headings (keeping each policy sub-section — e.g. "Accrual,"
  "Eligibility" — intact as one chunk with its heading captured in metadata as
  `section`), then any section still over 800 characters is further split with
  a sliding window (150-character overlap) so no chunk is unmanageably long.
  This produced 64 chunks from 8 source documents.

- **Embedding model — `sentence-transformers/all-MiniLM-L6-v2`, run locally**:
  free, no API key or rate limits, adequate quality for short policy-document
  chunks at this corpus size.

- **Vector store — Chroma, persisted locally**: zero external setup required.
  Anonymized telemetry is explicitly disabled at client construction
  (`Settings(anonymized_telemetry=False)`) — during development this telemetry
  call was found to add significant latency when network conditions were poor,
  so disabling it is both a privacy and a reliability choice.

- **Retrieval — top-k=5**: chosen to give reasonable topic coverage across a
  small, densely cross-referenced corpus (e.g. PTO and Holiday policies
  reference each other), at the cost of sometimes retrieving tangentially
  related sections alongside the most relevant one.

- **LLM — Groq, `openai/gpt-oss-20b`**: free-tier, ~1000 tokens/sec. Note:
  `llama-3.1-8b-instant`, originally selected, was found mid-project to have
  moved to Groq's Enterprise/Contact-Sales tier and was no longer reachable on
  a standard API key — a real example of a free-tier LLM API's availability
  shifting independently of application code. `openai/gpt-oss-20b` was
  substituted as the closest free-tier equivalent in speed and cost.

- **Guardrails**: the system prompt instructs the model to answer only from
  retrieved context and to return an exact fixed refusal string when the
  context is insufficient (enabling programmatic detection in evaluation).
  Output length is capped via `max_tokens` on the LLM client itself, not just
  documented — this is what actually enforces the limit.

- **Citations — structural, not LLM-generated**: rather than asking the model
  to cite its sources inline (which risks the model citing a source it didn't
  actually rely on, or hallucinating a citation), the API returns the
  retrieved chunks' real metadata (filename, section heading, snippet)
  directly alongside the generated answer. This guarantees citation accuracy
  reflects real retrieval, at the cost of not distinguishing which of the k=5
  retrieved chunks the model's answer actually drew from.

- **Warm-up at startup**: cold-starting the embedding model inside a request
  handler was measured at ~15 seconds (see Latency Notes below) — unacceptable
  for a first user. `get_retriever()` and `get_llm()` are now pre-loaded once
  in FastAPI's `lifespan` startup hook, so this cost is paid once at server
  boot, not per-request.

## 2. Latency Notes (pre-evaluation)

During development, per-request latency was observed in two regimes:
- **Cold start** (first call in a process, before the warm-up fix): ~15-27s,
  dominated by `HuggingFaceEmbeddings` initialization (PyTorch/transformers
  import and model load), not by retrieval or the LLM call itself.
- **Warm** (after startup pre-loading): ~1-2s end-to-end per question.

The formal evaluation below reports warm-state latency, since that reflects
real steady-state user experience; cold start is a one-time server-boot cost.

## 3. Evaluation Approach

*(pending — to be completed once `eval/run_eval.py` is built and run)*

## 4. Results

*(pending)*

## 5. Observations

*(pending)*