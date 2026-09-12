# Design and Evaluation

## 1. Design & Architecture Decisions

- **Backend framework — FastAPI**: chosen over Flask for built-in request/response
  validation (Pydantic), automatic OpenAPI docs, and a clean way to pre-load the
  embedding model and LLM client once at server startup (via a `lifespan` hook)
  rather than paying that cost on every request.

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
  "Eligibility" — intact as one chunk, with its heading captured in metadata as
  `section`), then any section still over 800 characters is further split with
  a sliding window (150-character overlap) so no chunk is unmanageably long.
  This produced 64 chunks from 8 source documents.

- **Embedding model — `sentence-transformers/all-MiniLM-L6-v2`, run locally**:
  free, no API key or rate limits, adequate quality for short policy-document
  chunks at this corpus size.

- **Vector store — Chroma, persisted locally**: zero external setup required.
  Anonymized telemetry is explicitly disabled at client construction
  (`Settings(anonymized_telemetry=False)`) — during development this telemetry
  call was found to add significant latency under some network conditions, so
  disabling it is both a privacy and a reliability choice.

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

- **Groundedness judge — a separate, larger model (`openai/gpt-oss-120b`)**:
  using a different model than the one that generates answers avoids
  self-preference bias (a model rating its own outputs favorably).

- **Guardrails**: the system prompt instructs the model to answer only from
  retrieved context and to return an exact fixed refusal string when the
  context is insufficient (enabling programmatic detection in evaluation).
  Output length is capped via `max_tokens` on the LLM client itself, not just
  documented — this is what actually enforces the limit.

- **Citations — structural, not LLM-generated**: rather than asking the model
  to cite its sources inline (which risks the model citing a source it didn't
  actually rely on, or hallucinating a citation), the API returns the
  retrieved chunks' real metadata