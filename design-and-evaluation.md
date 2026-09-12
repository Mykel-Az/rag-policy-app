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

- **Question set**: 30 questions in eval/eval_questions.json spanning all 8
  policy documents (~3-4 questions per doc), plus 2 intentionally
  out-of-scope questions to test the refusal guardrail.
- **Metrics** (see eval/run_eval.py):
  - **Groundedness**: LLM-as-judge (openai/gpt-oss-120b — a different, larger
    model than the one that generates answers, to avoid self-preference bias)
    checks whether every claim in the answer is supported by the retrieved
    context. Correct refusals are counted as trivially grounded rather than
    judged against context, since a refusal makes no factual claims to check.
  - **Citation accuracy**: for answerable questions, whether the expected
    source document appears among the returned citations. For the two
    out-of-scope questions, whether the app returned the exact refusal
    message.
  - **Latency (p50/p95)**: measured end-to-end per question, in a warm
    process (embedding model and LLM client pre-loaded before the timed loop
    began, matching the FastAPI `lifespan` warm-up behavior in production).

## 4. Results

\`\`\`json
{
  "num_questions": 30,
  "groundedness_pct": 96.7,
  "citation_accuracy_pct": 96.7,
  "latency_p50_ms": 641.6,
  "latency_p95_ms": 1133.5
}
\`\`\`
*(Re-run after the Q29 groundedness-judging fix below; expect groundedness_pct to reach 100.0.)*

## 5. Observations

- **Citation accuracy (29/30) — question 30 mislabeling, not a system failure**:
  question 30 ("Can I expense a personal vacation to Hawaii under the
  wellness benefit?") was designed as an out-of-scope guardrail test, but on
  inspection it's actually answerable by synthesizing two real policies — the
  Wellness Benefit definition (benefits-policy.md) and the general "expenses
  must not be primarily personal" principle (expense-policy.md). The model
  correctly declined to reimburse the expense with sound reasoning across
  both documents, rather than issuing a blanket refusal. The eval's citation
  check marked this "incorrect" only because it expected an exact-match
  refusal — the app's actual behavior (grounded multi-document synthesis) was
  arguably the better outcome. This is a lesson about eval-set construction:
  a question can look out-of-scope on its face while still being answerable
  through cross-document reasoning, and a binary refuse-or-cite-exact-source
  check doesn't capture that nuance.

- **Groundedness (originally 29/30, a methodology bug, not a model failure)**:
  the one failing case (question 29) was a correctly-refused, clearly
  out-of-scope question ("pet care leave on the moon") whose refusal text was
  then judged for whether it was "supported by the retrieved context" — an
  incoherent check, since a refusal asserts nothing that needs grounding.
  Fixed by skipping the groundedness judge for exact-match refusals.

- **Latency**: p50 of ~640ms and p95 of ~1.1s in the warm state, well within
  the range of a responsive chat experience. This depends entirely on the
  FastAPI `lifespan` startup hook pre-loading the embedding model and LLM
  client — without it, the first request in any process pays a one-time
  ~15-26s cold-start cost dominated by `sentence-transformers`/PyTorch
  initialization (see Latency Notes in Section 2).