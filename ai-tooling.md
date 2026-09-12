# AI Tooling Usage

## Tools Used

- **Claude (Anthropic)** — used as a guided pair-programming partner throughout
  this project, not as a code generator. The working process was: Claude
  explained one function or design decision at a time, presented real
  trade-offs (e.g. LangChain vs. manual chunking, structural vs. prompt-based
  citations, chunk size/k/embedding model choices), and I wrote and ran the
  code myself, one piece at a time, in my own editor and terminal.

  - **What worked well**: this piece-by-piece approach meant I understood
    *why* each design choice was made (e.g. why `MarkdownHeaderTextSplitter`
    needs a manual metadata-merge step but `RecursiveCharacterTextSplitter`
    doesn't; why citations are structural rather than LLM-generated) rather
    than just having working code I didn't fully own. It was also effective
    for debugging: when `python -m backend.rag_chain` returned a 26+ second
    latency, Claude walked me through isolating the cause step by step
    (timing instrumentation → ruling out Chroma telemetry → ruling out HF Hub
    network checks → isolating it to `HuggingFaceEmbeddings` cold-start) rather
    than guessing at a fix outright.

  - **What didn't work initially**: early in the project, Claude generated a
    full working implementation of the entire application in one pass instead
    of guiding me through writing it myself. I corrected this and the process
    switched to the piece-by-piece guided approach used for the rest of the
    build.

  - **Real-world debugging surfaced along the way**:
    - Groq's `llama-3.1-8b-instant` model moved to an Enterprise/Contact-Sales
      tier mid-project and returned a 404 on a standard API key; Claude
      searched Groq's current model documentation to find a still-available
      free-tier equivalent (`openai/gpt-oss-20b`).
    - Chroma's default client sends anonymized telemetry to PostHog on
      construction, which was slow on my network; disabled explicitly via
      `Settings(anonymized_telemetry=False)`.
    - `sentence-transformers`/HuggingFace makes a network call to check for
      model updates on every instantiation, adding latency; addressed via
      `HF_HUB_OFFLINE` and, more fundamentally, by only instantiating the
      embedding model once per server process (singleton pattern) rather than
      per-request.
    - `pytest` failed to import `backend` as a module until an empty
      `conftest.py` was added at the project root, so pytest's rootdir
      discovery would add the project root to `sys.path`.

## Human Review

All code in this repository was written by me, based on explanations and code
snippets provided turn-by-turn by Claude. Every snippet was typed in manually,
run, and its output (including errors) fed back into the conversation before
moving to the next piece — nothing was pasted in wholesale without being run
and understood first.