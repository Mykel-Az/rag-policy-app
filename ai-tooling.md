# AI Tooling Usage

> Briefly describe which AI coding tools you used and how, including what
> worked well and what didn't. Update this as you go rather than reconstructing
> it at the end.

## Tools Used

- **Claude** — used to scaffold the initial project structure (backend/frontend
  split, ingestion pipeline, RAG prompt, eval harness) and draft the synthetic
  policy corpus.
  - *What worked well*: fast scaffolding of boilerplate (FastAPI routes, Chroma
    ingestion, GitHub Actions workflow), and generating a coherent, varied
    policy corpus.
  - *What didn't*: (fill in — e.g., had to manually tune chunk size after
    testing retrieval quality, had to fix embedding function API version
    mismatches, etc.)

- (Add any other tools: GitHub Copilot, Cursor, ChatGPT, etc., and what you used
  each for.)

## Human Review

All AI-generated code was reviewed and tested locally before being committed.
(Note any spots where you changed AI-suggested code and why.)
