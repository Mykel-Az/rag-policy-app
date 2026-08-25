# Policy RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers employee questions
about company policies (PTO, remote work, security, expenses, holidays, code of
conduct, onboarding, and benefits) with citations back to the source policy documents.

## Architecture

- **Backend**: FastAPI (`backend/`) — owns ingestion, retrieval (Chroma), and
  generation (Groq LLM via an OpenAI-compatible API). Exposes `/chat` and `/health`.
- **Frontend**: Streamlit (`frontend/streamlit_app.py`) — a thin chat UI that calls
  the backend over HTTP. No RAG logic lives here.
- **Vector store**: Chroma, persisted locally to `backend/vectorstore/`.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, run locally (no API key
  required).

```
corpus/         -> policy source documents (markdown)
backend/        -> FastAPI app, ingestion, RAG pipeline
frontend/       -> Streamlit chat UI (calls the backend API)
eval/           -> evaluation question set + eval harness
tests/          -> smoke tests used by CI
.github/        -> GitHub Actions CI workflow
```

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # then edit .env and add your free Groq API key: https://console.groq.com
   ```

## Run

1. **Build the vector index** (run once, or whenever the corpus changes):

   ```bash
   python -m backend.ingest
   ```

2. **Start the backend API**:

   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

   Visit `http://localhost:8000/docs` for interactive API docs.

3. **Start the frontend** (in a second terminal, with the venv activated):

   ```bash
   streamlit run frontend/streamlit_app.py
   ```

   Visit `http://localhost:8501` to chat with the assistant.

## Evaluation

With the backend importable and `.env` configured, run:

```bash
python -m eval.run_eval
```

This scores groundedness, citation accuracy, and latency (p50/p95) over the
question set in `eval/eval_questions.json`, and writes full results to
`eval/eval_results.json`. See `design-and-evaluation.md` for a summary of results.

## Tests / CI

```bash
pytest -q
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same install + import + test
steps on every push/PR to `main`.

## Deployment (optional)

See `deployed.md` for the live URL, if deployed.
