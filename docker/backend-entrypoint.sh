#!/bin/sh
set -e

VECTORSTORE_DIR="/app/backend/vectorstore/chroma_db"

if [ ! -d "$VECTORSTORE_DIR" ] || [ -z "$(ls -A "$VECTORSTORE_DIR" 2>/dev/null)" ]; then
    echo "No existing vector index found at $VECTORSTORE_DIR — running ingestion..."
    python -m backend.ingest
else
    echo "Existing vector index found at $VECTORSTORE_DIR — skipping ingestion."
fi

echo "Starting FastAPI server..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000