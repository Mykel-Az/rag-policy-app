FROM python:3.14-slim

WORKDIR /app

# System deps some Python packages (torch, chromadb) may need at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY corpus/ ./corpus/
COPY conftest.py .

COPY docker/backend-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV HF_HUB_OFFLINE=0
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]