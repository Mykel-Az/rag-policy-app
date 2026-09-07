import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Path
CORPUS_DIR = BASE_DIR/ "corpus"
VECTORSTORE_DIR = BASE_DIR / "backend" / "vectorstore" /"chroma_db"
COLLECTION_NAME = "policy_docs"

RANDOM_SEED = 42

CHUNK_SIZE = 800        
CHUNK_OVERLAP = 150     

TOP_K = 5

# --- Embedding model (local, free, no API key) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM (Groq free tier) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# --- Guardrails ---
MAX_ANSWER_TOKENS = 400
OUT_OF_SCOPE_MESSAGE = (
    "I can only answer questions about our company policies."
)


