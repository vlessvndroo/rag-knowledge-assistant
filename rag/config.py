import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # must match vector(768) in sql/001_init.sql

# Chunking parameters, in characters. Tuned later by the evals harness (week 3).
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
