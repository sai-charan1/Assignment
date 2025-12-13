# Configurable parameters for the MVP
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # change to any sentence-transformers model
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # for re-ranking
CHUNK_MIN = 200
CHUNK_MAX = 800
OVERLAP = 50
TOP_K = 5
