"""
config.py - Shared settings for the RAG pipeline.

Every script (ingest, chain, app, evaluation) imports from here, so the
embedding model, database location and chunk settings are always identical.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Data locations ---------------------------------------------------------
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
URLS_FILE = os.path.join(DATA_DIR, "urls.txt")
YOUTUBE_FILE = os.path.join(DATA_DIR, "youtube.txt")

# --- Vector database ----------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

# --- Models -----------------------------------------------------------------
# all-MiniLM-L6-v2: 384-dimensional sentence embeddings, small and fast, runs
# fine on CPU. Note: it only reads the first ~256 tokens (~1000 characters)
# of a chunk, so chunks much longer than that get cut off when embedded.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# --- Chunking (in characters), per document type -------------------------------
# Chosen with experiments.py (15 size/overlap combinations, 12 test questions,
# results in results/chunking_summary.csv). Overlap 100 beat 0 and 200 for every size.
CHUNK_SETTINGS = {
    "pdf": {"chunk_size": 300, "chunk_overlap": 100},       # dense papers: small chunks match precisely (MRR 0.89)
    "web": {"chunk_size": 1000, "chunk_overlap": 100},      # doc pages: keep a whole section together (MRR 0.88)
    "youtube": {"chunk_size": 1500, "chunk_overlap": 0},    # = never split a 60 s window; windows are natural units (MRR 1.00)
}

# --- Retrieval defaults ---------------------------------------------------------
# MMR beat plain similarity search in experiments.py (hit@4 1.00 vs 0.92, more
# distinct results). k=6 because the small PDF chunks carry less text each.
DEFAULT_K = 6
DEFAULT_SEARCH = "mmr"

# Length of one YouTube transcript window in seconds (keeps timestamps for citations)
YOUTUBE_WINDOW_SECONDS = 60


def get_embeddings():
    """Return the embedding model used for both indexing and querying."""
    from langchain_huggingface import HuggingFaceEmbeddings

    # Normalised vectors -> L2 distance and cosine similarity give the same ranking
    # (squared L2 distance = 2 - 2 * cosine similarity).
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )