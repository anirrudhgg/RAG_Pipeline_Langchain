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
# results in results/chunking_summary.csv). Overlap 100 beat 0 at every size, and
# 300/100 is tied for best overall (MRR 0.82, with 1000/100) and best for PDFs (MRR 0.89),
# which are most of the knowledge base. (Before Wikipedia's reference list was removed it
# scored 0.86: that list had counted as a 'hit' for the Wikipedia question.)
#
# Per type, the experiment preferred web 1000 and video unsplit (~1000-1200 chars).
# But mixing sizes biased retrieval: 1,500 short PDF chunks outranked the long video
# chunks (diagnose.py: the right video chunk fell to rank 28 for "When should I use
# RAG instead of fine-tuning?"), and video chunks over ~1000 chars were also cut off
# by the embedding model's 256-token limit. So all types use the same size, and
# small-to-big expansion (CONTEXT_WINDOW) gives the LLM enough text to answer from.
CHUNK_SETTINGS = {
    "pdf": {"chunk_size": 300, "chunk_overlap": 100},
    "web": {"chunk_size": 300, "chunk_overlap": 100},
    "youtube": {"chunk_size": 300, "chunk_overlap": 100},
}

# --- Retrieval defaults ---------------------------------------------------------
# MMR beat plain similarity search in experiments.py (hit@4 1.00 vs 0.92, more
# distinct results). k=6 because the small PDF chunks carry less text each.
DEFAULT_K = 6
DEFAULT_SEARCH = "mmr"

# Small-to-big retrieval: the LLM gets each retrieved chunk plus this many
# neighbouring chunks from the same document on each side
# (~250 chars -> ~800 chars of continuous text).
# Small chunks are best for finding passages, but too short to answer from.
# Set to 0 to switch it off (e.g. to compare answer quality with and without).
CONTEXT_WINDOW = 2

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