"""
retrieval.py - Part B of the assignment: the two search methods.

  1. Similarity search (configurable k)
     Pure nearest-neighbour search: embed the question, return the k chunks
     whose vectors are closest to it (smallest distance in ChromaDB's HNSW index).

  2. Semantic search with MMR (Maximal Marginal Relevance)
     Also embedding-based, but re-ranks a larger candidate pool so the results
     are relevant to the question AND different from each other. This avoids
     returning several near-identical chunks (e.g. overlapping chunks of the
     same page), so the LLM sees more distinct information.

Both are used by the RAG chain later. You can also try them from the terminal:

    python retrieval.py "When should I use RAG instead of fine-tuning?"
    python retrieval.py "What is HNSW?" --k 6
    python retrieval.py "What is HNSW?" --method mmr --lambda-mult 0.3
"""
import argparse
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config

_store = None


def load_vector_store() -> Chroma:
    """Open the persistent ChromaDB built by ingest.py (loaded once, then reused)."""
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=config.get_embeddings(),
            persist_directory=config.CHROMA_DIR,
        )
        if _store._collection.count() == 0:
            raise RuntimeError("The vector database is empty - run 'python ingest.py' first.")
    return _store


def distance_to_similarity(distance: float) -> float:
    """
    Chroma's default distance is squared L2. Our embeddings are normalised to
    length 1, so squared L2 = 2 - 2 * cosine similarity. Converting back gives a
    more intuitive score: 1.0 = identical meaning, 0 = unrelated.
    """
    return 1 - distance / 2


# ---------------------------------------------------------------------------
# Method 1: similarity search
# ---------------------------------------------------------------------------
def similarity_search(query: str, k: int = 4, store: Chroma | None = None) -> list[tuple[Document, float]]:
    """Return the k closest chunks with their cosine similarity (highest first)."""
    store = store or load_vector_store()
    results = store.similarity_search_with_score(query, k=k)
    return [(doc, distance_to_similarity(dist)) for doc, dist in results]


# ---------------------------------------------------------------------------
# Method 2: semantic search with MMR
# ---------------------------------------------------------------------------
def semantic_search(query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5,
                    store: Chroma | None = None) -> list[Document]:
    """
    MMR search.
      fetch_k     - size of the candidate pool taken from plain similarity search
      lambda_mult - 1.0 = only relevance (same as similarity search),
                    0.0 = only diversity; 0.5 balances both
    """
    store = store or load_vector_store()
    return store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)


# ---------------------------------------------------------------------------
# Small-to-big: expand small hits into larger context for the LLM
# ---------------------------------------------------------------------------
def _merge_overlapping(a: str, b: str, max_overlap: int = 300) -> str:
    """Join two consecutive chunks, removing the text they share because of chunk overlap."""
    for size in range(min(len(a), len(b), max_overlap), 10, -1):
        if a.endswith(b[:size]):
            return a + b[size:]
    return a + " " + b


def parent_key(metadata: dict) -> str:
    """Which original document a chunk belongs to (one video, one PDF, one web page)."""
    if metadata.get("doc_type") == "youtube":
        return metadata.get("video_id") or metadata.get("title", "")
    return metadata.get("source", "")


def expand_context(docs: list[Document], scores: list | None = None, window: int = 2,
                   store: Chroma | None = None) -> tuple[list[Document], list]:
    """
    Small-to-big retrieval ("parent/neighbour expansion").

    Small chunks are best for FINDING the right passage (see experiments.py),
    but too short for the LLM to ANSWER from. So for every retrieved chunk we
    add the `window` chunks before and after it from the same document (same
    PDF, web page or video), and merge hits that are next to each other so no
    text is repeated.

    Returns new (docs, scores) lists in the original ranking order.
    """
    store = store or load_vector_store()
    scores = scores if scores is not None else [None] * len(docs)

    # 1. Turn each hit into an id range, merging overlapping ranges of the same document
    groups = []  # each: {"key", "lo", "hi", "score", "doc"} or a passthrough {"doc", "score"}
    for doc, score in zip(docs, scores):
        m = doc.metadata
        if "chunk_id" not in m:
            groups.append({"doc": doc, "score": score})
            continue
        key, cid = parent_key(m), int(m["chunk_id"])
        lo, hi = cid - window, cid + window
        for g in groups:
            if g.get("key") == key and lo <= g["hi"] + 1 and hi >= g["lo"] - 1:
                g["lo"], g["hi"] = min(g["lo"], lo), max(g["hi"], hi)
                break
        else:
            groups.append({"key": key, "lo": lo, "hi": hi, "score": score, "doc": doc})

    # 2. Fetch the neighbour chunks for all ranges in one database call
    wanted = [f"chunk-{i}" for g in groups if "key" in g for i in range(max(g["lo"], 0), g["hi"] + 1)]
    fetched = store.get(ids=wanted, include=["documents", "metadatas"]) if wanted else {"ids": []}
    by_id = {int(i.split("-")[1]): (text, meta) for i, text, meta
             in zip(fetched["ids"], fetched.get("documents", []), fetched.get("metadatas", []))}

    # 3. Build one expanded Document per group
    out_docs, out_scores = [], []
    for g in groups:
        parts = []
        if "key" in g:
            parts = [by_id[i] for i in range(g["lo"], g["hi"] + 1)
                     if i in by_id and parent_key(by_id[i][1]) == g["key"]]
        if not parts:
            out_docs.append(g["doc"])
            out_scores.append(g["score"])
            continue
        text = parts[0][0]
        for t, _ in parts[1:]:
            text = _merge_overlapping(text, t)
        first_meta = parts[0][1]
        meta = dict(g["doc"].metadata)
        if meta.get("doc_type") == "pdf":
            pages = sorted({int(p_meta.get("page", 0)) for _, p_meta in parts})
            meta["page"] = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
        elif meta.get("doc_type") == "youtube":
            # cite (and link to) the start of the expanded passage
            for field in ("timestamp", "start_seconds", "source"):
                if field in first_meta:
                    meta[field] = first_meta[field]
        out_docs.append(Document(page_content=text, metadata=meta))
        out_scores.append(g["score"])
    return out_docs, out_scores


# ---------------------------------------------------------------------------
# Helpers for printing / citations
# ---------------------------------------------------------------------------
def describe(doc: Document) -> str:
    """Short human-readable source label, e.g. 'pdf | lewis_2020_rag_original.pdf p. 3'."""
    m = doc.metadata
    if m.get("doc_type") == "pdf":
        where = f"{m.get('source')} p. {m.get('page')}"
    elif m.get("doc_type") == "youtube":
        where = f"{m.get('title')} @ {m.get('timestamp')}"
    else:
        where = m.get("title") or m.get("source")
    return f"{m.get('doc_type', '?'):<7} | {where}"


def print_results(title: str, rows: list[tuple[Document, float | None]], seconds: float):
    print(f"\n{title}  ({seconds * 1000:.0f} ms)")
    for i, (doc, score) in enumerate(rows, 1):
        score_text = f"sim {score:.3f}" if score is not None else " " * 9
        snippet = " ".join(doc.page_content.split())[:110]
        print(f"  {i}. {score_text} | {describe(doc)}")
        print(f"       {snippet}...")


def main():
    parser = argparse.ArgumentParser(description="Try the two search methods on the vector database.")
    parser.add_argument("query", help="the question to search for")
    parser.add_argument("--k", type=int, default=4, help="number of chunks to return (default 4)")
    parser.add_argument("--method", choices=["similarity", "mmr", "both"], default="both")
    parser.add_argument("--fetch-k", type=int, default=20, help="MMR candidate pool size (default 20)")
    parser.add_argument("--lambda-mult", type=float, default=0.5, help="MMR relevance/diversity balance (default 0.5)")
    args = parser.parse_args()

    store = load_vector_store()
    store.similarity_search(args.query, k=1)  # warm-up so timings exclude model loading

    print(f"Query: {args.query!r}   k={args.k}")

    if args.method in ("similarity", "both"):
        t = time.perf_counter()
        rows = similarity_search(args.query, k=args.k, store=store)
        print_results("Similarity search", rows, time.perf_counter() - t)

    if args.method in ("mmr", "both"):
        t = time.perf_counter()
        docs = semantic_search(args.query, k=args.k, fetch_k=args.fetch_k,
                               lambda_mult=args.lambda_mult, store=store)
        print_results(f"Semantic search (MMR, fetch_k={args.fetch_k}, lambda={args.lambda_mult})",
                      [(d, None) for d in docs], time.perf_counter() - t)


if __name__ == "__main__":
    main()