"""
experiments.py - Chunking experiments (Part A) and search-method comparison (Part B).

What it does:
  1. Loads all documents once (same loaders as ingest.py)
  2. For every chunk size / overlap combination:
       - splits the documents, embeds them into a temporary in-memory ChromaDB
       - runs the test questions and checks whether the correct source is found
  3. Compares similarity search vs. MMR semantic search on the current settings
  4. Saves everything as CSV files in results/ for the presentation slides

Metrics per setting (computed per document type, so each type gets its own optimum):
  hit@1  - correct source is the top result
  hit@4  - correct source is somewhere in the top 4 results
  MRR    - mean reciprocal rank: 1 if correct source is 1st, 0.5 if 2nd, ... (higher = better)

Run (takes a few minutes; much faster on GPU):
    python experiments.py
"""
import os
import time
import uuid

import pandas as pd
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import ingest
from test_questions import TEST_QUESTIONS

CHUNK_SIZES = [300, 500, 800, 1000, 1500]
CHUNK_OVERLAPS = [0, 100, 200]
K = 4
RESULTS_DIR = "results"


def first_hit_rank(docs, expected_sources) -> int | None:
    """1-based rank of the first chunk from a correct source, or None if not found."""
    for rank, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "")
        if any(s in source for s in expected_sources):
            return rank
    return None


def build_temp_store(chunks, embeddings) -> Chroma:
    """In-memory ChromaDB collection (nothing written to disk)."""
    store = Chroma(collection_name=f"exp_{uuid.uuid4().hex[:8]}", embedding_function=embeddings)
    for i in range(0, len(chunks), 500):
        store.add_documents(chunks[i:i + 500])
    return store


def evaluate(store, search) -> list[dict]:
    rows = []
    for question, doc_type, expected in TEST_QUESTIONS:
        docs = search(store, question)
        rank = first_hit_rank(docs, expected)
        rows.append({
            "question": question,
            "doc_type": doc_type,
            "rank": rank,
            "hit@1": int(rank == 1),
            f"hit@{K}": int(rank is not None),
            "rr": 1 / rank if rank else 0.0,
            "top_sources": " ; ".join(ingest_describe(d) for d in docs),
        })
    return rows


def ingest_describe(doc) -> str:
    m = doc.metadata
    if m.get("doc_type") == "pdf":
        return f"{m['source']} p.{m.get('page')}"
    if m.get("doc_type") == "youtube":
        return f"{m.get('title')} @{m.get('timestamp')}"
    return m.get("source", "?")


def similarity(store, q):
    return store.similarity_search(q, k=K)


def mmr(store, q):
    return store.max_marginal_relevance_search(q, k=K, fetch_k=20, lambda_mult=0.5)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading documents once...")
    docs = ingest.load_pdfs() + ingest.load_web() + ingest.load_youtube()
    embeddings = config.get_embeddings()
    embeddings.embed_query("warm-up")  # load the model before timing

    # ------------------------------------------------------------------
    # 1. Chunking grid
    # ------------------------------------------------------------------
    summary, details = [], []
    settings = [(s, o) for s in CHUNK_SIZES for o in CHUNK_OVERLAPS if o < s]
    print(f"\nRunning {len(settings)} chunk settings x {len(TEST_QUESTIONS)} questions (k={K})\n")

    for n, (size, overlap) in enumerate(settings, 1):
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        chunks = splitter.split_documents(docs)

        t0 = time.perf_counter()
        store = build_temp_store(chunks, embeddings)
        embed_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        rows = evaluate(store, similarity)
        query_ms = (time.perf_counter() - t0) / len(TEST_QUESTIONS) * 1000
        store.delete_collection()

        for r in rows:
            details.append({"chunk_size": size, "overlap": overlap, **r})

        df = pd.DataFrame(rows)
        for doc_type, part in [("all", df)] + list(df.groupby("doc_type")):
            n_chunks = len(chunks) if doc_type == "all" else sum(c.metadata["doc_type"] == doc_type for c in chunks)
            summary.append({
                "chunk_size": size, "overlap": overlap, "doc_type": doc_type,
                "chunks": n_chunks,
                "hit@1": part["hit@1"].mean(), f"hit@{K}": part[f"hit@{K}"].mean(), "MRR": part["rr"].mean(),
                "embed_s": embed_s if doc_type == "all" else None,
                "query_ms": query_ms if doc_type == "all" else None,
            })

        print(f"  [{n:>2}/{len(settings)}] size {size:>4}, overlap {overlap:>3}: {len(chunks):>5} chunks | "
              f"hit@1 {df['hit@1'].mean():.2f} | hit@{K} {df[f'hit@{K}'].mean():.2f} | "
              f"MRR {df['rr'].mean():.2f} | embed {embed_s:.1f}s")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "chunking_summary.csv"), index=False)
    pd.DataFrame(details).to_csv(os.path.join(RESULTS_DIR, "chunking_details.csv"), index=False)

    # Best setting per document type: highest MRR, ties broken by hit@1 then fewer chunks
    print("\nBest setting per document type (by MRR, then hit@1, then fewer chunks):")
    best_rows = []
    for doc_type, part in summary_df[summary_df.doc_type != "all"].groupby("doc_type"):
        best = part.sort_values(["MRR", "hit@1", "chunks"], ascending=[False, False, True]).iloc[0]
        best_rows.append(best)
        print(f"  {doc_type:<8} size {int(best.chunk_size):>4}, overlap {int(best.overlap):>3}  "
              f"(MRR {best.MRR:.2f}, hit@1 {best['hit@1']:.2f}, hit@{K} {best[f'hit@{K}']:.2f})")
    pd.DataFrame(best_rows).to_csv(os.path.join(RESULTS_DIR, "chunking_best.csv"), index=False)

    # ------------------------------------------------------------------
    # 2. Similarity vs. MMR on the current config.py settings
    # ------------------------------------------------------------------
    print("\nComparing similarity search vs. MMR semantic search (current config.py chunk settings)...")
    chunks = ingest.split_documents(docs)
    store = build_temp_store(chunks, embeddings)
    comparison = []
    for name, fn in [("similarity", similarity), ("mmr", mmr)]:
        t0 = time.perf_counter()
        rows = evaluate(store, fn)
        ms = (time.perf_counter() - t0) / len(TEST_QUESTIONS) * 1000
        # how many distinct sources appear in the top-k (diversity)
        distinct = sum(len(set(r["top_sources"].split(" ; "))) for r in rows) / len(rows)
        for r in rows:
            r["method"] = name
        comparison.extend(rows)
        df = pd.DataFrame(rows)
        print(f"  {name:<10} hit@1 {df['hit@1'].mean():.2f} | hit@{K} {df[f'hit@{K}'].mean():.2f} | "
              f"MRR {df['rr'].mean():.2f} | distinct results in top {K}: {distinct:.1f} | {ms:.0f} ms/query")
    store.delete_collection()
    pd.DataFrame(comparison).to_csv(os.path.join(RESULTS_DIR, "search_method_comparison.csv"), index=False)

    print(f"\nSaved CSV files in '{RESULTS_DIR}/':")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")
    print("\nNext: put the best values into CHUNK_SETTINGS in config.py and run 'python ingest.py' again.")


if __name__ == "__main__":
    main()
