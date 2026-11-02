"""
diagnose.py - Where does each document type rank for a question?

Shows, for each question, the best-ranked chunk of every document type
(pdf / web / youtube) among ALL chunks in the database, with its similarity.
Used to investigate why a question misses the source it should find.

    python diagnose.py
    python diagnose.py "your own question"
"""
import sys

import retrieval

QUESTIONS = sys.argv[1:] or [
    "When should I use RAG instead of fine-tuning?",
    "When should you use RAG instead of fine-tuning a model?",
    "What is the difference between RAG-Sequence and RAG-Token?",
]

store = retrieval.load_vector_store()
total = store._collection.count()

for q in QUESTIONS:
    results = retrieval.similarity_search(q, k=total)
    print(f"\nQ: {q}")
    seen = set()
    for rank, (doc, sim) in enumerate(results, 1):
        t = doc.metadata.get("doc_type")
        if t not in seen:
            seen.add(t)
            print(f"  best {t:<7} rank {rank:>4} of {total} | sim {sim:.3f} | {retrieval.describe(doc)} "
                  f"| {len(doc.page_content)} chars")
    mmr = retrieval.semantic_search(q, k=6)
    print("  MMR top 6 types:", [d.metadata.get("doc_type") for d in mmr])
