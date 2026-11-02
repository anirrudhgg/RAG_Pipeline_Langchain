"""
test_questions.py - Questions with a known correct source, used to measure
retrieval quality in the chunking experiments.

Each entry: (question, doc_type it targets, list of source substrings).
A retrieval counts as a "hit" if any of the top-k chunks comes from a source
containing one of the substrings.

Add or change questions here - keep them answerable from exactly one or two
known documents, so the measurement stays meaningful.
"""

TEST_QUESTIONS = [
    # --- PDFs ---
    ("What are the two components of the original RAG model and which retriever does it use?",
     "pdf", ["lewis_2020"]),
    ("What is the difference between RAG-Sequence and RAG-Token?",
     "pdf", ["lewis_2020"]),
    ("From which three perspectives are retrieval-augmented LLMs reviewed: retrieval, generation and what else?",
     "pdf", ["fan_2024"]),
    ("How can retrieval and reasoning work together, and what is reasoning-augmented retrieval?",
     "pdf", ["gao_2025"]),
    ("What research questions about RAG are relevant for Business and Information Systems Engineering?",
     "pdf", ["klesel_2025"]),
    ("How are RAG applications classified into task-based and discipline-based categories?",
     "pdf", ["procedia_2024"]),

    # --- Web pages ---
    ("Which distance functions can a Chroma collection use for its HNSW index?",
     "web", ["collections/configure"]),
    ("What is the default embedding function in Chroma?",
     "web", ["embedding-functions"]),
    ("What is the difference between 2-step RAG and agentic RAG?",
     "web", ["langchain/retrieval"]),
    ("Who introduced the term retrieval-augmented generation and when?",
     "web", ["wikipedia", "lewis_2020"]),

    # --- YouTube ---
    ("When should you use RAG instead of fine-tuning a model?",
     "youtube", ["00Q0G84kq3M"]),
    ("Which planet has the most moons, and why can an LLM answer this wrongly without retrieval?",
     "youtube", ["T-D1OfcDW1M"]),
]
