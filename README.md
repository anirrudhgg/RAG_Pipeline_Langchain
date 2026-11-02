# RAG Pipeline with LangChain

A Retrieval-Augmented Generation (RAG) system that answers questions about RAG itself, using only a curated knowledge base of research papers, documentation pages and videos. Every answer cites its sources, and the system says "I don't know" when the documents don't contain the answer.

Built with LangChain, ChromaDB, sentence-transformers and Llama 3.1 8B running locally through Ollama.

<!-- Add the architecture diagram here once it's ready, e.g. ![Architecture](docs/architecture.png) -->

## Features

- **Three document types**: PDFs (5 papers), web pages (4 documentation pages), YouTube transcripts (2 videos)
- **Two search methods**: similarity search with configurable k, and semantic search with MMR (Maximal Marginal Relevance)
- **Small-to-big retrieval**: search with small chunks for precision, then give the LLM each hit plus its neighbouring chunks
- **Four prompt styles**: zero-shot, few-shot, chain-of-thought, tree-of-thoughts
- **Citations** on every answer and a **"don't know" fallback** for questions outside the knowledge base
- **Streamlit UI** with a side-by-side comparison against the plain LLM (no retrieval)
- **Experiments** for chunk size, overlap and search method, with results saved as CSV

## Knowledge base

| Type | Sources |
|---|---|
| PDF | Lewis et al. 2020 (original RAG paper); Fan et al. 2024 (RAG meets LLMs survey, KDD); Gao et al. 2025 (RAG and reasoning review); Klesel & Wittmann 2025 (RAG catchword, BISE); Arslan et al. 2024 (RAG with LLMs survey, Procedia CS) |
| Web | Wikipedia: Retrieval-augmented generation; LangChain docs: Retrieval; Chroma docs: Embedding functions; Chroma docs: Configure collections |
| Video | IBM Technology: What is RAG?; IBM Technology: RAG vs. Fine-Tuning |

The files and links are in `data/`. Web pages and video transcripts are fetched when the database is built.

## Setup

**Requirements:** Python 3.10+ (developed on 3.12), [Ollama](https://ollama.com), about 6 GB of disk space for the model.

1. **Clone the repository and create a virtual environment**
   ```
   git clone https://github.com/anirrudhgg/RAG_Pipeline_Langchain.git
   cd RAG_Pipeline_Langchain
   python -m venv venv
   venv\Scripts\activate          # Windows
   source venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```

2. **Optional: GPU acceleration for embeddings** (NVIDIA GPUs; RTX 50-series needs CUDA 12.8)
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
   ```
   Embedding the knowledge base takes about 8 seconds on an RTX 5070, compared with about 3 minutes on CPU.

3. **Download the LLM**
   ```
   ollama pull llama3.1:8b
   ```
   On a laptop without a strong GPU, a smaller model works too: `ollama pull llama3.2:3b`, then set it in `.env`.

4. **Create the `.env` file** by copying `.env.example`:
   ```
   OLLAMA_MODEL=llama3.1:8b
   ```

5. **Build the vector database**
   ```
   python ingest.py
   ```
   This loads all documents, cleans and chunks them, embeds them and stores them in `chroma_db/`. Run it again whenever the documents or chunk settings change.

## Usage

**Web interface (for the demo)**
```
streamlit run app.py
```
Opens at http://localhost:8501. Choose the prompt style, search method, k and context window in the sidebar, and switch the plain-LLM comparison on or off.

**Terminal**
```
python rag_chain.py "What is the difference between RAG-Sequence and RAG-Token?"
python rag_chain.py "What is the capital of Australia?" --compare
python rag_chain.py "When should I use RAG instead of fine-tuning?" --style Chain-of-thought
python retrieval.py "What is HNSW?" --k 6            # compare the two search methods
python prompts.py                                    # preview all prompt templates
```

**Experiments and diagnostics**
```
python experiments.py     # chunk size / overlap grid and similarity vs MMR, saves results/*.csv
python diagnose.py        # where each document type ranks for a question
```

## How it works

```
question
  → MMR search over 1,845 chunks in ChromaDB (k=6)
  → small-to-big: each hit + 2 neighbouring chunks from the same document
  → numbered sources inserted into the prompt template
  → Llama 3.1 8B (temperature 0, 8k context)
  → answer with [n] citations, or "I don't know based on the provided documents."
```

| Step | Choice | Why |
|---|---|---|
| Loading | PyPDFLoader, WebBaseLoader, youtube-transcript-api | One LangChain-compatible loader per document type; transcripts grouped into 60-second windows with timestamps |
| Cleaning | Reference lists removed from PDFs and web pages | Bibliographies matched almost every query and ranked #1 without containing any explanation |
| Chunking | 300 characters, 100 overlap, for all types | Best overall setting in the experiments (MRR 0.86); equal sizes stop short chunks from outranking long ones |
| Embeddings | all-MiniLM-L6-v2, normalised | Small, fast, free, runs locally; normalised vectors make L2 distance equivalent to cosine similarity |
| Vector DB | ChromaDB (HNSW index), persistent | Local, simple, stores metadata (source, page, timestamp) for citations |
| Retrieval | MMR, k=6 | Same accuracy as similarity search but more diverse results (3.3 vs 2.8 distinct sources in the top 4) |
| Context | Small-to-big, window 2 | Small chunks retrieve best but are too short to answer from |
| LLM | Llama 3.1 8B via Ollama, temperature 0 | Free, private, runs on the GPU; temperature 0 makes the evaluation repeatable |

## Results

### Chunking experiments
15 chunk size / overlap combinations, 12 test questions with known sources, k=4 (`results/chunking_summary.csv`).

| Setting | Chunks | hit@1 | hit@4 | MRR |
|---|---|---|---|---|
| **300 / 100** | 1,911 | 0.75 | 1.00 | **0.86** |
| 1000 / 100 | 508 | 0.67 | 1.00 | 0.82 |
| 300 / 0 | 1,621 | 0.75 | 0.83 | 0.79 |
| 1000 / 0 | 489 | 0.50 | 0.92 | 0.67 |
| 500 / 0 | 957 | 0.50 | 0.83 | 0.62 |

Overlap 100 beat 0 and 200 at every size. The test set is small (12 questions), so trends matter more than exact decimals.

### Search methods (final settings)
| Method | hit@4 | MRR | Distinct results in top 4 |
|---|---|---|---|
| Similarity | 1.00 | 0.86 | 2.8 |
| MMR | 1.00 | 0.86 | 3.3 |

### Response times (RTX 5070)
| Step | Time |
|---|---|
| Building the database (embedding 1,845 chunks) | ~8 s (GPU) vs ~185 s (CPU) |
| Retrieval | ~10 ms |
| Generation (zero-shot) | ~1.5–2 s |
| Generation (chain-of-thought) | ~3 s |

### RAG vs plain LLM
<!-- TODO: fill in after running evaluate.py -->
*To be added: results of `evaluate.py` on the 10 questions in `eval_questions.csv` (6 inside, 4 outside the knowledge base).*

## Challenges and solutions

1. **Bibliographies ranked first.** The first test search returned a paper's reference list as the top result. Fix: cut PDFs at their "References" heading (keeping appendices), and later Wikipedia's reference section too, once smaller web chunks exposed the same problem.
2. **Small chunks gave thin answers.** 300-character chunks retrieved best, but the model could not find the RAG-Sequence definition in them. Fix: small-to-big retrieval, adding neighbouring chunks before generation. Same retrieved chunks, temperature 0, clearly better answer.
3. **Mixed chunk sizes buried the videos.** With small PDF chunks and unsplit video windows, the right video chunk ranked 28th for "When should I use RAG instead of fine-tuning?" (`diagnose.py`). Long chunks were also truncated by the embedding model's 256-token limit. Fix: the same chunk size for all types; the video then ranked 14th and was retrieved by MMR.
4. **Ollama's default context window is small.** Few-shot and tree-of-thoughts prompts with several sources could be cut off, losing the rules at the top. Fix: `num_ctx=8192`.

**Known limitations:** Llama 3.1 8B does not always cite every statement (often only one citation per answer); PDF page headers and footers appear as noise in some chunks; the test sets are small.

## Project structure

```
├── app.py                 Streamlit UI
├── rag_chain.py           RAG chain (LCEL) and plain-LLM baseline
├── prompts.py             Prompt templates, citation rules, "don't know" fallback
├── retrieval.py           Similarity search, MMR, small-to-big expansion
├── ingest.py              Loading, cleaning, chunking, embedding, storing
├── config.py              All settings (paths, models, chunk sizes, k, context window)
├── experiments.py         Chunking and search-method experiments
├── diagnose.py            Rank of each document type for a question
├── test_questions.py      Retrieval test questions with known sources
├── eval_questions.csv     Evaluation questions (inside / outside the knowledge base)
├── data/                  PDFs, web page URLs, YouTube URLs
├── results/               Experiment output (CSV)
└── .streamlit/config.toml Streamlit settings
```

## Team

<!-- Add names and who did what -->
