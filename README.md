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

3. **Download the LLM** (and the judge model for the evaluation metrics)
   ```
   ollama pull llama3.1:8b
   ollama pull qwen2.5:7b
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

**Experiments, diagnostics and evaluation**
```
python experiments.py           # chunk size / overlap grid and similarity vs MMR, saves results/*.csv
python diagnose.py              # where each document type ranks for a question
python evaluate.py              # RAG (4 prompt styles) vs plain LLM on eval_questions.csv
python evaluate.py --summarise  # rebuild the summary after labelling the answers
python metrics.py               # relevance + hallucination metrics (judge model: qwen2.5:7b)
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
| Chunking | 300 characters, 100 overlap, for all types | Tied best overall (MRR 0.82) and best for the PDFs (0.89); equal sizes stop short chunks from outranking long ones |
| Embeddings | all-MiniLM-L6-v2, normalised | Small, fast, free, runs locally; normalised vectors make L2 distance equivalent to cosine similarity |
| Vector DB | ChromaDB (HNSW index), persistent | Local, simple, stores metadata (source, page, timestamp) for citations |
| Retrieval | MMR, k=6 | Same accuracy as similarity search but more diverse results (3.4 vs 2.8 distinct sources in the top 4) |
| Context | Small-to-big, window 2 | Small chunks retrieve best but are too short to answer from |
| LLM | Llama 3.1 8B via Ollama, temperature 0 | Free, private, runs on the GPU; temperature 0 makes the evaluation repeatable |

## Results

### Chunking experiments
15 chunk size / overlap combinations, 12 test questions with known sources, k=4 (`results/chunking_summary.csv`).

| Setting | Chunks | hit@1 | hit@4 | MRR |
|---|---|---|---|---|
| **300 / 100** (chosen) | 1,846 | 0.75 | 0.92 | **0.82** |
| 1000 / 200 | 525 | 0.67 | 1.00 | 0.83 |
| 1000 / 100 | 492 | 0.67 | 1.00 | 0.82 |
| 300 / 0 | 1,564 | 0.75 | 0.83 | 0.77 |
| 1000 / 0 | 474 | 0.50 | 0.92 | 0.67 |
| 500 / 0 | 928 | 0.50 | 0.83 | 0.62 |

Overlap 100 beat 0 at every size. Per document type, 300/100 was best for PDFs (MRR 0.89), which make up most of the knowledge base. The one miss at 300/100 is "Who introduced the term retrieval-augmented generation?": before Wikipedia's reference list was removed, that list counted as a hit (MRR was 0.86), so the lower score is the more honest one. The test set is small (12 questions), so trends matter more than exact decimals.

### Search methods (final settings)
| Method | hit@4 | MRR | Distinct results in top 4 |
|---|---|---|---|
| Similarity | 0.92 | 0.82 | 2.8 |
| MMR | 0.92 | 0.82 | 3.4 |

### Response times (RTX 5070)
| Step | Time |
|---|---|
| Building the database (embedding 1,845 chunks) | ~8 s (GPU) vs ~185 s (CPU) |
| Retrieval | ~10 ms |
| Generation (zero-shot) | ~1.5–2 s |
| Generation (chain-of-thought) | ~3 s |

### RAG vs plain LLM
10 questions (`eval_questions.csv`): 6 answerable from the knowledge base, 4 outside it. Every question was asked to the plain LLM (no retrieval) and to RAG with each prompt style (`results/eval_summary.csv`).

| | Plain LLM | Zero-shot | Few-shot | Chain-of-thought | Tree-of-thoughts |
|---|---|---|---|---|---|
| Outside KB: correctly said "I don't know" | 2/4 | **4/4** | **4/4** | **4/4** | 2/4 |
| Inside KB: wrongly refused | 2/6 | 1/6 | 1/6 | 0/6 | 0/6 |
| Inside KB: answers with a citation | – | 4/5 | **5/5** | 6/6 | 6/6 |
| Average response time | 0.8 s | 0.6 s | 0.7 s | 2.1 s | 4.5 s |

The one question RAG refuses (Q6) needs two sources at once (a paper and a video); retrieval found only one, so the model declined instead of guessing.

### Evaluation metrics
Scored with `metrics.py`, using a different model (Qwen 2.5 7B) as the judge so Llama does not grade its own answers (`results/metrics_summary.csv`).

| | Plain LLM | Zero-shot | Few-shot | Chain-of-thought | Tree-of-thoughts |
|---|---|---|---|---|---|
| **Hallucinated answers** (inside KB) | **4/4** | 1/5 | **0/5** | **0/6** | **0/6** |
| Unsupported claims (inside KB) | 7/8 | 1/14 | 0/13 | 0/14 | 0/17 |
| Answered outside KB anyway | 2/4 | 0/4 | 0/4 | 0/4 | 2/4 |
| Answer relevance (judge, 1–5) | 3.8 | 4.6 | 4.2 | 3.8 | 4.5 |
| Answer–question similarity (embeddings) | 0.63 | 0.72 | 0.74 | 0.61 | 0.72 |
| Questions with a relevant retrieved chunk | – | 5/6 | 5/6 | 5/6 | 5/6 |
| Share of retrieved chunks that are relevant | – | 0.26 | 0.26 | 0.26 | 0.26 |

- **Hallucination:** every answer the plain LLM gave inside the knowledge base was invented: RAG as "Red-Amber-Green" status colours, "Recognition and Action", "Recurrent Attention-Generative", "Reinforced Actor-Guesser". RAG's only flagged answer (zero-shot) invented a part of the question it then said the sources did not cover.
- **Prompt styles:** few-shot and chain-of-thought were fully grounded and refused all out-of-scope questions. Tree-of-thoughts answered 2 of 4 out-of-scope questions: its long reasoning gives the 8B model more room to drift from the sources.
- **Retrieval:** with k=6 and MMR (deliberately diverse chunks) only a few chunks can contain the answer, so the share of relevant chunks is low by design; 5 of 6 questions had at least one chunk containing the answer.
- **Judge reliability:** the judge needed three prompt iterations (it rejected passages that did not repeat the paper's name, counted citations as claims, and copied an example answer). It still makes occasional mistakes, so human labels in `results/eval_answers.csv` remain the reference.

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
├── evaluate.py            RAG vs plain LLM evaluation (answers, timings, automatic checks)
├── metrics.py             Evaluation metrics: context/answer relevance, hallucination detection
├── test_questions.py      Retrieval test questions with known sources
├── eval_questions.csv     Evaluation questions (inside / outside the knowledge base)
├── data/                  PDFs, web page URLs, YouTube URLs
├── results/               Experiment output (CSV)
└── .streamlit/config.toml Streamlit settings
```

## Team

<!-- Add names and who did what -->
