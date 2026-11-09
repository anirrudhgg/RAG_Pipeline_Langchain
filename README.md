# RAG Explorer: a local RAG pipeline that cites its sources

Ask questions about Retrieval-Augmented Generation and get answers **grounded in a curated knowledge base** (research papers, documentation pages and videos), with numbered citations. If the documents don't contain the answer, the system says so instead of guessing.

Everything runs locally: LangChain, ChromaDB, sentence-transformers and Llama 3.1 8B through Ollama. No API keys, no data leaves your machine.

> **Headline result:** on our evaluation set, the plain LLM invented answers for 4 of 6 questions about the knowledge base. RAG hallucinated on none, and correctly refused all 4 out-of-scope questions. See [Results](#results).

```mermaid
flowchart LR
    Q[Question] --> R[MMR search<br/>ChromaDB, k=6]
    R --> E[Small-to-big<br/>add 2 neighbour chunks]
    E --> P[Numbered sources<br/>in prompt template]
    P --> L[Llama 3.1 8B<br/>temperature 0]
    L --> A["Answer with [n] citations<br/>or 'I don't know'"]
```

## Contents

- [Quick start](#quick-start)
- [Usage](#usage)
- [Features](#features)
- [Knowledge base](#knowledge-base)
- [How it works](#how-it-works)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Challenges and solutions](#challenges-and-solutions)
- [Known limitations](#known-limitations)
- [Project structure](#project-structure)
- [Team](#team)

## Quick start

**Requirements:** Python 3.10+ (developed on 3.12), [Ollama](https://ollama.com), about 6 GB of free disk space for the models.

```bash
# 1. Get the code and install dependencies
git clone https://github.com/anirrudhgg/RAG_Pipeline_Langchain.git
cd RAG_Pipeline_Langchain
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

# 2. Download the models (Ollama must be running)
ollama pull llama3.1:8b          # answers
ollama pull qwen2.5:7b           # judge for metrics.py (only needed for evaluation)

# 3. Build the vector database (loads, chunks and embeds all documents)
python ingest.py

# 4. Start the app
streamlit run app.py             # opens http://localhost:8501
```

**Optional: GPU embeddings** (NVIDIA only; RTX 50-series needs CUDA 12.8):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

Embedding the knowledge base takes about 8 s on an RTX 5070 and a few minutes on a CPU.

**Weaker laptop?** Use a smaller model: `ollama pull llama3.2:3b`, then copy `.env.example` to `.env` and set `OLLAMA_MODEL=llama3.2:3b` in it. The default is `llama3.1:8b`, so no `.env` is needed otherwise.

## Usage

### Web interface

```bash
streamlit run app.py
```

Pick a prompt style, search method, number of chunks (k) and context window in the sidebar. The app shows the RAG answer with clickable, numbered sources next to the plain LLM's answer to the same question, plus the exact prompt sent to the model and the timing of each step.

### Terminal

```bash
python rag_chain.py "What is the difference between RAG-Sequence and RAG-Token?"
python rag_chain.py "What is the capital of Australia?" --compare
python rag_chain.py "When should I use RAG instead of fine-tuning?" --style Chain-of-thought
python retrieval.py "What is HNSW?" --k 6       # compare the two search methods
python prompts.py                                # preview all prompt templates
```

### Experiments and evaluation

| Command | What it does |
|---|---|
| `python experiments.py` | Chunk size / overlap grid (15 settings) and similarity vs MMR; writes `results/*.csv` |
| `python diagnose.py` | Shows where each document type ranks for a question |
| `python evaluate.py` | Runs RAG (4 prompt styles) and the plain LLM on `eval_questions.csv` |
| `python evaluate.py --summarise` | Rebuilds the summary after you label the answers |
| `python metrics.py` | Relevance, hallucination and faithfulness metrics (judge: `qwen2.5:7b`) |
| `python benchmark.py` | GPU vs CPU timings. The CPU part is slow: several minutes per question. Close the Streamlit app first |

Typical order: `ingest.py` → `evaluate.py` → label `results/eval_answers.csv` → `evaluate.py --summarise` → `metrics.py`.

## Features

- **Three document types:** PDFs (5 papers), web pages (4 documentation pages), YouTube transcripts (2 videos)
- **Two search methods:** similarity search (configurable k) and MMR (Maximal Marginal Relevance) for relevant *and* diverse results
- **Small-to-big retrieval:** search with small chunks for precision, then give the LLM each hit plus its neighbours
- **Four prompt styles:** zero-shot, few-shot, chain-of-thought, tree-of-thoughts
- **Citations and refusal:** answers cite numbered sources; out-of-scope questions get "I don't know based on the provided documents."
- **Evaluation suite:** human labels plus automatic metrics (faithfulness, citation precision, hallucination rate) scored by a separate judge model
- **Hardware benchmark:** GPU vs CPU timings for embedding, retrieval and generation

## Knowledge base

| Type | Sources |
|---|---|
| PDF | Lewis et al. 2020 (original RAG paper); Fan et al. 2024 (RAG meets LLMs, KDD); Gao et al. 2025 (RAG and reasoning review); Klesel & Wittmann 2025 (RAG catchword, BISE); Arslan et al. 2024 (RAG with LLMs survey, Procedia CS) |
| Web | Wikipedia: Retrieval-augmented generation; LangChain docs: Retrieval; Chroma docs: Embedding functions; Chroma docs: Configure collections |
| Video | IBM Technology: What is RAG?; IBM Technology: RAG vs. Fine-Tuning |

PDFs live in `data/pdfs/`; web and video links are in `data/urls.txt` and `data/youtube.txt` (one entry per line, `#` for comments). Pages and transcripts are fetched when you run `ingest.py`. To use your own documents, edit those files and run `python ingest.py` again, which rebuilds the database from scratch.

## How it works

| Step | Choice | Why |
|---|---|---|
| Loading | PyPDFLoader, WebBaseLoader, youtube-transcript-api | One loader per document type; transcripts grouped into 60-second windows with timestamps |
| Cleaning | Reference lists removed from PDFs and web pages | Bibliographies matched almost every query and ranked first without explaining anything |
| Chunking | 300 characters, 100 overlap, all types | Within 0.01 of the best overall MRR (0.82 vs 0.83) and best for PDFs (0.89); equal sizes stop short chunks from outranking long ones |
| Embeddings | all-MiniLM-L6-v2, normalised | Small, fast, local; normalised vectors make L2 distance equivalent to cosine similarity |
| Vector DB | ChromaDB (HNSW), persistent, 1,845 chunks | Local, simple, stores source / page / timestamp metadata for citations |
| Retrieval | MMR, k=6 | Same accuracy as similarity search, more diverse results (3.4 vs 2.8 distinct sources in the top 4) |
| Context | Small-to-big, window 2 | Small chunks retrieve best but are too short to answer from |
| LLM | Llama 3.1 8B via Ollama, temperature 0, 8k context | Free, private, repeatable |

## Results

All numbers were measured on one machine (RTX 5070, 12 GB; Ryzen 7 9800X3D), and all evaluation numbers come from a single evaluation run. The test sets are small, so read trends rather than exact decimals.

### RAG vs plain LLM

10 questions (`eval_questions.csv`): 6 answerable from the knowledge base, 4 outside it. All 50 answers were labelled by a team member (`results/eval_answers.csv`).

| | Plain LLM | Zero-shot | Few-shot | Chain-of-thought | Tree-of-thoughts |
|---|---|---|---|---|---|
| Accuracy inside KB (correct + ½ partial) | 0.00 | 0.67 | **0.75** | **0.75** | **0.75** |
| Hallucinated inside KB (human labels) | 4/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| Outside KB: correctly said "I don't know" | 2/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| Inside KB: wrongly refused | 2/6 | 1/6 | 1/6 | 1/6 | 1/6 |
| Inside KB: answers with a citation | – | 4/5 | 5/5 | 5/5 | 5/5 |
| Average response time | 0.89 s | 0.75 s | 1.35 s | 2.01 s | 4.99 s |

- **Plain LLM:** invented 4 of 6 answers: RAG as "Red-Amber-Green" status colours, as "Recognition and Activation" of genes, as a "Recurrent Attention-Generative" model, and a made-up paper title ("Real Language Understanding for Dialogue Generation with Reinforced Actor-Guesser").
- **RAG:** zero hallucinations in every style. The missing accuracy comes from Q6 (needs a paper *and* a video; retrieval found only one, so the model declined) and Q5 (all styles explained the decoding difference but missed the core definition).
- **Repeatability:** even at temperature 0, two runs differed on 20 of 50 answers (GPU non-determinism), so all labels and metrics come from a single run.

### Automatic metrics

Scored with `metrics.py` using Qwen 2.5 7B as judge, so Llama does not grade its own answers.

| | Plain LLM | Zero-shot | Few-shot | Chain-of-thought | Tree-of-thoughts |
|---|---|---|---|---|---|
| Hallucinated answers (inside KB) | 4/4 | 0/5 | 0/5 | 1/5 | 0/5 |
| Faithfulness (claims backed by sources) | 0.09 | **1.00** | 0.95 | 0.95 | 0.90 |
| Citation precision | – | **1.00** | 0.93 | 0.80 | 0.77 |
| Answer relevance (judge, 1–5) | 3.8 | **4.6** | 4.2 | 4.0 | 4.4 |
| Judge agrees with human labels | 10/10 | 8/10 | 9/10 | 9/10 | 9/10 |

- **Faithfulness** asks whether an answer is backed by the retrieved documents. "Canberra" is true but not faithful. The script verifies the judge's evidence itself instead of trusting it.
- **Judge reliability:** it agrees with the humans on 45 of 50 answers. Most disagreements are on Q5, where the judge checks whether statements are *true*, not whether the answer is *complete*. Chain-of-thought's one "hallucinated" flag is a judge error: the cited source does state the claim.
- **Retrieval:** 5 of 6 questions had at least one chunk containing the answer. Only 29% of chunks are relevant, which is expected with k=6 and deliberately diverse MMR results.
- **Best prompt style:** zero-shot was the most faithful and had perfect citations; tree-of-thoughts cited the wrong source most often.

### Chunking and search experiments

15 chunk size / overlap combinations, 12 test questions with known sources, k=4 (`results/chunking_summary.csv`).

| Setting | Chunks | hit@1 | hit@4 | MRR |
|---|---|---|---|---|
| **300 / 100** (chosen) | 1,846 | 0.75 | 0.92 | 0.82 |
| 1000 / 200 | 525 | 0.67 | 1.00 | 0.83 |
| 1000 / 100 | 492 | 0.67 | 1.00 | 0.82 |
| 300 / 0 | 1,564 | 0.75 | 0.83 | 0.77 |
| 1000 / 0 | 474 | 0.50 | 0.92 | 0.67 |

Overlap helped at 300 and 1000 characters (MRR 0.77 → 0.82 and 0.67 → 0.82), but not at every size: at 800 and 1500 characters, no overlap scored higher. 1000/200 scored best overall (0.83), but 300/100 was best for the PDFs, which make up most of the knowledge base, and 1000-character chunks come close to the embedding model's 256-token limit. The final database has 1,845 chunks rather than 1,846 because the live Wikipedia page changed slightly between the two fetches. Removing Wikipedia's reference list lowered the 300/100 score from 0.86 to 0.82, because the reference list had counted as a false "hit", so the lower figure is the honest one.

| Search method | hit@4 | MRR | Distinct sources in top 4 |
|---|---|---|---|
| Similarity | 0.92 | 0.82 | 2.8 |
| MMR | 0.92 | 0.82 | 3.4 |

### Hardware: RTX 5070 vs CPU

`benchmark.py` runs the identical workload on the GPU and with everything forced onto the CPU.

| Step | RTX 5070 | CPU only | CPU slower by |
|---|---|---|---|
| Embedding 500 chunks | 0.1 s | 1.8 s | 18× |
| Embedding the first database (536 chunks, `ingest.py`) | 15.7 s | 185 s | 12× |
| 30 benchmark answers (6 questions × 5 modes) | ~1 min | stopped after 60+ min | ≥ 60× |
| Retrieval per question (MMR, k=6) | 7–17 ms | milliseconds | not significant |

Retrieval costs milliseconds on either device; almost all time is the LLM. Generating each token requires reading ~5 GB of weights from memory, so speed is limited by memory bandwidth (dual-channel DDR5 ≈ 60–80 GB/s vs ≈ 670 GB/s on the RTX 5070), not by CPU cores. The CPU sat at only 50–65% utilisation waiting for data. For interactive use, a GPU is effectively a requirement: about one second per answer instead of about one minute.

## Configuration

All settings live in `config.py`, shared by every script.

| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Answer model (override with `.env`) |
| `JUDGE_MODEL` | `qwen2.5:7b` | Judge for `metrics.py` (override with `.env`) |
| `CHUNK_SETTINGS` | 300 / 100 | Chunk size and overlap per document type |
| `DEFAULT_K` | 6 | Chunks retrieved per question |
| `DEFAULT_SEARCH` | `mmr` | `mmr` or `similarity` |
| `CONTEXT_WINDOW` | 2 | Neighbouring chunks added on each side of a hit (0 = off) |

After changing chunk settings or the documents, run `python ingest.py` again.

## Troubleshooting

| Problem | Fix |
|---|---|
| `connection refused` or model not found | Start Ollama and check that `ollama list` shows `llama3.1:8b` |
| "The vector database is empty" | Run `python ingest.py` first |
| A video is skipped during ingestion | It has no English captions, or the transcript API was blocked; try another video |
| A web page is skipped ("very little text") | The page needs JavaScript to render; use a different URL |
| Answers are very slow | Ollama is running on the CPU. Check with `ollama ps`: it should say `100% GPU` |
| `benchmark.py` stops with a placement error | Close other GPU programs and the Streamlit app, then rerun |

## Challenges and solutions

1. **Bibliographies ranked first.** The first test search returned a paper's reference list as the top hit. *Fix:* cut PDFs at their "References" heading (keeping appendices), and later Wikipedia's reference section too, once smaller chunks exposed the same problem.
2. **Small chunks gave thin answers.** 300-character chunks retrieved best, but the model could not find the RAG-Sequence definition in them. *Fix:* small-to-big retrieval. Same chunks, same temperature, clearly better answer.
3. **Mixed chunk sizes buried the videos.** With small PDF chunks and unsplit video windows, the right video chunk ranked 28th for "When should I use RAG instead of fine-tuning?" (`diagnose.py`); long chunks were also truncated by the embedding model's 256-token limit. *Fix:* one chunk size for all types; the video then ranked 14th and MMR retrieved it.
4. **Ollama's default context window is small.** Few-shot and tree-of-thoughts prompts could be cut off, losing the rules at the top. *Fix:* `num_ctx=8192`.
5. **Tree-of-thoughts prompt iterations.** Version 1 answered out-of-scope questions ("Canberra [1]") and gave bare "[1]" answers; version 2 added an early exit and then refused every question. *Fix:* version 3 replaces the early exit with a self-check, plus rules that a partial answer is allowed and that the model's own knowledge is not a source.
6. **An unreliable judge.** The first judge copied the last example score, counted citations as claims and accepted evidence that was only on the same topic. *Fix:* reason-first scoring, citations stripped from claims, and the judge must quote its evidence, which the script verifies against the sources.

## Known limitations

- Llama 3.1 8B does not always cite every statement; often it cites only once per answer.
- PDF page headers and footers appear as noise in some chunks.
- The knowledge base is small (11 documents) and the evaluation sets are small (12 retrieval questions, 10 answer questions).
- Dependencies in `requirements.txt` are not version-pinned, so a future install may behave differently.

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
├── evaluate.py            RAG vs plain LLM evaluation
├── metrics.py             Relevance, hallucination and faithfulness metrics
├── benchmark.py           GPU vs CPU timing benchmark
├── test_questions.py      Retrieval test questions with known sources
├── eval_questions.csv     Evaluation questions (inside / outside the knowledge base)
├── requirements.txt       Python dependencies
├── .env.example           Template for .env (choose the Ollama model)
├── data/                  PDFs, web page URLs, YouTube URLs
├── results/               Experiment and evaluation output (CSV)
└── .streamlit/config.toml Streamlit settings
```

## Team

Anirudh Gopishankar · Hussein Abdel Monem · Marc Baroud · Sakhib Ahmed Baig · Jeremie Boulain-Holstaine
