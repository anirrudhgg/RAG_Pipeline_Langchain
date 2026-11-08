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
python benchmark.py             # GPU vs CPU timings (the CPU part is slow: several minutes per question)
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

### Response times: RTX 5070 vs CPU
Same model, same questions, measured on one PC (GPU: NVIDIA RTX 5070, 12 GB; CPU: AMD Ryzen 7 9800X3D, 8 cores) with `benchmark.py`, which runs the identical workload once on the GPU and once with everything forced onto the CPU (embedding model on `cpu`, Ollama with `num_gpu=0`).

| Step | RTX 5070 | CPU only | CPU slower by |
|---|---|---|---|
| Embedding 500 chunks (benchmark) | 0.1 s | 1.8 s | 18× (measured) |
| Embedding the first database, 536 chunks (`ingest.py`) | 15.7 s | 185 s | 12× (measured) |
| All 30 benchmark answers (6 questions × plain LLM + 4 prompt styles) | ~1 min | stopped after 60+ min, unfinished | **at least 60×** (measured) |
| Retrieval per question (MMR, k=6) | 7–17 ms | milliseconds | not significant |

Per answer, average over the 10 evaluation questions. GPU measured; CPU **estimated** from the measured slowdown (GPU time × 60, so a lower bound):

| Mode | RTX 5070 (measured) | CPU (estimated, at least) |
|---|---|---|
| Plain LLM | 0.89 s | ~55 s |
| RAG zero-shot | 0.75 s | ~45 s |
| RAG few-shot | 1.35 s | ~1.5 min |
| RAG chain-of-thought | 2.01 s | ~2 min |
| RAG tree-of-thoughts | 4.99 s | ~5 min |

The final database (1,845 chunks) embeds in ~8 s on the GPU. Retrieval costs milliseconds on either device; almost all time is the LLM.

**Why a top gaming CPU is this slow for LLMs:** generating each token requires reading all ~5 GB of model weights from memory, so speed is limited by memory bandwidth (dual-channel DDR5 ≈ 60–80 GB/s vs ≈ 670 GB/s GDDR7 on the RTX 5070). Processing the ~2,000-token RAG prompt is parallel math, where 8 CPU cores compete with thousands of GPU cores. The 9800X3D's large 3D V-Cache helps games but not here: the weights are ~50× larger than the cache. This is why the CPU ran at only ~50–65 % utilisation: the cores were mostly waiting for data from memory. For an interactive RAG system, a GPU is not a luxury but a requirement: the same answer takes about a second instead of about a minute.

### RAG vs plain LLM
10 questions (`eval_questions.csv`): 6 answerable from the knowledge base, 4 outside it. Every question was asked to the plain LLM (no retrieval) and to RAG with each prompt style. All 50 answers were labelled by a team member (correct / partial / wrong / hallucinated / refused) in `results/eval_answers.csv`; the summary is in `results/eval_summary.csv`.

| | Plain LLM | Zero-shot | Few-shot | Chain-of-thought | Tree-of-thoughts |
|---|---|---|---|---|---|
| **Accuracy inside KB** (correct + ½ partial, human labels) | **0.00** | 0.67 | **0.75** | **0.75** | **0.75** |
| **Hallucinated inside KB** (human labels) | **4/6** | 0/6 | 0/6 | 0/6 | 0/6 |
| Outside KB: correctly said "I don't know" | 2/4 | **4/4** | **4/4** | **4/4** | **4/4** |
| Inside KB: refused | 2/6 | 1/6 | 1/6 | 1/6 | 1/6 |
| Inside KB: answers with a citation | – | 4/5 | 5/5 | 5/5 | 5/5 |
| Average retrieval time | – | 17 ms | 8 ms | 7 ms | 7 ms |
| **Average total response time** | 0.89 s | 0.75 s | 1.35 s | 2.01 s | 4.99 s |

Times measured on the demo machine (RTX 5070, Llama 3.1 8B via Ollama), averaged over the 10 questions. Retrieval takes milliseconds; almost all time is generation, which grows with the length of the prompt style's reasoning.

- **Plain LLM:** it answered none of the knowledge-base questions correctly and invented 4 of 6 answers outright: RAG as "Red-Amber-Green" status colours, as "Recognition and Activation" of genes, as a "Recurrent Attention-Generative" model, and an invented paper title ("Real Language Understanding for Dialogue Generation with Reinforced Actor-Guesser").
- **RAG:** no hallucinated answers in any style. The missing accuracy comes from Q6, which needs two sources at once (a paper and a video) — retrieval found only one, so the model declined instead of guessing — and Q5, where all styles explained the decoding difference but missed the core definition (one document per answer vs one per token).
- **Repeatability:** even at temperature 0, two evaluation runs gave slightly different answers for 20 of 50 questions (GPU non-determinism). All labels and metrics here therefore come from one single run.

### Evaluation metrics
Scored with `metrics.py`, using a different model (Qwen 2.5 7B) as the judge so Llama does not grade its own answers (`results/metrics_summary.csv`).

| | Plain LLM | Zero-shot | Few-shot | Chain-of-thought | Tree-of-thoughts |
|---|---|---|---|---|---|
| **Hallucinated answers** (judge, inside KB) | **4/4** | **0/5** | **0/5** | 1/5 | **0/5** |
| Unsupported claims (inside KB) | 12/13 | 0/12 | 0/13 | 1/10 | 0/10 |
| Answered outside KB anyway | 2/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| **Faithfulness** (claims backed by the sources) | **0.09** | **1.00** | **0.95** | **0.95** | **0.90** |
| Fully faithful answers (inside KB) | 0/4 | 5/5 | 4/5 | 4/5 | 4/5 |
| Citation precision (cited source states the claim) | – | 1.00 | 0.93 | 0.80 | 0.77 |
| Answer relevance (judge, 1–5) | 3.8 | 4.6 | 4.2 | 4.0 | 4.4 |
| Answer–question similarity (embeddings) | 0.65 | 0.73 | 0.73 | 0.77 | 0.69 |
| Questions with a relevant retrieved chunk | – | 5/6 | 5/6 | 5/6 | 5/6 |
| Share of retrieved chunks that are relevant | – | 0.29 | 0.29 | 0.29 | 0.29 |
| **Judge agrees with human labels** | 10/10 | 8/10 | 9/10 | 9/10 | 9/10 |

- **Hallucination vs faithfulness:** hallucination detection asks whether an answer is *true* (checked against the reference answer and the sources); faithfulness asks whether it is *backed by the retrieved documents* (sources only). "Canberra" is true but not faithful.
- **Faithfulness:** the judge must quote the sentence that supports each claim, and the script verifies the quote itself (it must occur in a source and share the claim's key words; if the judge's quote fails, the script searches the sources for a matching sentence). The plain LLM's 0.09 comes from one generic fragment; none of its invented content is backed.
- **Prompt styles:** zero-shot was fully faithful with perfect citations; tree-of-thoughts cited the wrong source most often (0.77). Chain-of-thought once ended with only a citation ("Answer: [1]") instead of a sentence.
- **Retrieval:** with k=6 and MMR (deliberately diverse chunks) only a few chunks can contain the answer, so the share of relevant chunks is low by design; 5 of 6 questions had at least one chunk containing the answer.
- **Judge reliability:** the judge agrees with the human labels on 45 of 50 answers (90%). Most disagreements are on Q5, where the judge calls answers "correct" that humans rated "partial": it checks whether each statement is *true*, not whether the answer is *complete*. Chain-of-thought's one "hallucinated" flag is a judge error (it called a citation that the source does state unverifiable). Getting the judge reliable took several iterations: it rejected passages that did not repeat the paper's name, counted citations as claims, accepted support that was only on the same topic, and once looped while copying text (now retried).

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
├── benchmark.py           GPU vs CPU timing benchmark
├── test_questions.py      Retrieval test questions with known sources
├── eval_questions.csv     Evaluation questions (inside / outside the knowledge base)
├── data/                  PDFs, web page URLs, YouTube URLs
├── results/               Experiment output (CSV)
└── .streamlit/config.toml Streamlit settings
```

## Team

- Anirudh Gopishankar
- Hussein A. Monem
- Marc Baroud
- Sakhib Ahmed Baig
- Jeremie HB
