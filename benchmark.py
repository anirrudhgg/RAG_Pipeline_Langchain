"""
benchmark.py - How much does the hardware matter? GPU (RTX 5070) vs CPU.

Runs exactly the same work twice, back to back, once on the GPU and once on the
CPU only, and measures the time of every step:
  1. Embedding: encoding 500 chunks from the database (what ingest.py does)
  2. Retrieval: embedding the question + MMR search + small-to-big expansion
  3. Generation: Llama 3.1 8B answering, for the plain LLM and every prompt style

How the CPU is forced:
  - Embeddings: the sentence-transformers model is loaded with device="cpu"
  - LLM: Ollama is asked to put 0 model layers on the GPU (num_gpu=0)

Only the timings matter here. The answers are not saved and the labelled
evaluation in results/eval_answers.csv is not touched. Results go to
results/benchmark_hardware.csv (every call) and results/benchmark_summary.csv.

Before each phase the script unloads all models from Ollama and checks with
'ollama ps' that the LLM really runs on the GPU / CPU, and stops if it does not.

Run (Ollama must be running; close the Streamlit app first). The CPU part is slow:
several minutes per question, because every answer runs an 8B model on the CPU.
    python benchmark.py                    # 2 questions (default)
    python benchmark.py --questions 6      # all in-KB questions (CPU part can take over an hour)
"""
import argparse
import os
import subprocess
import time

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

import config
import prompts
import rag_chain
import retrieval

RESULTS_DIR = "results"
EMBED_SAMPLE = 500


def ollama_ps() -> str:
    try:
        return subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def unload_all_models():
    """
    Unload every model Ollama has in memory. Otherwise a model still loaded on the
    CPU (e.g. from an interrupted run) or a judge model filling the GPU makes
    Ollama reuse the wrong placement, and the GPU phase silently runs on the CPU.
    """
    for line in ollama_ps().splitlines()[1:]:
        if line.strip():
            name = line.split()[0]
            subprocess.run(["ollama", "stop", name], capture_output=True, timeout=60)
    time.sleep(3)


def check_placement(device: str):
    """Verify with 'ollama ps' that the model really runs where we think it does."""
    line = next((l for l in ollama_ps().splitlines() if l.startswith(config.OLLAMA_MODEL)), "")
    print(f"  ollama ps: {' '.join(line.split()) or '(model not listed)'}")
    if device == "GPU" and "100% GPU" not in line:
        raise SystemExit("The GPU phase is not running fully on the GPU (see the line above). "
                         "Close other programs using the GPU or Ollama and run the benchmark again.")
    if device == "CPU" and "100% CPU" not in line:
        raise SystemExit("The CPU phase is not running fully on the CPU (see the line above).")


def setup_device(device: str):
    """Point the RAG pipeline at CPU-only or GPU components."""
    embeddings = config.get_embeddings("cpu" if device == "CPU" else "cuda")
    retrieval._store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )
    # num_gpu = number of model layers Ollama puts on the GPU: 0 = CPU only, 999 = all
    options = {"num_gpu": 0} if device == "CPU" else {"num_gpu": 999}
    rag_chain._llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0,
                                num_ctx=rag_chain.NUM_CTX, **options)
    rag_chain._chain = None
    return embeddings


def time_embedding(embeddings, texts: list[str]) -> float:
    embeddings.embed_documents(texts[:20])  # warm-up
    t = time.perf_counter()
    embeddings.embed_documents(texts)
    return time.perf_counter() - t


def main():
    parser = argparse.ArgumentParser(description="GPU vs CPU timing benchmark.")
    parser.add_argument("--questions", type=int, default=2,
                        help="number of in-KB evaluation questions to use (default 2; max 6). "
                             "The CPU phase takes several minutes per question.")
    args = parser.parse_args()

    try:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU found - this benchmark compares GPU and CPU.")
        gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        raise SystemExit("PyTorch is not installed.")

    questions = pd.read_csv("eval_questions.csv")
    questions = questions[questions.category == "in_kb"].head(args.questions)["question"].tolist()
    sample = Chroma(collection_name=config.COLLECTION_NAME, persist_directory=config.CHROMA_DIR,
                    embedding_function=config.get_embeddings()).get(limit=EMBED_SAMPLE)["documents"]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows, embed_times = [], {}
    for device in ["GPU", "CPU"]:
        print(f"\n=== {device}{' (' + gpu_name + ')' if device == 'GPU' else ' only'} ===")
        embeddings = setup_device(device)

        embed_times[device] = time_embedding(embeddings, sample)
        print(f"  embedding {len(sample)} chunks: {embed_times[device]:.1f} s")

        print("  unloading models from Ollama and loading the LLM on this device...")
        unload_all_models()
        rag_chain.warm_up()  # loads the model onto this device; not timed
        check_placement(device)

        for i, q in enumerate(questions, 1):
            print(f"  question {i}/{len(questions)}: {q[:60]}", flush=True)
            plain = rag_chain.answer_plain(q)
            rows.append({"device": device, "mode": "plain", "question": q,
                         "retrieval_ms": None, "generation_ms": plain["total_ms"], "total_ms": plain["total_ms"]})
            print(f"    plain LLM           {plain['total_ms'] / 1000:6.1f} s", flush=True)
            for style in prompts.STYLES:
                r = rag_chain.answer_with_rag(q, style=style)
                rows.append({"device": device, "mode": f"rag: {style}", "question": q,
                             "retrieval_ms": r["retrieval_ms"], "generation_ms": r["generation_ms"],
                             "total_ms": r["total_ms"]})
                print(f"    RAG {style:<17} {r['total_ms'] / 1000:6.1f} s", flush=True)

    calls = pd.DataFrame(rows)
    calls.to_csv(os.path.join(RESULTS_DIR, "benchmark_hardware.csv"), index=False, encoding="utf-8-sig")

    summary = []
    summary.append({"step": f"embedding {len(sample)} chunks",
                    "GPU": f"{embed_times['GPU']:.1f} s", "CPU": f"{embed_times['CPU']:.1f} s",
                    "CPU / GPU": f"{embed_times['CPU'] / embed_times['GPU']:.1f}x"})
    for step, col in [("retrieval", "retrieval_ms")]:
        g = calls[(calls.device == "GPU") & calls[col].notna()][col].mean()
        c = calls[(calls.device == "CPU") & calls[col].notna()][col].mean()
        summary.append({"step": "retrieval (per question)", "GPU": f"{g:.0f} ms", "CPU": f"{c:.0f} ms",
                        "CPU / GPU": f"{c / g:.1f}x"})
    for mode in calls["mode"].unique():
        g = calls[(calls.device == "GPU") & (calls["mode"] == mode)].total_ms.mean() / 1000
        c = calls[(calls.device == "CPU") & (calls["mode"] == mode)].total_ms.mean() / 1000
        summary.append({"step": f"answer: {mode}", "GPU": f"{g:.2f} s", "CPU": f"{c:.2f} s",
                        "CPU / GPU": f"{c / g:.1f}x"})
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(RESULTS_DIR, "benchmark_summary.csv"), index=False, encoding="utf-8-sig")

    print(f"\nGPU: {gpu_name} | CPU only | {len(questions)} questions x 5 modes per device\n")
    print(summary.to_string(index=False))
    print(f"\nSaved {RESULTS_DIR}/benchmark_hardware.csv and {RESULTS_DIR}/benchmark_summary.csv")


if __name__ == "__main__":
    main()