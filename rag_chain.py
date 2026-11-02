"""
rag_chain.py - Part C of the assignment: the retrieval chain.

    question -> retrieve chunks -> format context -> prompt -> Llama 3.1 (Ollama) -> answer with citations

Built with LangChain's runnable (LCEL) interface. Each step adds its result
to a dictionary, so the final output keeps everything the UI and the
evaluation need: the answer, the sources it cites, the full prompt and the
timing of each step.

Also contains the plain-LLM baseline (same model, no retrieval) for the
RAG-vs-plain-LLM comparison (Part D).

Try it from the terminal (Ollama must be running):
    python rag_chain.py "What is the difference between RAG-Sequence and RAG-Token?"
    python rag_chain.py "What is the capital of Australia?" --compare
    python rag_chain.py "How does MMR work?" --style Chain-of-thought --search similarity --k 4

Defaults (k=6, MMR search) come from config.py, chosen with experiments.py.
"""
import argparse
import re
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

import config
import prompts
import retrieval

SEARCH_METHODS = ["similarity", "mmr"]

# Llama 3.1 supports 128k tokens, but Ollama's default context window is much
# smaller. Few-shot and tree-of-thoughts prompts with several chunks can exceed
# it, and Ollama would silently cut off the start of the prompt (the rules!).
# 8192 tokens fits comfortably in the RTX 5070's 12 GB VRAM.
NUM_CTX = 8192

_llm = None


def get_llm() -> ChatOllama:
    """The local Llama model. temperature=0 makes answers repeatable for evaluation."""
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0, num_ctx=NUM_CTX)
    return _llm


# ---------------------------------------------------------------------------
# Chain steps
# ---------------------------------------------------------------------------
def _retrieve(inputs: dict) -> dict:
    """Step 1: get the k most relevant chunks with the chosen search method."""
    t = time.perf_counter()
    if inputs["search"] == "mmr":
        docs = retrieval.semantic_search(inputs["question"], k=inputs["k"])
        scores = [None] * len(docs)
    else:
        results = retrieval.similarity_search(inputs["question"], k=inputs["k"])
        docs = [d for d, _ in results]
        scores = [s for _, s in results]
    return {"docs": docs, "scores": scores, "retrieval_ms": (time.perf_counter() - t) * 1000}


def _build_prompt(inputs: dict) -> str:
    """Step 2 + 3: number the chunks as sources and insert them into the prompt template."""
    context = prompts.format_context(inputs["docs"])
    return prompts.build_prompt(inputs["style"], inputs["question"], context)


def _timed_generate(inputs: dict) -> dict:
    """Step 4: send the prompt to the LLM."""
    t = time.perf_counter()
    response = (get_llm() | StrOutputParser()).invoke(inputs["prompt"])
    return {"response": response, "generation_ms": (time.perf_counter() - t) * 1000}


def build_rag_chain():
    """
    The RAG chain. Input:  {"question", "style", "k", "search"}
    Output: the input plus docs, scores, prompt, response and timings.
    """
    return (
        RunnablePassthrough.assign(retrieved=RunnableLambda(_retrieve))
        | RunnableLambda(lambda x: {**x, **x.pop("retrieved")})
        | RunnablePassthrough.assign(prompt=RunnableLambda(_build_prompt))
        | RunnablePassthrough.assign(generated=RunnableLambda(_timed_generate))
        | RunnableLambda(lambda x: {**x, **x.pop("generated")})
    )


_chain = None


# ---------------------------------------------------------------------------
# Public functions (used by app.py and evaluate.py)
# ---------------------------------------------------------------------------
def source_link(metadata: dict) -> str | None:
    """Clickable link for web pages and videos (YouTube links jump to the timestamp)."""
    src = metadata.get("source", "")
    return src if src.startswith("http") else None


def cited_numbers(answer: str) -> list[int]:
    """Source numbers the model cited, e.g. 'RAG uses DPR [1][3].' -> [1, 3]."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def answer_with_rag(question: str, style: str = "Zero-shot", k: int = config.DEFAULT_K,
                    search: str = config.DEFAULT_SEARCH) -> dict:
    """
    Run the full RAG chain and return:
      answer         final answer (reasoning removed for CoT / ToT)
      response       full model output (including reasoning steps)
      sources        every retrieved chunk: number, label, link, score, text, cited (bool)
      dont_know      True if the model used the fallback
      uncited_claims True if the model answered without citing anything (a warning sign)
      prompt, retrieval_ms, generation_ms, total_ms
    """
    global _chain
    if style not in prompts.STYLES:
        raise ValueError(f"Unknown style {style!r}; choose from {list(prompts.STYLES)}")
    if search not in SEARCH_METHODS:
        raise ValueError(f"Unknown search method {search!r}; choose from {SEARCH_METHODS}")
    if _chain is None:
        _chain = build_rag_chain()

    t = time.perf_counter()
    out = _chain.invoke({"question": question, "style": style, "k": k, "search": search})
    total_ms = (time.perf_counter() - t) * 1000

    answer = prompts.extract_answer(out["response"])
    cited = set(cited_numbers(answer))
    dont_know = prompts.is_dont_know(answer)
    sources = [
        {
            "n": i,
            "label": prompts.source_label(doc.metadata),
            "doc_type": doc.metadata.get("doc_type"),
            "link": source_link(doc.metadata),
            "score": score,
            "text": doc.page_content,
            "cited": i in cited,
        }
        for i, (doc, score) in enumerate(zip(out["docs"], out["scores"]), 1)
    ]
    return {
        "question": question,
        "style": style,
        "search": search,
        "k": k,
        "answer": answer,
        "response": out["response"],
        "sources": sources,
        "dont_know": dont_know,
        "uncited_claims": not dont_know and not cited,
        "prompt": out["prompt"],
        "retrieval_ms": out["retrieval_ms"],
        "generation_ms": out["generation_ms"],
        "total_ms": total_ms,
    }


def answer_plain(question: str) -> dict:
    """Baseline: the same model answering from its own memory, without retrieval or rules."""
    t = time.perf_counter()
    response = (get_llm() | StrOutputParser()).invoke(prompts.plain_llm(question))
    return {"question": question, "answer": response.strip(), "total_ms": (time.perf_counter() - t) * 1000}


def warm_up():
    """Load the embedding model, the database and the LLM once, so the first real answer is fast."""
    retrieval.load_vector_store().similarity_search("warm-up", k=1)
    get_llm().invoke("Say OK.")


# ---------------------------------------------------------------------------
# Terminal demo
# ---------------------------------------------------------------------------
def print_rag(result: dict):
    print(f"\n--- RAG answer ({result['style']}, {result['search']} search, k={result['k']}) ---")
    if result["response"].strip() != result["answer"]:
        print("\n[Reasoning]\n" + result["response"].rsplit("Answer:", 1)[0].strip())
        print("\n[Final answer]")
    print(result["answer"])
    if result["uncited_claims"]:
        print("\n(!) The answer contains no citations - treat it with caution.")
    print("\nSources:")
    for s in result["sources"]:
        mark = "cited" if s["cited"] else "     "
        score = f"sim {s['score']:.2f}" if s["score"] is not None else ""
        print(f"  [{s['n']}] {mark} {s['label']} {score}" + (f"\n        {s['link']}" if s["link"] else ""))
    print(f"\nTime: retrieval {result['retrieval_ms']:.0f} ms | generation {result['generation_ms']:.0f} ms | "
          f"total {result['total_ms']:.0f} ms")


def main():
    parser = argparse.ArgumentParser(description="Ask the RAG system a question.")
    parser.add_argument("question")
    parser.add_argument("--style", default="Zero-shot", choices=list(prompts.STYLES))
    parser.add_argument("--k", type=int, default=config.DEFAULT_K)
    parser.add_argument("--search", default=config.DEFAULT_SEARCH, choices=SEARCH_METHODS)
    parser.add_argument("--compare", action="store_true", help="also show the plain LLM answer")
    parser.add_argument("--show-prompt", action="store_true", help="print the full prompt sent to the LLM")
    args = parser.parse_args()

    print("Loading models (first call takes a few seconds)...")
    warm_up()

    result = answer_with_rag(args.question, style=args.style, k=args.k, search=args.search)
    if args.show_prompt:
        print("\n--- Prompt sent to the LLM ---\n" + result["prompt"])
    print_rag(result)

    if args.compare:
        plain = answer_plain(args.question)
        print("\n--- Plain LLM answer (no retrieval) ---")
        print(plain["answer"])
        print(f"\nTime: {plain['total_ms']:.0f} ms")


if __name__ == "__main__":
    main()
