"""
evaluate.py - Part D of the assignment: RAG vs. plain LLM.

Runs every question in eval_questions.csv
  - once through the plain LLM (no retrieval), and
  - through the RAG chain with each prompt style (zero-shot, few-shot,
    chain-of-thought, tree-of-thoughts)
and saves all answers, sources and timings.

Automatic checks (no human needed):
  - out_of_kb questions: did the model correctly say "I don't know"?
  - in_kb questions: did it wrongly refuse? did retrieval find the expected
    source? did the answer cite any source?

Human labelling (teammates): results/eval_answers.csv has an empty "label"
column. Fill it with one of:
    correct | partial | wrong | hallucinated | refused
using expected_answer as the reference. Then run
    python evaluate.py --summarise
to add accuracy and hallucination rates to the summary.

Run (Ollama must be running; takes about 5-10 minutes):
    python evaluate.py
    python evaluate.py --styles Zero-shot Few-shot     # only some styles
    python evaluate.py --summarise                     # rebuild summary after labelling
    python evaluate.py --summarise --answers results/eval_answers_v1.csv   # summarise an older run
"""
import argparse
import os
import time

import pandas as pd

import config
import prompts
import rag_chain

QUESTIONS_FILE = "eval_questions.csv"
RESULTS_DIR = "results"
ANSWERS_FILE = os.path.join(RESULTS_DIR, "eval_answers.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "eval_summary.csv")
LABELS = ["correct", "partial", "wrong", "hallucinated", "refused"]


def expected_source_keys(expected: str) -> list[str]:
    """
    Turn the expected_source column into search keys, e.g.
      'lewis_2020_rag_original.pdf p. 3'                       -> ['lewis_2020_rag_original.pdf']
      'Chroma docs: embedding-functions'                       -> ['embedding-functions']
      'x.pdf p. 1 + IBM Technology: RAG vs. Fine-Tuning, ~3:48' -> ['x.pdf', 'IBM Technology: RAG vs. Fine-Tuning']
    """
    if not isinstance(expected, str) or not expected.strip():
        return []
    keys = []
    for part in expected.split("+"):
        key = part.split(" p.")[0].split(",")[0].replace("Chroma docs:", "").strip()
        if key:
            keys.append(key.lower())
    return keys


def sources_found(keys: list[str], sources: list[dict]) -> int:
    """How many of the expected sources appear among the retrieved chunks."""
    haystack = [f"{s['label']} {s.get('link') or ''}".lower() for s in sources]
    return sum(any(k in h for h in haystack) for k in keys)


def run(styles: list[str]):
    questions = pd.read_csv(QUESTIONS_FILE)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading models...")
    rag_chain.warm_up()

    rows = []
    total = len(questions) * (1 + len(styles))
    n = 0
    for _, q in questions.iterrows():
        base = {
            "id": q["id"], "question": q["question"], "category": q["category"],
            "expected_answer": q["expected_answer"], "expected_source": q.get("expected_source", ""),
        }
        keys = expected_source_keys(q.get("expected_source", ""))

        # Plain LLM baseline
        n += 1
        print(f"[{n:>2}/{total}] Q{q['id']} plain LLM")
        plain = rag_chain.answer_plain(q["question"])
        rows.append({
            **base, "mode": "plain", "answer": plain["answer"], "full_response": plain["answer"],
            "dont_know": prompts.is_dont_know(plain["answer"]),
            "cited_any": None, "expected_sources_retrieved": None, "expected_sources_total": None,
            "retrieved_sources": "", "cited_sources": "",
            "retrieval_ms": None, "generation_ms": round(plain["total_ms"]), "total_ms": round(plain["total_ms"]),
            "label": "",
        })

        # RAG with each prompt style
        for style in styles:
            n += 1
            print(f"[{n:>2}/{total}] Q{q['id']} RAG {style}")
            r = rag_chain.answer_with_rag(q["question"], style=style)
            rows.append({
                **base, "mode": f"rag: {style}", "answer": r["answer"], "full_response": r["response"],
                "dont_know": r["dont_know"],
                "cited_any": any(s["cited"] for s in r["sources"]),
                "expected_sources_retrieved": sources_found(keys, r["sources"]) if keys else None,
                "expected_sources_total": len(keys) if keys else None,
                "retrieved_sources": " ; ".join(f"[{s['n']}] {s['label']}" for s in r["sources"]),
                "cited_sources": " ; ".join(f"[{s['n']}] {s['label']}" for s in r["sources"] if s["cited"]),
                "retrieval_ms": round(r["retrieval_ms"]), "generation_ms": round(r["generation_ms"]),
                "total_ms": round(r["total_ms"]),
                "label": "",
            })

    answers = pd.DataFrame(rows)
    # utf-8-sig so Excel opens it with the right characters
    answers.to_csv(ANSWERS_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(answers)} answers to {ANSWERS_FILE}")
    summarise(answers)


def summarise(answers: pd.DataFrame | None = None, answers_file: str = ANSWERS_FILE):
    if answers is None:
        answers = pd.read_csv(answers_file, encoding="utf-8-sig")
    summary_file = answers_file.replace("eval_answers", "eval_summary")
    answers["label"] = answers["label"].astype("object").fillna("").astype(str).str.strip().str.lower()
    # Recompute refusals with the current detection rules, so fixes apply to older runs too
    answers["dont_know"] = answers["answer"].fillna("").astype(str).map(prompts.is_dont_know)

    summary = []
    for mode, part in answers.groupby("mode", sort=False):
        inside, outside = part[part.category == "in_kb"], part[part.category == "out_of_kb"]
        row = {
            "mode": mode,
            # Automatic metrics
            "out_of_kb: correctly refused": f"{int(outside.dont_know.sum())}/{len(outside)}",
            "in_kb: wrongly refused": f"{int(inside.dont_know.sum())}/{len(inside)}",
            "avg total time (s)": round(part.total_ms.mean() / 1000, 2),
            "avg generation time (s)": round(part.generation_ms.mean() / 1000, 2),
        }
        if mode != "plain":
            answered = inside[~inside.dont_know.astype(bool)]
            row["in_kb: answers with a citation"] = f"{int(answered.cited_any.astype(bool).sum())}/{len(answered)}"
            found = inside.expected_sources_retrieved.fillna(0).sum()
            needed = inside.expected_sources_total.fillna(0).sum()
            row["in_kb: expected sources retrieved"] = f"{int(found)}/{int(needed)}"
            row["avg retrieval time (ms)"] = round(part.retrieval_ms.mean())
        # Human labels, once filled in
        labelled = part[part.label.isin(LABELS)]
        if len(labelled):
            row["labelled"] = f"{len(labelled)}/{len(part)}"
            for label in LABELS:
                row[label] = int((labelled.label == label).sum())
            # In the knowledge base the right behaviour is a correct answer;
            # outside it, the right behaviour is to refuse. So score them separately.
            li = labelled[labelled.category == "in_kb"]
            lo = labelled[labelled.category == "out_of_kb"]
            if len(li):
                row["in_kb: accuracy (correct + partial/2)"] = round(
                    ((li.label == "correct").sum() + 0.5 * (li.label == "partial").sum()) / len(li), 2)
                row["in_kb: hallucinated (human)"] = f"{int((li.label == 'hallucinated').sum())}/{len(li)}"
            if len(lo):
                row["out_of_kb: correctly refused (human)"] = f"{int((lo.label == 'refused').sum())}/{len(lo)}"
            row["hallucination rate (all)"] = round((labelled.label == "hallucinated").sum() / len(labelled), 2)
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n" + summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_file}")
    unlabelled = (~answers.label.isin(LABELS)).sum()
    if unlabelled:
        print(f"{unlabelled} answers still need a label ({' | '.join(LABELS)}) in {answers_file}")


def main():
    parser = argparse.ArgumentParser(description="RAG vs. plain LLM evaluation.")
    parser.add_argument("--styles", nargs="+", default=list(prompts.STYLES), choices=list(prompts.STYLES))
    parser.add_argument("--summarise", action="store_true", help="only rebuild the summary from labelled answers")
    parser.add_argument("--answers", default=ANSWERS_FILE,
                        help="answers file to summarise (default results/eval_answers.csv)")
    args = parser.parse_args()
    if args.summarise:
        summarise(answers_file=args.answers)
    else:
        print(f"Settings: k={config.DEFAULT_K}, search={config.DEFAULT_SEARCH}, "
              f"context window={config.CONTEXT_WINDOW}, model={config.OLLAMA_MODEL}")
        run(args.styles)


if __name__ == "__main__":
    main()