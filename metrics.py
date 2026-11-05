"""
metrics.py - Part E of the assignment: automatic evaluation metrics.

Scores every answer in results/eval_answers.csv (made by evaluate.py) with:

  RELEVANCE TO QUERY
    context_relevance      RAG only, in-KB questions. Fraction of the retrieved chunks
                           that the judge scores 2 ("contains the answer or a clear part
                           of it") on a 0-2 scale (measures retrieval). Every chunk
                           judgment, with the judge's reason, is in metrics_context.csv.
                           With k=6 and MMR (diverse chunks), only a few chunks can hold
                           the answer, so this is never near 1. Also reported:
    questions with a relevant chunk   in how many in-KB questions at least one
                           retrieved chunk contains the answer (scores 2).
    answer_relevance       Judge score 1-5: does the answer address the question?
                           (ignores correctness). Not scored for refusals.
    answer_similarity      Cosine similarity between question and answer embeddings
                           (all-MiniLM-L6-v2). Cheap, deterministic second signal.

  HALLUCINATION DETECTION
    The judge lists the factual claims in the answer and checks each one against
    the human reference answer (eval_questions.csv) and the retrieved sources:
      supported     agrees with the reference or is stated in a source
      contradicted  conflicts with the reference or a source
      unverifiable  a specific fact found in neither
    hallucinated = any claim is contradicted or unverifiable.
    The same check runs for the plain LLM and for RAG, against the same sources,
    so the comparison is fair. Out-of-knowledge-base questions are reported
    separately as "answered without grounding" (e.g. "Canberra" is true, but it
    does not come from the documents).

  FAITHFULNESS (grounding in the retrieved sources only)
    The judge lists the claims in the answer and copies, for each, the sentence
    from the sources that states it (without the reference answer and without its
    own knowledge). The script then VERIFIES the evidence itself instead of trusting
    the judge: the quote must really occur in a source (which also gives the source
    number), and it must share enough key words with the claim. A claim without
    verified evidence is unfaithful, even if it is true ("Canberra" is true but not
    in any document).
      faithfulness         share of an answer's claims that a source states (0-1)
      citation_precision   share of the sources an answer cites ([n]) that actually
                           state one of its claims (are the citations correct?)
    For the plain LLM the same sources are used: its faithfulness shows how much of
    what it says happens to be backed by the documents it never saw.

The judge is a DIFFERENT model from the one that wrote the answers, to avoid a
model grading itself leniently. Default: qwen2.5:7b (ollama pull qwen2.5:7b).
If the answers have human labels, the summary also reports how often the judge
agrees with the humans, which shows how far the automatic metric can be trusted.

Run (Ollama must be running; ~5 minutes):
    python metrics.py
    python metrics.py --judge llama3.1:8b                      # use another judge model
    python metrics.py --answers results/eval_answers_v1.csv    # score an older run
"""
import argparse
import json
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from langchain_ollama import ChatOllama

import config
import prompts
import rag_chain

DEFAULT_ANSWERS = os.path.join("results", "eval_answers.csv")
QUESTIONS_FILE = "eval_questions.csv"

# ---------------------------------------------------------------------------
# Judge prompts (the judge must answer in JSON)
# ---------------------------------------------------------------------------
CONTEXT_RELEVANCE_PROMPT = """\
You are grading a search system. Decide how useful the PASSAGE is for answering the QUESTION.

QUESTION: {question}

PASSAGE (from: {source}):
{passage}

The passage is an excerpt from the source named above. If the question names a paper,
video or document, a passage from that source counts as coming from it even if the
passage text does not repeat its name. Judge the content: does it contain the facts
that answer the question, even if it uses different wording than the question?

Scoring:
2 = the passage contains the answer or a clear part of it
1 = the passage is on the same topic but does not help answer this question
0 = the passage is unrelated to the question

First explain your decision in one sentence, then give the score.
Reply with JSON only, with the keys "reason" (string) and "score" (0, 1 or 2)."""

ANSWER_RELEVANCE_PROMPT = """\
You are grading a question-answering system. Rate how well the ANSWER addresses
the QUESTION. Do NOT judge whether it is factually correct, only whether it is on topic
and responds to what was asked.

5 = directly and completely addresses the question
4 = addresses it, with small gaps or some off-topic content
3 = partly addresses it
2 = mostly off-topic, touches the question only slightly
1 = does not address the question at all

QUESTION: {question}

ANSWER: {answer}

Reply with JSON only: {{"score": 1-5, "reason": "one short sentence"}}"""

HALLUCINATION_PROMPT = """\
You are checking an answer for hallucinations.

QUESTION: {question}

REFERENCE ANSWER (written by a human, known to be correct):
{reference}

SOURCES (passages from the knowledge base):
{sources}

ANSWER TO CHECK:
{answer}

Instructions:
1. List the specific factual claims made in the ANSWER TO CHECK (names, numbers, definitions,
   how something works). At most 6. Only take claims from the ANSWER TO CHECK, never from
   the reference answer or the sources. Ignore generic filler that states no specific fact.
   Citations and source labels such as [1], [2][3] or (file.pdf, p. 4) are NOT claims:
   ignore them and check only the statement they are attached to.
2. Label each claim:
   "supported"    - agrees with the reference answer or is stated in the sources
   "contradicted" - conflicts with the reference answer or the sources
   "unverifiable" - a specific fact that appears in neither
3. Overall verdict compared with the reference answer:
   "correct" (covers the reference), "partial" (covers some of it), "incorrect" (does not).

Reply with JSON only:
{{"claims": [{{"claim": "...", "label": "supported|contradicted|unverifiable"}}],
  "verdict": "correct|partial|incorrect"}}"""

FAITHFULNESS_PROMPT = """\
You are checking whether an answer is grounded in its sources.

SOURCES (numbered passages):
{sources}

ANSWER TO CHECK:
{answer}

Instructions:
1. List the specific factual claims made in the ANSWER TO CHECK (names, numbers, definitions,
   how something works). At most 8. Only take claims from the ANSWER TO CHECK.
   Citations such as [1] or [1, p. 4] are NOT claims: ignore them.
2. For each claim, copy the sentence from the SOURCES that states it, word for word.
   Use ONLY the sources, not your own knowledge. Being on the same topic is not enough:
   the sentence must state the same fact. If no sentence states it, use an empty string.

Reply with JSON only:
{{"claims": [{{"claim": "...", "quote": "exact sentence copied from a source, or empty"}}]}}"""

NO_REFERENCE = "(none: the knowledge base does not cover this question)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_json(text: str) -> dict:
    """Parse the judge's JSON, tolerating extra text around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


class Judge:
    """
    The judge model. Copying sentences word for word can make a small model loop
    (repeat the same text until Ollama aborts with "token repeat limit reached").
    So every call has an output limit; a failed call is retried once with a little
    randomness and a repetition penalty; if that fails too, the check is skipped
    (empty result) and counted in self.failures instead of crashing the run.
    """

    def __init__(self, model: str):
        self.model = model
        self.llm = ChatOllama(model=model, temperature=0, num_ctx=8192, num_predict=1024, format="json")
        self.retry_llm = ChatOllama(model=model, temperature=0.3, repeat_penalty=1.15,
                                    num_ctx=8192, num_predict=1024, format="json")
        self.failures = 0

    def ask(self, prompt: str) -> dict:
        for llm in (self.llm, self.retry_llm):
            try:
                result = parse_json(llm.invoke(prompt).content)
                if result:
                    return result
            except Exception as e:  # Ollama aborts, timeouts, ...
                print(f"      judge call failed ({type(e).__name__}: {str(e)[:60]}), retrying")
        self.failures += 1
        return {}


STOPWORDS = set("""a an the and or of to in on for with by from as at is are was were be been being
this that these those it its into than then there their they them which who whom what when where how
not no can could would should will may might also such each other more most some any all both between
about over under only very so if but do does did has have had i we you he she our your his her""".split())


def _tokens(text: str) -> list[str]:
    """Normalised word tokens (ligatures like 'ﬁ' -> 'fi', lower case, plural 's' removed)."""
    text = unicodedata.normalize("NFKC", text).lower()
    words = re.findall(r"[a-z0-9%]+", text)  # hyphenated words split: "task-based" = "task -based" in PDFs
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words]


def _content(text: str) -> set[str]:
    return {w for w in _tokens(text) if w not in STOPWORDS and (len(w) > 2 or any(c.isdigit() for c in w))}


QUOTE_IN_SOURCE = 0.8     # share of the quote's words that must appear in the source
CLAIM_IN_QUOTE = 0.35     # share of the claim's key words that must appear in the quote


CLAIM_IN_SENTENCE = 0.6   # fallback: share of the claim's key words found in one source sentence


def _strip_citations(text: str) -> str:
    """Remove citation markers like [3], [1][2] or [1, p. 4] so they don't count as key words."""
    return re.sub(r"\[[^\]]*\]", " ", text)


def _sentences(text: str) -> list[str]:
    """Source text as single sentences plus pairs of neighbouring sentences."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if p.strip()]
    return parts + [a + " " + b for a, b in zip(parts, parts[1:])]


def find_support(claim: str, sources: list[str]) -> list[int]:
    """
    Deterministic fallback when the judge's quote fails: a source supports the claim
    if one of its sentences (or two neighbouring sentences) contains at least
    CLAIM_IN_SENTENCE of the claim's key words.
    """
    claim_words = _content(_strip_citations(claim))
    if len(claim_words) < 2:
        return []
    found = []
    for n, src in enumerate(sources, 1):
        best = max((len(claim_words & _content(s)) / len(claim_words) for s in _sentences(src)), default=0)
        if best >= CLAIM_IN_SENTENCE:
            found.append(n)
    return found


def verify_quote(claim: str, quote: str, sources: list[str]) -> tuple[list[int], str]:
    """
    Check a judge's evidence without trusting the judge:
      1. find which sources really contain the quote (by word overlap, robust to
         line breaks and PDF ligatures) - this assigns the source number ourselves;
      2. check the quote shares enough key words with the claim, so a sentence
         that is merely on the same topic does not count as support.
    Returns (supporting source numbers, reason).
    """
    # Drop a copied source label such as "[1] (lewis_2020_rag_original.pdf, p. 3)" or "[2]"
    quote = re.sub(r"^\s*\[\d+\]\s*(\([^)]*\))?\s*:?\s*", "", quote or "")
    claim = _strip_citations(claim)
    problem = None
    q_words = set(_tokens(quote))
    if not q_words:
        problem = "no quote"
    else:
        found = [n for n, src in enumerate(sources, 1)
                 if len(q_words & set(_tokens(src))) / len(q_words) >= QUOTE_IN_SOURCE]
        if not found:
            problem = "quote not found in any source"
        else:
            claim_words = _content(claim)
            overlap = len(claim_words & _content(quote)) / len(claim_words) if claim_words else 1
            if overlap < CLAIM_IN_QUOTE:
                problem = f"quote does not state the claim ({overlap:.0%} of key words)"
    if problem is None:
        return found, "ok"
    # The judge's evidence failed: search the sources ourselves before rejecting the claim
    fallback = find_support(claim, sources)
    if fallback:
        return fallback, f"found by key-word search ({problem})"
    return [], problem


def cited_numbers(text: str) -> set[int]:
    """Source numbers cited in an answer: [1], [2][3], [1, 3] and [1, p. 4] all count."""
    nums = set()
    for group in re.findall(r"\[([^\]]+)\]", text):
        for part in group.split(","):
            if part.strip().isdigit():
                nums.add(int(part.strip()))
    return nums


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "yes", "1")


def judge_to_label(verdict: str, hallucinated: bool, refused: bool) -> str:
    """Map the judge's result onto the human label scheme, for the agreement check."""
    if refused:
        return "refused"
    if hallucinated and verdict != "correct":
        return "hallucinated"
    return {"correct": "correct", "partial": "partial"}.get(verdict, "wrong")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(answers: pd.DataFrame, judge: Judge) -> pd.DataFrame:
    embeddings = config.get_embeddings()
    rag_chain.retrieval.load_vector_store()

    # Retrieve the context once per question, with the same settings as the evaluation
    contexts = {}
    for q in answers["question"].unique():
        out = rag_chain._retrieve({"question": q, "k": config.DEFAULT_K, "search": config.DEFAULT_SEARCH})
        contexts[q] = out["docs"]

    # Context relevance: one judge call per retrieved chunk, once per question
    print(f"\nContext relevance ({sum(len(d) for d in contexts.values())} chunks)...")
    context_relevance, context_hit, context_log = {}, {}, []
    for q, docs in contexts.items():
        votes = []
        for n, d in enumerate(docs, 1):
            parsed = judge.ask(CONTEXT_RELEVANCE_PROMPT.format(
                question=q, source=prompts.source_label(d.metadata), passage=d.page_content))
            raw = json.dumps(parsed)
            try:
                chunk_score = min(2, max(0, int(parsed.get("score"))))
            except (TypeError, ValueError):
                chunk_score = 0
            vote = chunk_score == 2  # relevant = contains (part of) the answer
            votes.append(vote)
            context_log.append({"question": q, "chunk": n, "source": prompts.source_label(d.metadata),
                                "score": chunk_score, "relevant": vote, "reason": parsed.get("reason", ""),
                                "judge_raw": raw.strip(), "passage": d.page_content[:300]})
        context_relevance[q] = sum(votes) / len(votes) if votes else np.nan
        context_hit[q] = any(votes)
    score.context_log = pd.DataFrame(context_log)

    rows = []
    print(f"Answer relevance and hallucination ({len(answers)} answers)...")
    for i, r in answers.iterrows():
        q, answer = r["question"], str(r["answer"])
        refused = prompts.is_dont_know(answer)
        in_kb = r["category"] == "in_kb"
        row = {
            "id": r["id"], "mode": r["mode"], "category": r["category"], "question": q, "answer": answer,
            "refused": refused,
            "context_relevance": context_relevance[q] if r["mode"] != "plain" else np.nan,
            "context_hit": context_hit[q] if r["mode"] != "plain" else np.nan,
            "answer_relevance": np.nan, "answer_similarity": np.nan,
            "claims": 0, "supported": 0, "contradicted": 0, "unverifiable": 0,
            "hallucinated": False, "judge_verdict": "refused" if refused else "",
            "answered_without_grounding": (not in_kb) and (not refused),
            "claim_details": "", "relevance_reason": "",
            "faith_claims": 0, "faithful_claims": 0, "faithfulness": np.nan,
            "cited_sources": "", "citation_precision": np.nan, "faith_details": "",
        }
        if not refused:
            # Answer relevance (judge + embeddings)
            rel = judge.ask(ANSWER_RELEVANCE_PROMPT.format(question=q, answer=answer))
            try:
                row["answer_relevance"] = min(5, max(1, int(rel.get("score"))))
            except (TypeError, ValueError):
                pass
            row["relevance_reason"] = str(rel.get("reason", ""))
            qv, av = embeddings.embed_query(q), embeddings.embed_query(answer)
            row["answer_similarity"] = round(float(np.dot(qv, av)), 3)  # vectors are normalised

            # Hallucination check against reference + the same sources for every mode
            reference = r["expected_answer"] if in_kb else NO_REFERENCE
            hal = judge.ask(HALLUCINATION_PROMPT.format(
                question=q, reference=reference,
                sources=prompts.format_context(contexts[q]), answer=answer))
            claims = [c for c in hal.get("claims", []) if isinstance(c, dict)]
            labels = [str(c.get("label", "")).lower() for c in claims]
            row["claims"] = len(claims)
            for name in ("supported", "contradicted", "unverifiable"):
                row[name] = labels.count(name)
            row["hallucinated"] = row["contradicted"] + row["unverifiable"] > 0
            row["judge_verdict"] = str(hal.get("verdict", "")).lower() if in_kb else "answered"
            row["claim_details"] = " | ".join(f"[{c.get('label')}] {c.get('claim')}" for c in claims)

            # Faithfulness: sources only (no reference, no world knowledge)
            n_sources = len(contexts[q])
            faith = judge.ask(FAITHFULNESS_PROMPT.format(
                sources=prompts.format_context(contexts[q]), answer=answer))
            fclaims = [c for c in faith.get("claims", []) if isinstance(c, dict) and str(c.get("claim", "")).strip()]
            source_texts = [d.page_content for d in contexts[q]]
            supporting, reasons = [], []
            for c in fclaims:
                nums, why = verify_quote(str(c.get("claim", "")), str(c.get("quote", "")), source_texts)
                supporting.append(nums)
                reasons.append(why)
            row["faith_claims"] = len(fclaims)
            row["faithful_claims"] = sum(1 for v in supporting if v)
            row["faithfulness"] = row["faithful_claims"] / len(fclaims) if fclaims else np.nan
            row["faith_details"] = " | ".join(
                f"[{','.join(map(str, v)) or 'none'}] {c.get('claim')}"
                + ("" if why == "ok" else f" ({why})")
                for c, v, why in zip(fclaims, supporting, reasons))
            # Citation precision: are the sources the answer cites the ones that state its claims?
            cited = sorted(n for n in cited_numbers(answer) if 1 <= n <= n_sources)
            if cited:
                backing = set().union(*map(set, supporting)) if supporting else set()
                row["cited_sources"] = ",".join(map(str, cited))
                row["citation_precision"] = sum(1 for n in cited if n in backing) / len(cited)

        row["judge_label"] = judge_to_label(row["judge_verdict"], row["hallucinated"], refused) if in_kb \
            else ("refused" if refused else "hallucinated" if row["hallucinated"] else "wrong")
        row["human_label"] = str(r.get("label", "") if pd.notna(r.get("label", "")) else "").strip().lower()
        rows.append(row)
        print(f"  [{len(rows):>2}/{len(answers)}] Q{r['id']} {r['mode']:<22} "
              f"{'refused' if refused else ('HALLUCINATED' if row['hallucinated'] else 'ok')}")
    return pd.DataFrame(rows)


def summarise(scored: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for mode, p in scored.groupby("mode", sort=False):
        inside = p[p.category == "in_kb"]
        outside = p[p.category == "out_of_kb"]
        answered_in = inside[~inside.refused]
        row = {
            "mode": mode,
            "context relevance, in_kb (RAG)": round(inside.context_relevance.mean(), 2) if mode != "plain" else None,
            "in_kb: questions with a relevant chunk": (f"{int(inside.context_hit.astype(bool).sum())}/{len(inside)}"
                                                       if mode != "plain" else None),
            "answer relevance 1-5 (judge)": round(p.answer_relevance.mean(), 2),
            "answer similarity (embeddings)": round(p.answer_similarity.mean(), 2),
            "in_kb: hallucinated answers": f"{int(answered_in.hallucinated.sum())}/{len(answered_in)}",
            "in_kb: hallucination rate": round(answered_in.hallucinated.mean(), 2) if len(answered_in) else None,
            "in_kb: unsupported claims": f"{int((answered_in.contradicted + answered_in.unverifiable).sum())}"
                                         f"/{int(answered_in.claims.sum())}",
            "out_of_kb: answered without grounding": f"{int(outside.answered_without_grounding.sum())}/{len(outside)}",
            "in_kb: faithfulness (claims backed by sources)": round(answered_in.faithfulness.mean(), 2)
                                                              if answered_in.faithfulness.notna().any() else None,
            "in_kb: fully faithful answers": f"{int((answered_in.faithfulness == 1).sum())}/{len(answered_in)}",
            "citation precision": round(p.citation_precision.mean(), 2) if p.citation_precision.notna().any() else None,
        }
        labelled = p[p.human_label.isin(["correct", "partial", "wrong", "hallucinated", "refused"])]
        if len(labelled):
            row["judge agrees with humans"] = f"{int((labelled.judge_label == labelled.human_label).sum())}/{len(labelled)}"
        summary.append(row)
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description="Relevance and hallucination metrics for evaluate.py answers.")
    parser.add_argument("--answers", default=DEFAULT_ANSWERS)
    parser.add_argument("--judge", default=config.JUDGE_MODEL, help="Ollama model used as judge")
    args = parser.parse_args()

    answers = pd.read_csv(args.answers, encoding="utf-8-sig")
    # Always use the current reference answers from eval_questions.csv, so corrections
    # there apply without rerunning evaluate.py
    reference = pd.read_csv(QUESTIONS_FILE).set_index("id")["expected_answer"]
    answers["expected_answer"] = answers["id"].map(reference).fillna(answers["expected_answer"])
    out_rows = args.answers.replace("eval_answers", "metrics_answers")
    out_summary = args.answers.replace("eval_answers", "metrics_summary")

    print(f"Judge model: {args.judge} | answers: {args.answers}")
    judge = Judge(args.judge)
    try:
        judge.llm.invoke("Reply with JSON: {\"ok\": true}")
    except Exception as e:
        raise SystemExit(f"Could not use judge model '{args.judge}': {e}\n"
                         f"Download it with:  ollama pull {args.judge}")

    t = time.perf_counter()
    scored = score(answers, judge)
    scored.to_csv(out_rows, index=False, encoding="utf-8-sig")
    out_context = args.answers.replace("eval_answers", "metrics_context")
    score.context_log.to_csv(out_context, index=False, encoding="utf-8-sig")
    summary = summarise(scored)
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    with pd.option_context("display.max_columns", None, "display.width", 250):
        print("\n" + summary.to_string(index=False))
    print(f"\nSaved {out_rows}, {out_summary} and {out_context} ({time.perf_counter() - t:.0f} s)")
    if judge.failures:
        print(f"Note: {judge.failures} judge call(s) failed twice and were skipped "
              f"(those checks are left empty in {out_rows}).")


if __name__ == "__main__":
    main()