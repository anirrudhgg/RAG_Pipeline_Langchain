"""
prompts.py - Part C of the assignment: prompt engineering for the RAG pipeline.

Four prompting styles, all grounded in the retrieved context:
  - zero-shot          : rules + context + question, no examples
  - few-shot           : the same, plus worked examples showing the expected
                         answer format (with citations and one "don't know" case)
  - chain-of-thought   : the model reasons step by step over the sources first
  - tree-of-thoughts   : the model explores several readings of the sources,
                         rates them, and continues with the best one

Every style enforces the two requirements from the assignment:
  - Citations: every statement must point to a numbered source, e.g. [2]
  - "Don't know" fallback: if the context does not contain the answer, the
    model must reply with exactly DONT_KNOW instead of guessing

There is also a PLAIN_LLM prompt without any context, used as the baseline in
the RAG-vs-plain-LLM comparison (Part D).

Each builder returns a plain string, so it works with any LLM client
(e.g. ChatOllama(...).invoke(prompt)).
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

# --------------------------------------------------------------------------
# Shared rules
# --------------------------------------------------------------------------

# Exact fallback sentence. evaluate.py checks for this string, so keep the two in sync.
DONT_KNOW = "I don't know based on the provided documents."

RAG_INSTRUCTION = f"""\
You are a research assistant answering questions about Retrieval-Augmented Generation (RAG).

Rules:
1. Use ONLY the information in the numbered sources under "Context". Do not use outside knowledge.
2. Cite every statement with the number of the source it comes from, e.g. [1] or [2][3].
3. If the sources do not contain the answer, reply with exactly: "{DONT_KNOW}"
   Do not guess and do not add citations to this sentence.
4. Be concise: answer in at most 5 sentences unless the question asks for a list."""

# --------------------------------------------------------------------------
# Raw templates
# --------------------------------------------------------------------------

ZERO_SHOT_TEMPLATE = """\
{instruction}

Context:
{context}

Question: {question}
Answer:"""

FEW_SHOT_TEMPLATE = """\
{instruction}

Here are examples of how to answer:

{examples}

Now answer the real question.

Context:
{context}

Question: {question}
Answer:"""

FEW_SHOT_EXAMPLE_TEMPLATE = """\
Example {n}
Context:
{context}

Question: {question}
Answer: {answer}"""

CHAIN_OF_THOUGHT_TEMPLATE = """\
{instruction}

Context:
{context}

Question: {question}

Think step by step before answering:
1. List which sources are relevant to the question and what each one says.
2. Combine that information to work out the answer. Note any gaps or contradictions.
3. If no source contains the answer, the final answer is the fallback sentence.

Write your numbered reasoning steps, then give the final answer with citations \
on a new line starting with "Answer:".

Reasoning:"""

TREE_OF_THOUGHTS_TEMPLATE = """\
{instruction}

Context:
{context}

Question: {question}

Solve this by exploring several reasoning paths:
1. Generate {n_branches} distinct ways to answer the question from the sources \
(e.g. relying on different sources, or different interpretations of the question). \
Label them "Branch A", "Branch B", etc.
2. For each branch, work through the first {depth} reasoning steps, citing the sources used.
3. Evaluate each branch: rate it 1-10 for how fully and directly the sources support it, \
and explain why.
4. Discard weak branches and expand the best one into a full answer. If it turns out \
the sources do not support it, backtrack to the next best branch.
5. If no branch is supported by the sources, the final answer is the fallback sentence.
6. Finish with the final answer with citations on a new line starting with "Answer:".

Branches:"""

# Baseline for Part D: the same question, no retrieval, no rules.
PLAIN_LLM_TEMPLATE = """\
Answer the following question.

Question: {question}
Answer:"""

# --------------------------------------------------------------------------
# Few-shot examples
# --------------------------------------------------------------------------
# Deliberately NOT taken from our knowledge base, so they teach the answer
# format without leaking answers to the test questions. The third example
# shows the "don't know" fallback.

DEFAULT_FEW_SHOT_EXAMPLES: list[Tuple[str, str, str]] = [
    (
        "[1] (glossary.pdf, p. 4)\n"
        "A tokenizer splits text into tokens, which are the units a language model reads.\n\n"
        "[2] (nlp_intro.html)\n"
        "Modern tokenizers such as BPE split rare words into several sub-word tokens.",
        "What does a tokenizer do?",
        "A tokenizer splits text into tokens, the units a language model reads [1]. "
        "Modern tokenizers like BPE split rare words into several sub-word tokens [2].",
    ),
    (
        "[1] (search_basics.pdf, p. 2)\n"
        "BM25 is a keyword-based ranking function that scores documents by term frequency.\n\n"
        "[2] (vector_search.html)\n"
        "Dense retrieval embeds queries and documents as vectors and ranks them by similarity.",
        "How does dense retrieval differ from BM25?",
        "BM25 ranks documents by keyword term frequency [1], while dense retrieval compares "
        "vector embeddings of the query and documents by similarity [2].",
    ),
    (
        "[1] (gpu_guide.pdf, p. 7)\n"
        "Inference speed depends on memory bandwidth and batch size.",
        "Who invented the transformer architecture?",
        DONT_KNOW,
    ),
]

# --------------------------------------------------------------------------
# Context formatting
# --------------------------------------------------------------------------


def source_label(metadata: dict) -> str:
    """Short citation label for a chunk, e.g. 'lewis_2020_rag_original.pdf, p. 3'."""
    doc_type = metadata.get("doc_type")
    if doc_type == "pdf":
        return f"{metadata.get('source')}, p. {metadata.get('page')}"
    if doc_type == "youtube":
        return f"{metadata.get('title')}, at {metadata.get('timestamp')}"
    return str(metadata.get("title") or metadata.get("source", "unknown source"))


def format_context(docs: Sequence) -> str:
    """
    Turn retrieved chunks into numbered sources for the prompt:

        [1] (lewis_2020_rag_original.pdf, p. 3)
        <chunk text>

    The numbers are what the model cites; rag_chain.py maps them back to the
    real documents to show them under the answer. Works with LangChain
    Documents (anything with .page_content and .metadata).
    """
    if not docs:
        return "(no sources found)"
    blocks = []
    for i, doc in enumerate(docs, 1):
        text = " ".join(doc.page_content.split())
        blocks.append(f"[{i}] ({source_label(doc.metadata)})\n{text}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def zero_shot(question: str, context: str, instruction: str = RAG_INSTRUCTION) -> str:
    """Rules + context + question, no examples."""
    return ZERO_SHOT_TEMPLATE.format(instruction=instruction, context=context, question=question)


def few_shot(
    question: str,
    context: str,
    examples: Iterable[Tuple[str, str, str]] = DEFAULT_FEW_SHOT_EXAMPLES,
    instruction: str = RAG_INSTRUCTION,
) -> str:
    """Rules + worked (context, question, answer) examples + context + question."""
    examples = list(examples)
    if not examples:
        raise ValueError("few_shot requires at least one example; use zero_shot instead.")
    rendered = "\n\n".join(
        FEW_SHOT_EXAMPLE_TEMPLATE.format(n=n, context=c, question=q, answer=a)
        for n, (c, q, a) in enumerate(examples, 1)
    )
    return FEW_SHOT_TEMPLATE.format(
        instruction=instruction, examples=rendered, context=context, question=question
    )


def chain_of_thought(question: str, context: str, instruction: str = RAG_INSTRUCTION) -> str:
    """Reason step by step over the sources, then answer."""
    return CHAIN_OF_THOUGHT_TEMPLATE.format(
        instruction=instruction, context=context, question=question
    )


def tree_of_thoughts(
    question: str,
    context: str,
    instruction: str = RAG_INSTRUCTION,
    n_branches: int = 3,
    depth: int = 2,
) -> str:
    """Branch, evaluate, prune and backtrack over the sources, then answer."""
    if n_branches < 2:
        raise ValueError("n_branches must be >= 2")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    return TREE_OF_THOUGHTS_TEMPLATE.format(
        instruction=instruction, context=context, question=question,
        n_branches=n_branches, depth=depth,
    )


def plain_llm(question: str) -> str:
    """Baseline prompt without retrieval, for the RAG-vs-plain-LLM comparison."""
    return PLAIN_LLM_TEMPLATE.format(question=question)


# Single entry point used by rag_chain.py and the UI dropdown
STYLES = {
    "Zero-shot": zero_shot,
    "Few-shot": few_shot,
    "Chain-of-thought": chain_of_thought,
    "Tree-of-thoughts": tree_of_thoughts,
}


def build_prompt(style: str, question: str, context: str) -> str:
    """Build the prompt for one of the STYLES names."""
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}; choose from {list(STYLES)}")
    return STYLES[style](question, context)


def extract_answer(response: str) -> str:
    """
    Return only the final answer from a model response.
    Chain-of-thought and tree-of-thoughts responses contain reasoning first and
    the answer after the last "Answer:"; other responses are returned unchanged.
    """
    marker = "Answer:"
    if marker in response:
        return response.rsplit(marker, 1)[1].strip()
    return response.strip()


def is_dont_know(answer: str) -> bool:
    """True if the model used the fallback (tolerant of small wording/punctuation changes)."""
    return "i don't know" in answer.lower() or "i do not know" in answer.lower()


__all__ = [
    "DONT_KNOW",
    "RAG_INSTRUCTION",
    "STYLES",
    "build_prompt",
    "zero_shot",
    "few_shot",
    "chain_of_thought",
    "tree_of_thoughts",
    "plain_llm",
    "format_context",
    "source_label",
    "extract_answer",
    "is_dont_know",
    "ZERO_SHOT_TEMPLATE",
    "FEW_SHOT_TEMPLATE",
    "CHAIN_OF_THOUGHT_TEMPLATE",
    "TREE_OF_THOUGHTS_TEMPLATE",
    "PLAIN_LLM_TEMPLATE",
]


if __name__ == "__main__":
    # Preview every style with a small fake context (no LLM or database needed)
    from types import SimpleNamespace

    fake_docs = [
        SimpleNamespace(
            page_content="RAG combines a pre-trained retriever (DPR) with a seq2seq generator (BART).",
            metadata={"doc_type": "pdf", "source": "lewis_2020_rag_original.pdf", "page": 2},
        ),
        SimpleNamespace(
            page_content="Use RAG when the knowledge changes often; fine-tune to change style or behaviour.",
            metadata={"doc_type": "youtube", "title": "IBM Technology: RAG vs. Fine-Tuning", "timestamp": "3:08"},
        ),
    ]
    ctx = format_context(fake_docs)
    q = "What are the two components of the original RAG model?"

    for name in STYLES:
        print("=" * 25, name, "=" * 25)
        print(build_prompt(name, q, ctx))
        print()
    print("=" * 25, "Plain LLM baseline", "=" * 25)
    print(plain_llm(q))
