"""
app.py - Part C of the assignment: the user interface.

A Streamlit web app for the live demo:
  - ask a question (or pick an example)
  - choose prompt style, search method, k and the small-to-big context window
  - see the RAG answer with clickable, numbered sources next to the plain LLM's
    answer to the same question (the before/after comparison)
  - inspect the exact prompt sent to the model and the timing of each step

Run it (Ollama must be running):
    streamlit run app.py
It opens in your browser at http://localhost:8501
"""
import re

import streamlit as st

import config
import prompts
import rag_chain

st.set_page_config(page_title="RAG Explorer", page_icon="📚", layout="wide")

# Example questions for the demo: short button label -> full question
EXAMPLES = {
    "In the documents": {
        "RAG-Sequence vs RAG-Token": "What is the difference between RAG-Sequence and RAG-Token?",
        "RAG vs fine-tuning": "When should I use RAG instead of fine-tuning?",
        "Enterprise search survey": "According to Klesel and Wittmann, what share of employees are "
                                    "dissatisfied with enterprise search interfaces?",
        "Chroma distance functions": "Which distance functions can a Chroma collection use for its HNSW index?",
    },
    "Not in the documents": {
        "Capital of Australia": "What is the capital of Australia?",
        "Pinecone pricing": "How much does Pinecone's enterprise plan cost per month?",
    },
}
TYPE_LABEL = {"pdf": "PDF", "web": "Web", "youtube": "Video"}


# ---------------------------------------------------------------------------
# Load models once per server start (not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the embedding model, database and Llama 3.1...")
def load_models():
    rag_chain.warm_up()
    return True


def render_markdown_safe(text: str) -> str:
    """Escape characters Streamlit would treat as LaTeX/markup, and highlight citations."""
    text = text.replace("$", "\\$")
    return re.sub(r"\[(\d+)\]", r"**[\1]**", text)


load_models()

# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    style = st.selectbox(
        "Prompt style", list(prompts.STYLES), index=0,
        help="Zero-shot: rules only. Few-shot: adds worked examples. Chain-of-thought: reasons "
             "step by step first. Tree-of-thoughts: explores and rates several reasoning paths.",
    )
    search = st.radio(
        "Search method", ["mmr", "similarity"],
        index=["mmr", "similarity"].index(config.DEFAULT_SEARCH),
        format_func=lambda s: "Semantic (MMR): relevant and diverse" if s == "mmr" else "Similarity: closest k chunks",
    )
    k = st.slider("Chunks to retrieve (k)", 1, 12, config.DEFAULT_K)
    window = st.slider(
        "Context window (small-to-big)", 0, 4, config.CONTEXT_WINDOW,
        help="Neighbouring chunks added on each side of every hit, so the model gets continuous "
             "text. 0 = only the small retrieved chunks.",
    )
    compare = st.toggle("Compare with plain LLM (no retrieval)", value=True)

    st.divider()
    st.caption(
        f"Model: **{config.OLLAMA_MODEL}** via Ollama  \n"
        f"Embeddings: all-MiniLM-L6-v2  \n"
        f"Vector DB: ChromaDB, {rag_chain.retrieval.load_vector_store()._collection.count():,} chunks"
    )

# ---------------------------------------------------------------------------
# Main area: question
# ---------------------------------------------------------------------------
st.title("RAG Explorer")
st.markdown(
    "Ask about Retrieval-Augmented Generation. Answers come only from 5 research papers, "
    "4 documentation pages and 2 IBM videos, with numbered citations. "
    "The plain LLM answers the same question from memory, for comparison."
)

if "question" not in st.session_state:
    st.session_state.question = ""

st.caption("Try an example")
for group, questions in EXAMPLES.items():
    cols = st.columns([1.4, 2, 2, 2, 2])
    cols[0].markdown(f"<div style='padding-top:8px;font-size:0.85rem;opacity:0.7'>{group}</div>",
                     unsafe_allow_html=True)
    for col, (label, q) in zip(cols[1:], questions.items()):
        if col.button(label, use_container_width=True, key=f"ex-{label}"):
            st.session_state.question = q
            st.session_state.run = True

with st.form("ask", clear_on_submit=False):
    question = st.text_input("Your question", key="question",
                             placeholder="e.g. How does MMR differ from similarity search?")
    submitted = st.form_submit_button("Ask", type="primary")

run = submitted or st.session_state.pop("run", False)

# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------
if run and st.session_state.question.strip():
    q = st.session_state.question.strip()
    config.CONTEXT_WINDOW = window  # read by rag_chain at query time

    with st.spinner("Retrieving sources and generating the RAG answer..."):
        result = rag_chain.answer_with_rag(q, style=style, k=k, search=search)
    plain = None
    if compare:
        with st.spinner("Asking the plain LLM..."):
            plain = rag_chain.answer_plain(q)
    st.session_state.last = {"rag": result, "plain": plain}

last = st.session_state.get("last")
if last:
    result, plain = last["rag"], last["plain"]
    left, right = st.columns(2) if plain else (st.container(), None)

    with left:
        st.subheader("With RAG")
        if result["dont_know"]:
            st.info("The documents don't contain the answer, so the model declined to guess.")
        elif result["uncited_claims"]:
            st.warning("This answer contains no citations. Treat it with caution.")
        else:
            cited = ", ".join(f"[{src['n']}]" for src in result["sources"] if src["cited"])
            st.success(f"Answer grounded in sources {cited}")

        st.markdown(render_markdown_safe(result["answer"]))

        if result["response"].strip() != result["answer"]:
            with st.expander(f"Reasoning ({result['style']})"):
                st.markdown(render_markdown_safe(result["response"].rsplit("Answer:", 1)[0].strip()))

        m1, m2, m3 = st.columns(3)
        m1.metric("Retrieval", f"{result['retrieval_ms']:.0f} ms")
        m2.metric("Generation", f"{result['generation_ms'] / 1000:.1f} s")
        m3.metric("Total", f"{result['total_ms'] / 1000:.1f} s")

        st.markdown("**Sources**")
        for s in result["sources"]:
            tag = TYPE_LABEL.get(s["doc_type"], s["doc_type"])
            status = "cited" if s["cited"] else "retrieved"
            score = f" · similarity {s['score']:.2f}" if s["score"] is not None else ""
            title = f"[{s['n']}] {tag} · {s['label']} · {status}{score}"
            with st.expander(title, expanded=False):
                if s["link"]:
                    st.markdown(f"[Open source]({s['link']})")
                st.write(s["text"])

        with st.expander("Prompt sent to the LLM"):
            st.code(result["prompt"], language=None)

    if right is not None:
        with right:
            st.subheader("Plain LLM")
            st.info("No retrieval: the model answers from its training data, without sources.")
            st.markdown(render_markdown_safe(plain["answer"]))
            st.metric("Total", f"{plain['total_ms'] / 1000:.1f} s")
            with st.expander("Prompt sent to the LLM"):
                st.code(prompts.plain_llm(plain["question"]), language=None)
