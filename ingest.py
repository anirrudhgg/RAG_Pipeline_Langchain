"""
ingest.py - Parts A and B of the assignment.

  1. Load three document types: PDFs, web pages, YouTube transcripts
  2. Clean the text and split it into chunks (settings per document type)
  3. Embed the chunks with all-MiniLM-L6-v2
  4. Store them in a persistent ChromaDB collection with metadata
     (doc_type, source, title, page / timestamp, chunk_id)

Run from the project folder (with the venv active):
    python ingest.py

Re-running rebuilds the database from scratch.
"""
import os
import re
import shutil
import time
from urllib.parse import parse_qs, urlparse

# WebBaseLoader warns if no user agent is set
os.environ.setdefault("USER_AGENT", "RAG-Pipeline-Student-Project/1.0")

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Basic cleanup: rejoin words hyphenated across lines, collapse whitespace."""
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def read_list(path: str) -> list[str]:
    """Read a text file with one entry per line (blank lines and # comments ignored)."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Loaders - one per document type
# ---------------------------------------------------------------------------
# Headings that start / end the bibliography of a paper
REFERENCES_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$", re.I | re.M)
APPENDIX_HEADING = re.compile(r"^\s*(appendix|appendices)\b", re.I | re.M)


def remove_reference_sections(pages: list[str]) -> list[str]:
    """
    Blank out the bibliography of a paper.

    Reference lists are full of RAG-related titles, so they match almost every
    query but contain no actual explanations - they pushed real content out of
    the top results in our first test. Text after a 'References' heading is
    dropped until an 'Appendix' heading (if any) starts real content again.
    Only headings in the second half of the paper count, to avoid false matches.
    """
    result, in_refs = [], False
    for i, text in enumerate(pages):
        if not in_refs:
            m = REFERENCES_HEADING.search(text)
            if m and i >= len(pages) * 0.4:
                text, in_refs = text[:m.start()], True
        else:
            m = APPENDIX_HEADING.search(text)
            if m:
                text, in_refs = text[m.start():], False
            else:
                text = ""
        result.append(text)
    return result


def load_pdfs() -> list[Document]:
    """One Document per PDF page (bibliography removed), keeping the page number for citations."""
    docs = []
    if not os.path.isdir(config.PDF_DIR):
        print(f"  ! Folder {config.PDF_DIR} not found, skipping PDFs")
        return docs

    for name in sorted(os.listdir(config.PDF_DIR)):
        if not name.lower().endswith(".pdf"):
            continue
        try:
            pages = PyPDFLoader(os.path.join(config.PDF_DIR, name)).load()
        except Exception as e:
            print(f"  ! Could not load {name}: {e}")
            continue

        texts = remove_reference_sections([p.page_content for p in pages])
        kept = 0
        for p, raw in zip(pages, texts):
            text = clean_text(raw)
            if len(text) < 50:  # skip empty / image-only / reference-only pages
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "doc_type": "pdf",
                    "source": name,
                    "title": str(p.metadata.get("title") or name),
                    "page": int(p.metadata.get("page", 0)) + 1,  # 1-based for humans
                },
            ))
            kept += 1
        print(f"  PDF      {name}: {kept} of {len(pages)} pages kept (references removed)")
    return docs


def load_web() -> list[Document]:
    """One Document per web page."""
    docs = []
    for url in read_list(config.URLS_FILE):
        try:
            loaded = WebBaseLoader(url).load()
        except Exception as e:
            print(f"  ! Could not load {url}: {e}")
            continue

        for d in loaded:
            text = clean_text(d.page_content)
            if len(text) < 200:
                print(f"  ! Very little text on {url}, skipped (page may need JavaScript)")
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "doc_type": "web",
                    "source": url,
                    "title": str(d.metadata.get("title") or url).strip(),
                },
            ))
            print(f"  Web      {url}: {len(text):,} characters")
    return docs


def get_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").replace("www.", "").replace("m.", "")
    if host == "youtu.be":
        return parsed.path.lstrip("/").split("/")[0] or None
    if parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [None])[0]
    for prefix in ("/shorts/", "/embed/", "/live/"):
        if parsed.path.startswith(prefix):
            return parsed.path[len(prefix):].split("/")[0] or None
    return None


def fetch_transcript(video_id: str) -> list[dict]:
    """Return [{'text': ..., 'start': seconds}, ...]; works with old and new youtube-transcript-api."""
    from youtube_transcript_api import YouTubeTranscriptApi

    languages = ["en", "en-US", "en-GB"]
    try:  # version 1.x
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        return [{"text": s.text, "start": s.start} for s in fetched]
    except AttributeError:  # version 0.x
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)


def load_youtube() -> list[Document]:
    """
    Transcripts grouped into ~60-second windows. Each window keeps its start
    time, so answers can cite e.g. 'video at 2:15' with a link to that moment.

    youtube.txt format, one video per line:   URL   or   URL | Video title
    """
    docs = []
    for line in read_list(config.YOUTUBE_FILE):
        url, _, title = (part.strip() for part in line.partition("|"))
        video_id = get_video_id(url)
        if not video_id:
            print(f"  ! Not a YouTube link: {url}")
            continue
        title = title or f"YouTube video {video_id}"

        try:
            snippets = fetch_transcript(video_id)
        except Exception as e:
            print(f"  ! No transcript for {url} ({type(e).__name__}) - does it have captions?")
            continue

        windows, current, start = [], [], None
        for s in snippets:
            if start is None:
                start = s["start"]
            current.append(s["text"].replace("\n", " "))
            if s["start"] - start >= config.YOUTUBE_WINDOW_SECONDS:
                windows.append((start, " ".join(current)))
                current, start = [], None
        if current:
            windows.append((start, " ".join(current)))

        for start, text in windows:
            docs.append(Document(
                page_content=clean_text(text),
                metadata={
                    "doc_type": "youtube",
                    "source": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
                    "title": title,
                    "start_seconds": int(start),
                    "timestamp": format_timestamp(start),
                },
            ))
        print(f"  YouTube  {title}: {len(windows)} windows of ~{config.YOUTUBE_WINDOW_SECONDS}s")
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def split_documents(docs: list[Document]) -> list[Document]:
    """Split each document type with its own chunk size / overlap."""
    chunks = []
    for doc_type, settings in config.CHUNK_SETTINGS.items():
        typed = [d for d in docs if d.metadata["doc_type"] == doc_type]
        if not typed:
            continue
        splitter = RecursiveCharacterTextSplitter(**settings)
        parts = splitter.split_documents(typed)
        avg = sum(len(p.page_content) for p in parts) / len(parts)
        print(f"  {doc_type:<8} {len(typed):>4} docs -> {len(parts):>5} chunks "
              f"(size {settings['chunk_size']}, overlap {settings['chunk_overlap']}, avg {avg:.0f} chars)")
        chunks.extend(parts)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def build_vector_store(chunks: list[Document], batch_size: int = 500) -> Chroma:
    """Embed all chunks and store them in a fresh, persistent ChromaDB collection."""
    if os.path.exists(config.CHROMA_DIR):
        shutil.rmtree(config.CHROMA_DIR)

    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=config.get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        store.add_documents(batch, ids=[f"chunk-{c.metadata['chunk_id']}" for c in batch])
        print(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
    return store


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()

    print("\n[1/4] Loading documents")
    docs = load_pdfs() + load_web() + load_youtube()
    if not docs:
        print("No documents loaded - check the data/ folder.")
        return
    t1 = time.perf_counter()

    print("\n[2/4] Splitting into chunks")
    chunks = split_documents(docs)
    t2 = time.perf_counter()

    print("\n[3/4] Embedding and storing in ChromaDB (first run downloads the model, ~90 MB)")
    store = build_vector_store(chunks)
    t3 = time.perf_counter()

    print("\n[4/4] Test search: 'What is retrieval-augmented generation?'")
    results = store.similarity_search_with_score("What is retrieval-augmented generation?", k=3)
    for doc, score in results:
        m = doc.metadata
        where = f"p. {m['page']}" if "page" in m else m.get("timestamp", "")
        print(f"  distance {score:.3f} | {m['doc_type']:<7} | {m['title'][:50]} {where}")
        print(f"     {doc.page_content[:120]!r}...")

    print(f"\nDone: {len(docs)} documents -> {len(chunks)} chunks in '{config.CHROMA_DIR}/'")
    print(f"Timing: loading {t1 - t0:.1f}s | chunking {t2 - t1:.1f}s | "
          f"embedding {t3 - t2:.1f}s | total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()