"""
csv_preprocessing.py
--------------------
Loads, cleans, and chunks ALL datasets into LangChain Documents,
then builds and persists a ChromaDB vector store via set_vectorstore().

Datasets
--------
CSV:
  1. Patient-Doctor Conversations (50K) — patient_input, doctor_response
  2. MedQuAD Medical Q&A             — question, answer
  3. MTSamples Transcriptions         — transcription, medical_specialty

JSON:
  4. PMC Articles (datasets_pmc/)     — publisher_content, disease, title

Usage
-----
    # Build vectorstore once (writes to chroma_db/)
    python csv_preprocessing.py

    # Import in app startup to load existing store
    from csv_preprocessing import load_vectorstore
    vs = load_vectorstore()
"""

import json
import logging
import os
import re

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


CSV_DIR     = "csv_datasets"
PMC_DIR     = "datasets_pmc"
CHROMA_DIR  = "chroma_db"

PATIENT_DOCTOR_FILE = os.path.join(
    CSV_DIR, "Healthcare Patient-Doctor Conversation Dataset (50K Samples).csv"
)
MEDQUAD_FILE   = os.path.join(CSV_DIR, "medquad.csv")
MTSAMPLES_FILE = os.path.join(CSV_DIR, "mtsamples.csv")


CHUNK_SIZE        = 800
CHUNK_OVERLAP     = 100
MIN_CONTENT_LEN   = 100   # skip PMC articles shorter than this

# Fallback: recursive character splitter (always available)
_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)

# Primary: semantic chunker — initialised lazily so a missing
# langchain_experimental install does not break the import.
_semantic_chunker = None

def _get_semantic_chunker():
    """Return a cached SemanticChunker backed by a local HuggingFace
    sentence-transformer (no API calls).  Returns None if unavailable."""
    global _semantic_chunker
    if _semantic_chunker is not None:
        return _semantic_chunker
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings
        _semantic_chunker = SemanticChunker(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            breakpoint_threshold_type="percentile",
        )
        logger.info("SemanticChunker (local HuggingFace) initialised successfully.")
    except Exception as exc:
        logger.warning("SemanticChunker unavailable (%s) — will use recursive fallback.", exc)
        _semantic_chunker = None
    return _semantic_chunker


# ── Shared utilities ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalize whitespace, strip citation markers and control characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[\d+[\d,\s]*\]", "", text)          # [1], [2,3]
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
    text = re.sub(r" +", " ", text)                        # multiple spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                 # 3+ newlines -> 2
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) >= 3]           # drop stub lines
    return "\n".join(lines).strip()


def _chunk(text: str, metadata: dict) -> list[Document]:
    """Split cleaned text into chunks, attach metadata.

    Strategy
    --------
    1. Try SemanticChunker (embedding-based boundary detection).
    2. Fall back to RecursiveCharacterTextSplitter if semantic chunking
       is unavailable or produces no output.
    """
    pieces: list[str] = []

    chunker = _get_semantic_chunker()
    if chunker is not None:
        try:
            pieces = chunker.split_text(text)
        except Exception as exc:
            logger.warning("SemanticChunker failed (%s) — falling back to recursive splitter.", exc)
            pieces = []

    if not pieces:
        pieces = _fallback_splitter.split_text(text)

    return [
        Document(page_content=piece, metadata={**metadata, "chunk_index": i})
        for i, piece in enumerate(pieces)
    ]


# ── Dataset 1 : Patient-Doctor Conversations ──────────────────────────────────

def load_patient_doctor() -> list[Document]:
    """
    Shape   : (50000, 10)
    Used    : patient_input, doctor_response, condition, symptom, category
    Strategy: One dialogue block per row → typically 1 chunk each.
    """
    df = pd.read_csv(PATIENT_DOCTOR_FILE, on_bad_lines="skip")
    df = df.dropna(subset=["patient_input", "doctor_response"])
    df["condition"] = df["condition"].fillna("")
    df["symptom"]   = df["symptom"].fillna("")
    df["category"]  = df["category"].fillna("")

    documents: list[Document] = []
    for _, row in df.iterrows():
        raw = (
            f"Patient: {row['patient_input']}\n"
            f"Doctor: {row['doctor_response']}"
        )
        cleaned = _clean(raw)
        if not cleaned:
            continue
        metadata = {
            "source":    "patient_doctor_conversations",
            "condition": str(row["condition"]),
            "symptom":   str(row["symptom"]),
            "category":  str(row["category"]),
        }
        documents.extend(_chunk(cleaned, metadata))

    print(f"[patient_doctor]  rows={len(df):,}  chunks={len(documents):,}")
    return documents


# ── Dataset 2 : MedQuAD Medical Q&A ───────────────────────────────────────────

def load_medquad() -> list[Document]:
    """
    Shape   : (16412, 4)
    Used    : question, answer, source, focus_area
    Strategy: Q + A block → long answers produce multiple chunks.
              5 rows with null answer are dropped.
    """
    df = pd.read_csv(MEDQUAD_FILE, on_bad_lines="skip")
    df = df.dropna(subset=["question", "answer"])
    df["focus_area"] = df["focus_area"].fillna("")
    df["source"]     = df["source"].fillna("")

    documents: list[Document] = []
    for _, row in df.iterrows():
        raw = f"Q: {row['question']}\nA: {row['answer']}"
        cleaned = _clean(raw)
        if not cleaned:
            continue
        metadata = {
            "source":     "medquad",
            "focus_area": str(row["focus_area"]),
            "origin":     str(row["source"]),
        }
        documents.extend(_chunk(cleaned, metadata))

    print(f"[medquad]         rows={len(df):,}  chunks={len(documents):,}")
    return documents


# ── Dataset 3 : MTSamples Medical Transcriptions ──────────────────────────────

def load_mtsamples() -> list[Document]:
    """
    Shape   : (4999, 6)
    Used    : transcription, medical_specialty, sample_name, description
    Strategy: description header + full transcription body.
              33 rows with null transcription dropped.
              keywords column ignored (67% null, low value).
    """
    df = pd.read_csv(MTSAMPLES_FILE, on_bad_lines="skip")
    df = df.dropna(subset=["transcription"])
    df["description"]       = df["description"].fillna("")
    df["medical_specialty"] = df["medical_specialty"].fillna("")
    df["sample_name"]       = df["sample_name"].fillna("")

    documents: list[Document] = []
    for _, row in df.iterrows():
        raw = (
            f"Case: {row['description'].strip()}\n\n"
            f"Specialty: {row['medical_specialty'].strip()}\n\n"
            f"Transcription:\n{row['transcription']}"
        )
        cleaned = _clean(raw)
        if not cleaned:
            continue
        metadata = {
            "source":    "mtsamples_transcriptions",
            "specialty": str(row["medical_specialty"]).strip(),
            "case_name": str(row["sample_name"]).strip(),
        }
        documents.extend(_chunk(cleaned, metadata))

    print(f"[mtsamples]       rows={len(df):,}  chunks={len(documents):,}")
    return documents


# ── Dataset 4 : PMC JSON Articles ─────────────────────────────────────────────

def _load_pmc_file(path: str) -> Document | None:
    """
    Parse a single PMC JSON file and return a list of chunks, or None if the
    file should be skipped (unreadable, too short, or empty after cleaning).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    content: str = data.get("publisher_content", "")
    if len(content) < MIN_CONTENT_LEN:
        return None

    cleaned = _clean(content)
    if not cleaned:
        return None

    return data


def load_pmc_articles() -> list[Document]:
    """
    Source  : datasets_pmc/*.json  (generated by scraping_pmc.py)
    Used    : publisher_content, disease, title, publisher_url
    Strategy: Clean raw article body text → chunk into 800-char pieces.
              Articles shorter than MIN_CONTENT_LEN chars are skipped.
              Returns empty list if datasets_pmc/ does not exist yet.
    """
    if not os.path.isdir(PMC_DIR):
        print(f"[pmc_articles]    folder '{PMC_DIR}' not found — skipping.")
        print(f"                  Run scraping_pmc.py first to populate it.")
        return []

    json_files = [
        entry.name
        for entry in os.scandir(PMC_DIR)
        if entry.is_file() and entry.name.endswith(".json")
    ]
    if not json_files:
        print(f"[pmc_articles]    no JSON files found in '{PMC_DIR}' — skipping.")
        return []

    documents: list[Document] = []
    skipped = 0

    for filename in json_files:
        path = os.path.join(PMC_DIR, filename)
        data = _load_pmc_file(path)
        if data is None:
            skipped += 1
            continue

        content = data.get("publisher_content", "")
        cleaned = _clean(content)
        metadata = {
            "source":  "pmc_articles",
            "disease": data.get("disease", ""),
            "title":   data.get("title", ""),
            "url":     data.get("publisher_url", ""),
        }
        documents.extend(_chunk(cleaned, metadata))

    loaded = len(json_files) - skipped
    print(f"[pmc_articles]    files={len(json_files):,}  loaded={loaded:,}  skipped={skipped:,}  chunks={len(documents):,}")
    return documents


# ── Unified loader ────────────────────────────────────────────────────────────

def load_all_documents() -> list[Document]:
    """
    Run all four pipelines and return a single merged list of Documents.
    Each Document carries .page_content (clean chunk) and .metadata.
    """
    print("\n--- Preprocessing Pipeline ---")
    docs: list[Document] = []
    docs.extend(load_patient_doctor())
    docs.extend(load_medquad())
    docs.extend(load_mtsamples())
    docs.extend(load_pmc_articles())
    print(f"\nTotal chunks     : {len(docs):,}")
    print("------------------------------\n")
    return docs


# ── Vectorstore builder ───────────────────────────────────────────────────────

def build_vectorstore(documents: list[Document]):
    """
    Embed all documents with OpenAI text-embedding-3-small and persist
    to CHROMA_DIR. Calls set_vectorstore() so retrieval_agent is live.
    Returns the Chroma vectorstore instance.
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    import importlib
    retrieval_module = importlib.import_module("agents.retrieval_agent")
    set_vectorstore = retrieval_module.set_vectorstore

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"Embedding {len(documents):,} chunks into ChromaDB ...")
    print(f"Persisting to   : {os.path.abspath(CHROMA_DIR)}")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    set_vectorstore(vectorstore)
    print("Vectorstore ready and wired into retrieval_agent.\n")
    return vectorstore


def load_vectorstore():
    """
    Load an existing ChromaDB store from disk (no re-embedding).
    Call this at app startup after build_vectorstore() has been run once.
    Calls set_vectorstore() so retrieval_agent is live.
    Returns the Chroma vectorstore instance.
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    import importlib
    retrieval_module = importlib.import_module("agents.retrieval_agent")
    set_vectorstore = retrieval_module.set_vectorstore

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    set_vectorstore(vectorstore)
    print(f"Vectorstore loaded from '{CHROMA_DIR}' and wired into retrieval_agent.")
    return vectorstore


# ── Entry point: build the full vectorstore ───────────────────────────────────

if __name__ == "__main__":
    docs = load_all_documents()

    # Print one sample chunk from each source
    seen: set[str] = set()
    for doc in docs:
        src = doc.metadata.get("source", "")
        if src not in seen:
            seen.add(src)
            print(f"--- Sample [{src}] ---")
            print(f"Metadata : {doc.metadata}")
            print(f"Content  : {doc.page_content[:200]}\n")

    # Build + persist vectorstore
    build_vectorstore(docs)
