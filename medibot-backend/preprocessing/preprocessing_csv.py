import os
import re

from dotenv import load_dotenv

import chromadb
import pandas as pd

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")

CSV_DIR = "preprocessing/datasets/csv_datasets/preprocessed_datasets"
CHROMA_DIR = "chroma_db"

PATIENT_DOCTOR_FILE = os.path.join(CSV_DIR, "dataset1.csv")
MEDQUAD_FILE = os.path.join(CSV_DIR, "dataset2.csv")
MTSAMPLES_FILE = os.path.join(CSV_DIR, "dataset3.csv")

EMBEDDING_MODEL = "text-embedding-3-small"
SEMANTIC_CHUNK_THRESHOLD = 1500
MIN_CHUNK_SIZE = 100

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

semantic_chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,
    min_chunk_size=MIN_CHUNK_SIZE
)

documents = []


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r"\[\d+(?:[,\s]+\d+)*\]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 3]

    return "\n".join(lines).strip()


def create_documents(text, metadata):
    cleaned = clean_text(text)

    if not cleaned:
        return []

    if len(cleaned) <= SEMANTIC_CHUNK_THRESHOLD:
        return [
            Document(
                page_content=cleaned,
                metadata={
                    **metadata,
                    "chunk_index": 0
                }
            )
        ]

    chunks = semantic_chunker.split_text(cleaned)

    temp_documents = []

    for index, chunk in enumerate(chunks):
        chunk = chunk.strip()

        if not chunk:
            continue

        temp_documents.append(
            Document(
                page_content=chunk,
                metadata={
                    **metadata,
                    "chunk_index": index
                }
            )
        )

    return temp_documents


def load_patient_doctor():
    print("\nLoading Patient-Doctor Conversations...")

    df = pd.read_csv(
        PATIENT_DOCTOR_FILE,
        on_bad_lines="skip"
    )

    starting_count = len(documents)

    for _, row in df.iterrows():
        text = (
            f"Patient: {row['patient_input']}\n"
            f"Doctor: {row['doctor_response']}"
        )

        metadata = {
            "source": "patient_doctor_conversations",
            "condition": str(row["condition"]),
            "symptom": str(row["symptom"]),
            "category": str(row["category"])
        }

        documents.extend(create_documents(text, metadata))

    added_documents = len(documents) - starting_count

    print(
        f"[patient_doctor] rows={len(df):,} "
        f"documents_added={added_documents:,}"
    )


def load_medquad():
    print("\nLoading MedQuAD...")

    df = pd.read_csv(
        MEDQUAD_FILE,
        on_bad_lines="skip"
    )

    starting_count = len(documents)

    for _, row in df.iterrows():
        text = (
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}"
        )

        metadata = {
            "source": "medquad",
            "focus_area": str(row["focus_area"]),
            "origin": str(row["source"])
        }

        documents.extend(create_documents(text, metadata))

    added_documents = len(documents) - starting_count

    print(
        f"[medquad] rows={len(df):,} "
        f"documents_added={added_documents:,}"
    )


def load_mtsamples():
    print("\nLoading MTSamples...")

    df = pd.read_csv(
        MTSAMPLES_FILE,
        on_bad_lines="skip"
    )

    starting_count = len(documents)

    for _, row in df.iterrows():
        text = (
            f"Case: {str(row['description']).strip()}\n\n"
            f"Specialty: {str(row['medical_specialty']).strip()}\n\n"
            f"Transcription:\n"
            f"{row['transcription']}"
        )

        metadata = {
            "source": "mtsamples_transcriptions",
            "specialty": str(row["medical_specialty"]).strip(),
            "case_name": str(row["sample_name"]).strip()
        }

        documents.extend(create_documents(text, metadata))

    added_documents = len(documents) - starting_count

    print(
        f"[mtsamples] rows={len(df):,} "
        f"documents_added={added_documents:,}"
    )


def save_to_chromadb():
    if not documents:
        raise ValueError(
            "No documents were loaded. Cannot create ChromaDB."
        )

    os.makedirs(CHROMA_DIR, exist_ok=True)

    chroma_path = os.path.abspath(CHROMA_DIR)

    print(f"\nChromaDB path: {chroma_path}")
    print(f"Embedding {len(documents):,} documents...")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    client = chromadb.PersistentClient(path=chroma_path)

    collection_name = "medical_knowledge"

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=client,
        collection_name=collection_name
    )

    print("\nSuccessfully created ChromaDB.")


if __name__ == "__main__":
    print("\nLoading and converting to chunks\n")

    load_patient_doctor()
    # load_medquad()
    # load_mtsamples()

    print(f"Total documents: {len(documents):,}")

    save_to_chromadb()