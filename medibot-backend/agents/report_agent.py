import os

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


def _get_documents(state):
    """
    Get PDF/document chunks from the graph state.
    Expected format:
        state["documents"] = [
            Document(
                page_content="...",
                metadata={
                    "source": "...",
                    "file_type": "pdf",
                    "page": 1,
                }
            )
        ]
    """
    return state.get("documents", [])


def _build_report_context(documents, max_chars=30000):
    """
    Combine document chunks into a context window while
    preserving source/page metadata.
    """

    context_parts = []
    total_chars = 0

    for document in documents:
        if not isinstance(document, Document):
            continue

        text = document.page_content.strip()

        if not text:
            continue

        metadata = document.metadata or {}

        source = metadata.get("source", "unknown")
        page = metadata.get(
            "page",
            metadata.get("page_number", "unknown"),
        )

        chunk = f"""
Source: {source}
Page: {page}

{text}
"""

        remaining = max_chars - total_chars

        if remaining <= 0:
            break

        chunk = chunk[:remaining]

        context_parts.append(chunk)
        total_chars += len(chunk)

    return "\n\n".join(context_parts)


def _retrieve_relevant_chunks(question, documents, top_k=8):
    """
    Lightweight RAG retrieval using the existing document chunks.

    If your project already has a vector store/retriever, this
    function can be replaced with that retriever.
    """

    if not documents:
        return []

    # Simple keyword-based fallback retrieval.
    # This keeps the agent independent of a specific vector DB.
    question_words = {
        word.lower()
        for word in question.split()
        if len(word) > 3
    }

    scored_documents = []

    for document in documents:
        if not isinstance(document, Document):
            continue

        text = document.page_content.lower()

        score = sum(
            1
            for word in question_words
            if word in text
        )

        scored_documents.append(
            (score, document)
        )

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for score, document in scored_documents[:top_k]
    ]


def _summarize_report(question, documents):
    """
    Generate a structured summary from retrieved report chunks.
    """

    context = _build_report_context(
        documents,
        max_chars=30000,
    )

    if not context:
        return (
            "I could not extract readable text from "
            "the supplied PDF report."
        )

    prompt = f"""
You are a document analysis assistant.

The user has uploaded a PDF report and wants a summary.

USER REQUEST:
{question}

REPORT CONTENT:
{context}

Your task is to summarize ONLY the information contained
in the supplied report.

Important rules:

- Do not invent information.
- Do not add facts that are not present in the report.
- Do not make medical, legal, financial, or other professional
  conclusions beyond what the report states.
- Preserve important numbers, dates, measurements, findings,
  diagnoses, recommendations, and conclusions when present.
- If something is unclear or missing, say so.
- Distinguish between findings reported in the document and
  your own explanation.
- Keep the summary easy to understand.
- Mention the relevant page number when available.

Structure the response as:

## Report Summary

Brief overall summary.

## Key Findings

- Important finding 1
- Important finding 2
- Important finding 3

## Important Details

Include important measurements, values, dates, or observations.

## Conclusions

Summarize the report's stated conclusions.

## Recommendations

Summarize recommendations explicitly stated in the report.

## Areas That Need Attention

Mention abnormal, concerning, incomplete, or unclear information
if the report contains any.

## Source

Mention the PDF/report name and page numbers when available.
"""

    response = llm.invoke(prompt)

    return response.content


def report_agent(state):
    """
    RAG-based PDF report summarization agent.

    Expected state:
        {
            "question": "...",
            "question_query": "...",
            "documents": [...]
        }

    Updates:
        state["report_info"]
        state["report_sources"]
    """

    print("[REPORT]")

    question = state.get(
        "question_query",
        state.get("question", ""),
    )

    documents = _get_documents(state)

    print(
        f"[REPORT] question={question}"
    )

    print(
        f"[REPORT] documents={len(documents)}"
    )

    if not documents:
        state["report_info"] = (
            "No PDF report was supplied or no readable "
            "document content was found."
        )

        state["report_sources"] = []

        return state

    # Retrieve relevant chunks.
    relevant_documents = _retrieve_relevant_chunks(
        question,
        documents,
        top_k=8,
    )

    # If retrieval returns nothing useful, fall back to
    # the available document chunks.
    if not relevant_documents:
        relevant_documents = documents

    print(
        f"[REPORT] retrieved_chunks="
        f"{len(relevant_documents)}"
    )

    for document in relevant_documents:
        metadata = document.metadata or {}

        print(
            "[REPORT] chunk="
            f"source={metadata.get('source', 'unknown')} "
            f"page={metadata.get('page', 'unknown')}"
        )

    summary = _summarize_report(
        question,
        relevant_documents,
    )

    state["report_info"] = summary

    # Preserve useful source/page information for downstream nodes.
    sources = []

    for document in relevant_documents:
        metadata = document.metadata or {}

        sources.append(
            {
                "source": metadata.get(
                    "source",
                    "unknown",
                ),
                "page": metadata.get(
                    "page",
                    metadata.get(
                        "page_number",
                        "unknown",
                    ),
                ),
            }
        )

    state["report_sources"] = sources

    print("[REPORT] summary generated")

    return state