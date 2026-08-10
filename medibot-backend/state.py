from typing import TypedDict


class HealthcareState(TypedDict, total=False):

    # --------------------------------
    # Chat
    # --------------------------------

    chat_id: str
    question: str

    # --------------------------------
    # Routing
    # --------------------------------

    route: str

    # --------------------------------
    # Uploaded documents
    # --------------------------------

    uploaded_files: list
    report_text: str

    # --------------------------------
    # RAG
    # --------------------------------

    documents: list

    # --------------------------------
    # Prescription
    # --------------------------------

    prescription_info: str
    prescription_sources: list

    # --------------------------------
    # Disease analysis
    # --------------------------------

    disease_analysis: str

    # --------------------------------
    # Report analysis
    # --------------------------------

    report_summary: str

    # --------------------------------
    # Final response
    # --------------------------------

    answer: str

    # --------------------------------
    # Judge
    # --------------------------------

    judge_verdict: str
    judge_block_reason: str
    ground_truth: str

    # --------------------------------
    # Sources
    # --------------------------------

    sources: list