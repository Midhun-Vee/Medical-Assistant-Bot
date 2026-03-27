from typing import Any, Dict, Optional, TypedDict

class HealthcareState(TypedDict, total=False):

    chat_id: str
    question: str
    language: str

    question_query: str

    route: str

    history: list

    documents: list

    prescription_info: str
    disease_analysis: str
    report_summary: str

    answer: str

    retrieval_relevant: bool
    retrieval_score: float