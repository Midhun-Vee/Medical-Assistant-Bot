from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class Route(BaseModel):
    route: Literal["retrieval", "prescription", "disease", "report"]


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
).with_structured_output(Route)


def supervisor_agent(state):

    print(f"\n\nsupervising agent\n\n")

    # Build an optional context hint so the supervisor knows what
    # material has already been provided alongside the question.
    context_hints = []

    if state.get("report_text"):
        context_hints.append(
            "The user has uploaded a medical report / lab results PDF."
        )

    if state.get("prescription_info"):
        context_hints.append(
            "The user has uploaded a prescription or medication PDF."
        )

    if state.get("documents"):
        context_hints.append(
            "A PDF has been loaded into the semantic search index."
        )

    context_section = (
        "\nAdditional context:\n" + "\n".join(f"- {h}" for h in context_hints)
        if context_hints
        else ""
    )

    prompt = f"""
You are the router for a healthcare chatbot.

Choose the most appropriate route based on the user's question
and the additional context below (if any).

Available routes:

retrieval
- general medical questions
- medical knowledge
- guidelines
- treatments
- questions about a PDF loaded for semantic search

prescription
- medicines
- dosage
- side effects
- drug interactions
- questions about an uploaded prescription PDF

disease
- symptoms
- possible diseases
- disease analysis

report
- uploaded medical reports
- lab reports
- discharge summaries
- questions about an uploaded medical report PDF
{context_section}

Question:

{state["question"]}
"""

    response = llm.invoke(prompt)

    state["route"] = (
        response.route
        if isinstance(response, Route)
        else response["route"]
    )

    return state