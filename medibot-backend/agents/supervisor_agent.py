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

    prompt = f"""
You are the router for a healthcare chatbot.

Choose the most appropriate route.

Available routes:

retrieval
- general medical questions
- medical knowledge
- guidelines
- treatments

prescription
- medicines
- dosage
- side effects
- drug interactions

disease
- symptoms
- possible diseases
- disease analysis

report
- uploaded medical reports
- lab reports
- discharge summaries

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