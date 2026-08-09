from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def disease_agent(state):

    context = "\n\n".join(
        doc.page_content
        for doc in state.get("documents", [])
    )

    prompt = f"""
You are a medical disease-analysis assistant.

Use the medical information below.

Medical information:

{context}

Patient question:

{state["question"]}

Analyze:
- symptoms mentioned
- potentially relevant conditions
- supporting information
- missing information

Do NOT make a definitive diagnosis.

Do NOT claim that the patient definitely has
a disease.

Explain the possibilities clearly.
"""

    response = llm.invoke(prompt)

    state["disease_analysis"] = response.content

    return state