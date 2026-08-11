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
You are a medical disease-analysis assistant. Give direct, specific answers based on the information available.

Medical information:

{context}

Patient question:

{state["question"]}

First, check whether the question contains enough detail to give a useful assessment:
- If critical information is missing (e.g. no symptoms described, no duration, no context), ask the patient up to 3 focused follow-up questions before attempting a diagnosis. List the questions clearly and explain briefly why each matters.
- If you have enough to work with, proceed with the analysis below.

When you have sufficient information, analyze and state clearly:
- Which symptoms the patient is describing
- The most likely condition(s) that match these symptoms, ranked by likelihood
- Key features that support each condition
- Any information that would change the assessment if present

Be direct and specific. State the most probable diagnosis plainly. If one condition is clearly the best fit, say so. Only express genuine uncertainty when the symptom picture is truly ambiguous.
"""

    response = llm.invoke(prompt)

    state["disease_analysis"] = response.content

    return state