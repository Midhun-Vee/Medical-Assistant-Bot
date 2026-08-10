from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults


# --------------------------------------------------
# Internet search
# --------------------------------------------------

search = TavilySearchResults(
    max_results=5
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def prescription_agent(state):

    question = state["question"]

    # --------------------------------------------------
    # Search authoritative medication sources first
    # --------------------------------------------------

    search_query = f"""
    {question}

    Find official medication information including:
    - generic and brand name
    - indications
    - dosage and administration
    - frequency
    - contraindications
    - side effects
    - drug interactions

    Prefer authoritative sources such as:
    FDA
    DailyMed
    MedlinePlus
    official government health organizations
    """

    results = search.invoke(search_query)

    # --------------------------------------------------
    # Convert search results into context
    # --------------------------------------------------

    context = "\n\n".join(
        result["content"]
        for result in results
        if result.get("content")
    )

    if not context:
        state["prescription_info"] = (
            "No medication information could be retrieved. "
            "Please try again or consult a healthcare professional."
        )
        state["prescription_sources"] = []
        return state

    # --------------------------------------------------
    # Extract medication information
    # --------------------------------------------------

    prompt = f"""
You are a medication information assistant.

Answer the user's medication question using ONLY
the web information provided below.

WEB SEARCH RESULTS:

{context}

USER QUESTION:

{question}

Provide:

1. Medicine name
2. Generic name
3. What it is used for
4. Standard labeled dosage information
5. How often it is normally taken
6. How it should be taken
7. Common side effects
8. Important warnings
9. Important drug interactions
10. Source information

IMPORTANT DOSAGE RULES:

- Only provide a dosage if it is explicitly supported
  by the retrieved sources.
- Do NOT calculate or invent a dose.
- Do NOT recommend a personalized dose.
- If dosage depends on age, weight, kidney function,
  liver function, indication, pregnancy, or another
  clinical factor, explicitly state that.
- Distinguish adult and pediatric dosing.
- If multiple indications have different doses,
  clearly separate them.
- Never tell the patient to start, stop, increase,
  or decrease a prescription medicine.
- If the sources disagree, say so.
- Prefer official drug labeling over general websites.

The answer should be clear and patient-friendly.
"""

    response = llm.invoke(prompt)

    # --------------------------------------------------
    # Save results for the conversational agent
    # --------------------------------------------------

    state["prescription_info"] = response.content

    # Keep the sources as well
    state["prescription_sources"] = results

    return state