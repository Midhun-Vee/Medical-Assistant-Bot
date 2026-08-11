import os

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

tavily_search = None

if os.getenv("TAVILY_API_KEY"):
    tavily_search = TavilySearchResults(
        max_results=5
    )

# * fallback if tavily fails
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def web_search(question):

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

    if tavily_search is not None:
        results = tavily_search.invoke(search_query)

        context = "\n\n".join(
                    result.get("content", "")
                    for result in results
                    if isinstance(result, dict) and result.get("content")
                )
        
        return context, results

    else:
        print(f"\n\ntavily failed\ninvoked backup openai\n\n")

        response = openai_client.responses.create(
                    model="gpt-4o-mini",
                    tools=[
                        {
                            "type": "web_search_preview"
                        }
                    ],
            input=search_query
        )

        context = response.output_text

        sources = [
            {
                "source": "OpenAI Web Search",
                "content": context
            }
        ]

        return context, sources


def prescription_agent(state):

    question = state["question"]
    
    context, sources = web_search(question)

    if not context:

        state["prescription_info"] = (
            "No medication information could be retrieved. "
            "Please try again later"
        )

        state["prescription_sources"] = []

        return state

    prompt = f"""
You are a medication information assistant. Give clear, direct, and complete answers about medications.

WEB SEARCH RESULTS:

{context}

USER QUESTION:

{question}

First, check whether the question is specific enough to give safe, accurate medication guidance:
- If the medicine name is missing or ambiguous, ask the patient to clarify which medication they mean.
- If the question involves dosage but no patient context is given (e.g. age, weight, indication, renal function), note the standard adult dose and ask for any relevant details that would affect the recommendation.
- Do not ask more than 2 clarifying questions at once.

When you have sufficient information, provide the following directly and specifically:

1. Medicine name and generic name
2. What it is used for (indications)
3. Standard dosage — state the actual dose, frequency, and route as found in the sources
4. How it should be taken (with food, timing, etc.)
5. Common side effects
6. Important warnings and contraindications
7. Notable drug interactions
8. Sources

Dosage guidance:
- State the specific dose figures from the sources (e.g. "500 mg twice daily").
- When dosage differs by indication, age group, or renal/hepatic function, list each variant clearly.
- If the sources provide a dose range, give the full range.
- Only omit a dosage if the sources contain none at all.
"""

    response = llm.invoke(prompt)

    state["prescription_info"] = response.content
    state["prescription_sources"] = sources

    return state