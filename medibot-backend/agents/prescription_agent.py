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
                    if result.get("content")
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

    state["prescription_info"] = response.content
    state["prescription_sources"] = sources

    return state