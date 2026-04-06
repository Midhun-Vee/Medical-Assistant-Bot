import os

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

tavily_search = None

if os.getenv("TAVILY_API_KEY"):
    tavily_search = TavilySearchResults(max_results=5)

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
        try:
            results = tavily_search.invoke(search_query)

            context = "\n\n".join(
                result.get("content", "")
                for result in results
                if isinstance(result, dict)
                and result.get("content")
            )

            return context, results

        except Exception as e:
            print(
                f"[PRESCRIPTION] Tavily failed: "
                f"{type(e).__name__}: {e}"
            )

    print(
        "[PRESCRIPTION] Tavily unavailable/failed; "
        "using OpenAI web search"
    )

    try:
        response = openai_client.responses.create(
            model="gpt-4o-mini",
            tools=[
                {
                    "type": "web_search_preview"
                }
            ],
            input=search_query,
        )

        context = response.output_text

        sources = [
            {
                "source": "OpenAI Web Search",
                "content": context,
            }
        ]

        return context, sources

    except Exception as e:
        print(
            f"[PRESCRIPTION] Web search failed: "
            f"{type(e).__name__}: {e}"
        )

        return "", []


def prescription_agent(state):
    print("[PRESCRIPTION]")

    question = state.get(
        "question_query",
        state.get("question", ""),
    )

    documents = state.get(
        "documents",
        [],
    )

    print(f"[PRESCRIPTION] question={question}")
    print(f"[PRESCRIPTION] documents={len(documents)}")

    attachment_context_parts = []

    for document in documents:
        source = document.metadata.get(
            "source",
            "unknown",
        )

        file_type = document.metadata.get(
            "file_type",
            "unknown",
        )

        text = document.page_content.strip()

        if not text:
            continue

        print(
            f"[PRESCRIPTION] attachment="
            f"{source} type={file_type}"
        )

        print(
            f"[PRESCRIPTION] OCR="
            f"{text[:2000]}"
        )

        attachment_context_parts.append(
            f"""
Attachment source: {source}
Attachment type: {file_type}

OCR text:
{text[:5000]}
"""
        )

    attachment_context = "\n\n".join(
        attachment_context_parts
    )

    if attachment_context:
        identification_prompt = f"""
You are analyzing a medical prescription or medication image.

The OCR text may contain spelling errors or incorrect
characters.

User question:
{question}

Prescription OCR:
{attachment_context}

Task:

Identify the specific medicine names that can be
reliably identified from the prescription.

Rules:

- Use ONLY the supplied prescription OCR.
- Do not invent medicine names.
- Do not silently correct an unclear medicine name.
- If a medicine name is reasonably clear, provide it.
- If a medicine name is ambiguous, mark it as unclear.
- Do not infer a medicine merely because it resembles
  a known drug.
- Do not invent dosage or frequency.
- Separate clearly identified medicines from uncertain text.

Return:

Clearly identified medicines:
- medicine 1
- medicine 2

Unclear/uncertain medicine names:
- ...

If no medicine can be reliably identified, say:
"No medicine names can be reliably identified from
the supplied prescription text."
"""

        identification_response = llm.invoke(
            identification_prompt
        )

        prescription_identification = (
            identification_response.content
        )

        print("[PRESCRIPTION] identification=")
        print(prescription_identification)

        state["prescription_info"] = (
            prescription_identification
        )

        lower_question = question.lower()

        needs_medical_details = any(
            keyword in lower_question
            for keyword in [
                "side effect",
                "side effects",
                "dosage",
                "dose",
                "used for",
                "use",
                "indication",
                "interaction",
                "interactions",
                "contraindication",
                "how to take",
            ]
        )

        if needs_medical_details:
            web_question = f"""
{question}

The following medicines were identified from
the uploaded prescription:

{prescription_identification}

Provide information only for medicines that are
clearly identified.
"""

            web_context, sources = web_search(
                web_question
            )

            if web_context:
                final_prompt = f"""
You are a healthcare assistant.

Answer the user's question using the supplied
prescription information and authoritative
medication information.

USER QUESTION:
{question}

MEDICINES IDENTIFIED FROM PRESCRIPTION:
{prescription_identification}

ADDITIONAL MEDICATION INFORMATION:
{web_context}

Rules:

- Do not invent medicine names.
- Do not guess unclear OCR.
- Clearly distinguish prescription information
  from general medication information.
- Do not claim that a medication is present if
  the prescription OCR does not reliably identify it.
- If a medicine name is uncertain, explicitly say so.
- Do not provide personalized prescribing decisions.
- If dosage information is requested, distinguish
  standard reference dosing from the patient's
  actual prescription.
- Be concise and direct.

Answer the user's question.
"""

                response = llm.invoke(final_prompt)

                state["prescription_info"] = (
                    response.content
                )

                state["prescription_sources"] = sources
            else:
                state["prescription_sources"] = []
        else:
            state["prescription_sources"] = []

        return state

    print("[PRESCRIPTION] no attachment found")

    context, sources = web_search(question)

    if not context:
        state["prescription_info"] = (
            "No medication information could "
            "be retrieved. Please try again later."
        )

        state["prescription_sources"] = []

        return state

    prompt = f"""
You are a medication information assistant.

USER QUESTION:
{question}

WEB SEARCH RESULTS:
{context}

Rules:

- Do not invent medication names.
- If the medicine name is ambiguous, ask for clarification.
- Give concise, evidence-based information.
- Distinguish standard reference dosage from
  individualized prescribing.
- Do not provide a personalized prescription.

When sufficient information is available, provide:

1. Medicine name and generic name
2. What it is used for
3. Standard dosage
4. How it is taken
5. Common side effects
6. Important warnings
7. Notable interactions
8. Sources
"""

    response = llm.invoke(prompt)

    state["prescription_info"] = response.content
    state["prescription_sources"] = sources

    return state