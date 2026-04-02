import json

from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

def supervisor_agent(state):
    print("[SUPERVISOR]")

    question = state.get("question", "")
    history = state.get("history", [])

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history[-10:]
    )

    prompt = f"""
You are a medical AI supervisor.

Determine the route for the current question and rewrite
the question into a standalone question when it depends
on previous conversation.

Conversation history:
{history_text}

Current question:
{question}

Routes:

prescription:
medicines, prescriptions, dosage, medication uses,
side effects, interactions, treatment instructions

disease:
diseases, symptoms, conditions, diagnosis

report:
lab reports, medical reports, test results

retrieval:
general medical knowledge requiring retrieval

out_of_context:
non-medical questions

Important:
The current question may contain references such as:
"they", "them", "these medicines", "this medicine",
"it", "this report", "those", etc.

Use the conversation history to resolve these references.

Return JSON only:

{{
    "route": "prescription|disease|report|retrieval|out_of_context",
    "retrieval_question": "standalone medical question"
}}
"""

    response = llm.invoke(prompt)

    data = json.loads(response.content)

    state["route"] = data["route"]
    state["retrieval_question"] = data[
        "retrieval_question"
    ]

    print(
        f"[SUPERVISOR] route={state['route']}"
    )

    print(
        f"[SUPERVISOR] retrieval_question="
        f"{state['retrieval_question']}"
    )

    return state