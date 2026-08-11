from langchain_openai import ChatOpenAI

from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationSummaryBufferMemory

from langchain_core.prompts import PromptTemplate


chains = {}


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ConversationChain only allows {history} and {input} in the prompt.
# Context is embedded directly into {input} at call time.
prompt = PromptTemplate.from_template(
    """
You are a knowledgeable healthcare assistant. Give clear, direct, and complete answers.

- Answer questions directly and specifically — do not hedge every statement with vague disclaimers.
- When context from specialists or documents is provided, use it to give precise answers.
- State findings, likely conditions, medication details, and clinical information plainly.
- Use plain language the patient can understand.
- Only flag uncertainty when the information genuinely is ambiguous or missing.
- If the question is too vague to answer safely and usefully (e.g. "I feel sick" with no other detail), ask up to 2 targeted follow-up questions — such as duration, severity, location, or relevant history — before or instead of giving a full answer. Frame them conversationally, not as a form.
- Never ask for clarification if enough detail is already present in the specialist information or conversation history.

Conversation history:

{history}

{input}

Answer:
"""
)


def get_chain(chat_id):

    if chat_id not in chains:

        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,
            memory_key="history",
            return_messages=False
        )

        chains[chat_id] = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt
        )

    return chains[chat_id]


def conversational_agent(state):

    print(f"\n\nconversational\n\n")

    chain = get_chain(
        state["chat_id"]
    )

    context = ""

    if state.get("documents"):

        context += "\nMedical knowledge:\n"

        context += "\n\n".join(
            doc.page_content
            for doc in state["documents"]
        )

    if state.get("prescription_info"):

        context += "\nPrescription information:\n"
        context += state["prescription_info"]

    if state.get("disease_analysis"):

        context += "\nDisease analysis:\n"
        context += state["disease_analysis"]

    if state.get("report_summary"):

        context += "\nReport summary:\n"
        context += state["report_summary"]

    # Embed context into input so ConversationChain sees only {history} + {input}
    if context:
        combined_input = (
            f"Specialist information:\n{context}\n\nUser question:\n{state['question']}"
        )
    else:
        combined_input = state["question"]

    response = chain.invoke({"input": combined_input})

    state["answer"] = (
        response.get("response")
        or "I'm sorry, I was unable to generate a response. Please try again."
    )

    return state