# from langchain_openai import ChatOpenAI
# from langchain_classic.chains import ConversationChain
# from langchain_classic.memory import ConversationSummaryBufferMemory
# from langchain_core.prompts import PromptTemplate

# chains = {}

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
# )

# prompt = PromptTemplate.from_template(
#     """
# You are a knowledgeable healthcare assistant. Give clear, direct, and complete answers.

# - Answer questions directly and specifically — do not hedge every statement with vague disclaimers.
# - When context from specialists or documents is provided, use it to give precise answers.
# - State findings, likely conditions, medication details, and clinical information plainly.
# - Use plain language the patient can understand.
# - Only flag uncertainty when the information genuinely is ambiguous or missing.
# - If the question is too vague to answer safely and usefully (e.g. "I feel sick" with no other detail), ask up to 2 targeted follow-up questions — such as duration, severity, location, or relevant history — before or instead of giving a full answer. Frame them conversationally, not as a form.
# - Never ask for clarification if enough detail is already present in the specialist information or conversation history.

# Conversation history:

# {history}

# Current information and user question:

# {input}

# Answer:
# """
# )


# def get_chain(chat_id):
#     if chat_id not in chains:
#         memory = ConversationSummaryBufferMemory(
#             llm=llm,
#             max_token_limit=1000,
#             memory_key="history",
#             return_messages=False,
#         )

#         chains[chat_id] = ConversationChain(
#             llm=llm,
#             memory=memory,
#             prompt=prompt,
#         )

#     return chains[chat_id]


# def conversational_agent(state):
#     print("\n\nconversational\n\n")

#     chain = get_chain(state["chat_id"])

#     context_parts = []

#     if state.get("documents"):
#         context_parts.append(
#             "Medical knowledge:\n"
#             + "\n\n".join(
#                 doc.page_content
#                 for doc in state["documents"]
#             )
#         )

#     if state.get("prescription_info"):
#         context_parts.append(
#             "Prescription information:\n"
#             + state["prescription_info"]
#         )

#     if state.get("disease_analysis"):
#         context_parts.append(
#             "Disease analysis:\n"
#             + state["disease_analysis"]
#         )

#     if state.get("report_summary"):
#         context_parts.append(
#             "Report summary:\n"
#             + state["report_summary"]
#         )

#     context = "\n\n".join(context_parts)

#     if context:
#         combined_input = (
#             f"Specialist information:\n\n"
#             f"{context}\n\n"
#             f"User question:\n"
#             f"{state['question']}\n\n"
#             f"Answer in {state['language']} language."
#         )
#     else:
#         combined_input = (
#             f"User question:\n"
#             f"{state['question']}\n\n"
#             f"Answer in {state['language']} language."
#         )

#     response = chain.invoke(
#         {
#             "input": combined_input,
#         }
#     )

#     state["answer"] = (
#         response.get("response")
#         or "I'm sorry, I was unable to generate a response. Please try again."
#     )

#     return state

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=500,
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a healthcare assistant.

Answer the user's question using the supplied medical context.

Rules:
- Be direct and concise.
- Use plain language.
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information, say so.
- Do not repeat the context unnecessarily.
- Answer in the requested language.

Medical context:
{context}
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


chain = prompt | llm


def conversational_agent(state):

    print("[CONVERSATION]")

    context_parts = []

    documents = state.get("documents", [])

    if documents:
        medical_context = "\n\n".join(
            doc.page_content[:2500]
            for doc in documents
        )

        context_parts.append(
            f"Medical knowledge:\n{medical_context}"
        )

    if state.get("prescription_info"):
        context_parts.append(
            "Prescription information:\n"
            + state["prescription_info"][:3000]
        )

    if state.get("disease_analysis"):
        context_parts.append(
            "Disease analysis:\n"
            + state["disease_analysis"][:3000]
        )

    if state.get("report_summary"):
        context_parts.append(
            "Report summary:\n"
            + state["report_summary"][:3000]
        )

    context = "\n\n".join(context_parts)

    response = chain.invoke(
        {
            "context": context,
            "question": (
                f"{state['question']}\n\n"
                f"Respond in {state['language']}."
            ),
        }
    )

    state["answer"] = response.content

    return state