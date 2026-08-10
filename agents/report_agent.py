from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def report_agent(state):

    report = state.get("report_text", "")

    if not report:
        state["report_summary"] = (
            "No report was provided. "
            "Please upload a medical report first."
        )
        return state

    prompt = f"""
You are a medical report summarization assistant.

Analyze the following medical report.

Report:

{report}

Provide:

1. Important findings
2. Abnormal values
3. Medical terminology explained simply
4. Overall summary
5. Items the patient may want to discuss
   with a healthcare professional

Do not invent information.
Do not make a definitive diagnosis.
"""

    response = llm.invoke(prompt)

    state["report_summary"] = response.content

    return state