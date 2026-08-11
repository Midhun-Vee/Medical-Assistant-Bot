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
You are a medical report analysis assistant. Give direct, specific interpretations of the report.

Report:

{report}

Before interpreting, check whether the patient has asked a specific question about this report. If they have, answer that question directly as part of your response.

If the report is incomplete or key context is missing (e.g. patient age/sex needed to apply reference ranges, or the report type is unclear), ask up to 2 focused clarifying questions before or alongside your interpretation.

Provide:

1. Key findings — state what each result means plainly (e.g. "Your LDL cholesterol is elevated at 160 mg/dL; the normal upper limit is 100 mg/dL for most adults")
2. Abnormal values — list every out-of-range result, its actual value, the reference range, and what the deviation indicates
3. Plain-language explanation of any medical terms used
4. Overall clinical picture — summarize what the results collectively indicate about the patient's health status
5. Specific follow-up points — concrete questions or actions based on what was found in this report

Be direct. Interpret findings clearly rather than simply restating the numbers.
"""

    response = llm.invoke(prompt)

    state["report_summary"] = response.content

    return state