from langgraph.graph import StateGraph, END

from state import HealthcareState

from agents.supervisor_agent import supervisor_agent
from agents.retrieval_agent import retrieval_agent
from agents.prescription_agent import prescription_agent
from agents.disease_agent import disease_agent
from agents.report_agent import report_agent
from agents.conversational_agent import conversational_agent

def route_after_supervisor(
    state: HealthcareState,
):
    route = state.get("route")

    print(
        f"[ROUTER] supervisor route = {route}"
    )

    if route == "out_of_context":
        return "out_of_context"

    if route == "retrieval":
        return "retrieval"

    if route == "prescription":
        return "retrieval"

    if route == "disease":
        return "retrieval"

    if route == "report":
        return "report"

    return "out_of_context"


def route_after_retrieval(
    state: HealthcareState,
):
    retrieval_relevant = state.get(
        "retrieval_relevant",
        False,
    )

    print(
        f"[ROUTER] retrieval_relevant = "
        f"{retrieval_relevant}"
    )

    if not retrieval_relevant:
        return "no_context"

    route = state.get("route")

    if route == "prescription":
        return "prescription"

    if route == "disease":
        return "disease"

    return "conversation"


def no_context_agent(
    state: HealthcareState,
):
    print("[NO_CONTEXT]")

    state["answer"] = (
        "I couldn't find sufficiently relevant "
        "information in the available medical "
        "context to answer this question reliably."
    )

    return state


builder = StateGraph(
    HealthcareState
)

builder.add_node(
    "supervisor",
    supervisor_agent,
)

builder.add_node(
    "retrieval",
    retrieval_agent,
)

builder.add_node(
    "prescription",
    prescription_agent,
)

builder.add_node(
    "disease",
    disease_agent,
)

builder.add_node(
    "report",
    report_agent,
)

builder.add_node(
    "conversation",
    conversational_agent,
)

builder.add_node(
    "no_context",
    no_context_agent,
)

builder.set_entry_point(
    "supervisor"
)

builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {
        "out_of_context": END,
        "retrieval": "retrieval",
        "prescription": "prescription",
        "disease": "retrieval",
        "report": "report",
    },
)

builder.add_conditional_edges(
    "retrieval",
    route_after_retrieval,
    {
        "no_context": "no_context",
        "prescription": "prescription",
        "disease": "disease",
        "conversation": "conversation",
    },
)
builder.add_edge(
    "no_context",
    END,
)

builder.add_edge(
    "prescription",
    "conversation",
)

builder.add_edge(
    "disease",
    "conversation",
)

builder.add_edge(
    "report",
    "conversation",
)

builder.add_edge(
    "conversation",
    END,
)

graph = builder.compile()