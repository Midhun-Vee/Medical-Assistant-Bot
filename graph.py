from langgraph.graph import StateGraph, END

from state import HealthcareState

from agents.supervisor_agent import supervisor_agent
from agents.retrieval_agent import retrieval_agent
from agents.prescription_agent import prescription_agent
from agents.disease_agent import disease_agent
from agents.report_agent import report_agent
from agents.conversational_agent import conversational_agent
from agents.judge_agent import judge_agent


# --------------------------------------------------
# Routing
# --------------------------------------------------

def route_agent(state):

    route = state["route"]

    if route == "retrieval":
        return "retrieval"

    elif route == "prescription":
        return "prescription"

    elif route == "disease":
        return "disease"

    elif route == "report":
        return "report"

    return "retrieval"


def route_after_retrieval(state):
    """After retrieval, go to disease_agent if that was the intended route,
    otherwise go straight to conversation."""
    if state.get("route") == "disease":
        return "disease"
    return "conversation"


# --------------------------------------------------
# Build graph
# --------------------------------------------------

builder = StateGraph(HealthcareState)


# --------------------------------------------------
# Add agents
# --------------------------------------------------

builder.add_node(
    "supervisor",
    supervisor_agent
)

builder.add_node(
    "retrieval",
    retrieval_agent
)

builder.add_node(
    "prescription",
    prescription_agent
)

builder.add_node(
    "disease",
    disease_agent
)

builder.add_node(
    "report",
    report_agent
)

builder.add_node(
    "conversation",
    conversational_agent
)

builder.add_node(
    "judge",
    judge_agent
)


# --------------------------------------------------
# Starting point
# --------------------------------------------------

builder.set_entry_point(
    "supervisor"
)


# --------------------------------------------------
# Supervisor routing
# --------------------------------------------------

builder.add_conditional_edges(
    "supervisor",

    route_agent,

    {
        "retrieval": "retrieval",
        "prescription": "prescription",
        "disease": "retrieval",  # disease fetches RAG context first
        "report": "report"
    }
)


# --------------------------------------------------
# Specialist agents → conversation
# --------------------------------------------------

builder.add_conditional_edges(
    "retrieval",
    route_after_retrieval,
    {
        "disease": "disease",
        "conversation": "conversation"
    }
)

builder.add_edge(
    "prescription",
    "conversation"
)

builder.add_edge(
    "disease",
    "conversation"
)

builder.add_edge(
    "report",
    "conversation"
)


# --------------------------------------------------
# Conversation → judge
# --------------------------------------------------

builder.add_edge(
    "conversation",
    "judge"
)


# --------------------------------------------------
# Judge routing
# --------------------------------------------------

def route_judge(state):
    if state.get("judge_verdict") == "block":
        return "block"
    return "end"


builder.add_conditional_edges(
    "judge",
    route_judge,
    {
        "end": END,
        "block": END
    }
)


# --------------------------------------------------
# Compile
# --------------------------------------------------

graph = builder.compile()