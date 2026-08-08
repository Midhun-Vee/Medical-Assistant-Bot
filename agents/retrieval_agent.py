TOP_K = 10

vectorstore = None


def set_vectorstore(vs):
    global vectorstore
    vectorstore = vs


def retrieval_agent(state):

    if vectorstore is None:
        state["documents"] = []
        return state

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    documents = retriever.invoke(
        state["question"]
    )

    state["documents"] = documents

    return state