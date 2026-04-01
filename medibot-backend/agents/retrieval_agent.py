from langchain_openai import OpenAIEmbeddings

TOP_K = 3

EMBEDDING = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = None


def set_vectorstore(vs):
    global vectorstore
    vectorstore = vs


def retrieval_agent(state):
    print("[RETRIEVAL]")

    question = state.get(
        "retrieval_question",
        state.get("question", ""),
    )

    attachment_documents = state.get(
        "documents",
        [],
    )

    print(
        f"[RETRIEVAL] question={question}"
    )

    if attachment_documents:
        print(
            f"[RETRIEVAL] using "
            f"{attachment_documents} attachment documents"
        )

        texts = [
            document.page_content
            for document in attachment_documents
            if document.page_content.strip()
        ]

        if texts:
            query_embedding = EMBEDDING.embed_query(
                question
            )

            document_embeddings = (
                EMBEDDING.embed_documents(texts)
            )

            scored = []

            for document, embedding in zip(
                attachment_documents,
                document_embeddings,
            ):
                distance = sum(
                    (a - b) ** 2
                    for a, b in zip(
                        query_embedding,
                        embedding,
                    )
                ) ** 0.5

                scored.append(
                    (
                        document,
                        distance,
                    )
                )

            scored.sort(
                key=lambda x: x[1]
            )

            results = scored[:TOP_K]

            print(
                f"[RETRIEVAL] attachment results="
                f"{len(results)}"
            )

            for document, score in results:
                print(
                    f"[RETRIEVAL] "
                    f"attachment score={score:.4f} "
                    f"source={document.metadata.get('source')} "
                    f"type={document.metadata.get('file_type')}"
                )

            state["documents"] = [
                document
                for document, _ in results
            ]

            state["retrieval_relevant"] = True
            state["retrieval_score"] = results[0][1]

            return state

    if vectorstore is None:
        print(
            "[RETRIEVAL] vectorstore is not initialized"
        )

        state["documents"] = []
        state["retrieval_relevant"] = False
        state["retrieval_score"] = None

        return state

    results = vectorstore.similarity_search_with_score(
        question,
        k=TOP_K,
    )

    if not results:
        print(
            "[RETRIEVAL] no results"
        )

        state["documents"] = []
        state["retrieval_relevant"] = False
        state["retrieval_score"] = None

        return state

    best_score = results[0][1]

    state["retrieval_score"] = best_score

    print(
        f"[RETRIEVAL] best_score={best_score:.4f}"
    )

    for document, score in results:
        print(
            f"[RETRIEVAL] "
            f"score={score:.4f} "
            f"source={document.metadata.get('source')} "
            f"type={document.metadata.get('file_type')}"
        )

    documents = [
        document
        for document, _ in results
    ]

    state["documents"] = documents
    state["retrieval_relevant"] = True

    print(
        f"[RETRIEVAL] accepted: "
        f"{len(documents)} chunks"
    )

    return state