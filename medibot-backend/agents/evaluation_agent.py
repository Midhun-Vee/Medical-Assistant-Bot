import asyncio
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory

from ragas.metrics.collections import (
    Faithfulness,
    ContextRecall,
    ContextPrecision,
    AnswerCorrectness,
    AnswerRelevancy,
)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()



# Evaluation models

EVALUATION_MODEL = os.getenv(
    "EVALUATION_MODEL",
    "gpt-4o-mini",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)



async_openai_client = AsyncOpenAI()
# sync_openai_client = OpenAI()


evaluator_llm = llm_factory(
    EVALUATION_MODEL,
    client=async_openai_client,
)

evaluator_embeddings = embedding_factory(
    "openai",
    model=EMBEDDING_MODEL,
    client=async_openai_client,
)

ground_truth_llm = ChatOpenAI(
    model=EVALUATION_MODEL,
    temperature=0,
)


def generate_ground_truth(
    question: str,
    contexts: List[str],
) -> str:

    limited_contexts = contexts[:2]

    context_text = "\n\n".join(
        f"[Context {i + 1}]\n{context[:2000]}"
        for i, context in enumerate(limited_contexts)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are evaluating a medical RAG system.

Create a SHORT reference answer using ONLY
the supplied context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- Keep the answer to 1-3 sentences.
- Include only facts necessary to answer the question.
- If the context cannot answer the question,
  return: "Insufficient information in the retrieved context."
""",
            ),
            (
                "human",
                """
Question:
{question}

Context:
{context}

Reference answer:
""",
            ),
        ]
    )

    chain = prompt | ground_truth_llm

    result = chain.invoke(
        {
            "question": question,
            "context": context_text,
        }
    )

    return result.content.strip()

async def _evaluate(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, Any]:

    faithfulness_metric = Faithfulness(
        llm=evaluator_llm,
    )

    context_recall_metric = ContextRecall(
        llm=evaluator_llm,
    )

    context_precision_metric = ContextPrecision(
        llm=evaluator_llm,
    )

    answer_correctness_metric = AnswerCorrectness(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    answer_relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    
    faithfulness = await faithfulness_metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    context_recall = await context_recall_metric.ascore(
        user_input=question,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )

    context_precision = await context_precision_metric.ascore(
        user_input=question,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )

    
    answer_correctness = await answer_correctness_metric.ascore(
        user_input=question,
        response=answer,
        reference=ground_truth,
    )

    
    answer_relevancy = await answer_relevancy_metric.ascore(
        user_input=question,
        response=answer,
    )

    
    return {
        "faithfulness": float(
            faithfulness.value
        ),

        "context_recall": float(
            context_recall.value
        ),

        "context_precision": float(
            context_precision.value
        ),

        "answer_correctness": float(
            answer_correctness.value
        ),

        "answer_relevancy": float(
            answer_relevancy.value
        ),
    }


def evaluation_agent(
    state: Dict[str, Any],
) -> Dict[str, Any]:

    question = state.get(
        "question",
        "",
    )

    answer = state.get(
        "answer",
        "",
    )

    documents = state.get(
        "documents",
        [],
    )

    
    if not documents:

        return {
            "evaluation": {
                "status": "no_data",
                "message": (
                    "No documents were retrieved. "
                    "RAG evaluation was skipped."
                ),
                "faithfulness": None,
                "context_recall": None,
                "context_precision": None,
                "answer_correctness": None,
                "answer_relevancy": None,
                "ground_truth": None,
            }
        }

    
    contexts = []

    for document in documents:

        if hasattr(document, "page_content"):
            contexts.append(
                document.page_content
            )

        elif isinstance(document, str):
            contexts.append(document)

        elif isinstance(document, dict):
            contexts.append(
                document.get(
                    "page_content",
                    str(document),
                )
            )

        else:
            contexts.append(str(document))

    contexts = [
        context.strip()
        for context in contexts
        if context and context.strip()
    ]

    if not contexts:

        return {
            "evaluation": {
                "status": "no_data",
                "message": (
                    "Retrieved documents contained "
                    "no usable text."
                ),
                "faithfulness": None,
                "context_recall": None,
                "context_precision": None,
                "answer_correctness": None,
                "answer_relevancy": None,
                "ground_truth": None,
            }
        }

    
    ground_truth = generate_ground_truth(
        question=question,
        contexts=contexts,
    )

    scores = asyncio.run(
        _evaluate(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
    )

    return {
        "evaluation": {
            "status": "evaluated",
            "ground_truth": ground_truth,
            "contexts_used": len(contexts),
            **scores,
        }
    }