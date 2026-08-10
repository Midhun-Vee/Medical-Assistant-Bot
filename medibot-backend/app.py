import os
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from agents.retrieval_agent import set_vectorstore

from agents.retrieval_agent import set_retriever
from graph import build_graph

# Load API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("LangGraph RAG")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:

    with open(pdf.name, "wb") as f:
        f.write(pdf.getbuffer())

    # Load PDF
    docs = PyPDFLoader(pdf.name).load()

    # Chunking
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    ).split_documents(docs)

    # Create Vector Store
    vs = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # Create Retriever
    set_vectorstore(vs)

    # Build LangGraph
    graph = build_graph()

    # User Question
    q = st.text_input("Question")

    if q:

        result = graph.invoke(
            {
                "question": q,
                "documents": [],
                "answer": "",
                "evaluation": {},
                "retry_count": 0
            }
        )

        st.write("Rewritten Question:", result["question"])
        st.write("Retrieved Chunks:", len(result["documents"]))

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Evaluation Scores")

        scores = result["evaluation"]

        st.write(f"Faithfulness : {scores['faithfulness']:.3f}")
        st.write(f"Answer Relevancy : {scores['answer_relevancy']:.3f}")
        st.write(f"Context Precision : {scores['context_precision']:.3f}")
        st.write(f"Context Recall : {scores['context_recall']:.3f}")
        st.write("Retries Performed:", result["retry_count"])

        st.write(
            "Chunks Retrieved:",
            len(result["documents"])
)