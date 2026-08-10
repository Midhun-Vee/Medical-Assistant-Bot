import uuid

import streamlit as st

from dotenv import load_dotenv

# Load API Key before any agent/graph imports so that
# OPENAI_API_KEY is in os.environ when ChatOpenAI is instantiated.
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from agents.retrieval_agent import set_vectorstore
from graph import graph

CHROMA_DIR = "chroma_db"
EMBEDDING = OpenAIEmbeddings(model="text-embedding-3-small")

# Cache the Chroma instance in session_state so it is created only once,
# but always re-register it with the retrieval agent global on every rerun.
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=EMBEDDING
    )

vs = st.session_state["vectorstore"]
set_vectorstore(vs)

# Streamlit UI
st.title("LangGraph RAG")

pdf = st.file_uploader("Upload PDF (optional)", type="pdf")

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

    # Add new chunks into the persistent store
    vs.add_documents(chunks)

# User Question (always visible)
q = st.text_input("Question")

if "chat_id" not in st.session_state:
    st.session_state["chat_id"] = str(uuid.uuid4())

if q:

    result = graph.invoke(
        {
            "chat_id": st.session_state["chat_id"],
            "question": q,
            "documents": [],
            "answer": "",
        }
    )

    st.write("Rewritten Question:", result["question"])
    st.write("Retrieved Chunks:", len(result["documents"]))

    st.subheader("Answer")
    st.write(result["answer"])

    if result.get("judge_verdict"):
        st.subheader("Safety Verdict")
        st.write(result["judge_verdict"])
        if result.get("judge_block_reason"):
            st.write("Block reason:", result["judge_block_reason"])