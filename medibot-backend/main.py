import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from agents.retrieval_agent import set_vectorstore
from graph import graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EVALUATION_STORE = []

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "medical_knowledge"

EMBEDDING = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


app = FastAPI(
    title="MediAssist Clinical AI API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=EMBEDDING,
)

set_vectorstore(vectorstore)


class ChatRequest(BaseModel):
    chat_id: str
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "documents": vectorstore._collection.count(),
    }


@app.post("/chat")
def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    result = graph.invoke(
        {
            "chat_id": request.chat_id,
            "question": request.question,
            "documents": [],
            "answer": "",
            "evaluation": None,
        }
    )

    evaluation = result.get(
        "evaluation"
    )

    evaluation_record = {
        "chat_id": request.chat_id,
        "question": request.question,
        "answer": result.get(
            "answer",
            "",
        ),
        "retrieved_chunks": len(
            result.get(
                "documents",
                [],
            )
        ),
        "evaluation": evaluation,
    }

    print(f"\n\n{evaluation_record}\n\n")

    EVALUATION_STORE.append(
        evaluation_record
    )

    return evaluation_record

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    upload_dir = os.path.join(
        BASE_DIR,
        "uploads",
    )

    os.makedirs(upload_dir, exist_ok=True)

    file_id = str(uuid.uuid4())

    file_path = os.path.join(
        upload_dir,
        f"{file_id}.pdf",
    )

    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    # Load PDF
    docs = PyPDFLoader(file_path).load()

    # Chunk
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    ).split_documents(docs)

    # Add to Chroma
    vectorstore.add_documents(chunks)

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "collection": COLLECTION_NAME,
        "total_documents": vectorstore._collection.count(),
    }

@app.get("/evaluations")
def get_evaluations():

    return {
        "evaluations": EVALUATION_STORE
    }