import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pymupdf
import easyocr

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from agents.retrieval_agent import set_vectorstore
from agents.evaluation_agent import evaluation_agent
from graph import graph

from sarvam_api import sarvam_call
from guardrails import check_input


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EVALUATION_STORE = []

ATTACHMENT_STORE = {}

CHAT_HISTORY = {}

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db",
)

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

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=False,
)


class ChatRequest(BaseModel):
    chat_id: str
    question: str


def get_chat_history(chat_id):
    return CHAT_HISTORY.get(
        chat_id,
        [],
    )


def add_chat_message(
    chat_id,
    role,
    content,
):
    if chat_id not in CHAT_HISTORY:
        CHAT_HISTORY[chat_id] = []

    CHAT_HISTORY[chat_id].append(
        {
            "role": role,
            "content": content,
        }
    )


def run_evaluation_background(
    chat_id,
    question,
    answer,
    documents,
):
    state = {
        "chat_id": chat_id,
        "question": question,
        "answer": answer,
        "documents": documents,
    }

    try:
        result = evaluation_agent(state)

        evaluation = result.get(
            "evaluation"
        )

        EVALUATION_STORE.append(
            {
                "chat_id": chat_id,
                "question": question,
                "answer": answer,
                "retrieved_chunks": len(
                    documents
                ),
                "evaluation": evaluation,
            }
        )

        print(
            f"[EVALUATION] completed "
            f"for chat_id={chat_id}"
        )

    except Exception as e:
        print(
            f"[EVALUATION ERROR] "
            f"{type(e).__name__}: {e}"
        )


def extract_attachment(
    file_path,
    filename,
    content_type,
    chat_id,
):
    documents = []

    if content_type == "application/pdf":
        pdf = pymupdf.open(file_path)

        try:
            for page_number, page in enumerate(pdf):
                text = page.get_text(
                    "text"
                ).strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "file_type": "pdf",
                            "page": page_number + 1,
                            "chat_id": chat_id,
                            "attachment": True,
                        },
                    )
                )
        finally:
            pdf.close()

    else:
        results = ocr_reader.readtext(
            file_path,
            detail=1,
        )

        text_parts = []

        for result in results:
            if len(result) >= 2:
                text_parts.append(
                    result[1]
                )

        text = "\n".join(
            text_parts
        ).strip()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "file_type": "image",
                        "chat_id": chat_id,
                        "attachment": True,
                    },
                )
            )

    return documents


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "documents": (
            vectorstore
            ._collection
            .count()
        ),
        "active_chats": len(
            CHAT_HISTORY
        ),
    }


@app.get("/chat/{chat_id}/history")
def get_history(chat_id: str):
    return {
        "chat_id": chat_id,
        "history": get_chat_history(
            chat_id
        ),
    }


@app.delete("/chat/{chat_id}/history")
def clear_history(chat_id: str):
    CHAT_HISTORY.pop(
        chat_id,
        None,
    )

    return {
        "chat_id": chat_id,
        "message": "Chat history cleared.",
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    chat_id: str = Form(...),
):
    allowed_types = {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF, JPG, JPEG, PNG, "
                "and WEBP files are supported."
            ),
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    suffix = os.path.splitext(
        file.filename or ""
    )[1].lower()

    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        delete=False,
    )

    temp_file.write(contents)
    temp_file.close()

    if chat_id not in ATTACHMENT_STORE:
        ATTACHMENT_STORE[chat_id] = []

    ATTACHMENT_STORE[chat_id].append(
        {
            "path": temp_file.name,
            "filename": file.filename,
            "content_type": file.content_type,
        }
    )

    print(
        f"[UPLOAD] stored temporary file "
        f"{file.filename}"
    )

    print(
        f"[UPLOAD] chat_id={chat_id}"
    )

    return {
        "filename": file.filename,
        "file_type": file.content_type,
        "chat_id": chat_id,
        "status": "stored",
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    chat_id = request.chat_id
    original_question = request.question

    history = get_chat_history(chat_id)

    print(
        f"[CHAT] chat_id={chat_id}"
    )

    print(
        f"[CHAT] history messages="
        f"{len(history)}"
    )

    question, language = sarvam_call(
        original_question
    )

    guardrail_response = check_input(
        question
    )

    if guardrail_response["blocked"]:
        reason = guardrail_response.get(
            "reason",
            "Please enter a valid medical question.",
        )

        return {
            "blocked": True,
            "message": reason,
            "answer": "",
            "evaluation": None,
        }

    attachment_documents = []

    attachments = ATTACHMENT_STORE.pop(
        chat_id,
        [],
    )

    try:
        for attachment in attachments:
            documents = extract_attachment(
                attachment["path"],
                attachment["filename"],
                attachment["content_type"],
                chat_id,
            )

            attachment_documents.extend(
                documents
            )

            print(
                f"[ATTACHMENT] "
                f"{attachment['filename']} "
                f"extracted "
                f"{len(documents)} documents"
            )

            for document in documents:
                print(
                    f"[ATTACHMENT TEXT] "
                    f"{document.page_content[:2000]}"
                )

        result = graph.invoke(
            {
                "chat_id": chat_id,
                "question": guardrail_response[
                    "query"
                ],
                "history": history,
                "documents": attachment_documents,
                "answer": "",
                "evaluation": None,
                "language": language,
                "has_attachment": bool(
                    attachment_documents
                ),
            }
        )

        answer = result.get(
            "answer",
            "",
        )

        documents = result.get(
            "documents",
            [],
        )

        add_chat_message(
            chat_id,
            "user",
            original_question,
        )

        add_chat_message(
            chat_id,
            "assistant",
            answer,
        )

        background_tasks.add_task(
            run_evaluation_background,
            chat_id,
            original_question,
            answer,
            documents,
        )

        return {
            "chat_id": chat_id,
            "question": original_question,
            "answer": answer,
            "retrieved_chunks": len(
                documents
            ),
        }

    finally:
        for attachment in attachments:
            path = attachment["path"]

            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.get("/evaluations")
def get_evaluations():
    return {
        "evaluations": EVALUATION_STORE,
    }