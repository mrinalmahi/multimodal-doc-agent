import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from .agent import build_agent, extract_text
from .ingestion.pipeline import ingest_document

app = FastAPI(title="Multimodal Document Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class QuestionRequest(BaseModel):
    document_ids: list[str]
    document_labels: dict[str, str] = {}
    messages: list[Message]


class AnswerResponse(BaseModel):
    answer: str
    image_base64: str | None = None
    image_caption: str | None = None


class DocumentResponse(BaseModel):
    document_id: str


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    try:
        agent, used_images = build_agent(request.document_ids, request.document_labels)
        result = agent.invoke({
            "messages": [{"role": m.role, "content": m.content} for m in request.messages]
        })
        final_content = result["messages"][-1].content
        answer = extract_text(final_content)

        image_base64 = None
        image_caption = None
        if used_images:
            last_image = used_images[-1]
            with open(last_image["path"], "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            image_caption = last_image["caption"]

        return AnswerResponse(
            answer=answer,
            image_base64=image_base64,
            image_caption=image_caption
        )

    except Exception as e:
        error_message = str(e)
        if "RESOURCE_EXHAUSTED" in error_message or "429" in error_message:
            raise HTTPException(
                status_code=429,
                detail="The AI service is temporarily rate-limited. Please try again in a moment."
            )
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {error_message}"
        )


@app.post("/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = Path("data/pdfs")
    upload_dir.mkdir(parents=True, exist_ok=True)
    # .name strips any directory parts, e.g. "../../x.pdf" -> "x.pdf"
    save_path = upload_dir / Path(file.filename).name

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        doc_id = ingest_document(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return DocumentResponse(document_id=doc_id)


@app.get("/health")
def health_check():
    return {"status": "ok"}