

from pathlib import Path
import hashlib
import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from pdf_reader import read_pdf_documents
from text_splitter import split_pdf_into_documents


BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "papers"
STATIC_DIR = BASE_DIR / "static"

# Use a NEW database because the old one contains Ollama embeddings.
DATABASE_PATH = str(BASE_DIR / "chroma_db_gemini")

CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
MAX_FILE_SIZE = 25 * 1024 * 1024

PAPERS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="ResearchPilot AI", version="1.1.0")


class GeminiEmbeddings(Embeddings):
    """Gemini embeddings compatible with LangChain and Chroma."""

    def _embed(self, text, task_type):
        if client is None:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return result.embeddings[0].values

    def embed_documents(self, texts):
        return [
            self._embed(text, "RETRIEVAL_DOCUMENT")
            for text in texts
        ]

    def embed_query(self, text):
        return self._embed(text, "RETRIEVAL_QUERY")


embeddings = GeminiEmbeddings()

vector_store = Chroma(
    collection_name="research_papers",
    embedding_function=embeddings,
    persist_directory=DATABASE_PATH,
)


class ChatRequest(BaseModel):
    question: str


def make_chunk_id(source: str, page: int, content: str) -> str:
    raw = f"{source}|{page}|{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def index_pdf(pdf_path: Path) -> int:
    chunks = split_pdf_into_documents(str(pdf_path))

    if not chunks:
        raise ValueError(
            "No extractable text found. Scanned PDFs may need OCR."
        )

    source = pdf_path.name
    ids = []

    for chunk in chunks:
        chunk.metadata["source"] = source
        page = int(chunk.metadata.get("page", 0))
        ids.append(make_chunk_id(source, page, chunk.page_content))

    old_records = vector_store.get(
        where={"source": source},
        include=["metadatas"],
    )

    if old_records["ids"]:
        vector_store.delete(ids=old_records["ids"])

    vector_store.add_documents(documents=chunks, ids=ids)
    return len(chunks)


@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail="static/index.html is missing.",
        )

    return FileResponse(index_file)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": CHAT_MODEL,
        "database": "ChromaDB",
        "ai_provider": "Gemini",
        "api_key_configured": client is not None,
    }


@app.get("/api/documents")
def list_documents():
    files = sorted(PAPERS_DIR.glob("*.pdf"))
    return {
        "documents": [
            {"name": file.name, "size_bytes": file.stat().st_size}
            for file in files
        ]
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Server AI configuration is missing.",
        )

    filename = Path(file.filename or "").name

    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 25 MB.",
        )

    destination = PAPERS_DIR / filename
    temporary = PAPERS_DIR / f".upload-{filename}"

    try:
        temporary.write_bytes(content)
        shutil.move(str(temporary), str(destination))
        chunk_count = index_pdf(destination)

        return {
            "message": "PDF uploaded and indexed successfully.",
            "filename": filename,
            "chunks_indexed": chunk_count,
        }

    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not index PDF: {error}",
        ) from error


@app.post("/api/chat")
def chat(request: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Server AI configuration is missing.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Enter a research question.",
        )

    try:
        documents = vector_store.similarity_search(question, k=4)

        if not documents:
            return {
                "answer": (
                    "I couldn't find relevant passages. "
                    "Upload and index a PDF first."
                ),
                "sources": [],
            }

        passages = []
        sources = []
        seen = set()

        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            passages.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Passage:\n{doc.page_content}"
            )

            key = (source, page)

            if key not in seen:
                sources.append({
                    "filename": source,
                    "page": page,
                    "excerpt": doc.page_content[:450],
                })
                seen.add(key)

        prompt = f"""
You are ResearchPilot AI, a research assistant.

Answer using the supplied research passages.
Be clear, accurate, and concise.
Do not invent facts, sources, or page numbers.
If the passages are insufficient, say so.
Cite claims using [Source: filename.pdf, Page: 3]
only when supported by the supplied passages.

RESEARCH PASSAGES:
{"\n\n---\n\n".join(passages)}

USER QUESTION:
{question}
"""

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )

        return {
            "answer": response.text or "No answer was generated.",
            "sources": sources,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Research request failed: {error}",
        ) from error
