# main.py
import os
import re
import uuid
import tempfile
import requests
import fitz  # PyMuPDF
import faiss
import unidecode
import numpy as np
import cohere
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

# 1. === Configuration ===

TEAM_TOKEN = "1072521ce486eb467b9bc36a8d3141814c43b779390db369589237eb87e940f7"
COHERE_API_KEY = "1bxiGdTKWg09SX91C9Cl62yzVOGqgyxWfZXBBayA"  # <-- Replace with your Cohere API key

co = cohere.Client(COHERE_API_KEY)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
security = HTTPBearer()

# 2. === FastAPI Setup ===

app = FastAPI(
    title="HackRx Retrieval System",
    version="1.0",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. === Input Schema ===

class QueryRequest(BaseModel):
    documents: str  # PDF Blob URL
    questions: List[str]

# 4. === Utilities ===

def preprocess_text(text: str) -> str:
    text = unidecode.unidecode(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return preprocess_text(" ".join([page.get_text() for page in doc]))


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 20) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += " " + sent
        else:
            chunks.append(current.strip())
            overlap_text = " ".join(current.strip().split()[-overlap:]) if overlap > 0 else ""
            current = overlap_text + " " + sent
    if current:
        chunks.append(current.strip())
    return chunks
def get_embeddings(texts: List[str]):
    return model.encode(texts, show_progress_bar=False)


def create_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def cohere_answer(question: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Answer based only on the policy below.

{context}

Q: {question}
A:"""
    response = co.generate(
        model="command-r-plus",  # or "command", "command-light", etc.
        prompt=prompt,
        max_tokens=256,
        temperature=0.0,
        stop_sequences=["\nQ:"]
    )
    return response.generations[0].text.strip()

# 5. === Main Endpoint with Bearer Auth ===

@app.post(
    "/api/v1/hackrx/run",
    dependencies=[Depends(security)]
)
async def run(
    payload: QueryRequest,
    request: Request,
    creds: HTTPAuthorizationCredentials = Depends(security)
):
    token = creds.credentials
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")
    MAX_CONTEXT_TOKENS = 1500  # adjust as needed
    def trim_context(chunks, max_tokens=MAX_CONTEXT_TOKENS):
        context = ""
        for chunk in chunks:
            if len(context.split()) + len(chunk.split()) > max_tokens:
                break
            context += "\n\n" + chunk
        return context.strip()

    # Download PDF
    try:
        pdf_path = tempfile.mktemp(suffix=".pdf")
        r = requests.get(payload.documents)
        with open(pdf_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document download failed: {e}")

    # Extract & index
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(np.array(embeddings))

    # Answer queries
    answers = []
    for q in payload.questions:
        q_emb = get_embeddings([q])[0].reshape(1, -1)
        D, I = index.search(q_emb, k=5)
        selected_chunks = [chunks[i] for i in I[0]]
        context = trim_context(selected_chunks)
        context = "\n\n".join(chunks[i] for i in I[0])
        answers.append(cohere_answer(q, context))

    return JSONResponse(content={"answers": answers})
