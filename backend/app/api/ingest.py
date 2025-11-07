# backend/app/api/ingest.py
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from typing import List
import uuid, os
from app.core.vectorstore import upsert_chunks

# Simple text-based ingestion (no external dependencies required)

router = APIRouter()

def simple_chunk(text: str, chunk_size=800, stride=100):
    chunks = []
    i = 0
    while i < len(text):
        chunk_text = text[i:i+chunk_size]
        chunks.append(chunk_text)
        i += (chunk_size - stride)
    return chunks

@router.post("/")
async def ingest_file(file: UploadFile = File(...)):
    contents = await file.read()
    # simple text extraction; for PDFs use PyMuPDF in production
    text = contents.decode("utf-8", errors="ignore")
    chunk_texts = simple_chunk(text)
    chunks = []
    for idx, t in enumerate(chunk_texts):
        chunks.append({
            "id": f"{file.filename}_{idx}",
            "text": t,
            "metadata": {"source": file.filename, "chunk": idx}
        })
    upsert_chunks(chunks)
    return {"status": "ok", "chunks_indexed": len(chunks)}
