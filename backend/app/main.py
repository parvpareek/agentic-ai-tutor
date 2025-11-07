# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, progress, ingest, simple_tutor, session

app = FastAPI(
    title="Agentic AI Tutor - Simplified 4-Agent System",
    description="Streamlined adaptive tutoring system with 4 core agents",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(progress.router, prefix="/progress", tags=["progress"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(simple_tutor.router, tags=["Simple Tutor"])
app.include_router(session.router, tags=["session"])

@app.get("/")
async def root():
    return {
        "message": "Agentic AI Tutor - Simplified 4-Agent System",
        "version": "2.0.0",
        "endpoints": {
            "simple_tutor": "/simple-tutor - Streamlined 4-agent tutoring system",
            "chat": "/chat - Multi-turn conversations with memory",
            "progress": "/progress - Get student progress",
            "ingest": "/ingest - Upload study materials",
            "session": "/session/{id}/summaries/lastk, /session/{id}/topic/{topic}/quiz_results"
        }
    }