# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, progress, ingest, simple_tutor, session
from app.core.config import settings

app = FastAPI(
    title="Agentic AI Tutor - Simplified 4-Agent System",
    description="Streamlined adaptive tutoring system with 4 core agents",
    version="2.0.0"
)

# CORS configuration - allow Vercel frontend
frontend_url = settings.FRONTEND_URL
allowed_origins = [frontend_url]
if settings.DEBUG:
    # In development, also allow localhost
    allowed_origins.extend(["http://localhost:5173", "http://localhost:3000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
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