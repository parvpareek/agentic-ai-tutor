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

# CORS configuration - allow Vercel frontend and local development
frontend_url = settings.FRONTEND_URL
allowed_origins = []

# Always add the configured frontend URL (without trailing slash)
if frontend_url:
    # Ensure no trailing slash for consistent origin matching
    frontend_url_clean = frontend_url.rstrip('/')
    allowed_origins.append(frontend_url_clean)
    # Also add with trailing slash just in case (though browsers send without)
    if frontend_url_clean and not frontend_url_clean.endswith('/'):
        allowed_origins.append(frontend_url_clean + '/')

# In development, also allow common localhost ports
if settings.DEBUG:
    allowed_origins.extend([
        "http://localhost:8080",  # Vite default port
        "http://localhost:5173",  # Vite alternative port
        "http://localhost:3000",  # React default port
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ])

# Remove duplicates while preserving order
seen = set()
allowed_origins = [x for x in allowed_origins if x and (x not in seen, seen.add(x))[0]]

print(f"[CORS] Allowed origins: {allowed_origins}")
print(f"[CORS] Frontend URL from config: {frontend_url}")

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