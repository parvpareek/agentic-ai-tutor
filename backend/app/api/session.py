from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.core.session_memory import get_redis_client, SessionMemory
from app.agents.workflow_orchestrator import workflow_orchestrator
from app.core.vectorstore import vectorstore

router = APIRouter()


def _get_memory(session_id: str) -> SessionMemory:
    client = get_redis_client()
    if not client:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return SessionMemory(client, session_id)


@router.get("/session/{session_id}/summaries/lastk")
async def get_lastk_summaries(session_id: str, k: int = Query(3, ge=1, le=5)):
    try:
        mem = _get_memory(session_id)
        summaries = mem.get_last_session_summaries(k=k)
        return {"success": True, "summaries": summaries}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/topic/{topic}/quiz_results")
async def get_recent_quiz_results(session_id: str, topic: str, k: int = Query(3, ge=1, le=10)):
    try:
        mem = _get_memory(session_id)
        results = mem.get_recent_quiz_results(topic, k=k)
        return {"success": True, "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/episodic_summary")
async def get_episodic_summary(session_id: str):
    try:
        mem = _get_memory(session_id)
        payload = mem.get_episodic_summary()
        return {"success": True, "episodic_summary": payload}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/episodic_summary/synthesize")
async def synthesize_episodic_summary(session_id: str):
    """Trigger summary synthesis using the global orchestrator (assumes same session)."""
    try:
        # Orchestrator uses current session context; ensure ids align in caller flow
        result = await workflow_orchestrator.synthesize_episodic_summary()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/sessions/saved")
async def list_saved_sessions():
    """List saved lesson sessions for selection in frontend."""
    try:
        from app.core.database import db
        sessions = db.get_all_lesson_sessions()
        return {"success": True, "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/document/{session_id}/chunks")
async def get_document_chunks(session_id: str, k: int = Query(50, ge=1, le=500)):
    """Return a preview of indexed chunks (text length + metadata) for quality checks."""
    try:
        # For Chroma, we can fetch the collection and filter by doc_id in metadata if present; here we return a best-effort sample
        data = vectorstore.collection.get(include=["documents", "metadatas", "ids"])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        ids = data.get("ids", [])
        preview = []
        for i in range(min(k, len(docs))):
            doc = docs[i] or ""
            meta = metas[i] or {}
            preview.append({
                "id": ids[i],
                "length": len(doc),
                "text_preview": doc[:180],
                "metadata": meta
            })
        return {"success": True, "chunks": preview, "total_indexed": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


