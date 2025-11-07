# backend/app/api/ask.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.workflow_orchestrator import workflow_orchestrator

router = APIRouter()

class AskRequest(BaseModel):
    query: str
    topic: str = None

@router.post("/")
async def ask_question(req: AskRequest):
    """Process student question with the simple tutor system"""
    try:
        response = await workflow_orchestrator.answer_student_question(
            question=req.query,
            topic=req.topic
        )
        
        if not response["success"]:
            raise HTTPException(status_code=500, detail=response["error"])
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))