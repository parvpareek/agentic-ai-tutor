# backend/app/api/quiz.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.agents.workflow_orchestrator import workflow_orchestrator

router = APIRouter()

class QuizRequest(BaseModel):
    topic: str
    difficulty: Optional[str] = None
    num_questions: Optional[int] = 3

@router.post("/")
async def generate_quiz(req: QuizRequest):
    """Generate adaptive quiz for a topic"""
    try:
        # Use the tutor evaluator to generate a quiz
        from app.agents.simple_workflow import TutorEvaluatorAgent
        from app.core.vectorstore import vectorstore
        from app.core.database import db
        from app.core.config import settings
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.7,
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        tutor_evaluator = TutorEvaluatorAgent(vectorstore, llm, db)
        
        result = await tutor_evaluator.execute(
            "generate_quiz",
            topic=req.topic,
            difficulty=req.difficulty or "medium",
            num_questions=req.num_questions
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.reasoning)
        
        return {
            "success": True,
            "quiz": result.data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))