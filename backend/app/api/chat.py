# backend/app/api/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.workflow_orchestrator import workflow_orchestrator

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    student_id: str = "default_student"
    target_exam: str = "general"
    topic: str = None

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    sources: list[str] = []
    session_type: str = "teaching"
    tools_used: list[str] = []
    reasoning: list[str] = []

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat with the simple AI tutor system"""
    
    try:
        # Use the simple tutor system to answer questions
        result = await workflow_orchestrator.answer_student_question(
            question=req.message,
            topic=req.topic
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return ChatResponse(
            response=result["answer"],
            thread_id=req.thread_id,
            sources=result.get("sources", []),
            session_type="teaching",
            tools_used=["tutor_evaluator"],
            reasoning=[result.get("reasoning", "Generated response using RAG")]
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """Get conversation history and state for a thread"""
    
    try:
        from app.core.database import db
        
        # Get lesson session details
        session = db.get_lesson_session(thread_id)
        if not session:
            return {
                "thread_id": thread_id,
                "messages": [],
                "session_type": "unknown",
                "current_topic": "",
                "weak_topics": [],
                "tools_used": []
            }
        
        # Get detailed messages
        messages = db.get_lesson_messages(thread_id)
        
        return {
            "thread_id": thread_id,
            "messages": [
                {
                    "type": msg["role"],
                    "content": msg["content"]
                }
                for msg in messages
            ],
            "session_type": "teaching",
            "current_topic": session.get("concepts", [{}])[0].get("label", "") if session.get("concepts") else "",
            "weak_topics": [],
            "tools_used": ["tutor_evaluator"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))