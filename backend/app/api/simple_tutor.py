# backend/app/api/simple_tutor.py
"""
FastAPI endpoints for the simplified 4-agent tutor system
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from app.agents.workflow_orchestrator import workflow_orchestrator

router = APIRouter(prefix="/simple-tutor", tags=["Simple Tutor"])

# Request/Response models
class StartSessionRequest(BaseModel):
    student_choice: str = "diagnostic"  # "diagnostic", "choose_topics", "from_beginning"

class SubmitAnswersRequest(BaseModel):
    answers: List[str]

class AskQuestionRequest(BaseModel):
    question: str
    topic: Optional[str] = None

class ExecuteStepRequest(BaseModel):
    step_index: Optional[int] = None
    exercise_answers: Optional[List[str]] = None  # Optional exercise answers from previous segment
    last_topic: Optional[str] = None  # Topic of previous segment (for exercise evaluation)
    last_segment_id: Optional[str] = None  # Segment ID of previous segment

class SubmitExerciseRequest(BaseModel):
    topic: str
    segment_id: str
    prompt: str
    student_answer: str
    correct: Optional[bool] = None

class SetExamContextRequest(BaseModel):
    exam_type: str  # "JEE", "UPSC", "SAT", "GRE", "general", etc.
    exam_details: Optional[Dict[str, Any]] = None

class SubmitDifficultyRatingRequest(BaseModel):
    topic: str
    segment_id: str
    rating: int  # 1-5 where 1 = too hard, 5 = too easy

# Endpoints
@router.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    student_choice: str = Form(default="diagnostic")
):
    """Upload a document and start a learning session"""
    print(f"[UPLOAD] Starting upload for file: {file.filename}, choice: {student_choice}")
    
    try:
        # Read file content
        print(f"[UPLOAD] Reading file content...")
        content = await file.read()
        print(f"[UPLOAD] File size: {len(content)} bytes")
        
        # Handle different file types
        file_extension = file.filename.lower().split('.')[-1] if '.' in file.filename else 'txt'
        print(f"[UPLOAD] File extension: {file_extension}")
        
        if file_extension == 'pdf':
            # For PDF files, pass the binary content directly to the hierarchical parser
            print(f"[UPLOAD] PDF file detected, passing binary content to parser")
            file_content = content  # Keep as bytes for PDF
        else:
            # For text files, try to decode as UTF-8, fallback to other encodings
            try:
                file_content = content.decode('utf-8')
                print(f"[UPLOAD] Successfully decoded as UTF-8")
            except UnicodeDecodeError as e:
                print(f"[UPLOAD] UTF-8 decode failed: {e}, trying latin-1")
                try:
                    file_content = content.decode('latin-1')
                    print(f"[UPLOAD] Successfully decoded as latin-1")
                except UnicodeDecodeError as e2:
                    print(f"[UPLOAD] latin-1 decode failed: {e2}, using utf-8 with errors='ignore'")
                    file_content = content.decode('utf-8', errors='ignore')
                    print(f"[UPLOAD] Decoded with errors ignored")
        
        print(f"[UPLOAD] Content length: {len(file_content)} {'characters' if isinstance(file_content, str) else 'bytes'}")
        if isinstance(file_content, str):
            print(f"[UPLOAD] Content preview: {file_content[:200]}...")
        else:
            print(f"[UPLOAD] Content preview (bytes): {file_content[:50]}...")
        
        # Start learning session
        print(f"[UPLOAD] Starting learning session...")
        result = await workflow_orchestrator.start_learning_session(
            file_content=file_content,
            filename=file.filename,
            student_choice=student_choice
        )
        
        print(f"[UPLOAD] Learning session result: {result.get('success', False)}")
        
        if not result["success"]:
            print(f"[UPLOAD] Learning session failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        print(f"[UPLOAD] Success! Returning response with {len(result.get('concepts', []))} concepts")
        return {
            "message": "Document uploaded and processed successfully",
            "session_id": result["session_id"],
            "concepts": result["concepts"],
            "study_plan": result["study_plan"],
            "next_action": result["next_action"]
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"[UPLOAD] Unexpected error: {str(e)}")
        import traceback
        print(f"[UPLOAD] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

class StartFromSavedRequest(BaseModel):
    saved_session_id: str
    student_choice: str = "from_beginning"

@router.post("/start-from-saved")
async def start_from_saved(req: StartFromSavedRequest):
    """Start a session reusing previously saved concepts/segments and existing vector index.
    Does not reuse past learning progress.
    """
    try:
        result = await workflow_orchestrator.start_learning_session_from_saved(
            saved_session_id=req.saved_session_id,
            student_choice=req.student_choice
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to start from saved"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start from saved: {str(e)}")

@router.post("/execute-step")
async def execute_step(request: ExecuteStepRequest):
    """Execute the next step in the study plan"""
    # Clear agent thoughts from previous request
    from app.core.agent_thoughts import thoughts_tracker
    thoughts_tracker.clear()
    
    try:
        # OPTIMIZATION: Spawn async exercise evaluation with context reuse
        if request.exercise_answers and request.last_topic and request.last_segment_id:
            import asyncio
            # Try to get context from current state to avoid Redis fetch
            taught_context = None
            try:
                segment_key = f"{request.last_topic}:{request.last_segment_id}"
                taught_context = workflow_orchestrator.current_state.taught_content.get(segment_key)
            except:
                pass
            
            asyncio.create_task(
                workflow_orchestrator.evaluate_exercise_answers_async(
                    topic=request.last_topic,
                    segment_id=request.last_segment_id,
                    exercise_answers=request.exercise_answers,
                    taught_context=taught_context  # Reuse context!
                )
            )
            print(f"[API] 🚀 Spawned exercise evaluation (context={'reused' if taught_context else 'will fetch'})")
        
        result = await workflow_orchestrator.execute_plan_step(
            step_index=request.step_index
        )
        # Diagnostics: log outgoing result shape
        try:
            print(f"[API] execute-step result keys: {list(result.keys())} success={result.get('success')} next_action={result.get('next_action')}")
            print(f"[API] execute-step result: {result}")
        except Exception:
            pass
        
        if not result["success"]:
            try:
                import json as _json
                print(f"[API] execute-step full error payload: {_json.dumps(result, ensure_ascii=False)[:1000]}")
            except Exception:
                print(f"[API] execute-step error payload (non-json): {result}")
            raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
        
        # Add agent thoughts to response
        result["agent_thoughts"] = thoughts_tracker.get_all()
        
        return result
        
    except HTTPException as he:
        # Let FastAPI return the original status code (e.g., 400)
        raise he
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[API] execute-step failed: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Step execution failed: {str(e)}")

@router.post("/submit-quiz-answers")
async def submit_quiz_answers(request: SubmitAnswersRequest):
    """Submit answers for the current quiz"""
    try:
        result = await workflow_orchestrator.submit_quiz_answers(
            answers=request.answers
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer submission failed: {str(e)}")

@router.post("/ask-question")
async def ask_question(request: AskQuestionRequest):
    """Ask a question to the tutor"""
    # Clear agent thoughts from previous request
    from app.core.agent_thoughts import thoughts_tracker
    thoughts_tracker.clear()
    
    try:
        result = await workflow_orchestrator.answer_student_question(
            question=request.question,
            topic=request.topic
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add agent thoughts to response
        result["agent_thoughts"] = thoughts_tracker.get_all()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@router.post("/submit-exercise")
async def submit_exercise(req: SubmitExerciseRequest):
    """Record a simple student exercise interaction in session memory."""
    try:
        from app.core.session_memory import get_redis_client, SessionMemory
        state = workflow_orchestrator.get_current_state()
        session_id = state.get('session_id') or (f"session_{state.get('doc_id')}" if state.get('doc_id') else None)
        client = get_redis_client()
        if not (client and session_id):
            return {"success": True, "stored": False}
        mem = SessionMemory(client, session_id)
        stored = mem.add_exercise_interaction(req.topic, req.segment_id, {
            "prompt": req.prompt,
            "student_answer": req.student_answer,
            "correct": req.correct,
        })
        return {"success": True, "stored": bool(stored)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/current-state")
async def get_current_state():
    """Get the current workflow state"""
    try:
        state = workflow_orchestrator.get_current_state()
        return {
            "success": True,
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")

@router.get("/spaced-repetition-schedule")
async def get_spaced_repetition_schedule():
    """Get the spaced repetition schedule for review"""
    try:
        schedule = workflow_orchestrator.get_spaced_repetition_schedule()
        return {
            "success": True,
            "schedule": schedule
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schedule: {str(e)}")

@router.get("/concepts")
async def get_concepts():
    """Get the extracted concepts from the current document"""
    try:
        state = workflow_orchestrator.get_current_state()
        concepts = state.get("concepts", [])
        
        return {
            "success": True,
            "concepts": concepts,
            "count": len(concepts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get concepts: {str(e)}")

@router.get("/study-plan")
async def get_study_plan():
    """Get the current study plan"""
    try:
        state = workflow_orchestrator.get_current_state()
        study_plan = state.get("study_plan", [])
        
        return {
            "success": True,
            "study_plan": study_plan,
            "total_steps": len(study_plan),
            "current_step": state.get("current_step", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get study plan: {str(e)}")

@router.post("/reset-session")
async def reset_session():
    """Reset the current session (clears state and memory)"""
    try:
        # Clear Redis memory if it exists
        if workflow_orchestrator.memory:
            try:
                workflow_orchestrator.memory.clear_session()
                print("[RESET] Cleared Redis memory")
            except Exception as e:
                print(f"[RESET] Failed to clear memory: {e}")
        
        # Reset the orchestrator state
        workflow_orchestrator.current_state = workflow_orchestrator.SimpleState()
        workflow_orchestrator.memory = None
        workflow_orchestrator.adaptive_state = None
        
        print("[RESET] Session reset successfully")
        return {
            "success": True,
            "message": "Session reset successfully (state and memory cleared)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for the simple tutor system"""
    try:
        # Check if all components are available
        state = workflow_orchestrator.get_current_state()
        
        return {
            "status": "healthy",
            "components": {
                "ingest_agent": "ready",
                "concept_agent": "ready", 
                "planner_agent": "ready",
                "tutor_evaluator": "ready",
                "spaced_repetition": "ready"
            },
            "current_state": {
                "has_document": state.get("doc_id") is not None,
                "has_concepts": len(state.get("concepts", [])) > 0,
                "has_plan": len(state.get("study_plan", [])) > 0
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Lesson History Endpoints
@router.get("/lesson-sessions")
async def get_lesson_sessions():
    """Get all lesson sessions for the user"""
    try:
        from app.core.database import db
        sessions = db.get_all_lesson_sessions()
        return {
            "success": True,
            "sessions": sessions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lesson sessions: {str(e)}")

@router.get("/lesson-sessions/{session_id}")
async def get_lesson_session(session_id: str):
    """Get a specific lesson session with full details"""
    try:
        from app.core.database import db
        session = db.get_lesson_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Lesson session not found")
        
        # Get detailed conversation history
        messages = db.get_lesson_messages(session_id)
        session["detailed_messages"] = messages
        
        return {
            "success": True,
            "session": session
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lesson session: {str(e)}")

@router.get("/lesson-sessions/{session_id}/messages")
async def get_lesson_messages(session_id: str):
    """Get conversation messages for a specific lesson session"""
    try:
        from app.core.database import db
        messages = db.get_lesson_messages(session_id)
        return {
            "success": True,
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lesson messages: {str(e)}")

@router.post("/lesson-sessions/{session_id}/complete")
async def complete_lesson_session(session_id: str):
    """Mark a lesson session as completed"""
    try:
        from app.core.database import db
        success = db.update_lesson_session(
            session_id=session_id,
            status="completed"
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to complete lesson session")
        
        return {
            "success": True,
            "message": "Lesson session marked as completed"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to complete lesson session: {str(e)}")

@router.post("/set-exam-context")
async def set_exam_context(request: SetExamContextRequest):
    """
    Set the exam context for tailored learning.
    This directs all teaching and quiz content to match the exam style.
    
    Supported exam types:
    - JEE: Technical depth, numerical problem-solving, rigorous derivations
    - UPSC: Broad coverage, analytical thinking, comprehensive understanding
    - SAT/GRE: Balanced conceptual and applied knowledge, reasoning skills
    - general: Default learning mode
    """
    try:
        from app.core.session_memory import get_redis_client, SessionMemory
        
        # Get current session
        state = workflow_orchestrator.get_current_state()
        session_id = state.get('session_id') or (f"session_{state.get('doc_id')}" if state.get('doc_id') else "default")
        
        client = get_redis_client()
        if not client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        memory = SessionMemory(client, session_id)
        success = memory.set_exam_context(request.exam_type, request.exam_details)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to set exam context")
        
        print(f"[API] Set exam context to: {request.exam_type}")
        return {
            "success": True,
            "message": f"Exam context set to {request.exam_type}",
            "exam_type": request.exam_type,
            "details": request.exam_details or {}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set exam context: {str(e)}")

@router.post("/submit-difficulty-rating")
async def submit_difficulty_rating(request: SubmitDifficultyRatingRequest):
    """
    Submit difficulty rating for a segment.
    Rating scale: 1-5 where:
    - 1: Too hard / Confused / Uncomfortable
    - 2: Challenging but manageable
    - 3: Just right
    - 4: Easy / Comfortable
    - 5: Too easy / Boring
    
    This rating is used to:
    - Adapt future teaching complexity
    - Adjust quiz difficulty dynamically
    - Track student confidence levels
    """
    try:
        from app.core.session_memory import get_redis_client, SessionMemory
        
        # Validate rating
        if not 1 <= request.rating <= 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Get current session
        state = workflow_orchestrator.get_current_state()
        session_id = state.get('session_id') or (f"session_{state.get('doc_id')}" if state.get('doc_id') else "default")
        
        client = get_redis_client()
        if not client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        memory = SessionMemory(client, session_id)
        success = memory.store_difficulty_rating(request.topic, request.segment_id, request.rating)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store difficulty rating")
        
        # Get updated average for feedback
        avg_difficulty = memory.get_avg_difficulty_for_topic(request.topic)
        
        # Provide feedback based on rating
        feedback = ""
        if request.rating <= 2:
            feedback = "Thanks for the feedback! I'll simplify future explanations and provide more examples."
        elif request.rating >= 4:
            feedback = "Great! I'll increase the depth and challenge level for upcoming content."
        else:
            feedback = "Perfect! The difficulty level seems just right."
        
        print(f"[API] Difficulty rating: {request.topic}/{request.segment_id} = {request.rating} (avg: {avg_difficulty:.1f})")
        
        return {
            "success": True,
            "message": feedback,
            "rating": request.rating,
            "topic_avg_difficulty": round(avg_difficulty, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit difficulty rating: {str(e)}")

@router.get("/exam-context")
async def get_exam_context():
    """Get the current exam context settings"""
    try:
        from app.core.session_memory import get_redis_client, SessionMemory
        
        state = workflow_orchestrator.get_current_state()
        session_id = state.get('session_id') or (f"session_{state.get('doc_id')}" if state.get('doc_id') else "default")
        
        client = get_redis_client()
        if not client:
            return {"success": True, "exam_context": {"exam_type": "general", "exam_details": {}}}
        
        memory = SessionMemory(client, session_id)
        exam_context = memory.get_exam_context()
        
        return {
            "success": True,
            "exam_context": exam_context
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get exam context: {str(e)}")

@router.get("/spaced-repetition/due-reviews")
async def get_due_reviews():
    """Get topics that are due for review based on spaced repetition"""
    try:
        from app.core.session_memory import get_redis_client, SessionMemory
        
        state = workflow_orchestrator.get_current_state()
        session_id = state.get('session_id') or (f"session_{state.get('doc_id')}" if state.get('doc_id') else "default")
        
        client = get_redis_client()
        if not client:
            return {"success": True, "due_reviews": []}
        
        memory = SessionMemory(client, session_id)
        due_topics = memory.get_due_reviews()
        
        # Get next review time for each topic
        reviews_with_details = []
        for topic in due_topics:
            next_review = memory.get_next_review_time(topic)
            reviews_with_details.append({
                "topic": topic,
                "next_review": next_review.isoformat() if next_review else None,
                "is_overdue": True  # If it's in due_reviews, it's overdue
            })
        
        return {
            "success": True,
            "due_reviews": reviews_with_details,
            "count": len(due_topics)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get due reviews: {str(e)}")
