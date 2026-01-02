# backend/app/agents/simple_state.py
"""
Simplified State Schema for 4-Agent System
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

class SimpleTutorState(TypedDict):
    """Simplified state for the 4-agent system"""
    
    # Student context
    student_id: str
    target_exam: str
    
    # Document processing
    uploaded_files: List[Dict[str, Any]]  # [{"file_path": str, "file_type": str, "doc_id": str}]
    doc_id: Optional[str]
    ingest_status: str  # "pending", "processing", "completed", "failed"
    
    # Concept extraction
    concepts: List[Dict[str, Any]]  # [{"concept_id": str, "label": str, "supporting_chunk_ids": List[str]}]
    
    # Planning
    study_mode: str  # "diagnostic", "student_choice", "beginning"
    selected_concepts: List[str]  # concepts student wants to study
    study_plan: List[Dict[str, Any]]  # [{"step_id": str, "action": str, "topic": str, "est_minutes": int, "why_assigned": str}]
    current_step: int
    
    # Tutoring and evaluation
    current_topic: Optional[str]  # Track current topic being studied
    current_question: Optional[str]
    student_answer: Optional[str]
    quiz_questions: List[Dict[str, Any]]
    quiz_results: Dict[str, Any]
    
    # Student profile
    student_profile: Dict[str, Any]  # topic proficiency tracking
    session_history: List[Dict[str, Any]]
    
    # Spaced repetition
    review_schedule: Dict[str, str]  # topic -> next_review_date
    practice_queue: List[str]  # topics ready for practice
    
    # System state
    current_agent: str
    next_action: str
    session_id: str
    timestamp: str

