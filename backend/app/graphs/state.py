# backend/app/graphs/state.py
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class TutorState(TypedDict):
    """Enhanced state schema for single-user Agentic AI Tutor with Assessment & Progress Tracking"""
    
    # Conversation history
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Student Profile (single user, no ID needed)
    target_exam: str  # "JEE", "SAT", "GRE", "general"
    current_level: str  # "beginner", "intermediate", "advanced"
    
    # Planning & Reasoning
    plan: list[dict]  # List of planned actions
    plan_reasoning: str
    current_step: int
    
    # Current Session Context
    current_topic: str
    current_difficulty: str  # "easy", "medium", "hard"
    session_type: str  # "teaching", "assessment", "review", "planning"
    
    # Enhanced Assessment System
    active_quiz: dict | None  # {
        # "quiz_id": "quiz_123",
        # "questions": [...],
        # "current_question_idx": 0,
        # "student_answers": [],
        # "time_started": "2024-01-15T10:00:00Z"
    #}
    quiz_results: dict | None  # {
        # "total_score": 7,
        # "max_score": 10,
        # "percentage": 70.0,
        # "topics_tested": ["probability"],
        # "question_breakdown": [...],
        # "misconceptions": [...],
        # "strengths": [...],
        # "weaknesses": [...]
    #}
    
    # Enhanced Progress Tracking
    topic_proficiency: dict  # {
        # "probability": {
        #     "current_score": 0.65,
        #     "trend": "improving",
        #     "attempts": 5,
        #     "last_tested": "2024-01-15"
        # }
        # }
    performance_trends: dict  # {
        # "probability": [0.3, 0.5, 0.7],  # historical scores
        # "algebra": [0.9, 0.9, 0.9]
        # }
    weak_topics: list[str]
    strong_topics: list[str]
    
    # Study Plan & Recommendations
    study_plan: dict | None  # {
        # "current_focus": "probability",
        # "next_topics": ["statistics", "permutations"],
        # "review_schedule": {"algebra": "in_2_days"},
        # "estimated_completion": "2_weeks"
    #}
    
    # Content Retrieval
    context: str
    sources: list[str]
    
    # Tool Usage Log & Reasoning
    tools_used: list[str]
    reasoning_log: list[str]
    
    # Assessment-specific fields
    current_question: dict | None  # Current question being answered
    answer_evaluation: dict | None  # Latest answer evaluation
    misconceptions_detected: list[str]  # List of detected misconceptions