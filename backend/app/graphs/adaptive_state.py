"""
Adaptive Learning State for LangGraph-based Agent Coordination
"""
from typing import TypedDict, Dict, List, Any, Optional
from typing_extensions import NotRequired

class EngagementSignals(TypedDict):
    """Signals tracking student engagement and comprehension"""
    questions_asked: int
    time_spent_seconds: float
    confidence_rating: NotRequired[float]  # 0-1, optional student input
    follow_ups_attempted: int
    exercises_completed: int

class AdaptiveState(TypedDict):
    """State managed by the adaptive learning graph"""
    # Session identifiers
    session_id: str
    doc_id: str
    user_id: str
    
    # Current learning context
    current_topic: str
    current_segment: NotRequired[str]
    study_plan: List[Dict[str, Any]]  # Full study plan from planner
    current_step_index: int
    
    # Memory and proficiency
    proficiency: Dict[str, float]  # topic → EWMA score (0-1)
    last_k_summaries: List[Dict[str, Any]]  # Recent taught summaries
    unclear_segments: List[str]  # Segments needing remediation
    
    # Engagement tracking
    engagement_signals: EngagementSignals
    
    # Adaptive routing
    next_action: str  # teach | quiz | remediate | assess | advance | complete
    planner_reason: str  # Why this action was chosen
    
    # Results and outputs
    last_result: NotRequired[Dict[str, Any]]  # Last agent output
    messages: List[Dict[str, str]]  # Chat history for UI
    
    # Quiz and assessment
    quiz_results: List[Dict[str, Any]]
    needs_assessment: bool  # Flag to trigger mid-segment check

def create_initial_state(session_id: str, doc_id: str, user_id: str) -> AdaptiveState:
    """Create initial state for a new learning session"""
    return AdaptiveState(
        session_id=session_id,
        doc_id=doc_id,
        user_id=user_id,
        current_topic="",
        study_plan=[],
        current_step_index=0,
        proficiency={},
        last_k_summaries=[],
        unclear_segments=[],
        engagement_signals=EngagementSignals(
            questions_asked=0,
            time_spent_seconds=0.0,
            follow_ups_attempted=0,
            exercises_completed=0
        ),
        next_action="teach",
        planner_reason="Starting new learning session",
        messages=[],
        quiz_results=[],
        needs_assessment=False
    )
