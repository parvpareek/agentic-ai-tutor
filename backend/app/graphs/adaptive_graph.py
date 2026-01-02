"""
LangGraph-based Adaptive Learning Graph
Coordinates Planner (Supervisor) and Worker Agents (Tutor, Quiz, Assess, Remediate)
"""
from langgraph.graph import StateGraph, END
from .adaptive_state import AdaptiveState, create_initial_state
from .adaptive_nodes import (
    planner_node,
    tutor_node,
    quiz_node,
    assess_node,
    remediate_node,
    route_after_planner
)
from app.agents.llm_planner import LLMPlannerAgent
from app.agents.simple_workflow import TutorEvaluatorAgent
from app.core.session_memory import SessionMemory
from app.core.vectorstore import vectorstore
from app.core.database import db
from langchain_openai import ChatOpenAI
from app.core.config import settings
import time

class AdaptiveLearningGraph:
    """
    Main graph coordinating adaptive learning flow.
    Uses LangGraph StateGraph with conditional routing.
    """
    
    def __init__(self, orchestrator=None):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize agents
        self.planner = LLMPlannerAgent(self.llm, db, vectorstore)
        self.tutor = TutorEvaluatorAgent(vectorstore, self.llm, db)
        
        # Reference to orchestrator for delegating teaching/quiz logic
        self.orchestrator = orchestrator
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Construct the adaptive learning state graph"""
        
        # Create graph with our state schema
        workflow = StateGraph(AdaptiveState)
        
        # Add nodes
        # Planner is the supervisor that decides what to do next
        workflow.add_node("planner", self._planner_wrapper)
        
        # Worker nodes
        workflow.add_node("tutor", self._tutor_wrapper)
        workflow.add_node("quiz", self._quiz_wrapper)
        workflow.add_node("assess", self._assess_wrapper)
        workflow.add_node("remediate", self._remediate_wrapper)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add conditional edges from planner (routes based on decision)
        workflow.add_conditional_edges(
            "planner",
            route_after_planner,
            {
                "tutor": "tutor",
                "quiz": "quiz",
                "assess": "assess",
                "remediate": "remediate",
                "end": END
            }
        )
        
        # After each worker node, decide whether to continue or wait for user
        # Tutor: STOP after teaching one segment (wait for user to click "Next")
        workflow.add_edge("tutor", END)
        # Quiz: STOP and wait for submission
        workflow.add_edge("quiz", END)
        # Assessment: STOP and wait for answer
        workflow.add_edge("assess", END)
        # Remediation: Continue to planner (may need multiple remediations in one go)
        workflow.add_edge("remediate", "planner")
        
        return workflow.compile()
    
    # ========================================================================
    # Node Wrappers (inject dependencies)
    # ========================================================================
    
    async def _planner_wrapper(self, state: AdaptiveState) -> AdaptiveState:
        """Wrapper to inject planner and memory into node"""
        # Get memory from orchestrator if available
        if self.orchestrator and self.orchestrator.memory:
            memory = self.orchestrator.memory
        else:
            from app.core.session_memory import get_redis_client
            redis_client = get_redis_client()
            memory = SessionMemory(redis_client, state["session_id"])
        return await planner_node(state, self.planner, memory)
    
    async def _tutor_wrapper(self, state: AdaptiveState) -> AdaptiveState:
        """Wrapper to inject tutor and memory into node"""
        if self.orchestrator and self.orchestrator.memory:
            memory = self.orchestrator.memory
        else:
            from app.core.session_memory import get_redis_client
            redis_client = get_redis_client()
            memory = SessionMemory(redis_client, state["session_id"])
        return await tutor_node(state, self.tutor, memory, self.orchestrator)
    
    async def _quiz_wrapper(self, state: AdaptiveState) -> AdaptiveState:
        """Wrapper to inject tutor (for quiz gen) and memory into node"""
        if self.orchestrator and self.orchestrator.memory:
            memory = self.orchestrator.memory
        else:
            from app.core.session_memory import get_redis_client
            redis_client = get_redis_client()
            memory = SessionMemory(redis_client, state["session_id"])
        return await quiz_node(state, self.tutor, memory, self.orchestrator)
    
    async def _assess_wrapper(self, state: AdaptiveState) -> AdaptiveState:
        """Wrapper to inject tutor (for assessment) and memory into node"""
        if self.orchestrator and self.orchestrator.memory:
            memory = self.orchestrator.memory
        else:
            from app.core.session_memory import get_redis_client
            redis_client = get_redis_client()
            memory = SessionMemory(redis_client, state["session_id"])
        return await assess_node(state, self.tutor, memory)
    
    async def _remediate_wrapper(self, state: AdaptiveState) -> AdaptiveState:
        """Wrapper to inject tutor (for remediation) and memory into node"""
        if self.orchestrator and self.orchestrator.memory:
            memory = self.orchestrator.memory
        else:
            from app.core.session_memory import get_redis_client
            redis_client = get_redis_client()
            memory = SessionMemory(redis_client, state["session_id"])
        return await remediate_node(state, self.tutor, memory, self.orchestrator)
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    async def execute_step(self, state: AdaptiveState) -> AdaptiveState:
        """
        Execute one step of the adaptive learning graph.
        Returns updated state.
        """
        print(f"[ADAPTIVE_GRAPH] Executing step | Action: {state.get('next_action', 'unknown')} | Index: {state.get('current_step_index')} | AwaitingNext: {state.get('awaiting_user_next')} ")
        
        # Run the graph (it will execute planner → routed node → return)
        result = await self.graph.ainvoke(state)
        print(f"[ADAPTIVE_GRAPH] Step complete | NextAction: {result.get('next_action')} | Index: {result.get('current_step_index')} | AwaitingNext: {result.get('awaiting_user_next')} ")

        # If we've reached the end of the current topic and quiz is done, advance to next topic's first segment
        try:
            plan = result.get("study_plan", [])
            idx = result.get("current_step_index", 0)
            # Skip any residual non-teach items and move to next available study_segment
            while idx < len(plan) and plan[idx].get("action") != "study_segment":
                idx += 1
            result["current_step_index"] = idx
        except Exception:
            pass
        
        return result
    
    def create_session(self, session_id: str, doc_id: str, user_id: str = "default") -> AdaptiveState:
        """Create initial state for new session"""
        return create_initial_state(session_id, doc_id, user_id)
    
    async def handle_student_answer(self, state: AdaptiveState, answer: str) -> AdaptiveState:
        """
        Process student answer (to question or assessment).
        Updates engagement signals and routes to planner.
        """
        print(f"[ADAPTIVE_GRAPH] Student answered: {answer[:50]}...")
        
        # Record answer
        state["messages"].append({
            "role": "student",
            "content": answer
        })
        
        # Update engagement
        state["engagement_signals"]["questions_asked"] += 1
        
        # Reset to normal flow
        state["next_action"] = "teach"
        
        return state
    
    async def handle_quiz_submission(self, state: AdaptiveState, answers: dict, score: float, unclear_segments: list) -> AdaptiveState:
        """
        Process quiz submission.
        Updates proficiency, unclear segments, and routes to planner.
        
        PHASE 4: Stores proficiency in memory store for cross-session tracking.
        """
        print(f"[ADAPTIVE_GRAPH] Quiz submitted | Score: {score:.1%} | Unclear: {unclear_segments}")
        
        topic = state["current_topic"]
        user_id = state["user_id"]
        
        # Update proficiency (EWMA: 0.7 * old + 0.3 * new)
        old_prof = state["proficiency"].get(topic, 0.5)
        new_prof = 0.7 * old_prof + 0.3 * score
        state["proficiency"][topic] = new_prof
        
        # PHASE 4: Store proficiency in memory store
        from .memory_store import adaptive_memory
        adaptive_memory.store_proficiency(user_id, topic, new_prof)
        
        # Store quiz result
        state["quiz_results"].append({
            "topic": topic,
            "score": score,
            "unclear_segments": unclear_segments,
            "timestamp": int(time.time())
        })
        
        # Mark unclear segments for remediation
        state["unclear_segments"] = unclear_segments
        
        # Reset action so planner decides next
        state["next_action"] = "decide"
        
        return state


# Factory function - graph will be initialized with orchestrator reference
def create_adaptive_graph(orchestrator):
    return AdaptiveLearningGraph(orchestrator)