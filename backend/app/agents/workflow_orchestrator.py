# backend/app/agents/workflow_orchestrator.py
"""
Workflow Orchestrator for Simplified 4-Agent System
Manages the complete student learning workflow
"""

from typing import Dict, List, Any, Optional
from .simple_workflow import (
    SimpleState, IngestAgent, ConceptExtractionAgent, 
    TutorEvaluatorAgent, SpacedRepetitionScheduler,
    AgentType
)
from .llm_planner import LLMPlannerAgent
from app.core.vectorstore import vectorstore
from app.core.database import db
from app.core.config import settings
from app.core.session_memory import SessionMemory, get_redis_client
from app.core.request_context import RequestContext
from app.core.rich_student_context import RichStudentContext
import json
from langchain_openai import ChatOpenAI
from app.utils.session_logger import init_session_logger, get_logger
import asyncio
import time
from app.graphs.adaptive_graph import create_adaptive_graph
from app.graphs.adaptive_state import AdaptiveState, create_initial_state

class WorkflowOrchestrator:
    """Main orchestrator for the simplified 4-agent workflow"""
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.2,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize agents
        self.ingest_agent = IngestAgent(vectorstore)
        self.concept_agent = ConceptExtractionAgent(vectorstore, self.llm)
        self.planner_agent = LLMPlannerAgent(self.llm, db, vectorstore)
        self.tutor_evaluator = TutorEvaluatorAgent(vectorstore, self.llm, db)
        
        # Initialize spaced repetition
        self.spaced_rep = SpacedRepetitionScheduler()
        
        # Initialize Redis memory (with fallback to in-memory)
        self.redis_client = get_redis_client()
        self.memory = None  # Will be initialized per session
        
        # Current state
        self.current_state = SimpleState()
        self.current_state.taught_content = {}  # Initialize taught content storage
        
        # Adaptive graph state (optional, for graph-based routing)
        self.adaptive_state: Optional[AdaptiveState] = None
        self.adaptive_graph = None
        self.use_adaptive_graph = True  # Feature flag - ENABLED

    def _extract_full_text_from_possible_json(self, raw: str) -> str:
        """If raw looks like JSON (possibly fenced), extract content/full_text; otherwise return raw."""
        try:
            content_raw = (raw or "").strip()
            
            # Step 1: Remove code fences if present
            if '```json' in content_raw:
                start = content_raw.find('```json') + len('```json')
                end = content_raw.find('```', start)
                if end != -1:
                    content_raw = content_raw[start:end].strip()
            elif '```' in content_raw and content_raw.startswith('```'):
                parts = content_raw.split('```')
                if len(parts) >= 2:
                    content_raw = parts[1].strip()
            
            # Step 2: Extract JSON object if wrapped in text
            import json as _json
            import re as _re
            obj_txt = content_raw
            if not obj_txt.strip().startswith('{'):
                m = _re.search(r"\{[\s\S]*\}", content_raw)
                if m:
                    obj_txt = m.group(0)
            
            # Step 3: Parse and extract content field
            obj = _json.loads(obj_txt)
            if isinstance(obj, dict):
                # Try 'content' field first (new format), then 'full_text' (old format)
                extracted = obj.get("content") or obj.get("full_text")
                if extracted:
                    print(f"[WORKFLOW] 🔍 Extracted content from JSON (prevented JSON from showing to user)")
                    return extracted
        except Exception as e:
            # If it looks like JSON but we can't parse it, try regex extraction
            if raw.strip().startswith('{'):
                try:
                    import re as _re
                    content_match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, _re.DOTALL)
                    if content_match:
                        extracted = content_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                        print(f"[WORKFLOW] 🔍 Extracted content via regex (JSON parse failed)")
                        return extracted
                except:
                    pass
        
        return raw
    
    def _generate_search_queries(self, topic: str) -> List[str]:
        """Generate multiple search queries for better retrieval coverage.
        Uses simple heuristics - no LLM call needed.
        """
        queries = []
        
        # Main query
        queries.append(topic)
        
        # Add explanatory query
        queries.append(f"What is {topic} and how does it work")
        
        # Add technical query
        queries.append(f"{topic} definition explanation examples")
        
        # Add components query for complex topics
        if any(word in topic.lower() for word in ['architecture', 'model', 'system', 'framework']):
            queries.append(f"{topic} components structure")
        
        # Add mechanism query for processes
        if any(word in topic.lower() for word in ['attention', 'mechanism', 'process', 'algorithm']):
            queries.append(f"{topic} mechanism mathematical formulation")
        
        return queries[:4]  # Limit to 4 queries max
    
    def _init_session_memory(self, session_id: str, metadata: Dict, clear_existing: bool = True) -> bool:
        """
        Initialize Redis memory for a session.
        
        Args:
            session_id: Unique session identifier
            metadata: Session metadata to store
            clear_existing: If True, clear any existing data for this session (default: True for new uploads)
        """
        try:
            if self.redis_client:
                # Create new SessionMemory instance for this session
                self.memory = SessionMemory(self.redis_client, session_id)
                
                # Initialize with automatic cleanup of any stale data
                success = self.memory.init_session(metadata, clear_existing=clear_existing)
                if success:
                    print(f"✅ Redis memory initialized for session: {session_id} (cleared={clear_existing})")
                else:
                    print(f"⚠️ Redis memory initialization failed, using in-memory fallback")
                    self.memory = None
                return success
            else:
                print(f"⚠️ Redis not available, using in-memory fallback")
                self.memory = None
                return False
        except Exception as e:
            print(f"⚠️ Memory initialization error: {e}, using in-memory fallback")
            self.memory = None
            return False
    
    async def start_learning_session(self, file_content, filename: str, 
                                   student_choice: str = "diagnostic") -> Dict[str, Any]:
        """Start a complete learning session"""
        print("🚀 Starting new learning session...")
        print(f"[WORKFLOW] File: {filename}, Choice: {student_choice}, Content length: {len(file_content)}")
        
        try:
            # Enforce: clear cross-session summaries only (preserve document index)
            try:
                from app.core.vectorstore import clear_taught_summaries
                clear_taught_summaries()
                print("[WORKFLOW] Cleared taught summaries for new upload")
            except Exception as e:
                print(f"[WORKFLOW] Warning: failed to clear taught summaries on upload: {e}")
            # Step 1: Ingest document
            print("📚 Step 1: Ingesting document...")
            # Log agent action
            try:
                get_logger().log_agent_action("WorkflowOrchestrator", "start_session", f"filename={filename}, choice={student_choice}")
            except Exception:
                pass
            ingest_result = await self.ingest_agent.execute(file_content, filename)
            
            if not ingest_result.success:
                return {
                    "success": False,
                    "error": "Document ingestion failed",
                    "details": ingest_result.reasoning
                }
            
            self.current_state.doc_id = ingest_result.data["doc_id"]
            self.current_state.ingest_status = ingest_result.data["ingest_status"]
            document_outline = ingest_result.data.get("document_outline", "")
            learning_roadmap = ingest_result.data.get("learning_roadmap", {})
            hierarchy = ingest_result.data.get("hierarchy", {})
            
            # Initialize session with a unique session_id per run (avoid collisions across same doc)
            try:
                import uuid
                session_id = f"session_{self.current_state.doc_id}_{uuid.uuid4().hex[:8]}"
                self.current_state.session_id = session_id
                init_session_logger(session_id)
                logger = get_logger()
                logger.log_agent_action("IngestAgent", "parse_document", f"doc_id={self.current_state.doc_id}")
                # Log complete structures
                logger.log_pdf_structure(hierarchy)
                logger.log_data("document_outline", document_outline)
                logger.log_data("learning_roadmap", learning_roadmap)
            except Exception:
                pass
            
            # Initialize Redis memory for this session
            session_metadata = {
                "filename": filename,
                "student_choice": student_choice,
                "doc_id": self.current_state.doc_id,
                "total_chapters": learning_roadmap.get('total_chapters', 0),
                "total_sections": learning_roadmap.get('total_sections', 0)
            }
            self._init_session_memory(session_id, session_metadata)
            
            # Update agents with memory reference
            self.concept_agent.memory = self.memory
            self.planner_agent.memory = self.memory

            # Log parsed structure
            print(f"\n{'='*80}")
            print(f"📄 PARSED DOCUMENT STRUCTURE")
            print(f"{'='*80}")
            print(f"Document ID: {self.current_state.doc_id}")
            print(f"Roadmap: {learning_roadmap.get('total_chapters', 0)} chapters, {learning_roadmap.get('total_sections', 0)} sections")
            print(f"Outline (first 500 chars):\n{document_outline[:500]}...")
            print(f"{'='*80}\n")
            
            # Step 2: Extract concepts using deep document structure
            print("🔍 Step 2: Extracting concepts from deep document structure...")
            try:
                get_logger().log_agent_action("ConceptExtractionAgent", "extract_concepts", "using outline + roadmap")
            except Exception:
                pass
            concept_result = await self.concept_agent.execute(
                self.current_state.doc_id, 
                document_outline=document_outline,
                learning_roadmap=learning_roadmap,
                target_exam="JEE",  # Could be parameterized
                student_context=None  # Could add student context here
            )
            
            if not concept_result.success:
                return {
                    "success": False,
                    "error": "Concept extraction failed",
                    "details": concept_result.reasoning
                }
            
            self.current_state.concepts = concept_result.data["concepts"]
            self.current_state.learning_roadmap = concept_result.data.get("learning_roadmap", {})
            
            # Step 3: Create study plan using LLM planner with deep structure context
            print("📋 Step 3: Creating intelligent study plan with roadmap...")
            try:
                get_logger().log_agent_action("LLMPlannerAgent", "create_initial_plan", f"concepts={len(self.current_state.concepts)}")
            except Exception:
                pass
            plan_result = await self.planner_agent.create_initial_plan(
                self.current_state.concepts, 
                student_choice,
                document_outline=document_outline,
                learning_roadmap=learning_roadmap  # Pass roadmap to planner
            )
            
            if not plan_result.success:
                return {
                    "success": False,
                    "error": "Study planning failed",
                    "details": plan_result.reasoning
                }
            
            self.current_state.study_plan = plan_result.plan
            self.current_state.student_choice = student_choice
            
            # Create lesson session for history tracking
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            
            # Get exam context from Redis if set
            exam_context_data = None
            if self.memory:
                try:
                    exam_context_data = self.memory.get_exam_context()
                    if exam_context_data and exam_context_data.get("exam_type") == "general":
                        exam_context_data = None  # Don't store default
                except Exception:
                    pass
            
            db.create_lesson_session(
                session_id=session_id,
                document_name=filename,
                document_type=filename.split('.')[-1] if '.' in filename else 'text',
                concepts=self.current_state.concepts,
                study_plan=self.current_state.study_plan,
                document_structure=learning_roadmap,
                exam_context=exam_context_data
            )
            
            # Initialize adaptive graph if enabled
            if self.use_adaptive_graph:
                self.adaptive_graph = create_adaptive_graph(self)
                self.adaptive_state = create_initial_state(
                    session_id=session_id,
                    doc_id=self.current_state.doc_id,
                    user_id="default"
                )
                self.adaptive_state["study_plan"] = self.current_state.study_plan
                self.adaptive_state["current_topic"] = self.current_state.study_plan[0]["topic"] if self.current_state.study_plan else ""
                print(f"[WORKFLOW] Adaptive graph initialized for session {session_id}")
            
            # Return initial response
            return {
                "success": True,
                "session_id": session_id,
                "concepts": self.current_state.concepts,
                "study_plan": self.current_state.study_plan,
                "next_action": "start_plan",
                "message": f"Found {len(self.current_state.concepts)} key concepts. Ready to start your {student_choice} session!"
            }
            
        except Exception as e:
            print(f"[WORKFLOW] Error in start_learning_session: {str(e)}")
            import traceback
            print(f"[WORKFLOW] Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": "Session start failed",
                "details": str(e)
            }

    async def start_learning_session_from_saved(self, saved_session_id: str, student_choice: str = "from_beginning") -> Dict[str, Any]:
        """Start a session reusing persisted concepts/segments and existing vector index.
        Does NOT reuse past progress; creates a fresh plan and fresh Redis memory.
        """
        print(f"🚀 Starting session from saved data: {saved_session_id}")
        try:
            # Enforce: clear cross-session summaries when starting from saved (preserve doc index)
            try:
                from app.core.vectorstore import clear_taught_summaries
                clear_taught_summaries()
                print("[WORKFLOW] Cleared taught summaries for start-from-saved")
            except Exception as e:
                print(f"[WORKFLOW] Warning: failed to clear taught summaries on saved: {e}")
            # Load saved lesson session
            saved = db.get_lesson_session(saved_session_id)
            if not saved:
                return {"success": False, "error": "Saved session not found"}

            # Derive doc_id from session_id convention
            doc_id = saved_session_id.replace("session_", "") if saved_session_id.startswith("session_") else saved_session_id
            self.current_state.doc_id = doc_id
            self.current_state.ingest_status = "completed"
            self.current_state.concepts = saved.get("concepts", []) or []
            self.current_state.learning_roadmap = saved.get("document_structure", {}) or {}

            # Initialize FRESH Redis memory for this session (DO NOT reuse old memory!)
            import uuid
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}_{uuid.uuid4().hex[:8]}"
            self.current_state.session_id = session_id
            # Reset any in-memory runtime fields to ensure no carryover
            self.current_state.taught_content = {}
            self.current_state.current_quiz = None
            self.current_state.student_answers = []
            
            # CRITICAL FIX: Initialize planner with loaded concepts!
            self.planner_agent.all_concepts = self.current_state.concepts
            self.planner_agent.completed_topics = set()  # Fresh start
            print(f"[WORKFLOW] 🔧 Loaded {len(self.planner_agent.all_concepts)} concepts into planner")
            self.current_state.current_step = 0
            self._init_session_memory(
                session_id=session_id,
                metadata={
                    "doc_id": self.current_state.doc_id,
                    "filename": saved.get("document_name", "Unknown"),
                    "loaded_from": saved_session_id,
                    "num_concepts": len(self.current_state.concepts)
                },
                clear_existing=True  # Clear any stale data from previous sessions
            )
            # Populate ONLY segment plans into fresh Redis (no progress/taught restoration)
            if self.memory and self.current_state.concepts:
                try:
                    from typing import List, Dict
                    populated = 0
                    for concept in self.current_state.concepts:
                        topic = concept.get("label")
                        segments: List[Dict] = concept.get("learning_segments") or []
                        if topic and segments:
                            ok = self.memory.store_segment_plan(topic, segments)
                            if ok:
                                populated += 1
                    print(f"[WORKFLOW] Stored segment plans for {populated} topics into fresh session memory")
                except Exception as e:
                    print(f"[WORKFLOW] Failed to store segment plans into memory: {e}")
            
            # Restore exam context from saved session to NEW Redis memory
            if self.memory:
                saved_exam_context = saved.get("exam_context")
                if saved_exam_context:
                    try:
                        self.memory.set_exam_context(
                            exam_type=saved_exam_context.get("exam_type", "general"),
                            exam_details=saved_exam_context.get("exam_details", {})
                        )
                        print(f"[WORKFLOW] Restored exam context: {saved_exam_context.get('exam_type', 'general')}")
                    except Exception as e:
                        print(f"[WORKFLOW] Failed to restore exam context: {e}")

            # Fast path: deterministically build a simple segment-first plan without LLM to reduce latency
            plan: List[Dict[str, Any]] = []
            for concept in (self.current_state.concepts or [])[:6]:  # cap to first 6 for quick start
                topic = concept.get("label", "Topic")
                concept_id = concept.get("concept_id", "concept")
                segments = sorted(concept.get("learning_segments", []), key=lambda s: s.get("order", 0))
                for seg in segments:
                    plan.append({
                        "step_id": f"{concept_id}_{seg.get('segment_id','seg')}",
                        "action": "study_segment",
                        "topic": topic,
                        "concept_id": concept_id,
                        "segment_id": seg.get("segment_id", "seg_1"),
                        "segment_title": seg.get("title", "Segment"),
                        "difficulty": "medium",
                        "est_minutes": seg.get("estimated_minutes", 8),
                        "why_assigned": "Learn this segment"
                    })
                # Do NOT append quiz steps here; planner will trigger quiz only after all segments are completed

            self.current_state.study_plan = plan
            self.current_state.student_choice = student_choice

            # Initialize adaptive graph if enabled
            if self.use_adaptive_graph:
                self.adaptive_graph = create_adaptive_graph(self)
                self.adaptive_state = create_initial_state(
                    session_id=session_id,
                    doc_id=self.current_state.doc_id,
                    user_id="default"
                )
                self.adaptive_state["study_plan"] = self.current_state.study_plan
                self.adaptive_state["current_topic"] = self.current_state.study_plan[0]["topic"] if self.current_state.study_plan else ""
                print(f"[WORKFLOW] Adaptive graph initialized for saved session {session_id}")

            # Return response similar to upload flow (no previous messages)
            return {
                "success": True,
                "session_id": session_id,
                "concepts": self.current_state.concepts,
                "study_plan": self.current_state.study_plan,
                "messages": [],
                "next_action": "start_plan",
                "message": f"Loaded saved concepts ({len(self.current_state.concepts)}) and created a fresh plan."
            }
        except Exception as e:
            print(f"[WORKFLOW] Error in start_learning_session_from_saved: {str(e)}")
            import traceback
            print(f"[WORKFLOW] Traceback: {traceback.format_exc()}")
            return {"success": False, "error": "Failed to start from saved", "details": str(e)}
    
    async def execute_plan_step(self, step_index: int = None) -> Dict[str, Any]:
        """Execute the next step in the study plan"""
        if not self.current_state.study_plan:
            return {
                "success": False,
                "error": "No study plan available"
            }
        
        # Use provided step or current step
        if step_index is None:
            step_index = self.current_state.current_step
        
        # If using adaptive graph, let the graph decide end-of-plan behavior
        if (not self.use_adaptive_graph) and step_index >= len(self.current_state.study_plan):
            return {
                "success": False,
                "error": "All plan steps completed"
            }
        
        # Adaptive graph path (if enabled and initialized)
        if self.use_adaptive_graph and self.adaptive_graph and self.adaptive_state:
            print(f"[WORKFLOW] Using adaptive graph for step execution")
            try:
                # Update adaptive state with current step index
                self.adaptive_state["current_step_index"] = step_index
                
                # If awaiting explicit user action, do not advance automatically
                if self.adaptive_state.get("awaiting_user_next"):
                    print("[WORKFLOW] Awaiting user action; not advancing graph")
                    # Return last result if available, otherwise a minimal prompt
                    last = self.adaptive_state.get("last_result", {})
                    return last or {"success": True, "message": "Ready. Click Next Step to continue."}
                
                # Execute graph (planner decides, then routes to worker)
                updated_state = await self.adaptive_graph.execute_step(self.adaptive_state)
                
                # Sync back to legacy state
                self.current_state.current_step = updated_state["current_step_index"]
                self.adaptive_state = updated_state
                
                # Extract result for API response
                result = updated_state.get("last_result", {})
                if result.get("success"):
                    result["planner_reason"] = updated_state.get("planner_reason", "")
                    # If worker set awaiting_user_next, persist it so we halt next time
                    if updated_state.get("awaiting_user_next"):
                        self.adaptive_state["awaiting_user_next"] = True
                    return result
                else:
                    # If graph returned end/wait state, translate to response
                    next_action = updated_state.get("next_action")
                    if next_action == "wait_for_quiz_submission":
                        # Normalize waiting response to success True
                        normalized = {**(result or {})}
                        normalized["step_type"] = "practice_quiz"
                        normalized["next_action"] = "wait_for_quiz_submission"
                        normalized["success"] = True  # enforce success
                        return normalized
                    elif next_action == "wait_for_answer":
                        normalized = {**(result or {})}
                        normalized["step_type"] = "assessment_wait"
                        normalized["next_action"] = "wait_for_answer"
                        normalized["success"] = True
                        return normalized
                    elif next_action == "complete":
                        return {"success": True, "message": "Learning complete!", "next_action": "complete"}
                    else:
                        return result or {"success": True, "message": "Ready. Click Next Step to continue."}
            except Exception as e:
                print(f"[WORKFLOW] Adaptive graph error, falling back to legacy: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to legacy execution
        
        # Legacy execution path
        step = self.current_state.study_plan[step_index]
        action = step["action"]
        topic = step["topic"]
        concept_id = step.get("concept_id")
        
        # Memory-driven simplification: prefer segment-based study when segment plan exists
        if action in ("study_topic", "study_segment") and self.memory:
            try:
                next_seg = self.memory.get_next_segment(topic)
                if next_seg:
                    # Ensure step uses study_segment with concrete segment_id/title
                    step["action"] = "study_segment"
                    step["segment_id"] = next_seg.get("segment_id", "seg_1")
                    step["segment_title"] = next_seg.get("title", "Segment")
                    action = step["action"]
            except Exception:
                pass

        print(f"🎯 Executing step {step_index + 1}: {action} for {topic}")
        
        try:
            result = None
            if action == "diagnostic_quiz":
                result = await self._execute_diagnostic_quiz(step, concept_id)
            elif action == "calibration_quiz":
                result = await self._execute_calibration_quiz(step, concept_id)
            elif action == "study_topic":
                result = await self._execute_study_topic(step, concept_id)
            elif action == "study_segment":
                result = await self._execute_study_segment(step, concept_id)
            elif action == "practice_quiz":
                # Enforce: do not quiz until all segments for topic are completed
                print(f"[WORKFLOW] Practice quiz encountered at step {self.current_state.current_step} for topic={topic}")
                if self.memory:
                    try:
                        remaining_next = self.memory.get_next_segment(topic)
                        print(f"[WORKFLOW] Memory next segment for topic '{topic}': {remaining_next}")
                        if remaining_next:
                            # Insert the next segment step before quiz and execute it now
                            print(f"[WORKFLOW] Injecting remaining segment before quiz: {remaining_next}")
                            insert = {
                                "step_id": f"auto_{remaining_next.get('segment_id', 'seg')}",
                                "action": "study_segment",
                                "topic": topic,
                                "concept_id": concept_id,
                                "segment_id": remaining_next.get("segment_id", "seg_1"),
                                "segment_title": remaining_next.get("title", "Segment"),
                                "difficulty": "medium",
                                "est_minutes": remaining_next.get("estimated_minutes", 8),
                                "why_assigned": "Complete remaining segment before topic-level quiz"
                            }
                            try:
                                upcoming = self.current_state.study_plan[self.current_state.current_step:self.current_state.current_step+2]
                                print(f"[WORKFLOW] Upcoming before insert: {upcoming}")
                            except Exception:
                                pass
                            self.current_state.study_plan.insert(self.current_state.current_step, insert)
                            try:
                                upcoming2 = self.current_state.study_plan[self.current_state.current_step:self.current_state.current_step+2]
                                print(f"[WORKFLOW] Upcoming after insert: {upcoming2}")
                            except Exception:
                                pass
                            step = insert
                            action = "study_segment"
                            result = await self._execute_study_segment(step, concept_id)
                        else:
                            print(f"[WORKFLOW] No remaining segments. Proceeding to topic-level quiz for {topic}")
                            result = await self._execute_practice_quiz(step, concept_id)
                    except Exception as e:
                        print(f"[WORKFLOW] Error checking remaining segments: {e}. Proceeding to quiz")
                        result = await self._execute_practice_quiz(step, concept_id)
                else:
                    print(f"[WORKFLOW] No memory available. Proceeding to quiz for {topic}")
                    result = await self._execute_practice_quiz(step, concept_id)
            elif action == "optional_exercise":
                result = await self._execute_optional_exercise(step, concept_id)
            elif action == "review_results":
                result = await self._execute_review_results(step)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
            
            # Log what we're returning
            try:
                print(f"[WORKFLOW] Returning result with keys: {list(result.keys())}")
                if 'content' in result and isinstance(result.get('content'), str):
                    print(f"[WORKFLOW] Content preview: {result['content'][:100]}...")
                else:
                    print(f"[WORKFLOW] WARNING: No 'content' field in result! Full result: {result}")
            except Exception:
                pass
            # Also write step result to session log
            try:
                logger = get_logger()
                logger.log_agent_action("WorkflowOrchestrator", "execute_plan_step", f"action={action}, topic={topic}")
                logger.log_data("step_result", result)
            except Exception:
                pass
            
            return result
                
        except Exception as e:
            print(f"[WORKFLOW] Exception: {str(e)}")
            import traceback
            print(f"[WORKFLOW] Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Step execution failed: {str(e)}"
            }
    
    async def _execute_diagnostic_quiz(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute diagnostic quiz step"""
        topic = step["topic"]
        
        # Generate quiz
        quiz_result = await self.tutor_evaluator.execute(
            "generate_quiz",
            topic=topic,
            concept_id=concept_id,
            difficulty="medium",
            num_questions=2
        )
        
        if not quiz_result.success:
            return {
                "success": False,
                "error": "Quiz generation failed",
                "details": quiz_result.reasoning
            }
        
        quiz = quiz_result.data
        self.current_state.current_quiz = quiz
        
        return {
            "success": True,
            "step_type": "diagnostic_quiz",
            "quiz": quiz,
            "instructions": f"Answer these {len(quiz['questions'])} questions about {topic}. This is a quick diagnostic to assess your current level.",
            "next_action": "submit_answers"
        }
    
    async def _execute_calibration_quiz(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute calibration quiz step"""
        topic = step["topic"]
        
        # Generate easy quiz for calibration
        quiz_result = await self.tutor_evaluator.execute(
            "generate_quiz",
            topic=topic,
            concept_id=concept_id,
            difficulty="easy",
            num_questions=1
        )
        
        if not quiz_result.success:
            return {
                "success": False,
                "error": "Quiz generation failed",
                "details": quiz_result.reasoning
            }
        
        quiz = quiz_result.data
        self.current_state.current_quiz = quiz
        
        return {
            "success": True,
            "step_type": "calibration_quiz",
            "quiz": quiz,
            "instructions": f"Answer this quick question about {topic} to help me calibrate the difficulty level for you.",
            "next_action": "submit_answers"
        }
    
    async def _execute_study_topic(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute study topic step with automatic introduction"""
        topic = step["topic"]
        print(f"[WORKFLOW] Executing study_topic for: {topic}")
        
        # Update current topic in state
        self.current_state.current_topic = topic
        
        # IMPROVED RETRIEVAL: Use LLM-generated queries if available, otherwise fallback
        print(f"[WORKFLOW] Retrieving content for topic: {topic}")
        
        # Try to get LLM-generated queries from concept
        concept_queries = None
        for concept in self.current_state.concepts:
            if concept.get("label") == topic or concept.get("concept_id") == concept_id:
                concept_queries = concept.get("search_queries", [])
                break
        
        # Use LLM queries if available, otherwise generate heuristic queries
        if concept_queries and len(concept_queries) > 0:
            queries = concept_queries[:4]  # Use up to 4 LLM-generated queries
            print(f"[WORKFLOW] Using {len(queries)} LLM-generated search queries")
        else:
            queries = self._generate_search_queries(topic)
            print(f"[WORKFLOW] Using {len(queries)} heuristic search queries")
        
        # Retrieve with all queries and deduplicate
        all_documents = []
        all_metadatas = []
        seen_texts = set()
        
        for query in queries:
            # Use hybrid retrieval (vector + BM25). Falls back to vector if BM25 unavailable.
            results = vectorstore.query_hybrid(query, k=5, alpha=0.6)
            docs = results.get('documents', [[]])[0] if 'documents' in results else []
            metas = results.get('metadatas', [[]])[0] if 'metadatas' in results else []
            
            for doc, meta in zip(docs, metas):
                if doc not in seen_texts:
                    seen_texts.add(doc)
                    all_documents.append(doc)
                    all_metadatas.append(meta)
        
        documents = all_documents
        print(f"[WORKFLOW] Retrieved {len(documents)} unique documents for {topic}")
        # Log retrieved chunks for evaluation
        try:
            logger = get_logger()
            preview = [
                {
                    "text_preview": (doc[:220] + "...") if isinstance(doc, str) and len(doc) > 220 else doc,
                    "metadata": meta
                }
                for doc, meta in zip(all_documents[:10], all_metadatas[:10])
            ]
            logger.log_data("retrieved_chunks", {
                "topic": topic,
                "queries": queries,
                "count": len(documents),
                "samples": preview
            })
        except Exception:
            pass
        
        if not documents or len(documents) == 0:
            # Fallback to regular search
            print(f"[WORKFLOW] No documents with filter, trying regular search")
            results = vectorstore.query_top_k(topic, k=5)
        documents = results.get('documents', [[]])[0] if 'documents' in results else []
        
        if not documents or len(documents) == 0:
            error_msg = f"I couldn't find detailed content about {topic} in the uploaded document. Let me explain based on general knowledge."
            return {
                "success": True,
                "step_type": "study_topic",
                "topic": topic,
                "study_content": "",
                "content": error_msg,
                "instructions": "Ask me specific questions about this topic!",
                "next_action": "ask_questions_or_continue"
            }
        
        # Format study material with proper context (use more chunks for better context)
        # Sanitize: remove heading-only lines and keep substantive sentences
        def clean_snippet(s: str) -> str:
            import re
            lines = [ln.strip() for ln in s.splitlines()]
            cleaned = []
            for ln in lines:
                # Skip very short heading-like lines
                if len(ln) < 25 and (ln.lower().startswith('section:') or ln.istitle()):
                    continue
                cleaned.append(ln)
            text = ' '.join(cleaned)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        cleaned_docs = [clean_snippet(d) for d in documents[:6] if isinstance(d, str) and d.strip()]
        study_content = "\n\n---\n\n".join([d for d in cleaned_docs if len(d) > 120])
        
        # Generate grounded, structured explanation using LLM (with timeout)
        print(f"[WORKFLOW] Generating grounded explanation for {topic}")
        recent_summaries = []
        if self.memory:
            try:
                recent_summaries = self.memory.get_last_session_summaries(k=3)
            except Exception:
                recent_summaries = []
        educator_prompt = f"""Explain this topic using the provided context.

            TOPIC: {topic}

CONTEXT:
{study_content[:1500]}

RECENT SESSION (for continuity):
{json.dumps(recent_summaries)[:600]}

Return ONLY JSON (no code fences):
{{
  "full_text": "4-6 paragraph explanation grounded in context",
  "summary": "1-2 sentence summary",
  "excerpts": ["quote from context"],
  "student_facing_next_steps": ["exercise 1"],
  "memory_patch": {{"session_summary_delta": "note", "confidence": 0.8}}
}}"""
        
        
        try:
            import asyncio
            # Add 10 second timeout
            start_time = time.time()
            intro_response = await asyncio.wait_for(
                self.llm.ainvoke(educator_prompt),
                timeout=60.0  # Increased to 60s; simplified prompt should reduce actual time
            )
            duration = time.time() - start_time
            # Try to parse JSON contract; fallback to plain text
            parsed = None
            try:
                content_raw = intro_response.content.strip()
                if content_raw.startswith("```"):
                    if '```json' in content_raw:
                        content_raw = content_raw.split('```json', 1)[1]
                    parts = content_raw.split('```')
                    if len(parts) >= 2:
                        content_raw = parts[1]
                if not content_raw.strip().startswith('{'):
                    import re
                    m = re.search(r"\{[\s\S]*\}", content_raw)
                    if m:
                        content_raw = m.group(0)
                parsed = json.loads(content_raw)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("full_text"):
                grounded_explanation = parsed.get("full_text", "")
                summary_text = parsed.get("summary", "")
                memory_patch = parsed.get("memory_patch", {})
                # Store structured taught segment
                if self.memory:
                    try:
                        self.memory.store_taught_segment_json(
                            topic,
                            "full_topic",
                            {
                                "full_text": grounded_explanation,
                                "summary": summary_text,
                                "excerpts": parsed.get("excerpts", []),
                                "sources": parsed.get("sources", []),
                                "segment_id": "full_topic",
                                "topic": topic
                            }
                        )
                    except Exception:
                        pass
            else:
                grounded_explanation = intro_response.content
                # Fallback summary: first 180 chars
                summary_text = (intro_response.content or "").strip()[:180]
                memory_patch = {}
            
            # Log LLM call
            print(f"\n{'='*80}")
            print(f"🤖 LLM CALL - Generating Grounded Explanation")
            print(f"{'='*80}")
            print(f"Topic: {topic}")
            print(f"Duration: {duration:.2f}s")
            print(f"Prompt (first 200 chars): {educator_prompt[:200]}...")
            print(f"Response (first 200 chars): {grounded_explanation[:200]}...")
            print(f"{'='*80}\n")
            
            print(f"[WORKFLOW] Grounded explanation generated successfully")
        except asyncio.TimeoutError:
            print(f"[WORKFLOW] Explanation generation timed out, using fallback")
            grounded_explanation = (
                f"**{topic}**\n\n"
                f"Here is a concise overview based on the document.\n\n"
                f"- What it is: [INFERRED] A high-level summary will appear here.\n"
                f"- Why it matters: [INFERRED] Brief motivation.\n\n"
                f"Key takeaways:\n- [INFERRED] Takeaway 1\n- [INFERRED] Takeaway 2\n- [INFERRED] Takeaway 3\n- [INFERRED] Takeaway 4\n- [INFERRED] Takeaway 5\n\n"
                f"Examples:\n1) [INFERRED] Short example.\n2) [INFERRED] Short example.\n\n"
                f"Practice questions:\n1) Write a short question.\n2) Write a short question.\n\n"
                f"Study plan (next steps):\n1) Skim the section.\n2) Work an example.\n3) Answer practice questions.\n"
            )
        except Exception as e:
            print(f"[WORKFLOW] Error generating explanation: {e}")
            grounded_explanation = f"**{topic}**\n\n[INFERRED] Let's explore this concept step by step using the document context."
        
        # Use normalized explanation (extract full_text from JSON if needed)
        full_explanation = self._extract_full_text_from_possible_json(grounded_explanation)
        
        # STEP 1 FIX: Store taught content for quiz grounding
        self.current_state.taught_content[topic] = full_explanation
        print(f"[WORKFLOW] Stored taught content for {topic} ({len(full_explanation)} chars)")
        
        # Store in Redis memory if available (for segment-based teaching)
        if self.memory:
            # For now, store as a single segment since we're still using topic-based teaching
            # Later this will be updated to store per-segment content
            self.memory.store_taught_segment(topic, "full_topic", full_explanation)
            self.memory.update_context({
                "current_topic": topic,
                "current_segment": "full_topic",
                "last_taught_content": f"{topic}:full_topic"
            })
            # Ensure last_k summary is pushed even if model omitted it
            try:
                if summary_text:
                    self.memory.push_session_summary({
                        "segment_id": "full_topic",
                        "topic": topic,
                        "summary": summary_text,
                        "timestamp": time.time(),
                        "memory_delta": memory_patch.get("session_summary_delta") if isinstance(memory_patch, dict) else None
                    }, k=3)
                # Write to long-term summary store (Chroma) for semantic recall
                vectorstore.upsert_taught_summary(
                    _id=f"{self.current_state.doc_id}:{topic}:full_topic",
                    text=summary_text,
                    metadata={"doc_id": self.current_state.doc_id, "topic": topic, "segment_id": "full_topic"}
                )
            except Exception:
                pass
            # Push last-k summary if available
            try:
                if summary_text:
                    self.memory.push_session_summary({
                        "segment_id": "full_topic",
                        "topic": topic,
                        "summary": summary_text,
                        "timestamp": time.time(),
                        "memory_delta": memory_patch.get("session_summary_delta")
                    }, k=3)
            except Exception:
                pass
        
        # Log the study content in lesson history
        if self.current_state.doc_id:
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=full_explanation,
                sources=[f"Document: {self.current_state.doc_id}"]
            )
        
        print(f"[WORKFLOW] Successfully prepared study content for {topic}")
        
        return {
            "success": True,
            "step_type": "study_topic",
            "topic": topic,
            "study_content": study_content,
            "content": full_explanation,  # This will be displayed in chat automatically
            "exercises": parsed.get("student_facing_next_steps", []) if isinstance(parsed, dict) else [],
            "instructions": "Feel free to ask me any questions about this topic!",
            "next_action": "ask_questions_or_continue"
        }
    
    def _get_engagement_profile_cached(self, topic: str) -> Dict[str, Any]:
        """
        OPTIMIZATION: Compute engagement profile once and cache for this request.
        Reduces pattern analysis overhead by 80%.
        """
        cache_key = f"_engagement_profile_{topic}"
        if hasattr(self, cache_key):
            return getattr(self, cache_key)
        
        engagement_profile = {}
        if self.memory:
            try:
                from app.core.learning_patterns import LearningPatternAnalyzer
                analyzer = LearningPatternAnalyzer()
                engagement_profile = analyzer.extract_engagement_profile(self.memory, topic)
                print(f"[WORKFLOW] 📊 Engagement: {engagement_profile.get('engagement_level')}, Style: {engagement_profile.get('preferred_learning_style')}")
            except Exception as e:
                print(f"[WORKFLOW] Could not extract engagement profile: {e}")
        
        # Cache for this request
        setattr(self, cache_key, engagement_profile)
        return engagement_profile
    
    async def _determine_quiz_difficulty_agentic(
        self,
        topic: str,
        recent_quiz_results: List[Dict],
        exercise_assessments: List[tuple],
        student_questions: List[str],
        confusion_patterns: List[str]
    ) -> tuple[str, bool, str]:
        """
        Use LLM to intelligently decide quiz difficulty and format.
        
        Returns: (difficulty, include_subjective, reasoning)
        """
        try:
            from app.core.agent_thoughts import thoughts_tracker
            
            # Build context for LLM
            quiz_history_str = "No prior quizzes"
            if recent_quiz_results:
                scores = [q.get("score_percentage", 0) for q in recent_quiz_results]
                quiz_history_str = f"Recent quiz scores: {[f'{s:.0f}%' for s in scores]}"
                if len(scores) >= 2:
                    trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
                    quiz_history_str += f" (Trend: {trend})"
            
            exercise_summary = "No exercises completed"
            if exercise_assessments:
                performances = [p for _, p in exercise_assessments]
                difficulties = [d for d, _ in exercise_assessments]
                exercise_summary = f"Exercise performance: {performances}. Recommended difficulties: {difficulties}"
            
            confusion_str = "No confusion detected"
            if confusion_patterns:
                confusion_str = f"Confusion areas: {', '.join(confusion_patterns)}"
            
            questions_str = "No questions asked"
            if student_questions:
                questions_str = f"Asked {len(student_questions)} questions: {student_questions[:2]}"
            
            prompt = f"""Analyze this student's learning state and recommend quiz parameters for "{topic}".

QUIZ HISTORY:
{quiz_history_str}

EXERCISE PERFORMANCE:
{exercise_summary}

CONFUSION PATTERNS:
{confusion_str}

STUDENT ENGAGEMENT:
{questions_str}

Recommend quiz parameters considering:
1. **Trajectory**: Are they improving or struggling?
2. **Confusion Type**: Do they understand concepts or are they lost?
3. **Engagement**: Are they actively learning or passive?
4. **Challenge Balance**: Push them to grow, but don't overwhelm

Return ONLY valid JSON (no code fences):
{{
  "difficulty": "easy|medium|hard",
  "include_subjective": true|false,
  "reasoning": "2-3 sentence explanation of why this difficulty and format",
  "focus_areas": ["area to emphasize in questions"],
  "question_mix": ["conceptual", "application", "analysis"]
}}

RULES:
- If struggling consistently (scores <50%) → "easy" + no subjective
- If improving steadily → challenge with "medium" or "hard"
- If mastered basics (>80%) → "hard" + include subjective
- If confused about concepts → focus on conceptual understanding
- If weak on application → include application questions
- Balance challenge with achievability - aim for 60-70% success rate"""
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import json
            import re
            
            # Strip code fences
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Extract JSON
            if not response_text.startswith('{'):
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            decision = json.loads(response_text)
            
            difficulty = decision.get("difficulty", "medium")
            include_subjective = decision.get("include_subjective", False)
            reasoning = decision.get("reasoning", "LLM-based difficulty decision")
            
            # Log the agentic decision
            thoughts_tracker.add(
                "Quiz Generator (Agentic)",
                f"Decided {difficulty} difficulty, subjective={include_subjective}: {reasoning}",
                "🤖",
                {"decision": decision}
            )
            
            print(f"[WORKFLOW] 🤖 Agentic Quiz Decision: {difficulty}, subjective={include_subjective}")
            print(f"[WORKFLOW] Reasoning: {reasoning}")
            
            return (difficulty, include_subjective, reasoning)
            
        except Exception as e:
            print(f"[WORKFLOW] Agentic decision failed: {e}, falling back to rule-based")
            # Fallback to simple rule
            if recent_quiz_results:
                avg_score = sum(q.get("score_percentage", 0) for q in recent_quiz_results) / len(recent_quiz_results)
                if avg_score >= 80:
                    return ("hard", True, "Rule-based fallback: high scores")
                elif avg_score < 50:
                    return ("easy", False, "Rule-based fallback: low scores")
            return ("medium", False, "Rule-based fallback: moderate performance")
    
    async def _determine_quiz_difficulty_with_rich_context(
        self,
        topic: str,
        rich_context: RichStudentContext
    ) -> tuple[str, bool, str]:
        """
        🎯 IMPROVED: Use rich context instead of fragmented memory fetches
        
        Benefits:
        - Single memory fetch (vs 5-10 separate calls)
        - Temporal awareness (recent weighted higher)
        - Coherent student state
        - Much richer LLM prompt
        """
        try:
            from app.core.agent_thoughts import thoughts_tracker
            
            # Get focused context for quiz difficulty decision
            context_str = rich_context.to_llm_context(focus="quiz_difficulty")
            
            prompt = f"""Analyze this student's learning state and recommend quiz parameters for "{topic}".

{context_str}

Recommend quiz parameters considering:
1. **Trajectory**: Are they improving or struggling?
2. **Confusion Type**: Do they understand concepts or are they lost?
3. **Engagement**: Are they actively learning or passive?
4. **Challenge Balance**: Push them to grow, but don't overwhelm
5. **Cognitive Load**: Are they fresh or tired?

Return ONLY valid JSON (no code fences):
{{
  "difficulty": "easy|medium|hard",
  "include_subjective": true|false,
  "reasoning": "2-3 sentence explanation of why this difficulty and format",
  "confidence": 0.85
}}

RULES:
- If struggling consistently (scores <50%) → "easy" + no subjective
- If improving steadily → challenge with "medium" or "hard"
- If mastered basics (>80%) → "hard" + include subjective
- If confused about concepts → focus on conceptual understanding
- If cognitive load heavy → easier quiz to build confidence
- Balance challenge with achievability - aim for 60-70% success rate"""
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import json
            import re
            
            # Strip code fences
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Extract JSON
            if not response_text.startswith('{'):
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            decision = json.loads(response_text)
            
            difficulty = decision.get("difficulty", "medium")
            include_subjective = decision.get("include_subjective", False)
            reasoning = decision.get("reasoning", "LLM-based difficulty decision")
            confidence = decision.get("confidence", 0.75)
            
            # Log the agentic decision
            thoughts_tracker.add(
                "Quiz Generator (Rich Context)",
                f"Decided {difficulty} difficulty, subjective={include_subjective}: {reasoning}",
                "🎯",
                {"decision": decision, "cognitive_load": rich_context.cognitive_load.level}
            )
            
            print(f"[WORKFLOW] 🎯 Rich Context Quiz Decision: {difficulty}, subjective={include_subjective}, confidence={confidence}")
            
            return (difficulty, include_subjective, reasoning)
            
        except Exception as e:
            print(f"[WORKFLOW] Rich context decision failed: {e}, falling back")
            import traceback
            traceback.print_exc()
            # Fallback
            if rich_context.performance.quiz_trajectory == "declining":
                return ("easy", False, "Fallback: declining performance")
            elif rich_context.performance.quiz_trajectory == "improving":
                return ("hard", True, "Fallback: improving performance")
            return ("medium", False, "Fallback: stable performance")
    
    async def _determine_content_strategy_with_rich_context(
        self,
        segment: Dict,
        rich_context: RichStudentContext,
        prior_segments: List[str]
    ) -> Dict:
        """
        🎯 IMPROVED: Determine content strategy using rich context
        Cross-agent aware: Can see quiz difficulty and other decisions
        """
        try:
            from app.core.agent_thoughts import thoughts_tracker
            
            # Get focused context for content strategy
            context_str = rich_context.to_llm_context(focus="content_strategy")
            
            segment_title = segment.get("title", "Unknown")
            difficulty = segment.get("difficulty", "medium")
            objectives = segment.get("learning_objectives", [])
            
            prior_str = f"Completed {len(prior_segments)} prior segments" if prior_segments else "First segment"
            
            prompt = f"""Plan how to teach "{segment_title}" to this student.

{context_str}

SEGMENT DETAILS:
- Title: {segment_title}
- Difficulty: {difficulty}
- Objectives: {objectives[:2] if objectives else "General understanding"}
- Prior Context: {prior_str}

Decide the optimal teaching strategy considering:
1. **Cognitive Load**: Student's current mental state
2. **Confusion Patterns**: What they struggle with
3. **Learning Style**: How they learn best
4. **Performance Trajectory**: Are they improving or struggling?
5. **Prior Knowledge**: What they already know

Return ONLY valid JSON (no code fences):
{{
  "content_length": "brief|standard|comprehensive",
  "teaching_approach": "definition-first|example-driven|analogy-heavy|step-by-step",
  "detail_level": "high-level|balanced|deep-dive",
  "estimated_words": 500,
  "rationale": "Why this approach works for this student",
  "key_focus": "What to emphasize based on their needs"
}}

GUIDELINES:
- Brief (300-400 words): Easy topics, strong understanding, low confusion
- Standard (400-600 words): Medium complexity, some confusion
- Comprehensive (600-800 words): Hard topics, high confusion, weak foundation

- Definition-first: For conceptual learners
- Example-driven: For practical learners, high confusion
- Analogy-heavy: When student is confused (more analogies = clearer)
- Step-by-step: For processes, when student asks "how" questions

Consider cognitive load: If heavy → briefer content to avoid overwhelm"""
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import json, re
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            if not response_text.startswith('{'):
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            strategy = json.loads(response_text)
            
            # Log decision
            thoughts_tracker.add(
                "Content Strategist (Rich Context)",
                f"{strategy.get('content_length', 'standard')} content, {strategy.get('teaching_approach', 'balanced')} approach: {strategy.get('rationale', '')[:60]}...",
                "📚",
                {"strategy": strategy, "cognitive_load": rich_context.cognitive_load.level}
            )
            
            print(f"[WORKFLOW] 🎯 Content Strategy: {strategy.get('content_length')} ({strategy.get('estimated_words')} words), {strategy.get('teaching_approach')}")
            
            return strategy
            
        except Exception as e:
            print(f"[WORKFLOW] Content strategy with rich context failed: {e}")
            return {
                "content_length": "standard",
                "teaching_approach": "balanced",
                "detail_level": "balanced",
                "estimated_words": 500,
                "rationale": "Fallback strategy"
            }
    
    async def _generate_adaptive_exercises_with_rich_context(
        self,
        segment: Dict,
        rich_context: RichStudentContext,
        content_strategy: Dict = None
    ) -> str:
        """
        🎯 IMPROVED: Generate exercises using rich context
        Cross-agent aware: Knows what content strategy was chosen
        """
        try:
            from app.core.agent_thoughts import thoughts_tracker
            
            # Get focused context for exercise generation
            context_str = rich_context.to_llm_context(focus="exercise_generation")
            
            segment_title = segment.get("title", "Unknown")
            difficulty = segment.get("difficulty", "medium")
            
            # Cross-agent awareness
            content_approach = "balanced"
            if content_strategy:
                content_approach = content_strategy.get("teaching_approach", "balanced")
            
            prompt = f"""Design exercises for "{segment_title}" based on student needs.

{context_str}

SEGMENT: {segment_title} (Difficulty: {difficulty})

CROSS-AGENT CONTEXT:
- Content approach chosen: {content_approach}
  (Exercises should align with content teaching style)

Generate 0-3 exercises that:
1. **Target Weak Areas**: Practice exactly what they struggle with
2. **Progress Logically**: Easy → Medium → Hard
3. **Match Cognitive Load**: If tired, fewer/easier exercises
4. **Align with Content**: If content is example-driven, exercises reinforce those examples

Return ONLY valid JSON (no code fences):
{{
  "exercise_count": 2,
  "exercises": [
    {{
      "question": "specific question text",
      "difficulty": "easy|medium|hard",
      "type": "recall|application|analysis",
      "targets_weakness": "what this targets",
      "hint": "optional hint"
    }}
  ],
  "rationale": "Why this number and type of exercises"
}}

RULES:
- 0 exercises: No confusion + low cognitive load → skip practice
- 1 exercise: Minor confusion → one targeted question
- 2 exercises: Moderate confusion → easy + medium
- 3 exercises: High confusion → easy + medium + hard progression
- If cognitive load is heavy → max 1-2 exercises (don't overwhelm)
- If weak areas identified → target those specifically"""
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import json, re
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            if not response_text.startswith('{'):
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            exercise_strategy = json.loads(response_text)
            
            exercise_count = exercise_strategy.get("exercise_count", 1)
            exercises = exercise_strategy.get("exercises", [])
            rationale = exercise_strategy.get("rationale", "")
            
            # Log decision
            thoughts_tracker.add(
                "Exercise Generator (Rich Context)",
                f"Generating {exercise_count} exercises: {rationale}",
                "✍️",
                {"count": exercise_count, "cognitive_load": rich_context.cognitive_load.level}
            )
            
            print(f"[WORKFLOW] 🎯 Exercise Strategy: {exercise_count} exercises - {rationale}")
            
            # Build instruction
            if exercise_count == 0:
                return ""
            else:
                exercise_instruction = f"""
🤖 EXERCISE STRATEGY (AI-Generated with Rich Context):
Generate exactly {exercise_count} exercise(s) following these specifications:
{json.dumps(exercises, indent=2)}

Rationale: {rationale}

Create exercises that match the difficulty, type, and purpose specified above.
Target the student's specific weak areas identified in context.
"""
                return exercise_instruction
            
        except Exception as e:
            print(f"[WORKFLOW] Exercise generation with rich context failed: {e}")
            return ""
    
    async def _generate_adaptive_exercises_agentic(
        self,
        segment: Dict,
        student_context: Dict,
        confusion_areas: List[str]
    ) -> str:
        """
        Use LLM to decide exercise generation strategy.
        
        Returns: Exercise instruction string for the segment teaching prompt
        """
        try:
            from app.core.agent_thoughts import thoughts_tracker
            
            segment_title = segment.get("title", "Unknown")
            difficulty = segment.get("difficulty", "medium")
            
            confusion_str = "No specific confusion detected"
            if confusion_areas:
                confusion_str = f"Student confused about: {', '.join(confusion_areas[:2])}"
            
            weak_topics = student_context.get("weak_topics", [])
            recent_questions = student_context.get("recent_questions", [])
            
            engagement_str = "Moderate engagement"
            if len(recent_questions) > 2:
                engagement_str = "High engagement - asking many questions"
            elif len(recent_questions) == 0:
                engagement_str = "Low engagement - not asking questions"
            
            prompt = f"""Design exercises for "{segment_title}" based on student needs.

SEGMENT: {segment_title} (Difficulty: {difficulty})

STUDENT CONTEXT:
- {confusion_str}
- Weak Topics: {weak_topics[:2] if weak_topics else "None identified"}
- {engagement_str}

Generate 0-3 exercises that:
1. Target their specific confusion areas (if any)
2. Progress logically (easier → harder)
3. Test different cognitive levels (recall → application → analysis)

Return ONLY valid JSON (no code fences):
{{
  "exercise_count": 2,
  "exercises": [
    {{
      "question": "specific question text",
      "difficulty": "easy|medium|hard",
      "type": "recall|application|analysis",
      "targets_confusion": "tokenization",
      "hint": "optional hint"
    }}
  ],
  "rationale": "Why this number and type of exercises"
}}

GUIDELINES:
- 0 exercises: Student mastered basics, no confusion → skip practice
- 1 exercise: Basic understanding, minor confusion → one targeted question
- 2 exercises: Some confusion → one easy + one medium
- 3 exercises: Major confusion → easy + medium + hard progression

QUESTION TYPES:
- recall: Test if they remember key facts/definitions
- application: Test if they can use the concept in new situations
- analysis: Test deeper understanding, connections, synthesis"""
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import json
            import re
            
            # Strip code fences
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Extract JSON
            if not response_text.startswith('{'):
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            exercise_strategy = json.loads(response_text)
            
            exercise_count = exercise_strategy.get("exercise_count", 1)
            exercises = exercise_strategy.get("exercises", [])
            rationale = exercise_strategy.get("rationale", "")
            
            # Log the decision
            thoughts_tracker.add(
                "Exercise Generator (Agentic)",
                f"Generating {exercise_count} exercises: {rationale}",
                "✍️",
                {"count": exercise_count, "strategy": exercise_strategy}
            )
            
            print(f"[WORKFLOW] 🤖 Exercise Strategy: {exercise_count} exercises - {rationale}")
            
            # Build exercise instruction for the prompt
            if exercise_count == 0:
                return '  "exercises": [],'
            else:
                exercise_specs = []
                for ex in exercises:
                    spec = f'{{"question": "{ex.get("question", "")}", "difficulty": "{ex.get("difficulty", "medium")}", "type": "{ex.get("type", "application")}"}}'
                    exercise_specs.append(spec)
                
                exercise_instruction = f"""
🤖 EXERCISE STRATEGY (AI-Generated):
Generate exactly {exercise_count} exercise(s) following these specifications:
{json.dumps(exercises, indent=2)}

Rationale: {rationale}

Create exercises that match the difficulty, type, and purpose specified above.
"""
                return exercise_instruction
            
        except Exception as e:
            print(f"[WORKFLOW] Agentic exercise generation failed: {e}, using default")
            # Fallback to 1 medium exercise
            return ""
    
    async def _determine_content_strategy_agentic(
        self,
        segment: Dict,
        student_context: Dict,
        prior_segments: List[str]
    ) -> Dict:
        """
        Use LLM to decide HOW to teach this segment.
        
        Returns: { content_length, teaching_approach, detail_level, estimated_words, rationale }
        """
        try:
            from app.core.agent_thoughts import thoughts_tracker
            
            # Build context
            segment_title = segment.get("title", segment.get("segment_title", "Unknown"))
            difficulty = segment.get("difficulty", "medium")
            prerequisites = segment.get("prerequisites", [])
            objectives = segment.get("learning_objectives", [])
            
            prior_knowledge_str = "No prior segments"
            if prior_segments:
                prior_knowledge_str = f"Completed {len(prior_segments)} prior segments: {', '.join(prior_segments[:3])}"
            
            confusion_str = "No confusion detected"
            if student_context.get("confusion", []):
                confusion_str = f"Confused about: {', '.join(student_context['confusion'][:2])}"
            
            questions_str = "No questions asked"
            if student_context.get("recent_questions", []):
                questions_str = f"Asked {len(student_context['recent_questions'])} questions recently"
            
            learning_style = student_context.get("learning_style", "balanced")
            
            prompt = f"""You're planning how to teach "{segment_title}" to a student.

SEGMENT DETAILS:
- Difficulty: {difficulty}
- Prerequisites: {prerequisites or "None"}
- Learning Objectives: {objectives[:2] if objectives else "General understanding"}

STUDENT CONTEXT:
- Prior Knowledge: {prior_knowledge_str}
- {questions_str}
- {confusion_str}
- Preferred Learning Style: {learning_style}

Decide the optimal teaching strategy considering:
1. **Segment Complexity**: Easy concepts need brief explanation, hard concepts need depth
2. **Student's Prior Knowledge**: Build on what they know vs start from scratch
3. **Confusion Signals**: If confused, use more examples and analogies
4. **Learning Style**: Adapt to their preference (practical vs conceptual)

Return ONLY valid JSON (no code fences):
{{
  "content_length": "brief|standard|comprehensive",
  "teaching_approach": "definition-first|example-driven|analogy-heavy|step-by-step",
  "detail_level": "high-level|balanced|deep-dive",
  "estimated_words": 400,
  "rationale": "Why this approach works for this student and segment"
}}

GUIDELINES:
- Brief (300-400 words): Easy topics, strong prerequisites, no confusion
- Standard (400-600 words): Medium complexity, some confusion, balanced approach
- Comprehensive (600-800 words): Hard topics, weak prerequisites, lots of confusion

Teaching Approaches:
- definition-first: Start with clear definition, then examples (good for concepts)
- example-driven: Lead with concrete examples (good for practical learners)
- analogy-heavy: Use multiple analogies (good when confused)
- step-by-step: Break down into numbered steps (good for processes)"""
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import json
            import re
            
            # Strip code fences
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Extract JSON
            if not response_text.startswith('{'):
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            strategy = json.loads(response_text)
            
            # Log the agentic decision
            thoughts_tracker.add(
                "Content Strategist (Agentic)",
                f"{strategy.get('content_length', 'standard')} content, {strategy.get('teaching_approach', 'balanced')} approach: {strategy.get('rationale', '')[:60]}...",
                "📚",
                {"strategy": strategy}
            )
            
            print(f"[WORKFLOW] 🤖 Content Strategy: {strategy.get('content_length')} ({strategy.get('estimated_words')} words), {strategy.get('teaching_approach')} approach")
            
            return strategy
            
        except Exception as e:
            print(f"[WORKFLOW] Agentic content strategy failed: {e}, using default")
            # Fallback to balanced approach
            return {
                "content_length": "standard",
                "teaching_approach": "balanced",
                "detail_level": "balanced",
                "estimated_words": 500,
                "rationale": "Default balanced approach"
        }
    
    async def _execute_study_segment(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute study segment step - focused teaching of one segment"""
        topic = step["topic"]
        segment_id = step.get("segment_id")
        segment_title = step.get("segment_title") or step.get("title")
        # If missing, fill from memory's next segment
        if (not segment_id or not segment_title) and self.memory:
            try:
                next_seg = self.memory.get_next_segment(topic)
                if next_seg:
                    segment_id = segment_id or next_seg.get("segment_id", "seg_1")
                    segment_title = segment_title or next_seg.get("title", "Segment")
                    step["segment_id"] = segment_id
                    step["segment_title"] = segment_title
                    print(f"[WORKFLOW] Filled missing segment from memory: id={segment_id} title={segment_title}")
            except Exception as e:
                print(f"[WORKFLOW] Could not fill missing segment: {e}")
        # Final fallback
        segment_id = segment_id or "seg_1"
        segment_title = segment_title or "Segment"
        
        print(f"[WORKFLOW] Executing study_segment for: {topic} - {segment_title}")
        
        # Update current topic in state
        self.current_state.current_topic = topic
        
        # Get segment details from concept
        segment_info = None
        for concept in self.current_state.concepts:
            if concept.get("concept_id") == concept_id:
                segments = concept.get("learning_segments", [])
                segment_info = next((s for s in segments if s.get("segment_id") == segment_id), None)
                break
        
        if not segment_info:
            print(f"[WORKFLOW] Segment {segment_id} not found, falling back to topic teaching")
            return await self._execute_study_topic(step, concept_id)
        
        # Get segment-specific search queries
        concept_queries = None
        for concept in self.current_state.concepts:
            if concept.get("concept_id") == concept_id:
                concept_queries = concept.get("search_queries", [])
                break
        
        # Use LLM queries if available, otherwise generate heuristic queries
        if concept_queries and len(concept_queries) > 0:
            queries = concept_queries[:4]  # Use up to 4 LLM-generated queries
            print(f"[WORKFLOW] Using {len(queries)} LLM-generated search queries for segment")
        else:
            # Generate segment-specific queries
            queries = [
                f"{topic} {segment_title}",
                f"{segment_title} explanation examples",
                f"{topic} {segment_title} key concepts"
            ]
            # Add objectives-based query terms to strengthen retrieval
            objectives = segment_info.get('learning_objectives', [])
            if isinstance(objectives, list) and objectives:
                primary_obj = objectives[0]
                if isinstance(primary_obj, str) and primary_obj.strip():
                    queries.append(f"{topic} {segment_title} {primary_obj}")
                if len(objectives) > 1 and isinstance(objectives[1], str):
                    queries.append(f"{segment_title} {objectives[1]}")
            print(f"[WORKFLOW] Using {len(queries)} heuristic search queries for segment")
        
        # Retrieve with all queries and deduplicate
        all_documents = []
        all_metadatas = []
        seen_texts = set()
        
        for query in queries:
            # Use hybrid retrieval (vector + BM25). Falls back to vector if BM25 unavailable.
            results = vectorstore.query_hybrid(query, k=5, alpha=0.6)
            docs = results.get('documents', [[]])[0] if 'documents' in results else []
            metas = results.get('metadatas', [[]])[0] if 'metadatas' in results else []
            
            for doc, meta in zip(docs, metas):
                if doc not in seen_texts:
                    seen_texts.add(doc)
                    all_documents.append(doc)
                    all_metadatas.append(meta)
        
        documents = all_documents
        print(f"[WORKFLOW] Retrieved {len(documents)} unique documents for segment {segment_title}")
        
        # Log retrieved chunks for evaluation
        try:
            logger = get_logger()
            preview = [
                {
                    "text_preview": (doc[:220] + "...") if isinstance(doc, str) and len(doc) > 220 else doc,
                    "metadata": meta
                }
                for doc, meta in zip(all_documents[:10], all_metadatas[:10])
            ]
            logger.log_data("retrieved_chunks", {
                "topic": topic,
                "segment": segment_title,
                "queries": queries,
                "count": len(documents),
                "samples": preview
            })
        except Exception:
            pass
        
        if not documents or len(documents) == 0:
            error_msg = f"I couldn't find detailed content about {segment_title} in the uploaded document. Let me explain based on general knowledge."
            return {
                "success": True,
                "step_type": "study_segment",
                "topic": topic,
                "segment_id": segment_id,
                "segment_title": segment_title,
                "study_content": "",
                "content": error_msg,
                "instructions": "Ask me specific questions about this segment!",
                "next_action": "ask_questions_or_continue"
            }
        
        # Format study material with proper context (use more chunks for better context)
        def clean_snippet(s: str) -> str:
            import re
            lines = [ln.strip() for ln in s.splitlines()]
            cleaned = []
            for ln in lines:
                # Skip very short heading-like lines
                if len(ln) < 25 and (ln.lower().startswith('section:') or ln.istitle()):
                    continue
                cleaned.append(ln)
            text = ' '.join(cleaned)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        cleaned_docs = [clean_snippet(d) for d in documents[:6] if isinstance(d, str) and d.strip()]
        # Drop snippets that look like headings-only lines
        cleaned_docs = [d for d in cleaned_docs if not d.lower().startswith('section:') or len(d.split()) > 25]
        study_content = "\n\n---\n\n".join([d for d in cleaned_docs if len(d) > 120])
        
        # Generate segment-specific explanation using LLM
        print(f"[WORKFLOW] Generating explanation for segment: {segment_title}")
        recent_summaries = []
        if self.memory:
            try:
                recent_summaries = self.memory.get_last_session_summaries(k=3)
            except Exception:
                recent_summaries = []
        
        # Get exam context for tailored teaching
        exam_context = {}
        exam_style_instruction = ""
        if self.memory:
            try:
                exam_context = self.memory.get_exam_context()
                exam_type = exam_context.get("exam_type", "general")
                
                # Create exam-specific teaching instructions
                if exam_type == "JEE":
                    exam_style_instruction = "\n**EXAM FOCUS: JEE (Technical Depth)**\n- Emphasize mathematical rigor and derivations\n- Include numerical problem-solving approaches\n- Focus on concept application in physics/chemistry/math\n- Use technical terminology precisely\n"
                elif exam_type == "UPSC":
                    exam_style_instruction = "\n**EXAM FOCUS: UPSC (Broad Coverage)**\n- Provide comprehensive conceptual understanding\n- Include real-world applications and current affairs connections\n- Cover breadth of topic with multiple perspectives\n- Use clear, articulate language suitable for essays\n"
                elif exam_type in ["SAT", "GRE"]:
                    exam_style_instruction = f"\n**EXAM FOCUS: {exam_type}**\n- Balance conceptual clarity with problem-solving\n- Include standardized test-style examples\n- Focus on reasoning and analytical skills\n"
                elif exam_type != "general":
                    exam_style_instruction = f"\n**EXAM FOCUS: {exam_type}**\n- Tailor explanations for {exam_type} preparation\n- Include exam-relevant examples and applications\n"
            except Exception as e:
                print(f"[WORKFLOW] Could not get exam context: {e}")
        
        # OPTIMIZATION: Use cached engagement profile (computed once per request)
        engagement_profile = self._get_engagement_profile_cached(topic)
        complexity_instruction = ""  # Removed difficulty-based complexity adjustment
        
        # Check for agentic strategy (AI-generated) OR old remediation strategy
        agentic_strategy = step.get("agentic_strategy", None)
        remediation_strategy = step.get("remediation_strategy", None) if not agentic_strategy else None
        teaching_strategy = ""
        custom_prompt_override = None  # Will override the standard prompt if set
        
        # Track teaching decision
        from app.core.agent_thoughts import thoughts_tracker
        
        if agentic_strategy:
            # 🧠 AGENTIC REMEDIATION: Use AI-generated custom strategy
            approach_name = agentic_strategy.get("approach_name", "Custom Approach")
            custom_prompt = agentic_strategy.get("custom_prompt", "")
            teaching_hook = agentic_strategy.get("teaching_hook", "")
            diagnosis = agentic_strategy.get("diagnosis", {})
            root_cause = diagnosis.get("root_cause", "addressing confusion")
            
            # Use the custom prompt as the primary teaching instruction
            custom_prompt_override = custom_prompt
            teaching_strategy = f"\n**🧠 AGENTIC STRATEGY: {approach_name}**\n{teaching_hook}\n"
            
            thoughts_tracker.add("Workflow", f"Using AI-generated strategy '{approach_name}' for {segment_id}", "🧠", {
                "approach": approach_name,
                "segment": segment_id,
                "root_cause": root_cause[:100]
            })
            
            print(f"[WORKFLOW] 🧠 Using agentic strategy '{approach_name}' for {segment_id}")
            print(f"[WORKFLOW] Root cause: {root_cause[:100]}")
            
        elif remediation_strategy:
            # FALLBACK: Old hardcoded remediation (for backward compatibility)
            if remediation_strategy == "fundamentals":
                teaching_strategy = """
**🔄 REMEDIATION: FUNDAMENTALS** - Student showed fundamental misunderstanding.
- Start from absolute basics, assume NO prior knowledge of this segment
- Use VERY simple language and concrete everyday examples
- Break down each concept into smallest possible pieces
- Provide multiple analogies from everyday life
- Build strong foundation before adding complexity"""
                thoughts_tracker.add("Workflow", f"Re-teaching {segment_id} from FUNDAMENTALS - student showed basic misunderstanding", "🔄", {
                    "strategy": "fundamentals",
                    "segment": segment_id
                })
            elif remediation_strategy == "examples":
                teaching_strategy = """
**🔄 REMEDIATION: EXAMPLES-FOCUSED** - Student needs concrete illustrations.
- Lead with 3+ DETAILED real-world examples
- Show step-by-step walkthroughs with explanations
- Use visual descriptions and relatable analogies
- Connect abstract concepts to tangible scenarios
- Less theory, more practical demonstration"""
                thoughts_tracker.add("Workflow", f"Re-teaching {segment_id} with EXAMPLES - student needs concrete illustrations", "🔄", {
                    "strategy": "examples",
                    "segment": segment_id
                })
            elif remediation_strategy == "practice":
                teaching_strategy = """
**🔄 REMEDIATION: PRACTICE-ORIENTED** - Student has partial understanding.
- Quick concept refresh (1-2 paragraphs max)
- Focus on application and problem-solving
- Provide guided practice scenarios with solutions
- Address common mistakes and pitfalls explicitly
- Build confidence through practical mastery"""
                thoughts_tracker.add("Workflow", f"Re-teaching {segment_id} with PRACTICE - student has partial understanding", "🔄", {
                    "strategy": "practice",
                    "segment": segment_id
                })
            
            print(f"[WORKFLOW] 🔄 Using fallback remediation strategy: {remediation_strategy} for {segment_id}")
        else:
            # Normal teaching strategy based on engagement
            if engagement_profile:
                style = engagement_profile.get('preferred_learning_style', 'balanced')
                needs = engagement_profile.get('needs', [])
                
                if style == "practical/applied":
                    teaching_strategy = "\n**TEACHING STRATEGY: Practical Focus**\n- Lead with real-world examples and applications\n- Show concrete how-to steps\n- Include hands-on practice suggestions\n"
                    thoughts_tracker.add("Workflow", f"Teaching {segment_id} with PRACTICAL focus - student prefers applied learning", "📚", {
                        "style": "practical",
                        "segment": segment_id
                    })
                elif style == "conceptual/theoretical":
                    teaching_strategy = "\n**TEACHING STRATEGY: Conceptual Depth**\n- Start with underlying principles and 'why'\n- Build logical connections between ideas\n- Emphasize theoretical framework\n"
                    thoughts_tracker.add("Workflow", f"Teaching {segment_id} with CONCEPTUAL depth - student prefers theory", "📚", {
                        "style": "conceptual",
                        "segment": segment_id
                    })
                else:
                    thoughts_tracker.add("Workflow", f"Teaching {segment_id} with BALANCED approach", "📚", {
                        "style": "balanced",
                        "segment": segment_id
                    })
                
                if needs:
                    teaching_strategy += f"**STUDENT NEEDS:** {'; '.join(needs[:2])}\n"
        
        # 🎯 RICH CONTEXT: Build once, use for ALL decisions
        rich_context = None
        if not custom_prompt_override and not remediation_strategy:
            try:
                # Build rich context if not already built
                rich_context = RichStudentContext(
                    session_id=self.current_state.session_id,
                    topic=topic,
                    segment_id=segment_id
                )
                await rich_context.build_from_memory(self.memory)
                print(f"[WORKFLOW] 🎯 Rich context built: cognitive_load={rich_context.cognitive_load.level}, trajectory={rich_context.performance.quiz_trajectory}")
            except Exception as e:
                print(f"[WORKFLOW] Rich context build failed: {e}, using fallback")
        
        # 🤖 AGENTIC: Determine content strategy with rich context
        content_strategy = None
        if rich_context:
            try:
                # Get segment details
                segment_for_strategy = {
                    "title": segment_title,
                    "difficulty": step.get("difficulty", "medium"),
                    "prerequisites": segment_info.get("prerequisites", []) if segment_info else [],
                    "learning_objectives": segment_info.get("learning_objectives", []) if segment_info else []
                }
                
                # Get prior segments
                prior_segment_names = []
                if recent_summaries:
                    for summ in recent_summaries:
                        if isinstance(summ, dict):
                            seg_id = summ.get('segment_id', '')
                            if seg_id:
                                prior_segment_names.append(seg_id)
                
                # Call agentic method with rich context
                content_strategy = await self._determine_content_strategy_with_rich_context(
                    segment=segment_for_strategy,
                    rich_context=rich_context,
                    prior_segments=prior_segment_names
                )
                
                # Store decision for cross-agent awareness
                rich_context.add_decision("content_strategy", content_strategy)
                
            except Exception as e:
                print(f"[WORKFLOW] Content strategy decision failed: {e}")
        
        # 🤖 AGENTIC: Determine exercise strategy with rich context
        exercise_instruction = ""
        if rich_context:
            try:
                # Get content strategy decision (cross-agent awareness)
                content_decision = rich_context.get_decision("content_strategy")
                
                # Call agentic exercise method with rich context
                exercise_instruction = await self._generate_adaptive_exercises_with_rich_context(
                    segment=segment_for_strategy,
                    rich_context=rich_context,
                    content_strategy=content_decision
                )
                
                # Store decision for future reference
                rich_context.add_decision("exercise_strategy", exercise_instruction)
                
            except Exception as e:
                print(f"[WORKFLOW] Exercise strategy decision failed: {e}")
        
        # Build minimal prior context (segment names only)
        prior_context = ""
        if recent_summaries and len(recent_summaries) > 0:
            covered = []
            for summ in recent_summaries[-2:]:
                if isinstance(summ, dict):
                    seg_id = summ.get('segment_id', '')
                    if seg_id:
                        covered.append(seg_id)
            if covered:
                prior_context = f"(Previous segments: {', '.join(covered)})\n"
        
        # Use custom AI-generated prompt if available (agentic mode)
        if custom_prompt_override:
            # Extract targeted exercises from agentic strategy if available
            targeted_exercises_instruction = ""
            if agentic_strategy and agentic_strategy.get("targeted_exercises"):
                exercises_list = agentic_strategy.get("targeted_exercises", [])
                targeted_exercises_instruction = f"""

🎯 TARGETED EXERCISES (Pre-designed for this student's confusion):
Generate these specific exercises:
{json.dumps(exercises_list, indent=2)}

Each exercise was designed to address the specific confusion identified in diagnosis."""
            
            segment_prompt = f"""You are teaching "{segment_title}" to a student learning {topic}.

{prior_context}
CURRENT SEGMENT: {segment_title}

DOCUMENT CONTENT:
{study_content[:2000]}

🧠 CUSTOM TEACHING INSTRUCTION (AI-Generated Strategy):
{custom_prompt_override}
{targeted_exercises_instruction}
CRITICAL: Follow the custom instruction above precisely. Return ONLY a JSON object with these EXACT keys (no markdown, no code fences):
{{
  "content": "Your explanation following the custom instruction above. Minimum 300-500 words with examples and clarity as specified.",
  "summary": "One sentence summary capturing the key insight",
  "exercises": [
    {{"question": "Exercise question as specified above", "difficulty": "as specified", "type": "as specified"}}
  ],
  "memory_delta": "Brief summary of what was covered. 10-15 words"
}}

Start your response with {{ and end with }}. Nothing else!"""
        else:
            # 🤖 AGENTIC: Use content strategy to build adaptive prompt
            if content_strategy:
                content_length = content_strategy.get("content_length", "standard")
                teaching_approach = content_strategy.get("teaching_approach", "balanced")
                estimated_words = content_strategy.get("estimated_words", 500)
                rationale = content_strategy.get("rationale", "")
                
                # Map content length to word counts
                word_range_map = {
                    "brief": (300, 400),
                    "standard": (400, 600),
                    "comprehensive": (600, 800)
                }
                min_words, max_words = word_range_map.get(content_length, (400, 600))
                
                # Map teaching approach to instructions
                approach_instructions = {
                    "definition-first": "Start with a clear, precise definition. Then provide 2-3 concrete examples that illustrate the definition. Finally, explain why this concept matters.",
                    "example-driven": "Lead with a concrete, relatable example. Then extract the underlying concept from the example. Provide 1-2 more examples with increasing complexity.",
                    "analogy-heavy": "Begin with a powerful analogy to something familiar. Extend the analogy to cover key aspects. Then transition to the actual concept. Use 2-3 different analogies if needed.",
                    "step-by-step": "Break down into clear, numbered steps. For each step, explain what, why, and how. Provide an example after covering all steps."
                }
                approach_instruction = approach_instructions.get(teaching_approach, "Provide a balanced explanation with examples and clear structure.")
                
                content_instruction = f"""
🤖 ADAPTIVE TEACHING STRATEGY (AI-Generated for this student):
- Content Length: {content_length.upper()} ({estimated_words} words target)
- Teaching Approach: {teaching_approach.upper()}
- Rationale: {rationale}

TEACHING APPROACH INSTRUCTION:
{approach_instruction}
"""
            else:
                # Fallback to standard if no strategy
                min_words = 400
                max_words = 600
                content_instruction = ""
            
            # Standard prompt with agentic adaptations
            segment_prompt = f"""You are teaching "{segment_title}" to a student learning {topic}.

{prior_context}
CURRENT SEGMENT: {segment_title}
LEARNING OBJECTIVES:
{segment_info.get('learning_objectives', [])}
{exam_style_instruction}{complexity_instruction}{teaching_strategy}

DOCUMENT CONTENT:
{study_content[:2000]}
{content_instruction}
TEACHING PRIORITIES:
1. Focus ONLY on {segment_title} - no repetition of previous segments
2. Ground explanation in DOCUMENT CONTENT above
3. Follow the teaching approach instruction above
4. Pre-empt common confusions with clear examples
5. Target {min_words}-{max_words} words for content (prioritize quality over exact count)

PROACTIVE TEACHING:
- Anticipate where students typically struggle with {segment_title}
- Address potential confusion BEFORE it happens
- Use concrete examples that connect to prior knowledge
{exercise_instruction}
CRITICAL: Return ONLY a JSON object with these EXACT keys (no markdown, no code fences):
{{
  "content": "Your explanation of {segment_title} following the teaching approach above. Target {min_words}-{max_words} words with examples, analogies, and clear structure.",
  "summary": "One sentence summary capturing the key insight",
  "exercises": [
    {{"question": "Exercise question following the specifications above", "difficulty": "as specified", "type": "as specified"}}
  ],
  "memory_delta": "Brief summary of what was covered. 10-15 words"
}}

REMEMBER: Focus on quality explanation following the specified teaching approach.
Start your response with {{ and end with }}. Nothing else!"""
        
        # Initialize variables at the start
        parsed = None
        summary_text = ""
        memory_patch = {}
        exercises = []
        session_summary_delta = ""
        
        try:
            import asyncio
            start_time = time.time()
            segment_response = await asyncio.wait_for(
                self.llm.ainvoke(segment_prompt),
                timeout=60.0  # Increased to 60s for safety; simplified prompt should reduce actual time
            )
            duration = time.time() - start_time
            
            # Parse JSON response with robust extraction
            parsed = None
            try:
                content_raw = segment_response.content.strip()
                
                # Step 1: Remove markdown code fences
                if "```json" in content_raw:
                    start = content_raw.index("```json") + 7
                    end = content_raw.rindex("```")
                    content_raw = content_raw[start:end].strip()
                elif "```" in content_raw and content_raw.startswith("```"):
                    parts = content_raw.split("```")
                    if len(parts) >= 2:
                        content_raw = parts[1].strip()
                
                # Step 2: Extract JSON object (find first { to last })
                if '{' in content_raw and '}' in content_raw:
                    start_idx = content_raw.index('{')
                    end_idx = content_raw.rindex('}') + 1
                    content_raw = content_raw[start_idx:end_idx]
                
                # Step 3: Parse JSON
                parsed = json.loads(content_raw)
                print(f"[WORKFLOW] ✅ Successfully parsed JSON response")
                
            except json.JSONDecodeError as e:
                print(f"[WORKFLOW] ❌ JSON parse error: {e}")
                print(f"[WORKFLOW] Raw response (first 500 chars): {content_raw[:500] if 'content_raw' in locals() else 'N/A'}")
                parsed = None
            except Exception as e:
                print(f"[WORKFLOW] ❌ Parse error: {e}")
                parsed = None
            
            # Extract content from parsed JSON or use raw response
            if isinstance(parsed, dict) and (parsed.get("content") or parsed.get("full_text")):
                # Support both old and new field names
                segment_explanation = parsed.get("content") or parsed.get("full_text", "")
                summary_text = parsed.get("summary", "")
                session_summary_delta = parsed.get("memory_delta") or parsed.get("session_summary_delta", "")
                key_excerpts = parsed.get("key_excerpts", [])
                exercises = parsed.get("exercises") or parsed.get("student_facing_next_steps", [])
                
                # Log content metrics
                word_count = len(segment_explanation.split()) if segment_explanation else 0
                char_count = len(segment_explanation) if segment_explanation else 0
                print(f"[WORKFLOW] ✅ Parsed JSON:")
                print(f"  - Content: {word_count} words, {char_count} chars")
                print(f"  - Summary: {bool(summary_text)}")
                print(f"  - Memory delta: {bool(session_summary_delta)}")
                print(f"  - Exercises: {len(exercises)}")
                if exercises:
                    print(f"  - First exercise: {exercises[0]}")
                else:
                    print(f"  - WARNING: No exercises in parsed JSON!")
                    print(f"  - Parsed keys: {list(parsed.keys())}")
                
                # Warn if content is too                 
                # Store structured taught segment
                if self.memory:
                    try:
                        self.memory.store_taught_segment_json(
                            topic,
                            segment_id,
                            {
                                "full_text": segment_explanation,
                                "summary": summary_text,
                                "excerpts": key_excerpts,
                                "session_summary_delta": session_summary_delta,
                                "segment_id": segment_id,
                                "segment_title": segment_title,
                                "topic": topic
                            }
                        )
                        
                        # CRITICAL: Store session summary delta for continuity
                        if session_summary_delta:
                            self.memory.push_session_summary({
                                "segment_id": segment_id,
                                "topic": topic,
                                "summary": session_summary_delta,  # Use the concise delta
                                "timestamp": time.time()
                            }, k=3)
                            print(f"[WORKFLOW] 💾 Stored session summary: '{session_summary_delta}'")
                        
                    except Exception as e:
                        print(f"[WORKFLOW] Failed to store segment data: {e}")
                
                # Store for return
                memory_patch = {
                    "session_summary_delta": session_summary_delta,
                    "exercises": exercises
                }
            else:
                # Fallback: try to extract content even from malformed JSON
                print(f"[WORKFLOW] ⚠️ JSON parsing failed, attempting content extraction")
                raw_content = segment_response.content
                
                # Try to extract just the content field value using regex
                import re
                content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_content, re.DOTALL)
                if content_match:
                    # Found content field in JSON
                    segment_explanation = content_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                    print(f"[WORKFLOW] ✅ Extracted content field from malformed JSON")
                elif raw_content.strip().startswith('{'):
                    # It's JSON but we couldn't parse it - don't show JSON to user!
                    segment_explanation = f"I apologize, but I encountered an error generating the explanation for {segment_title}. Let me try to explain this concept briefly: {segment_info.get('learning_objectives', ['this topic'])[0] if segment_info.get('learning_objectives') else 'this important concept'}."
                    print(f"[WORKFLOW] ❌ Could not extract content from JSON, using error message")
                else:
                    # Use raw content if it's not JSON
                    segment_explanation = raw_content
                
                summary_text = f"{segment_title}: basic explanation"
                memory_patch = {}
                exercises = []
                parsed = None  # Ensure parsed is None for later checks
            
            # Log LLM call
            print(f"\n{'='*80}")
            print(f"🤖 LLM CALL - Generating Segment Explanation")
            print(f"{'='*80}")
            print(f"Topic: {topic}")
            print(f"Segment: {segment_title}")
            print(f"Duration: {duration:.2f}s")
            print(f"JSON Parsed: {'Yes' if parsed else 'No (using raw content)'}")
            print(f"Content Length: {len(segment_explanation)} chars")
            print(f"Response (first 200 chars): {segment_explanation[:200]}...")
            print(f"{'='*80}\n")
            
            print(f"[WORKFLOW] Segment explanation generated successfully")
            
        except asyncio.TimeoutError:
            print(f"[WORKFLOW] ⚠️ Segment explanation generation timed out (>45s), using fallback")
            segment_explanation = (
                f"**{segment_title}**\n\n"
                f"This segment focuses on key concepts in {topic}.\n\n"
                f"**Learning Objectives:**\n" +
                "\n".join([f"- {obj}" for obj in segment_info.get('learning_objectives', ['Understand this concept'])]) +
                f"\n\n**Key Points:**\n"
                f"- This topic is essential for understanding {topic}\n"
                f"- Focus on the core concepts and their applications\n"
                f"- Practice with examples to reinforce learning\n"
            )
            parsed = None
            summary_text = f"{segment_title} - Core concepts"
            
        except Exception as e:
            print(f"[WORKFLOW] ❌ Error generating segment explanation: {e}")
            import traceback
            traceback.print_exc()
            segment_explanation = (
                f"**{segment_title}**\n\n"
                f"Let's explore this topic step by step.\n\n"
                f"**Learning Objectives:**\n" +
                "\n".join([f"- {obj}" for obj in segment_info.get('learning_objectives', ['Understand this concept'])])
            )
            parsed = None
            summary_text = segment_title
        
        # Use normalized segment explanation
        full_explanation = self._extract_full_text_from_possible_json(segment_explanation)
        
        # STEP 1 FIX: Store taught content for quiz grounding (with segment info)
        segment_key = f"{topic}:{segment_id}"
        self.current_state.taught_content[segment_key] = full_explanation
        print(f"[WORKFLOW] Stored taught content for {segment_key} ({len(full_explanation)} chars)")
        
        # Store in Redis memory (OPTIMIZED: Single JSON storage, no duplication)
        if self.memory:
            # Build comprehensive taught segment object
            taught_json = {
                "full_text": full_explanation,
                "summary": summary_text,
                "session_summary_delta": session_summary_delta if 'session_summary_delta' in locals() else "",
                "topic": topic,
                "segment_id": segment_id,
            }
            # Add parsed data if available
            if parsed and isinstance(parsed, dict):
                if "key_excerpts" in parsed:
                    taught_json["excerpts"] = parsed["key_excerpts"]
                if "student_facing_next_steps" in parsed:
                    taught_json["exercises"] = parsed["student_facing_next_steps"]
            
            # SINGLE storage point - structured JSON only
            self.memory.store_taught_segment_json(topic, segment_id, taught_json)
            print(f"[WORKFLOW] 💾 Stored taught JSON for {segment_key} ({len(full_explanation)} chars)")
            
            self.memory.mark_segment_started(topic, segment_id)
            self.memory.update_context({
                "current_topic": topic,
                "current_segment": segment_id,
                "last_taught_content": f"{topic}:{segment_id}"
            })
            # Ensure last_k summary is pushed even if model omitted it
            try:
                if summary_text:
                    self.memory.push_session_summary({
                        "segment_id": segment_id,
                        "topic": topic,
                        "summary": summary_text,
                        "timestamp": time.time(),
                        "memory_delta": memory_patch.get("session_summary_delta") if isinstance(memory_patch, dict) else None
                    }, k=3)
                vectorstore.upsert_taught_summary(
                    _id=f"{self.current_state.doc_id}:{topic}:{segment_id}",
                    text=summary_text,
                    metadata={"doc_id": self.current_state.doc_id, "topic": topic, "segment_id": segment_id}
                )
            except Exception:
                pass
        
            # Mark this segment completed to allow planner to advance; quiz only after all segments
            try:
                from app.core.agent_thoughts import thoughts_tracker
                ok = self.memory.mark_segment_completed(topic, segment_id)
                if ok:
                    print(f"[WORKFLOW] ✅ Marked segment completed: {topic}/{segment_id}")
                    thoughts_tracker.add("Workflow", f"Marked {segment_id} as completed", "✅", {
                        "topic": topic,
                        "segment": segment_id
                    })
                    # Verify it was marked
                    progress = self.memory.get_topic_progress(topic) or {}
                    print(f"[WORKFLOW] Verification - {segment_id} status: {progress.get(segment_id)}")
                else:
                    print(f"[WORKFLOW] ⚠️ mark_segment_completed returned False for {topic}/{segment_id}")
            except Exception as e:
                print(f"[WORKFLOW] ❌ Failed to mark segment completed: {e}")

        # Log the study content in lesson history
        if self.current_state.doc_id:
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=full_explanation,
                sources=[f"Document: {self.current_state.doc_id}"]
            )
        
        print(f"[WORKFLOW] Successfully prepared segment content for {segment_title}")
        
        # 🤖 AGENTIC: Decide if NOW is the right time to quiz
        from app.core.agent_thoughts import thoughts_tracker
        
        try:
            # Gather context for agentic quiz timing decision
            segments_for_topic = []
            for i, plan_step in enumerate(self.current_state.study_plan):
                if plan_step.get("action") == "study_segment" and plan_step.get("topic") == topic:
                    segments_for_topic.append({
                        "index": i,
                        "segment_id": plan_step.get("segment_id"),
                        "is_current": i == self.current_state.current_step
                    })
            
            total_segments_in_plan = len(segments_for_topic)
            current_segment_index = next((i for i, s in enumerate(segments_for_topic) if s["is_current"]), -1)
            is_last_segment = (current_segment_index == total_segments_in_plan - 1)
            
            # Find segments learned since last quiz (guard against index drift)
            segments_since_last_quiz = []
            last_quiz_index = -1
            plan_len = len(self.current_state.study_plan or [])
            if plan_len > 0:
                start_index = min(self.current_state.current_step, plan_len - 1)
                for i in range(start_index, -1, -1):
                    step = self.current_state.study_plan[i]
                    if step.get("action") == "practice_quiz":
                        last_quiz_index = i
                        break
                    elif step.get("action") == "study_segment" and step.get("topic") == topic:
                        segments_since_last_quiz.insert(0, step.get("segment_id", "unknown"))
            
            # Estimate time since last quiz (rough estimate: 8 min per segment)
            time_since_quiz_min = len(segments_since_last_quiz) * 8
            
            # Determine confusion level
            confusion_level = "low"  # Default
            if self.memory:
                try:
                    recent_qa = self.memory.get_recent_qa(k=3)
                    memory_context = self.memory.get_context() or {}
                    unclear_segments = memory_context.get("last_unclear_segments", [])
                    
                    if len(unclear_segments) > 0:
                        confusion_level = "high"
                    elif len(recent_qa) > 2:
                        confusion_level = "moderate"
                except:
                    pass
            
            # Determine cognitive load
            cognitive_load = "light" if len(segments_since_last_quiz) <= 1 else "moderate" if len(segments_since_last_quiz) <= 3 else "heavy"
            
            print(f"[WORKFLOW] 📊 Quiz Timing Check for {topic}:")
            print(f"  - Segments since last quiz: {len(segments_since_last_quiz)} ({segments_since_last_quiz})")
            print(f"  - Est. time since quiz: {time_since_quiz_min} min")
            print(f"  - Confusion level: {confusion_level}")
            print(f"  - Cognitive load: {cognitive_load}")
            print(f"  - Is last segment: {is_last_segment}")
            
            # 🤖 Call agentic quiz timing method
            should_insert_quiz = False
            quiz_reasoning = "Default: no quiz"
            
            try:
                should_insert_quiz, quiz_reasoning = await self.planner_agent._decide_quiz_timing_agentic(
                    topic=topic,
                    segments_since_last_quiz=segments_since_last_quiz,
                    time_since_last_quiz_minutes=time_since_quiz_min,
                    confusion_level=confusion_level,
                    cognitive_load=cognitive_load
                )
            except Exception as e:
                print(f"[WORKFLOW] Agentic quiz timing failed: {e}, using fallback")
                # Fallback: quiz if last segment OR 3+ segments learned OR high confusion
                should_insert_quiz = is_last_segment or len(segments_since_last_quiz) >= 3 or confusion_level == "high"
                quiz_reasoning = f"Rule fallback: last={is_last_segment}, segments={len(segments_since_last_quiz)}, confusion={confusion_level}"
            
            if should_insert_quiz and total_segments_in_plan > 0:
                thoughts_tracker.add("Workflow", f"✅ INSERTING QUIZ for {topic}: {quiz_reasoning}", "🎯", {
                    "topic": topic,
                    "reasoning": quiz_reasoning,
                    "segments_learned": len(segments_since_last_quiz),
                    "confusion": confusion_level
                })
                
                quiz_index = self.current_state.current_step + 1  # Right after current step
                already_has_quiz = False
                if quiz_index < len(self.current_state.study_plan):
                    nxt = self.current_state.study_plan[quiz_index]
                    already_has_quiz = nxt.get("action") == "practice_quiz" and nxt.get("topic") == topic
                
                if not already_has_quiz:
                    self.current_state.study_plan.insert(quiz_index, {
                        "step_id": f"quiz_{topic}_{int(time.time())}",
                        "action": "practice_quiz",
                        "topic": topic,
                        "concept_id": concept_id,
                        "difficulty": "medium",
                        "est_minutes": 6,
                        "why_assigned": f"✅ Quiz for {topic} - all {total_segments_in_plan} segments complete!"
                    })
                    print(f"[WORKFLOW] ✅✅✅ QUIZ INSERTED at index {quiz_index} for {topic}")
                    print(f"[WORKFLOW] Study plan now has {len(self.current_state.study_plan)} steps")
                else:
                    print(f"[WORKFLOW] Quiz already exists at index {quiz_index}")
            else:
                thoughts_tracker.add("Workflow", f"Not last segment for {topic} ({current_segment_index + 1}/{total_segments_in_plan})", "⏳", {
                    "topic": topic,
                    "current": current_segment_index + 1,
                    "total": total_segments_in_plan
                })
                print(f"[WORKFLOW] Not inserting quiz - not the last segment")
                
        except Exception as e:
            print(f"[WORKFLOW] ❌ Error in quiz insertion logic: {e}")
            import traceback
            traceback.print_exc()

        # Don't add text-based difficulty request - frontend will show buttons instead

        # Get planner reasoning (why this step was assigned)
        planner_reason = step.get("why_assigned", "")

        return {
            "success": True,
            "step_type": "study_segment",
            "topic": topic,
            "segment_id": segment_id,
            "segment_title": segment_title,
            "study_content": study_content,
            "content": full_explanation,
            "exercises": exercises,
            "memory_patch": memory_patch,
            "planner_reason": planner_reason,
            "instructions": f"Feel free to ask me any questions about {segment_title}!",
            "next_action": "ask_questions_or_continue"
        }
    
    async def _execute_practice_quiz(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute practice quiz step with notification"""
        topic = step["topic"]
        print(f"[WORKFLOW] Executing practice_quiz for: {topic}")
        
        # Update current topic
        self.current_state.current_topic = topic
        
        # Topic-level quiz: aggregate taught content across all segments under the topic
        segment_id = step.get("segment_id")  # optional
        taught_content = ""

        if self.memory:
            # Prefer structured JSONs, concatenate full_texts in order (most recent first reasonable for MVP)
            taught_jsons = self.memory.get_taught_segments_for_topic(topic)
            if taught_jsons:
                texts = [j.get("full_text", "") for j in taught_jsons if j.get("full_text")]
                taught_content = "\n\n---\n\n".join(texts)
        # Fallbacks
        if not taught_content:
            if segment_id:
                segment_key = f"{topic}:{segment_id}"
                taught_content = self.current_state.taught_content.get(segment_key, "") or (
                    self.memory.get_taught_segment(topic, segment_id) if self.memory else ""
                )
            else:
                taught_content = self.current_state.taught_content.get(topic, "") or (
                    self.memory.get_taught_segment(topic, "full_topic") if self.memory else ""
                )
        print(f"[WORKFLOW] Using {len(taught_content)} chars of taught content for topic quiz grounding")
        
        # 🤖 RICH CONTEXT: Build unified student context ONCE
        quiz_difficulty = "medium"  # Default fallback
        include_subjective = False
        
        try:
            # Build rich context (single memory fetch, structured output)
            rich_context = RichStudentContext(
                session_id=self.current_state.session_id,
                topic=topic,
                segment_id=concept_id
            )
            await rich_context.build_from_memory(self.memory)
            
            # Call agentic method with rich context (much cleaner!)
            quiz_difficulty, include_subjective, reasoning = await self._determine_quiz_difficulty_with_rich_context(
                topic=topic,
                rich_context=rich_context
            )
            
            print(f"[WORKFLOW] 🎯 Quiz difficulty: {quiz_difficulty}, subjective: {include_subjective}")
            print(f"[WORKFLOW] Reasoning: {reasoning}")
            
        except Exception as e:
            print(f"[WORKFLOW] Rich context quiz difficulty failed: {e}, using default medium")
            import traceback
            traceback.print_exc()
        
        # Get exam context for quiz style
        exam_context = {}
        if self.memory:
            try:
                exam_context = self.memory.get_exam_context()
            except Exception:
                pass
        
        # ENHANCED: Gather student context for better quiz questions
        student_context = {
            "recent_questions": [],
            "exercise_responses": [],
            "unclear_segments": [],
            "confusion_areas": []
        }
        
        if self.memory:
            try:
                # Get recent questions/doubts from student
                recent_qa = self.memory.get_recent_qa(k=5)
                if recent_qa:
                    student_context["recent_questions"] = [
                        qa.get("q", "") for qa in recent_qa if qa.get("q")
                    ][:3]  # Top 3 most recent questions
                
                # Get exercise assessment data to identify weak areas
                segment_plan = self.memory.get_segment_plan(topic) or []
                for seg in segment_plan:
                    seg_id = seg.get("segment_id")
                    if seg_id:
                        # Check for exercise assessments
                        assessment_key = f"exercise_assessment:{topic}:{seg_id}"
                        try:
                            assessment = self.memory.redis.get(assessment_key)
                            if assessment:
                                import json
                                assessment_data = json.loads(assessment)
                                if assessment_data.get("misconceptions"):
                                    student_context["confusion_areas"].extend(
                                        assessment_data["misconceptions"][:2]
                                    )
                        except:
                            pass
                
                # Get unclear segments from recent quizzes
                recent_quizzes = self.memory.get_recent_quiz_results(topic, k=2)
                for quiz_res in recent_quizzes:
                    unclear = quiz_res.get("unclear_segments", [])
                    student_context["unclear_segments"].extend(unclear[:2])
                
                # Remove duplicates
                student_context["unclear_segments"] = list(set(student_context["unclear_segments"]))[:3]
                student_context["confusion_areas"] = list(set(student_context["confusion_areas"]))[:3]
                
                print(f"[WORKFLOW] Student context: {len(student_context['recent_questions'])} questions, "
                      f"{len(student_context['unclear_segments'])} unclear segments, "
                      f"{len(student_context['confusion_areas'])} confusion areas")
            except Exception as e:
                print(f"[WORKFLOW] Error gathering student context: {e}")
        
        # Generate practice quiz with 5 questions
        print(f"[WORKFLOW] Preparing to call tutor_evaluator.generate_quiz | difficulty={quiz_difficulty}, subjective={include_subjective}")
        quiz_result = await self.tutor_evaluator.execute(
            "generate_quiz",
            topic=topic,
            concept_id=concept_id,
            difficulty=quiz_difficulty,  # Dynamic difficulty
            num_questions=5,  # ENHANCED: 5 questions instead of 3
            taught_content=taught_content,  # Pass what was taught
            segment_id=segment_id,  # Pass segment info
            exam_context=exam_context,  # Pass exam context for question style
            student_context=student_context,  # Pass student questions, doubts, confusion areas
            include_subjective=include_subjective  # NEW: Include subjective questions based on performance
        )
        
        if not quiz_result.success:
            return {
                "success": False,
                "error": "Quiz generation failed",
                "details": quiz_result.reasoning
            }
        
        quiz = quiz_result.data
        # Diagnostics: validate quiz structure
        try:
            if not isinstance(quiz, dict):
                raise ValueError(f"Quiz is not a dict: {type(quiz)}")
            if "questions" not in quiz or not isinstance(quiz["questions"], list):
                raise ValueError("Quiz missing 'questions' array")
            print(f"[WORKFLOW] Quiz OK | questions={len(quiz['questions'])} | has_intro={bool(quiz.get('intro'))}")
            if quiz.get('questions'):
                q0 = quiz['questions'][0]
                print(f"[WORKFLOW] Q1 type={q0.get('type')} | options={len(q0.get('options', [])) if isinstance(q0.get('options'), list) else 'na'}")
        except Exception as e:
            print(f"[WORKFLOW] Quiz validation error: {e}")
        self.current_state.current_quiz = quiz
        
        # Generate quiz notification message
        quiz_notification = f"📝 **Quiz Time: {topic}**\n\nI've prepared {len(quiz['questions'])} practice questions to help you master this topic. Take your time and think through each one carefully.\n\nClick the quiz button below to begin!"
        
        # Log quiz notification in lesson history
        if self.current_state.doc_id:
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=quiz_notification,
                sources=[]
            )
        
        # Final safeguard: ensure quiz is JSON-serializable
        try:
            import json
            json.dumps(quiz)
        except Exception as e:
            print(f"[WORKFLOW] Quiz not JSON-serializable: {e}; attempting to sanitize")
            try:
                def to_jsonable(x):
                    if isinstance(x, (str, int, float, bool)) or x is None:
                        return x
                    if isinstance(x, list):
                        return [to_jsonable(i) for i in x]
                    if isinstance(x, dict):
                        return {str(k): to_jsonable(v) for k, v in x.items()}
                    # Fallback: string cast
                    return str(x)
                quiz = to_jsonable(quiz)
            except Exception as e2:
                print(f"[WORKFLOW] Quiz sanitize failed: {e2}; returning empty quiz")
                quiz = {"questions": []}

        return {
            "success": True,
            "step_type": "practice_quiz",
            "quiz": quiz,
            "content": quiz_notification,  # This will be displayed in chat
            "instructions": f"Practice {topic} with these {len(quiz['questions'])} questions. Take your time and think through each one.",
            "next_action": "submit_answers"
        }
    
    async def evaluate_exercise_answers_async(self, topic: str, segment_id: str, exercise_answers: List[str], 
                                             taught_context: str = None) -> None:
        """
        Asynchronously evaluate exercise answers and store insights in memory.
        OPTIMIZATION: Accepts pre-fetched context to avoid redundant Redis calls.
        """
        try:
            # Filter out empty answers
            answered = [a for a in exercise_answers if a and a.strip()]
            if not answered:
                print(f"[WORKFLOW] No exercise answers to evaluate for {segment_id}")
                return
            
            print(f"[WORKFLOW] 🔄 Evaluating {len(answered)} exercise answers for {segment_id} (async)")
            
            # OPTIMIZATION: Reuse context if provided, otherwise fetch
            if taught_context:
                context = taught_context[:1500]
                print(f"[WORKFLOW] ⚡ Reusing provided context ({len(context)} chars)")
            else:
                taught_json = self.memory.get_taught_segment_json(topic, segment_id) if self.memory else None
                context = taught_json.get("full_text", "")[:1500] if taught_json else ""
            
            # Enhanced evaluation prompt with performance tracking
            prompt = f"""You are evaluating a student's exercise answers for {segment_id}.

TAUGHT CONTENT:
{context}

STUDENT'S ANSWERS (Easy → Medium → Hard):
{chr(10).join(f'{i+1}. {a}' for i, a in enumerate(answered))}

Provide a JSON assessment:
{{
  "overall_performance": "excellent|good|fair|weak",
  "strengths": "brief strength summary",
  "areas_to_improve": "brief improvement areas",
  "difficulty_mastery": {{
    "easy": true/false,
    "medium": true/false,
    "hard": true/false
  }},
  "recommended_quiz_difficulty": "easy|medium|hard|mixed"
}}

Base your assessment on answer quality and depth of understanding."""
            
            response = await self.llm.ainvoke(prompt)
            assessment_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON assessment
            import json
            try:
                # Try to extract JSON
                if '{' in assessment_text:
                    json_start = assessment_text.index('{')
                    json_end = assessment_text.rindex('}') + 1
                    assessment_json = json.loads(assessment_text[json_start:json_end])
                else:
                    assessment_json = {"overall_performance": "fair", "recommended_quiz_difficulty": "medium"}
            except:
                assessment_json = {"overall_performance": "fair", "recommended_quiz_difficulty": "medium"}
            
            # Store in memory as a Q&A entry with performance data
            if self.memory:
                self.memory.store_qa(
                    question=f"Exercise: {segment_id}",
                    answer=json.dumps(assessment_json),
                    topic=topic,
                    metadata={
                        "type": "exercise_assessment", 
                        "segment_id": segment_id,
                        "performance": assessment_json.get("overall_performance"),
                        "recommended_difficulty": assessment_json.get("recommended_quiz_difficulty")
                    }
                )
                print(f"[WORKFLOW] ✅ Exercise assessment stored: {assessment_json.get('overall_performance')} → {assessment_json.get('recommended_quiz_difficulty')} quiz")
                
                # Track exercise evaluation
                from app.core.agent_thoughts import thoughts_tracker
                performance = assessment_json.get('overall_performance', 'fair')
                rec_diff = assessment_json.get('recommended_quiz_difficulty', 'medium')
                thoughts_tracker.add("Exercise Evaluator", f"Student's {segment_id} exercises: {performance} performance → recommends {rec_diff} quiz", "✍️", {
                    "performance": performance,
                    "recommended_difficulty": rec_diff,
                    "segment": segment_id
                })
            
        except Exception as e:
            print(f"[WORKFLOW] Error evaluating exercises: {e}")
    
    async def _execute_optional_exercise(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute an optional exercise after a segment - student can answer or skip"""
        topic = step["topic"]
        segment_id = step.get("segment_id")
        print(f"[WORKFLOW] Executing optional exercise for: {topic} / {segment_id}")
        
        # Get the taught content for this segment to ground the exercise
        segment_key = f"{topic}:{segment_id}"
        taught_content = self.current_state.taught_content.get(segment_key, "")
        
        # If not in state, try memory
        if not taught_content and self.memory and segment_id:
            taught_json = self.memory.get_taught_segment_json(topic, segment_id)
            if taught_json:
                taught_content = taught_json.get("full_text", "")
        
        print(f"[WORKFLOW] Using {len(taught_content)} chars of taught content for exercise")
        
        # Generate a single practice question using the tutor evaluator
        exercise_result = await self.tutor_evaluator.execute(
            "generate_quiz",
            topic=topic,
            concept_id=concept_id,
            difficulty="easy",
            num_questions=1,  # Just one question
            taught_content=taught_content,
            segment_id=segment_id,
            exam_context=self.memory.get_exam_context() if self.memory else {}
        )
        
        if not exercise_result.success:
            # Fallback: just move to next step
            return {
                "success": True,
                "step_type": "optional_exercise",
                "content": "💡 Great work on that segment! Ready to continue?",
                "next_action": "continue",
                "skippable": True
            }
        
        exercise = exercise_result.data
        question = exercise["questions"][0] if exercise.get("questions") else None
        
        if not question:
            return {
                "success": True,
                "step_type": "optional_exercise",
                "content": "💡 Great work on that segment! Ready to continue?",
                "next_action": "continue",
                "skippable": True
            }
        
        # Store exercise in state for later evaluation
        self.current_state.current_quiz = {
            "quiz_id": f"exercise_{segment_id}",
            "topic": topic,
            "concept_id": concept_id,
            "segment_id": segment_id,
            "difficulty": "easy",
            "questions": [question],
            "is_optional": True
        }
        
        # Get planner reasoning
        planner_reason = step.get("why_assigned", "Optional exercise to reinforce understanding")
        
        return {
            "success": True,
            "step_type": "optional_exercise",
            "exercise": {
                "question": question["question"],
                "options": question["options"],
                "hint": question.get("hint", ""),
                "segment_id": segment_id
            },
            "content": f"💪 **Quick Practice Exercise**\n\nBefore we move on, here's an optional question to help reinforce what you just learned. Feel free to try it or skip to the next segment!\n\n**Question:** {question['question']}",
            "instructions": "Try this optional exercise to reinforce your understanding, or click 'Skip' to continue.",
            "planner_reason": planner_reason,
            "next_action": "answer_or_skip",
            "skippable": True
        }
    
    async def _execute_review_results(self, step: Dict) -> Dict[str, Any]:
        """Execute review results step"""
        # Get student profile
        student_profile = db.get_topic_proficiency()
        
        # Analyze results
        weak_topics = []
        strong_topics = []
        
        for topic, data in student_profile.items():
            if isinstance(data, dict):
                accuracy = data.get("accuracy", 0.0)
                if accuracy < 0.6:
                    weak_topics.append(topic)
                elif accuracy >= 0.8:
                    strong_topics.append(topic)
        
        return {
            "success": True,
            "step_type": "review_results",
            "student_profile": student_profile,
            "weak_topics": weak_topics,
            "strong_topics": strong_topics,
            "recommendations": self._generate_recommendations(weak_topics, strong_topics),
            "next_action": "continue_learning"
        }
    
    def _generate_recommendations(self, weak_topics: List[str], strong_topics: List[str]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if weak_topics:
            recommendations.append(f"Focus on strengthening: {', '.join(weak_topics[:3])}")
        
        if strong_topics:
            recommendations.append(f"Great progress in: {', '.join(strong_topics[:2])}")
        
        if not weak_topics and not strong_topics:
            recommendations.append("Keep practicing to maintain your current level")
        
        return recommendations
    
    async def submit_quiz_answers(self, answers: List[str]) -> Dict[str, Any]:
        """Submit and evaluate quiz answers, then trigger plan adaptation"""
        if not self.current_state.current_quiz:
            return {
                "success": False,
                "error": "No active quiz to submit answers for"
            }
        
        quiz = self.current_state.current_quiz
        topic = quiz["topic"]
        concept_id = quiz.get("concept_id")
        
        # Evaluate answers
        eval_result = await self.tutor_evaluator.execute(
            "evaluate_quiz",
            quiz=quiz,
            student_answers=answers
        )
        
        if not eval_result.success:
            return {
                "success": False,
                "error": "Quiz evaluation failed",
                "details": eval_result.reasoning
            }
        
        evaluation = eval_result.data
        score_percentage = evaluation["score_percentage"]
        
        # Update student profile
        profile_result = await self.tutor_evaluator.execute(
            "update_profile",
            topic=topic,
            score_percentage=score_percentage,
            concept_id=concept_id
        )
        
        # Get current step info
        current_step = self.current_state.study_plan[self.current_state.current_step] if self.current_state.current_step < len(self.current_state.study_plan) else None
        
        # Tag unclear segments from question-level segment_hints
        unclear_counts = {}
        for r in evaluation.get("results", []):
            seg = r.get("segment_hint", "general")
            if not r.get("is_correct", False):
                unclear_counts[seg] = unclear_counts.get(seg, 0) + 1
        unclear_segments = [seg for seg, cnt in sorted(unclear_counts.items(), key=lambda x: x[1], reverse=True) if seg != "general"]

        # Save quiz results and unclear segments in session memory
        if self.memory:
            try:
                ev = dict(evaluation)
                ev["unclear_segments"] = unclear_segments
                self.memory.append_quiz_result(topic, ev)
                self.memory.update_context({"last_unclear_segments": unclear_segments, "last_quiz_score": score_percentage, "current_topic": topic})
                
                # Update learning objectives mastery based on quiz performance
                segment_id = current_step.get("segment_id") if current_step else None
                if segment_id:
                    segment_plan = self.memory.get_segment_plan(topic)
                    segment_info = next((s for s in segment_plan if s.get("segment_id") == segment_id), None)
                    
                    if segment_info and "learning_objectives" in segment_info:
                        # Calculate confidence based on quiz score and difficulty rating
                        confidence_score = score_percentage / 100.0  # Convert to 0-1
                        
                        for objective in segment_info["learning_objectives"]:
                            self.memory.mark_objective_mastered(topic, objective, confidence_score)
                            print(f"[WORKFLOW] Marked objective: '{objective[:50]}...' with confidence {confidence_score:.2f}")
                
                # Schedule spaced repetition based on quiz performance
                mastery_score = score_percentage / 100.0  # Convert to 0-1
                self.memory.schedule_review(topic, mastery_score)
                next_review = self.memory.get_next_review_time(topic)
                if next_review:
                    print(f"[WORKFLOW] Scheduled review for '{topic}' at {next_review.strftime('%Y-%m-%d %H:%M')}")
                
            except Exception as e:
                print(f"[WORKFLOW] Error updating objectives/spaced repetition: {e}")
                pass

        # Evaluate progress and adapt plan using LLM planner
        print("🔄 Evaluating progress and adapting plan...")
        adaptation_result = await self.planner_agent.evaluate_and_adapt(
            completed_step=current_step or {"topic": topic, "action": "quiz"},
            quiz_results=evaluation,
            student_questions=[]  # Could track questions asked during session
        )
        
        # Log quiz results in lesson history
        if self.current_state.doc_id:
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=f"Quiz completed! Your score: {score_percentage:.1f}%. {adaptation_result.reasoning}",
                sources=[]
            )
            
            # Update lesson session with quiz results
            db.update_lesson_session(
                session_id=session_id,
                quiz_results=evaluation,
                final_evaluation=adaptation_result.evaluation
            )
            # Append quiz results to session memory for topic-level analytics
            if self.memory and isinstance(evaluation, dict):
                try:
                    ev = dict(evaluation)
                    ev["topic"] = topic
                    self.memory.append_quiz_result(topic, ev)
                except Exception:
                    pass
        
        # Update state based on adaptation
        if adaptation_result.success:
            # CRITICAL DEBUG: Log what planner returned
            print(f"[WORKFLOW] Planner returned plan with {len(adaptation_result.plan)} steps")
            print(f"[WORKFLOW] Plan first 3 steps: {adaptation_result.plan[:3] if adaptation_result.plan else 'EMPTY'}")
            print(f"[WORKFLOW] Next action: {adaptation_result.next_action}")
            
            self.current_state.study_plan = adaptation_result.plan
            next_action = adaptation_result.next_action
            
            # CRITICAL FIX: Don't increment step if plan is empty
            if not adaptation_result.plan:
                print(f"[WORKFLOW] ⚠️ WARNING: Planner returned empty plan after quiz!")
                # Try to regenerate plan for next uncompleted concept
                if hasattr(self.planner_agent, 'all_concepts') and self.planner_agent.all_concepts:
                    print(f"[WORKFLOW] Attempting to recover by finding next concept...")
                    next_concept = self.planner_agent._get_next_uncompleted_concept()
                    if next_concept:
                        print(f"[WORKFLOW] Found next concept: {next_concept.get('label')}")
                        import asyncio
                        recovery_plan = await self.planner_agent._create_plan_for_concept(next_concept)
                        self.current_state.study_plan = recovery_plan
                        self.current_state.current_step = 0  # Reset to start of new plan
                        print(f"[WORKFLOW] Recovery successful! New plan has {len(recovery_plan)} steps")
                    else:
                        print(f"[WORKFLOW] No more concepts available. All learning complete!")
                        return {
                            "success": True,
                            "evaluation": evaluation,
                            "profile_updated": profile_result.success,
                            "next_action": "complete",
                            "message": f"🎉 Quiz completed! Score: {score_percentage:.1f}%. You've finished all available content!"
                        }
            
            # Move to next step (or stay at 0 if we just recovered)
            if adaptation_result.plan and self.current_state.current_step >= len(self.current_state.study_plan) - 1:
                self.current_state.current_step = 0  # Reset for new concept
            elif adaptation_result.plan:
                self.current_state.current_step += 1
            
            # Return success response
            return {
                "success": True,
                "evaluation": evaluation,
                "profile_updated": profile_result.success,
                "next_action": next_action,
                "planner_reasoning": adaptation_result.reasoning,
                "planner_reason": adaptation_result.reasoning,  # For frontend compatibility
                "planner_evaluation": adaptation_result.evaluation,
                "adapted_plan": adaptation_result.plan if next_action == "adapt_plan" else None,
                "message": f"Quiz completed! Score: {score_percentage:.1f}%. {adaptation_result.reasoning}"
            }
        else:
            # Move to next step even if adaptation fails
            self.current_state.current_step += 1
            
            return {
                "success": True,
                "evaluation": evaluation,
                "profile_updated": profile_result.success,
                "next_action": "continue_plan",
                "message": f"Quiz completed! Score: {score_percentage:.1f}%. Your profile has been updated."
            }

    async def synthesize_episodic_summary(self) -> Dict[str, Any]:
        """Create a concise episodic summary from recent taught segment summaries and quiz results."""
        try:
            # Gather recent summaries and last quiz results for current topic
            recent_summaries = self.memory.get_last_session_summaries(k=5) if self.memory else []
            topic = self.current_state.current_topic
            recent_quiz = self.memory.get_recent_quiz_results(topic, k=2) if (self.memory and topic) else []

            prompt = f"""
You are summarizing a learning session.
RECENT_SUMMARIES:
{json.dumps(recent_summaries)[:1500]}

RECENT_QUIZ_RESULTS:
{json.dumps(recent_quiz)[:800]}

Write a 3-5 sentence episodic summary capturing what was taught and current mastery.
Keep it concise and student-friendly. No markdown, plain text only.
"""
            import asyncio
            response = await asyncio.wait_for(self.llm.ainvoke(prompt), timeout=30.0)
            text = response.content.strip()
            if self.memory:
                self.memory.set_episodic_summary(text, meta={"topic": topic})
            return {"success": True, "summary": text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def answer_student_question(self, question: str, topic: str = None) -> Dict[str, Any]:
        """Answer student's question using RAG with current topic context"""
        # Use current topic from state if not provided
        if not topic and self.current_state.current_topic:
            topic = self.current_state.current_topic
            print(f"[WORKFLOW] Using current topic from state: {topic}")
        
        # Track engagement: increment questions_asked
        if self.adaptive_state:
            self.adaptive_state["engagement_signals"]["questions_asked"] += 1
        
        # Fetch short-term memory summaries for coherence and current segment taught content
        recent_summaries = []
        recent_qa = []
        current_segment_id = None
        current_segment_text = ""
        if self.memory:
            try:
                recent_summaries = self.memory.get_last_session_summaries(k=3) or []
                recent_qa = self.memory.get_recent_qa(k=3) or []  # NEW: Q&A history
                ctx = self.memory.get_context() or {}
                current_segment_id = ctx.get("current_segment")
                if topic and current_segment_id:
                    current_segment_text = self.memory.get_taught_segment(topic, current_segment_id) or ""
                
                # Log what we're passing for debugging
                print(f"[WORKFLOW] Answer context: topic={topic}, segment={current_segment_id}")
                print(f"[WORKFLOW] Current segment text: {len(current_segment_text)} chars")
                print(f"[WORKFLOW] Recent summaries: {len(recent_summaries)} items")
                print(f"[WORKFLOW] Recent Q&A: {len(recent_qa)} exchanges")
                if recent_summaries:
                    for i, summ in enumerate(recent_summaries[:2]):
                        print(f"  Summary {i+1}: {str(summ)[:80]}...")
                if recent_qa:
                    for i, qa in enumerate(recent_qa[:2]):
                        print(f"  Q&A {i+1}: Q={qa.get('q', '')[:50]}...")
            except Exception as e:
                print(f"[WORKFLOW] Error fetching answer context: {e}")
                recent_summaries = []
                recent_qa = []
        
        answer_result = await self.tutor_evaluator.execute(
            "answer_question",
            question=question,
            topic=topic,
            recent_summaries=recent_summaries,
            recent_qa=recent_qa,  # NEW: Q&A history
            current_segment_id=current_segment_id,
            current_segment_text=current_segment_text
        )
        
        # Store Q&A exchange in memory
        if answer_result.success and self.memory:
            try:
                self.memory.append_qa_exchange(question, answer_result.data["answer"])
                print(f"[WORKFLOW] Stored Q&A exchange in conversation history")
            except Exception as e:
                print(f"[WORKFLOW] Failed to store Q&A: {e}")
        
        if not answer_result.success:
            return {
                "success": False,
                "error": "Answer generation failed",
                "details": answer_result.reasoning
            }
        
        # Log the conversation in lesson history
        if self.current_state.doc_id:
            session_id = f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="student",
                content=question
            )
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=answer_result.data["answer"],
                sources=answer_result.data.get("sources", [])
            )
        
        return {
            "success": True,
            "answer": answer_result.data["answer"],
            "sources": answer_result.data.get("sources", []),
            "supporting_chunks": answer_result.data.get("supporting_chunks", [])
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current workflow state"""
        return {
            "doc_id": self.current_state.doc_id,
            "session_id": getattr(self.current_state, 'session_id', None),
            "ingest_status": self.current_state.ingest_status,
            "concepts": self.current_state.concepts,
            "study_plan": self.current_state.study_plan,
            "current_step": self.current_state.current_step,
            "student_choice": self.current_state.student_choice,
            "current_quiz": self.current_state.current_quiz
        }
    
    def get_spaced_repetition_schedule(self) -> Dict[str, Any]:
        """Get spaced repetition schedule for review"""
        student_profile = db.get_topic_proficiency()
        due_topics = self.spaced_rep.get_topics_for_review(student_profile)
        
        return {
            "due_topics": due_topics,
            "total_topics": len(student_profile),
            "next_reviews": {
                topic: self.spaced_rep.calculate_next_review(
                    topic, 
                    data.get("accuracy", 0.0), 
                    data.get("attempts", 0)
                ).isoformat()
                for topic, data in student_profile.items()
                if isinstance(data, dict)
            }
        }

# Global orchestrator instance
workflow_orchestrator = WorkflowOrchestrator()