"""
LangGraph nodes for adaptive learning workflow
Each node is a pure function: State -> State
"""
from typing import Dict, Any
from .adaptive_state import AdaptiveState, EngagementSignals
from app.core.session_memory import SessionMemory
from app.agents.llm_planner import LLMPlannerAgent
from app.agents.simple_workflow import TutorEvaluatorAgent
from app.core.vectorstore import vectorstore
from app.core.database import db
from .memory_store import adaptive_memory

# Feature flag: disable cross-session memory retrieval/storage by default
CROSS_SESSION_MEMORY_ENABLED = False
import time

# ============================================================================
# Planner Node (Supervisor)
# ============================================================================

async def planner_node(state: AdaptiveState, planner: LLMPlannerAgent, memory: SessionMemory) -> AdaptiveState:
    """
    Planner decides next action based on proficiency, engagement, and quiz results.
    Acts as supervisor controlling the flow.
    """
    print(f"[PLANNER_NODE] Analyzing state for topic: {state['current_topic']}")
    
    # Read current context
    proficiency = state["proficiency"]
    engagement = state["engagement_signals"]
    unclear_segments = state["unclear_segments"]
    quiz_results = state["quiz_results"]
    current_topic = state["current_topic"]
    
    # Decision logic
    next_action = "teach"
    reason = ""
    
    # 1. Check if we need assessment (mid-segment)
    if state.get("needs_assessment"):
        next_action = "assess"
        reason = f"Student asked {engagement['questions_asked']} questions; checking understanding"
    
    # 2. Check if remediation needed (after quiz)
    elif unclear_segments:
        next_action = "remediate"
        reason = f"Remediating {len(unclear_segments)} unclear segments from quiz"
    
    # 3. Check proficiency after quiz
    elif quiz_results and current_topic:
        last_quiz = quiz_results[-1]
        score = last_quiz.get("score", 0)
        
        if score < 0.5:
            next_action = "remediate"
            reason = f"Low quiz score ({score:.1%}); reviewing topic"
        elif score < 0.8:
            # Check if there's a next segment
            next_seg = memory.get_next_segment(current_topic)
            if next_seg:
                next_action = "teach"
                reason = f"Moderate score ({score:.1%}); continuing to next segment"
            else:
                next_action = "review"
                reason = f"Moderate score ({score:.1%}); reviewing before advancing"
        else:
            # High score - advance
            next_seg = memory.get_next_segment(current_topic)
            if next_seg:
                next_action = "teach"
                reason = f"Strong score ({score:.1%}); advancing to next segment"
            else:
                next_action = "advance"
                reason = f"Topic mastered ({score:.1%}); advancing to next topic"
    
    # 4. Normal teaching flow
    else:
        study_plan = state["study_plan"]
        current_index = state["current_step_index"]
        
        if current_index < len(study_plan):
            step = study_plan[current_index]
            # Check both step_type and action for quiz (backward compatibility)
            step_type = step.get("step_type", step.get("action", "study_segment"))
            # Diagnostics: log current step dict for visibility
            try:
                print(f"[PLANNER_NODE] Current step @ {current_index}: action={step.get('action')} step_type={step.get('step_type')} topic={step.get('topic')} segment_id={step.get('segment_id')} title={step.get('segment_title')}")
            except Exception:
                pass
            if step_type == "practice_quiz":
                # 🤖 AGENTIC QUIZ TIMING: If quiz is in plan, it was dynamically inserted at optimal time
                # Trust the agentic decision and route to quiz_node
                next_action = "quiz"
                reason = step.get("why_assigned", "Quiz scheduled by agentic timing system")
                print(f"[PLANNER_NODE] Practice quiz detected - routing to quiz_node: {reason}")
            elif step_type == "optional_exercise":
                # CRITICAL FIX: Route exercises to quiz node
                next_action = "quiz"
                reason = f"Optional exercise for {step.get('segment_id', 'segment')}"
            else:
                next_action = "teach"
                segment_title = step.get('segment_title') or step.get('segment_id') or 'segment'
                reason = f"Teaching segment: {segment_title}"
        else:
            next_action = "complete"
            reason = "All study plan steps completed"
    
    print(f"[PLANNER_NODE] Decision: {next_action} | Reason: {reason}")
    
    state["next_action"] = next_action
    state["planner_reason"] = reason
    state["needs_assessment"] = False  # Reset flag
    
    return state


# ============================================================================
# Tutor Node (Teaching Worker)
# ============================================================================

async def tutor_node(state: AdaptiveState, tutor: TutorEvaluatorAgent, memory: SessionMemory, orchestrator=None) -> AdaptiveState:
    """
    Tutor teaches current segment, stores summary, updates engagement signals.
    Delegates to orchestrator's existing _execute_study_segment/_execute_study_topic methods.
    
    PHASE 4: Retrieves related summaries from memory store for context.
    """
    print(f"[TUTOR_NODE] Teaching segment for topic: {state['current_topic']}")
    
    study_plan = state["study_plan"]
    current_index = state["current_step_index"]
    
    if current_index >= len(study_plan):
        state["next_action"] = "complete"
        return state
    
    step = study_plan[current_index]
    topic = step.get("topic", state["current_topic"])
    segment_id = step.get("segment_id")
    action = step.get("action", "study_segment")
    concept_id = step.get("concept_id")
    
    # PHASE 4: Retrieve top-2 related summaries from memory store
    user_id = state["user_id"]
    segment_title = step.get("segment_title", segment_id)
    related_summaries = []
    if CROSS_SESSION_MEMORY_ENABLED:
        related_summaries = adaptive_memory.retrieve_related_summaries(
            user_id=user_id,
            query=f"{topic} {segment_title}",
            topic=topic,
            k=2
        )
    
    # Log related summaries for context engineering
    if related_summaries:
        print(f"[TUTOR_NODE] Retrieved {len(related_summaries)} related summaries for context")
        for i, summ in enumerate(related_summaries):
            print(f"  [{i+1}] {summ['topic']}:{summ['segment_id']} - {summ['summary'][:60]}...")
    
    start_time = time.time()
    
    # Delegate to orchestrator's existing teaching logic
    result = {}
    if orchestrator:
        if action == "study_segment":
            result = await orchestrator._execute_study_segment(step, concept_id)
        elif action == "study_topic":
            result = await orchestrator._execute_study_topic(step, concept_id)
        elif action in ["optional_exercise", "practice_quiz"]:
            # CRITICAL FIX: Don't teach for exercises/quizzes - wrong node!
            print(f"[TUTOR_NODE] ⚠️ Action '{action}' should not be in tutor_node - routing issue!")
            result = {
                "success": True,
                "content": f"Routing error: {action} in tutor node",
                "next_action": "continue"
            }
        else:
            # Planner may have deferred a quiz to teach; synthesize next segment
            try:
                next_seg = memory.get_next_segment(topic) if memory else None
                if next_seg:
                    print(f"[TUTOR_NODE] Overriding action '{action}' with next segment: {next_seg}")
                    synthesized = {
                        "step_id": f"auto_{concept_id}_{next_seg.get('segment_id','seg')}",
                        "action": "study_segment",
                        "topic": topic,
                        "concept_id": concept_id,
                        "segment_id": next_seg.get("segment_id", "seg_1"),
                        "segment_title": next_seg.get("title", "Segment"),
                        "difficulty": "medium",
                        "est_minutes": next_seg.get("estimated_minutes", 8),
                        "why_assigned": "Complete remaining segment before topic-level quiz"
                    }
                    # Update plan in place so state is consistent
                    try:
                        study_plan[current_index] = synthesized
                    except Exception:
                        pass
                    step = synthesized
                    segment_id = synthesized["segment_id"]
                    result = await orchestrator._execute_study_segment(step, concept_id)
                else:
                    # Fallback minimal content if no next segment info available
                    result = {
                        "success": True,
                        "content": f"Teaching {topic} - overview",
                        "exercises": [],
                        "follow_ups": []
                    }
            except Exception as e:
                print(f"[TUTOR_NODE] Failed to override non-teach action: {e}")
                result = {
                    "success": True,
                    "content": f"Teaching {topic} - overview",
                    "exercises": [],
                    "follow_ups": []
                }
    else:
        # Fallback when orchestrator not provided
        result = {
            "success": True,
            "content": f"Teaching {topic} - {segment_id or 'overview'}",
            "exercises": [],
            "follow_ups": []
        }
    
    elapsed = time.time() - start_time
    
    # Update engagement
    state["engagement_signals"]["time_spent_seconds"] += elapsed
    
    # Update state
    state["current_topic"] = topic
    state["current_segment"] = segment_id or ""
    state["current_step_index"] += 1
    state["last_result"] = result
    # Enforce: stop after teaching; wait for explicit Next from user
    state["awaiting_user_next"] = True
    
    # Add to messages
    state["messages"].append({
        "role": "tutor",
        "content": result.get("content", "")
    })
    
    # PHASE 4: Store this segment's summary in memory store for future context
    # Extract summary from result if available
    if result.get("success") and CROSS_SESSION_MEMORY_ENABLED:
        # Try to get summary from taught segment JSON or create a minimal one
        summary_text = ""
        try:
            taught_json = memory.get_taught_segment_json(topic, segment_id or "full_topic")
            if taught_json and isinstance(taught_json, dict):
                summary_text = taught_json.get("summary", "")
        except Exception:
            pass
        
        # Fallback: use first 180 chars of content
        if not summary_text and result.get("content"):
            summary_text = result["content"][:180]
        
        if summary_text:
            adaptive_memory.store_taught_summary(
                user_id=user_id,
                topic=topic,
                segment_id=segment_id or "full_topic",
                summary=summary_text,
                metadata={"session_id": state["session_id"]}
            )
    
    print(f"[TUTOR_NODE] Taught segment {segment_id}; advancing index to {state['current_step_index']}")
    
    return state


# ============================================================================
# Assessment Node (Mid-Segment Check)
# ============================================================================

async def assess_node(state: AdaptiveState, tutor: TutorEvaluatorAgent, memory: SessionMemory) -> AdaptiveState:
    """
    Triggers mini-quiz/check when engagement signals show confusion.
    """
    print(f"[ASSESS_NODE] Running quick assessment for {state['current_topic']}")
    
    # Generate 2-3 quick questions on current segment
    topic = state["current_topic"]
    segment = state.get("current_segment", "")
    
    # Simplified: ask a reflective question
    assessment_prompt = f"Quick check: Can you explain the main idea of what we just covered in {segment or topic}?"
    
    state["messages"].append({
        "role": "tutor",
        "content": assessment_prompt
    })
    
    # Mark that we're waiting for student response
    state["next_action"] = "wait_for_answer"
    state["planner_reason"] = "Awaiting student self-assessment response"
    
    print(f"[ASSESS_NODE] Assessment question posed; waiting for response")
    
    return state


# ============================================================================
# Quiz Node (Full Topic Quiz)
# ============================================================================

async def quiz_node(state: AdaptiveState, tutor: TutorEvaluatorAgent, memory: SessionMemory, orchestrator=None) -> AdaptiveState:
    """
    Generates quiz or optional exercise.
    Delegates to orchestrator's _execute_practice_quiz or _execute_optional_exercise.
    """
    topic = state["current_topic"]
    step = state["study_plan"][state["current_step_index"]] if state["current_step_index"] < len(state["study_plan"]) else {}
    action = step.get("action", "practice_quiz")
    concept_id = step.get("concept_id")
    
    print(f"[QUIZ_NODE] Handling '{action}' for topic: {topic}")
    
    # Delegate to appropriate executor
    result = {}
    if orchestrator:
        if action == "optional_exercise":
            result = await orchestrator._execute_optional_exercise(step, concept_id)
        else:
            result = await orchestrator._execute_practice_quiz(step, concept_id)
    else:
        # Fallback
        result = {
            "success": True,
            "quiz_id": f"quiz_{topic}_{int(time.time())}",
            "questions": []
        }
    
    state["last_result"] = result
    # Do NOT advance plan index here because quiz is not a plan step
    state["next_action"] = "wait_for_quiz_submission"
    state["planner_reason"] = "Quiz generated; awaiting student answers"
    # Enforce: stop after generating quiz; wait for explicit action
    state["awaiting_user_next"] = True
    
    print(f"[QUIZ_NODE] Quiz ready")
    
    return state


# ============================================================================
# Remediation Node (Re-teach unclear segments)
# ============================================================================

async def remediate_node(state: AdaptiveState, tutor: TutorEvaluatorAgent, memory: SessionMemory, orchestrator=None) -> AdaptiveState:
    """
    Re-teaches unclear segments with:
    - Example-focused retrieval (prioritize concrete examples)
    - Simplified language prompts
    - Interactive exercises
    """
    print(f"[REMEDIATE_NODE] Remediating unclear segments: {state['unclear_segments']}")
    
    if not state["unclear_segments"]:
        # Nothing to remediate
        state["next_action"] = "teach"
        state["planner_reason"] = "No unclear segments; resuming normal flow"
        return state
    
    # Take first unclear segment
    segment_to_remediate = state["unclear_segments"].pop(0)
    topic = state["current_topic"]
    
    # Different retrieval strategy: focus on examples and analogies
    from app.core.vectorstore import vectorstore
    example_queries = [
        f"{topic} {segment_to_remediate} simple example",
        f"{segment_to_remediate} analogy explanation",
        f"how {segment_to_remediate} works step by step",
        f"{topic} {segment_to_remediate} for beginners"
    ]
    
    all_docs = []
    seen = set()
    for query in example_queries:
        results = vectorstore.query_hybrid(query, k=3, alpha=0.7)
        for doc in results.get('documents', [[]])[0]:
            if doc not in seen and len(doc) > 100:
                seen.add(doc)
                all_docs.append(doc)
    
    context = "\n\n".join(all_docs[:4]) if all_docs else "No additional examples found."
    
    # Simplified remediation prompt
    remediation_prompt = f"""
    You are a patient tutor helping a student who struggled with this concept.
    
    TOPIC: {topic}
    SEGMENT: {segment_to_remediate}
    
    CONTEXT (use for examples and analogies):
    {context[:1500]}
    
    INSTRUCTIONS:
    - Use SIMPLE language (avoid jargon unless you define it first)
    - Start with a real-world analogy or concrete example
    - Break down into 3-4 small steps
    - Include 1-2 interactive "try this" exercises
    - Be encouraging and patient
    
    Return a clear, example-focused explanation (3-4 paragraphs).
    """
    
    try:
        from langchain_openai import ChatOpenAI
        from app.core.config import settings
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        response = await llm.ainvoke(remediation_prompt)
        remediation_content = response.content
        
        print(f"[REMEDIATE_NODE] Generated simplified explanation for {segment_to_remediate}")
    except Exception as e:
        print(f"[REMEDIATE_NODE] Error generating remediation: {e}")
        remediation_content = f"Let's revisit {segment_to_remediate} with a simpler explanation and concrete examples..."
    
    # Store remediation in memory
    if memory:
        try:
            memory.store_taught_segment(topic, f"remediate_{segment_to_remediate}", remediation_content)
            memory.push_session_summary({
                "segment_id": f"remediate_{segment_to_remediate}",
                "topic": topic,
                "summary": f"Remediated {segment_to_remediate} with examples",
                "timestamp": time.time()
            }, k=3)
        except Exception:
            pass
    
    state["messages"].append({
        "role": "tutor",
        "content": remediation_content
    })
    
    state["last_result"] = {
        "success": True,
        "content": remediation_content,
        "is_remediation": True,
        "remediated_segment": segment_to_remediate
    }
    
    # If more unclear segments, stay in remediate; else return to normal flow
    if state["unclear_segments"]:
        state["next_action"] = "remediate"
        state["planner_reason"] = f"Continuing remediation; {len(state['unclear_segments'])} segments remaining"
    else:
        state["next_action"] = "teach"
        state["planner_reason"] = "Remediation complete; resuming study plan"
    
    print(f"[REMEDIATE_NODE] Remediated {segment_to_remediate}")
    
    return state


# ============================================================================
# Routing Functions (Conditional Edges)
# ============================================================================

def route_after_planner(state: AdaptiveState) -> str:
    """Route to appropriate node based on planner's decision"""
    action = state["next_action"]
    
    if action == "teach":
        return "tutor"
    elif action == "quiz":
        return "quiz"
    elif action == "assess":
        return "assess"
    elif action == "remediate":
        return "remediate"
    elif action == "complete":
        return "end"
    elif action in ["wait_for_answer", "wait_for_quiz_submission"]:
        return "end"  # Exit graph, wait for user input
    else:
        return "tutor"  # Default

def should_trigger_assessment(state: AdaptiveState) -> str:
    """Check if engagement signals warrant mid-segment assessment"""
    signals = state["engagement_signals"]
    
    # Trigger if student asked many questions or low confidence
    if signals["questions_asked"] >= 3:
        state["needs_assessment"] = True
        return "planner"
    
    if signals.get("confidence_rating", 1.0) < 0.4:
        state["needs_assessment"] = True
        return "planner"
    
    return "continue"
