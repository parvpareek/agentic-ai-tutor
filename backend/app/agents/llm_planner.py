# backend/app/agents/llm_planner.py
"""
LLM-Based Intelligent Planner Agent
Creates adaptive study plans, evaluates progress, and adapts dynamically
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class PlannerResponse:
    success: bool
    plan: List[Dict[str, Any]]
    reasoning: str
    next_action: str
    evaluation: Optional[Dict] = None

class LLMPlannerAgent:
    """Intelligent LLM-based planner that creates, evaluates, and adapts study plans"""
    
    def __init__(self, llm, database, vectorstore, memory=None):
        self.llm = llm
        self.db = database
        self.vectorstore = vectorstore
        self.memory = memory  # Redis memory for reading segment plans
        self.name = "LLMPlannerAgent"
        
        # Agentic components for adaptive remediation
        from app.agents.diagnostic_agent import DiagnosticAgent
        from app.agents.strategy_generator import StrategyGenerator
        
        self.diagnostic_agent = DiagnosticAgent(llm=llm)
        self.strategy_generator = StrategyGenerator(llm=llm)
        
        # Track plan execution state
        self.current_plan = None
        self.completed_steps = []
        self.concept_attempts = {}  # Track how many times we've worked on each concept
        self.max_attempts_per_concept = 3  # Prevent getting stuck
        
        # CRITICAL FIX: Store all available concepts and track which are completed
        self.all_concepts = []  # Store all extracted concepts
        self.completed_topics = set()  # Track completed topic names
        
        # Store last decision reasoning for transparency
        self.last_decision_reason = ""
    
    async def create_initial_plan(self, concepts: List[Dict], student_choice: str = "from_beginning", 
                                 document_outline: str = None, learning_roadmap: Dict = None) -> PlannerResponse:
        """Create initial study plan based on concepts, student choice, and document structure"""
        print(f"[{self.name}] Creating initial plan for {len(concepts)} concepts, choice: {student_choice}")
        
        # CRITICAL FIX: Store all concepts for later use when transitioning between topics
        self.all_concepts = concepts
        
        try:
            # Get student profile
            student_profile = self._get_student_profile()
            
            # Analyze concept relationships
            concept_relationships = await self._analyze_concept_relationships(concepts)
            
            # Create intelligent plan using LLM with document context and roadmap
            plan = await self._llm_create_plan(
                concepts=concepts,
                student_choice=student_choice,
                student_profile=student_profile,
                concept_relationships=concept_relationships,
                document_outline=document_outline,
                learning_roadmap=learning_roadmap
            )
            
            self.current_plan = plan
            
            return PlannerResponse(
                success=True,
                plan=plan,
                reasoning=f"Created {len(plan)}-step plan based on student choice: {student_choice}",
                next_action="execute_plan"
            )
            
        except Exception as e:
            return PlannerResponse(
                success=False,
                plan=[],
                reasoning=f"Plan creation failed: {str(e)}",
                next_action="retry"
            )
    
    async def evaluate_and_adapt(self, completed_step: Dict, quiz_results: Dict = None, 
                                student_questions: List[str] = None) -> PlannerResponse:
        """Evaluate progress and adapt plan based on feedback"""
        print(f"[{self.name}] Evaluating progress and adapting plan...")
        
        try:
            # Mark step as completed
            self.completed_steps.append(completed_step)
            
            # Track concept attempts
            topic = completed_step.get("topic", "")
            self.concept_attempts[topic] = self.concept_attempts.get(topic, 0) + 1
            
            # SIMPLIFIED: Read memory context for intelligent planning
            progress = {}
            next_segment = None
            recent_quiz = None
            unclear_segments = []
            unmastered_objectives = []
            recent_qa_count = 0
            
            if self.memory and topic:
                try:
                    progress = self.memory.get_topic_progress(topic) or {}
                    next_seg = self.memory.get_next_segment(topic)
                    next_segment = next_seg.get("segment_id") if next_seg else None
                    rq = self.memory.get_recent_quiz_results(topic, k=1)
                    if rq:
                        recent_quiz = rq[0]
                        unclear_segments = recent_quiz.get("unclear_segments", [])
                    
                    # Get engagement signals (no difficulty rating)
                    unmastered_objectives = self.memory.get_unmastered_objectives(topic)
                    recent_qa = self.memory.get_recent_qa(k=3)
                    recent_qa_count = len(recent_qa)
                    
                    print(f"[{self.name}] Memory signals: unmastered_obj={len(unmastered_objectives)}, questions={recent_qa_count}")
                except Exception as e:
                    print(f"[{self.name}] Error reading memory signals: {e}")

            # Get current student profile
            student_profile = self._get_student_profile()
            
            # Evaluate progress using LLM
            evaluation = await self._llm_evaluate_progress(
                completed_steps=self.completed_steps,
                current_plan=self.current_plan,
                quiz_results=quiz_results,
                student_profile=student_profile,
                student_questions=student_questions
            )
            
            # Decide next action with memory signals (await since it's async)
            next_action = await self._decide_next_action(
                evaluation, 
                quiz_results,
                unmastered_count=len(unmastered_objectives),
                engagement_level=recent_qa_count
            )
            
            # Adapt plan if needed
            if next_action == "adapt_plan":
                # 🤖 AGENTIC: Reorder remaining segments based on confusion
                adapted_plan = list(self.current_plan or [])
                
                # Extract remaining segments (not yet completed)
                completed_seg_ids = [s.get("segment_id") for s in self.completed_steps if s.get("segment_id")]
                remaining_steps = [s for s in adapted_plan if s.get("action") == "study_segment" and s.get("segment_id") not in completed_seg_ids]
                
                # If unclear_segments is empty but student did poorly, identify segments to re-teach
                score = quiz_results.get("score_percentage", 100) if quiz_results else 100
                if not unclear_segments and score < 70 and completed_step and completed_step.get("topic"):
                    # Student did poorly but no unclear_segments identified
                    # Get all segments for this topic and mark for remediation
                    current_topic = completed_step.get("topic")
                    segment_plan = self.memory.get_segment_plan(current_topic) if self.memory else []
                    
                    if segment_plan:
                        # Take the most recent taught segments (up to 2)
                        recent_completed = [s for s in self.completed_steps if s.get("topic") == current_topic and s.get("segment_id")][-2:]
                        unclear_segments = [s.get("segment_id") for s in recent_completed if s.get("segment_id")]
                        print(f"[{self.name}] ⚠️ Poor score ({score:.0f}%) but no unclear_segments identified")
                        print(f"[{self.name}] 🔄 Defaulting to re-teach recent segments: {unclear_segments}")
                
                if remaining_steps and (unclear_segments or student_questions):
                    print(f"[{self.name}] 🤖 Reordering {len(remaining_steps)} remaining segments based on student needs...")
                    
                    # Get confusion patterns from memory
                    confusion_patterns = unclear_segments if unclear_segments else []
                    
                    # Call agentic reordering
                    try:
                        reordered = await self._reorder_segments_agentic(
                            remaining_segments=remaining_steps,
                            completed_segments=completed_seg_ids,
                            quiz_results=quiz_results,
                            confusion_patterns=confusion_patterns
                        )
                        
                        # Rebuild plan: completed steps + reordered remaining
                        non_segment_steps = [s for s in adapted_plan if s.get("action") != "study_segment" or s.get("segment_id") in completed_seg_ids]
                        adapted_plan = non_segment_steps + reordered
                        
                        print(f"[{self.name}] ✅ Reordered plan with {len(reordered)} segments")
                    except Exception as e:
                        print(f"[{self.name}] Segment reordering failed: {e}, continuing with original order")
                
                if unclear_segments:
                    # Get segment performance details to determine remediation approach
                    segment_perf = quiz_results.get("segment_performance", {})
                    
                    # Track remediation attempts to avoid infinite loops
                    remediation_key = f"remediated:{topic}"
                    if not hasattr(self, 'remediation_tracker'):
                        self.remediation_tracker = {}
                    
                    # 🧠 AGENTIC REMEDIATION: Use diagnostic + strategy agents
                    for seg in unclear_segments[:2]:
                        # Check if already remediated once
                        already_remediated = self.remediation_tracker.get(f"{topic}:{seg}", 0)
                        if already_remediated >= 2:
                            print(f"[{self.name}] Segment {seg} already remediated {already_remediated} times, moving on")
                            continue
                        
                        # Get student context
                        student_questions = student_questions or []
                        past_attempts = self._get_past_remediation_attempts(topic, seg)
                        
                        # STEP 1: Diagnose the confusion
                        try:
                            diagnosis = await self.diagnostic_agent.analyze_confusion(
                                quiz_results=quiz_results,
                                segment_id=seg,
                                topic=topic,
                                student_questions=student_questions,
                                past_attempts=past_attempts
                            )
                            print(f"[{self.name}] 🔍 Diagnosis for {seg}: {diagnosis.get('root_cause', 'unknown')[:80]}")
                        except Exception as e:
                            print(f"[{self.name}] Diagnosis failed for {seg}: {e}, using fallback")
                            diagnosis = {"root_cause": "comprehension issue", "recommended_approach": {"style": "step-by-step"}}
                        
                        # STEP 2: Generate custom strategy
                        try:
                            student_profile = self._get_student_profile()
                            strategy_data = await self.strategy_generator.generate_strategy(
                                diagnosis=diagnosis,
                                segment_id=seg,
                                topic=topic,
                                student_profile=student_profile
                            )
                            approach_name = strategy_data.get("approach_name", "Custom Approach")
                            custom_prompt = strategy_data.get("custom_prompt", "")
                            teaching_hook = strategy_data.get("teaching_hook", "")
                            print(f"[{self.name}] 🎯 Strategy for {seg}: {approach_name}")
                        except Exception as e:
                            print(f"[{self.name}] Strategy generation failed for {seg}: {e}, using fallback")
                            approach_name = "Adaptive Re-teaching"
                            custom_prompt = f"Re-teach {seg} with clear examples and step-by-step explanations"
                            teaching_hook = f"Let's revisit {seg} with a clearer approach"
                        
                        # STEP 3: Insert remediation step with custom strategy
                        adapted_plan.insert(self._current_index_or_end(), {
                            "step_id": f"remediate_{seg}_{already_remediated+1}",
                            "action": "study_segment",
                            "topic": topic,
                            "concept_id": completed_step.get("concept_id", ""),
                            "segment_id": seg,
                            "segment_title": f"Re-learn: {seg}",
                            "difficulty": "easy",
                            "est_minutes": 5,
                            "agentic_strategy": {  # NEW: AI-generated strategy
                                "diagnosis": diagnosis,
                                "approach_name": approach_name,
                                "custom_prompt": custom_prompt,
                                "teaching_hook": teaching_hook,
                                "strategy_data": strategy_data
                            },
                            "why_assigned": f"🧠 {approach_name}: {diagnosis.get('root_cause', 'addressing confusion')[:100]}"
                        })
                        
                        # Track remediation
                        self.remediation_tracker[f"{topic}:{seg}"] = already_remediated + 1
                        print(f"[{self.name}] 🔄 Inserting agentic remediation for {seg}: {approach_name}, attempt={already_remediated+1}")
                else:
                    adapted_plan = await self._llm_adapt_plan(
                        current_plan=self.current_plan,
                        evaluation=evaluation,
                        quiz_results=quiz_results,
                        completed_steps=self.completed_steps
                    )
                self.current_plan = adapted_plan
                
                return PlannerResponse(
                    success=True,
                    plan=adapted_plan,
                    reasoning=self.last_decision_reason or evaluation.get("reasoning", "Plan adapted based on performance"),
                    next_action="continue_plan",
                    evaluation=evaluation
                )
            
            elif next_action == "continue_plan":
                # CRITICAL FIX: Check if current plan has remaining steps
                current_plan_copy = list(self.current_plan or [])
                
                # If current plan is empty or exhausted, try to move to next concept
                if not current_plan_copy or len(current_plan_copy) <= len(self.completed_steps):
                    print(f"[{self.name}] Current plan exhausted, checking for next segment/concept...")
                    
                    # Check if there's a next segment for current topic
                    if next_segment:
                        print(f"[{self.name}] Found next segment: {next_segment}")
                        current_plan_copy.append({
                            "step_id": f"study_{next_segment}",
                            "action": "study_segment",
                            "topic": topic,
                            "concept_id": completed_step.get("concept_id", ""),
                            "segment_id": next_segment,
                            "segment_title": f"Next segment {next_segment}",
                            "difficulty": "medium",
                            "est_minutes": 8,
                            "why_assigned": "Continue with next segment"
                        })
                        self.current_plan = current_plan_copy
                        return PlannerResponse(
                            success=True,
                            plan=self.current_plan,
                            reasoning=self.last_decision_reason or "Progress is good, continuing with next segment",
                            next_action="continue_plan",
                            evaluation=evaluation
                        )
                    else:
                        # No more segments, move to next concept
                        print(f"[{self.name}] No more segments, moving to next concept...")
                        if topic:
                            self.completed_topics.add(topic)
                        
                        next_concept = self._get_next_uncompleted_concept()
                        if next_concept:
                            print(f"[{self.name}] Moving to next concept: {next_concept['label']}")
                            new_plan = await self._create_plan_for_concept(next_concept)
                            self.current_plan = new_plan
                            return PlannerResponse(
                                success=True,
                                plan=new_plan,
                                reasoning=self.last_decision_reason or "Great work! Moving to next concept.",
                                next_action="continue_plan",
                                evaluation=evaluation
                            )
                        else:
                            print(f"[{self.name}] No more concepts available!")
                            return PlannerResponse(
                                success=True,
                                plan=[],
                                reasoning="All concepts completed! Great work!",
                                next_action="complete",
                                evaluation=evaluation
                            )
                
                # Plan has remaining steps, continue with current plan
                return PlannerResponse(
                    success=True,
                    plan=self.current_plan,
                    reasoning=self.last_decision_reason or "Progress is good, continuing with current plan",
                    next_action="continue_plan",
                    evaluation=evaluation
                )
            
            elif next_action == "move_forward":
                # Student has mastered this concept, move to next major concept
                # Prefer next segment if available, else move to next concept via LLM
                if next_segment:
                    next_plan = list(self.current_plan or [])
                    next_plan.insert(self._current_index_or_end(), {
                        "step_id": f"study_{next_segment}",
                        "action": "study_segment",
                        "topic": topic,
                        "concept_id": completed_step.get("concept_id", ""),
                        "segment_id": next_segment,
                        "segment_title": f"Next segment {next_segment}",
                        "difficulty": "medium",
                        "est_minutes": 8,
                        "why_assigned": "Advance to next segment in sequence"
                    })
                else:
                    # CRITICAL FIX: No more segments for current topic, find next concept
                    # Mark current topic as completed
                    if topic:
                        self.completed_topics.add(topic)
                        print(f"[{self.name}] Marked topic '{topic}' as completed")
                    
                    # Find next concept from original concept list
                    next_concept = self._get_next_uncompleted_concept()
                    
                    if next_concept:
                        print(f"[{self.name}] Moving to next concept: {next_concept['label']}")
                        # Generate segment plan for this next topic
                        next_plan = await self._create_plan_for_concept(next_concept)
                    else:
                        # No more concepts, try LLM fallback
                        print(f"[{self.name}] No more concepts available, using LLM fallback")
                    next_plan = await self._llm_create_next_concept_plan(
                        evaluation=evaluation,
                        student_profile=student_profile,
                        previous_concepts=self.completed_steps
                    )
                
                self.current_plan = next_plan
                
                return PlannerResponse(
                    success=True,
                    plan=next_plan,
                    reasoning=self.last_decision_reason or "Great progress! Moving to next major concept",
                    next_action="continue_plan",
                    evaluation=evaluation
                )
            
            else:  # clarify_concept
                # Student is struggling, need more clarification
                return PlannerResponse(
                    success=True,
                    plan=self.current_plan,
                    reasoning=self.last_decision_reason or evaluation.get("reasoning", "Student needs clarification"),
                    next_action="clarify_concept",
                    evaluation=evaluation
                )
                
        except Exception as e:
            return PlannerResponse(
                success=False,
                plan=self.current_plan or [],
                reasoning=f"Evaluation failed: {str(e)}",
                next_action="retry"
            )
    
    async def _llm_create_plan(self, concepts: List[Dict], student_choice: str, 
                              student_profile: Dict, concept_relationships: Dict,
                              document_outline: str = None, learning_roadmap: Dict = None) -> List[Dict]:
        """Use LLM to create intelligent study plan with document structure awareness"""
        
        # Format concepts with segments for LLM (with Redis memory fallback)
        concepts_str = "\n".join([
            f"- {c.get('label', 'Unknown')} (ID: {c.get('concept_id', 'unknown')})\n  "
            f"Section: {c.get('section_title', 'N/A')}\n  "
            f"Segments: {len(c.get('learning_segments', []))} segments\n  "
            f"Segment Details: " + ", ".join([
                f"{seg.get('segment_id', 'unknown')}: {seg.get('title', 'Untitled')} ({seg.get('estimated_minutes', 0)}min)"
                for seg in c.get('learning_segments', [])
            ])
            for c in concepts
        ])
        
        # If concepts don't have segments, try to get them from Redis memory
        if self.memory:
            for concept in concepts:
                topic = concept.get('label', '')
                if not concept.get('learning_segments') and topic:
                    segments = self.memory.get_segment_plan(topic)
                    if segments:
                        concept['learning_segments'] = segments
                        print(f"[{self.name}] Retrieved {len(segments)} segments from memory for topic: {topic}")
        
        # Re-format concepts string with updated segments
        concepts_str = "\n".join([
            f"- {c.get('label', 'Unknown')} (ID: {c.get('concept_id', 'unknown')})\n  "
            f"Section: {c.get('section_title', 'N/A')}\n  "
            f"Segments: {len(c.get('learning_segments', []))} segments\n  "
            f"Segment Details: " + ", ".join([
                f"{seg.get('segment_id', 'unknown')}: {seg.get('title', 'Untitled')} ({seg.get('estimated_minutes', 0)}min)"
                for seg in c.get('learning_segments', [])
            ])
            for c in concepts
        ])
        
        # Format student profile
        profile_str = self._format_profile_for_llm(student_profile)
        
        # Format relationships
        relationships_str = json.dumps(concept_relationships, indent=2)
        
        # Add document outline and roadmap context if available
        outline_context = ""
        if document_outline and document_outline != "Document structure not available":
            outline_context = f"""
DOCUMENT STRUCTURE (for context):
{document_outline[:500]}...
(This shows how concepts are organized in the source material)
"""
        
        # Add learning roadmap context if available
        roadmap_context = ""
        if learning_roadmap and learning_roadmap.get("chapters"):
            roadmap_context = f"""
LEARNING ROADMAP (for progression planning):
Total Chapters: {learning_roadmap.get('total_chapters', 0)}
Total Sections: {learning_roadmap.get('total_sections', 0)}
Estimated Learning Hours: {learning_roadmap.get('estimated_learning_hours', 0)}

Chapter Progression:
"""
            for i, chapter in enumerate(learning_roadmap.get('chapters', [])[:3]):
                roadmap_context += f"\nChapter {chapter.get('chapter_number', i+1)}: {chapter.get('title', 'Unknown')}"
                roadmap_context += f" (Est. {chapter.get('estimated_hours', 0)} hours)"
                for section in chapter.get('sections', [])[:2]:
                    roadmap_context += f"\n  - {section.get('title', 'Unknown')}"
            
            roadmap_context += "\n\nUse this roadmap to create a logical learning progression that follows the document structure."
        
        prompt = f"""You are an expert tutor creating a personalized study plan.

AVAILABLE CONCEPTS:
{concepts_str}
{outline_context}
{roadmap_context}
STUDENT PROFILE:
{profile_str}

CONCEPT RELATIONSHIPS:
{relationships_str}

STUDENT CHOICE: {student_choice}

Create a study plan that teaches concepts through their segments. Each concept has multiple segments that build on each other.

PLANNING RULES:
1. For "from_beginning": Teach segments in order (seg_1 → seg_2 → seg_3...)
2. Do NOT schedule a quiz after each segment. Schedule ONE topic-level "practice_quiz" AFTER all segments for that topic are completed.
3. For "diagnostic": Use "diagnostic_quiz" to test understanding first
4. For "choose_topics": Start with "calibration_quiz" to assess level
5. Follow segment prerequisites (some segments require others first)
6. Use estimated_minutes from segments for timing

Return a JSON array with this structure:
[
  {{
    "step_id": "step_1",
    "action": "study_segment|practice_quiz|diagnostic_quiz",
    "topic": "Concept name",
    "concept_id": "concept_id",
    "segment_id": "seg_1",
    "segment_title": "Segment title from learning_segments",
    "chapter": "Chapter X: Chapter Title",
    "section": "Section Y: Section Title",
    "difficulty": "easy|medium|hard",
    "est_minutes": 5-10,
    "why_assigned": "Clear explanation of why this segment is needed",
    "prerequisites": ["segment_ids that should be completed first"],
    "roadmap_position": "early|middle|late"
  }}
]

IMPORTANT: Create steps for individual segments, not entire concepts. Each segment should be a separate step.
After finishing all segments for a topic, add ONE topic-level practice_quiz step.
Make the plan logical, achievable, and personalized. Use segment timing estimates.
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            plan = json.loads(response.content)
            
            # Validate and clean plan
            return self._validate_plan(plan, concepts)
            
        except Exception as e:
            # Fallback plan if LLM fails
            return self._create_fallback_plan(concepts, student_choice)
    
    async def _llm_evaluate_progress(self, completed_steps: List[Dict], current_plan: List[Dict],
                                    quiz_results: Dict, student_profile: Dict,
                                    student_questions: List[str]) -> Dict:
        """Use LLM to evaluate student progress"""
        
        completed_str = "\n".join([f"- {s['topic']}: {s.get('action', 'completed')}" for s in completed_steps[-5:]])
        quiz_str = json.dumps(quiz_results, indent=2) if quiz_results else "No quiz results yet"
        profile_str = self._format_profile_for_llm(student_profile)
        questions_str = "\n".join(student_questions[-3:]) if student_questions else "No questions asked"
        
        prompt = f"""Evaluate student's learning progress and recommend next action.

COMPLETED STEPS:
{completed_str}

LATEST QUIZ RESULTS:
{quiz_str}

STUDENT PROFILE:
{profile_str}

RECENT QUESTIONS:
{questions_str}

Analyze:
1. Understanding level: How well has the student grasped the concepts?
2. Struggle areas: What specific topics need more work?
3. Progress rate: Is the student moving too fast or too slow?
4. Concept mastery: Has the student sufficiently mastered current concepts?
5. Readiness: Is the student ready to move to next major concept?

Return JSON:
{{
  "understanding_level": "strong|moderate|weak",
  "mastery_score": 0.0-1.0,
  "struggle_topics": ["topic1", "topic2"],
  "ready_for_next": true|false,
  "needs_clarification": true|false,
  "recommendation": "move_forward|continue_practice|clarify_concept|adapt_plan",
  "reasoning": "Detailed explanation of your assessment",
  "next_focus": "What to focus on next"
}}
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            evaluation = json.loads(response.content)
            return evaluation
            
        except Exception as e:
            # Fallback evaluation
            return {
                "understanding_level": "moderate",
                "mastery_score": 0.5,
                "ready_for_next": False,
                "recommendation": "continue_practice",
                "reasoning": "Need more practice with current concepts"
            }
    
    async def _llm_adapt_plan(self, current_plan: List[Dict], evaluation: Dict,
                             quiz_results: Dict, completed_steps: List[Dict]) -> List[Dict]:
        """Use LLM to adapt the plan based on evaluation"""
        
        plan_str = json.dumps(current_plan, indent=2)
        eval_str = json.dumps(evaluation, indent=2)
        quiz_str = json.dumps(quiz_results, indent=2) if quiz_results else "No quiz results"
        
        prompt = f"""Adapt the study plan based on student performance.

CURRENT PLAN:
{plan_str}

EVALUATION:
{eval_str}

QUIZ RESULTS:
{quiz_str}

Adapt the plan to:
1. Address struggle areas identified
2. Add clarification steps if needed
3. Adjust difficulty level appropriately
4. Add more practice if mastery is low
5. Move forward if student is ready
6. Keep momentum going (don't get stuck on one concept)

Return adapted plan as JSON array with same structure as original plan.
Keep it to 3-5 steps. Focus on immediate next actions.
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            adapted_plan = json.loads(response.content)
            return self._validate_plan(adapted_plan, current_plan)
            
        except Exception as e:
            # Return current plan if adaptation fails
            return current_plan
    
    async def _llm_create_next_concept_plan(self, evaluation: Dict, student_profile: Dict,
                                           previous_concepts: List[Dict]) -> List[Dict]:
        """Create plan for next major concept"""
        
        eval_str = json.dumps(evaluation, indent=2)
        profile_str = self._format_profile_for_llm(student_profile)
        prev_str = "\n".join([f"- {s['topic']}" for s in previous_concepts[-5:]])
        
        prompt = f"""Student has mastered current concepts. Create plan for next major concept.

PREVIOUS CONCEPTS LEARNED:
{prev_str}

EVALUATION:
{eval_str}

STUDENT PROFILE:
{profile_str}

Create a plan that:
1. Introduces the next logical concept in the progression
2. Connects it to previously learned concepts
3. Starts with appropriate difficulty
4. Includes learning + practice steps
5. Maintains engagement and momentum

Return JSON array with 3-5 steps following the standard plan structure.
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            next_plan = json.loads(response.content)
            return next_plan
            
        except Exception as e:
            # Fallback: continue with review
            return self._create_review_plan(previous_concepts)
    
    async def _analyze_concept_relationships(self, concepts: List[Dict]) -> Dict:
        """Analyze how concepts relate to each other"""
        
        # Get sample content for each concept
        relationships = {}
        
        for concept in concepts[:10]:  # Limit to prevent overload
            concept_id = concept['concept_id']
            label = concept['label']
            
            # Query vectorstore for related content
            try:
                results = self.vectorstore.query_top_k(label, k=2)
                documents = results.get('documents', [[]])[0] if 'documents' in results else []
                
                relationships[concept_id] = {
                    "label": label,
                    "prerequisites": self._infer_prerequisites(label, documents),
                    "related_to": self._infer_related_concepts(label, [c['label'] for c in concepts])
                }
            except:
                relationships[concept_id] = {
                    "label": label,
                    "prerequisites": [],
                    "related_to": []
                }
        
        return relationships
    
    def _infer_prerequisites(self, concept: str, documents: List[str]) -> List[str]:
        """Infer prerequisite concepts from content"""
        # Simple heuristic-based prerequisite detection
        prerequisites = []
        
        common_prerequisites = {
            "derivatives": ["limits", "functions"],
            "integration": ["derivatives", "functions"],
            "thermodynamics": ["mechanics", "energy"],
            "optics": ["waves", "light"],
            "electricity": ["charge", "forces"]
        }
        
        concept_lower = concept.lower()
        for key, prereqs in common_prerequisites.items():
            if key in concept_lower:
                prerequisites.extend(prereqs)
        
        return prerequisites
    
    def _infer_related_concepts(self, concept: str, all_concepts: List[str]) -> List[str]:
        """Infer related concepts"""
        related = []
        
        concept_lower = concept.lower()
        for other in all_concepts:
            if other != concept:
                other_lower = other.lower()
                # Simple similarity check
                if any(word in other_lower for word in concept_lower.split()) or \
                   any(word in concept_lower for word in other_lower.split()):
                    related.append(other)
        
        return related[:3]  # Limit to 3 related concepts
    
    async def _decide_next_action_llm(self, evaluation: Dict, quiz_results: Dict = None, 
                                      unmastered_count: int = 0, engagement_level: int = 0) -> tuple[str, str]:
        """
        🧠 AGENTIC PLANNER: Let LLM decide next action using rich context.
        This makes the planner truly adaptive instead of rule-based.
        
        Returns:
            tuple[str, str]: (action, reasoning)
        """
        # Build rich context for LLM decision
        context_parts = []
        
        # 1. Quiz Performance
        if quiz_results:
            topic = quiz_results.get("topic", "Unknown")
            score = quiz_results.get("score_percentage", 0.0)
            unclear = quiz_results.get("unclear_segments", [])
            context_parts.append(f"**Quiz Results:** {topic} - {score}% score")
            if unclear:
                context_parts.append(f"  - Unclear segments: {', '.join(unclear[:3])}")
            
            attempts = self.concept_attempts.get(topic, 0)
            if attempts > 0:
                context_parts.append(f"  - Previous attempts: {attempts}")
        
        # 2. Learning Patterns (if memory available)
        if self.memory and quiz_results:
            topic = quiz_results.get("topic")
            try:
                from app.core.learning_patterns import LearningPatternAnalyzer
                analyzer = LearningPatternAnalyzer()
                
                # Get confusion patterns
                recent_qa = self.memory.get_recent_qa(k=5)
                if recent_qa:
                    confusion = analyzer.detect_confusion_type(recent_qa[-1].get('q', ''), recent_qa)
                    context_parts.append(f"**Confusion Pattern:** {confusion['confusion_type']} ({confusion['suggested_approach']})")
                
                # Get engagement profile
                profile = analyzer.extract_engagement_profile(self.memory, topic)
                context_parts.append(f"**Engagement:** {profile.get('engagement_level')} engagement, {profile.get('preferred_learning_style')} learner")
                if profile.get('needs'):
                    context_parts.append(f"  - Needs: {'; '.join(profile['needs'][:2])}")
            except Exception as e:
                print(f"[{self.name}] Could not analyze patterns: {e}")
        
        # 3. Mastery Status
        if unmastered_count > 0:
            context_parts.append(f"**Mastery:** {unmastered_count} objectives not yet mastered")
        
        if engagement_level > 0:
            context_parts.append(f"**Engagement:** {engagement_level} recent questions asked")
        
        # 4. Evaluation insights
        if evaluation:
            if evaluation.get("needs_clarification"):
                context_parts.append("**Evaluation:** Student needs clarification on concepts")
            if evaluation.get("ready_for_next"):
                context_parts.append("**Evaluation:** Student appears ready for next topic")
        
        context_str = "\n".join(context_parts) if context_parts else "Limited context available"
        
        # Ask LLM to decide
        decision_prompt = f"""You are an adaptive learning planner. Based on student performance, decide the BEST next action.

{context_str}

Your options:
1. **clarify_concept** - Student struggling, needs re-teaching with different approach
2. **adapt_plan** - Moderate performance, needs targeted practice on weak areas
3. **move_forward** - Strong performance, ready for next topic/concept
4. **continue_plan** - Steady progress, continue current trajectory

Consider:
- Low scores (<50%) → usually need clarification
- Moderate scores (50-80%) with confusion patterns → adapt teaching approach
- High scores (>80%) with mastery → move forward
- High engagement (many questions) might indicate confusion OR deep interest

Respond with ONLY the action name (e.g., "clarify_concept") and brief 1-sentence reasoning.
Format: ACTION: reasoning"""
        
        try:
            response = await self.llm.ainvoke(decision_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            action_map = {
                "clarify": "clarify_concept",
                "adapt": "adapt_plan",
                "move_forward": "move_forward",
                "move forward": "move_forward",
                "continue": "continue_plan"
            }
            
            for keyword, action in action_map.items():
                if keyword in response_text.lower():
                    reasoning = response_text.split(":")[-1].strip() if ":" in response_text else response_text
                    print(f"[{self.name}] 🧠 LLM Decision: {action} - {reasoning[:100]}")
                    
                    # Log planner thinking
                    from app.core.agent_thoughts import thoughts_tracker
                    thoughts_tracker.add("Planner", f"{action.replace('_', ' ').title()}: {reasoning[:100]}", "🎯", {
                        "action": action,
                        "full_reasoning": reasoning
                    })
                    
                    return (action, reasoning)
            
            print(f"[{self.name}] Could not parse LLM decision: {response_text[:200]}")
        except Exception as e:
            print(f"[{self.name}] LLM decision failed: {e}")
        
        # Fallback to simple rules
        action = self._decide_next_action_fallback(evaluation, quiz_results, unmastered_count, engagement_level)
        fallback_reason = self._get_fallback_reasoning(action, quiz_results)
        return (action, fallback_reason)
    
    def _decide_next_action_fallback(self, evaluation: Dict, quiz_results: Dict = None, 
                                     unmastered_count: int = 0, engagement_level: int = 0) -> str:
        """Fallback rule-based decision if LLM fails"""
        if quiz_results and "score_percentage" in quiz_results:
            score = quiz_results.get("score_percentage", 0.0) / 100.0 if quiz_results.get("score_percentage") > 1 else quiz_results.get("score_percentage", 0.0)
            
            if score < 0.5:
                return "clarify_concept"
            elif score >= 0.8:
                return "move_forward"
            else:
                return "adapt_plan"
        
        return "continue_plan"
    
    def _get_fallback_reasoning(self, action: str, quiz_results: Dict = None) -> str:
        """Generate reasoning for fallback decisions"""
        if quiz_results:
            score = quiz_results.get("score_percentage", 0.0)
            topic = quiz_results.get("topic", "this topic")
            
            if action == "clarify_concept":
                return f"Score of {score}% indicates need for clarification on {topic} fundamentals"
            elif action == "move_forward":
                return f"Strong score of {score}% - ready to advance to next concept"
            elif action == "adapt_plan":
                return f"Moderate score of {score}% - adding targeted practice for {topic}"
        
        return "Continuing with steady progress through learning plan"
    
    async def _decide_next_action(self, evaluation: Dict, quiz_results: Dict = None, 
                                   unmastered_count: int = 0, engagement_level: int = 0) -> str:
        """
        Async wrapper for LLM decision making.
        Stores reasoning in self.last_decision_reason for use in PlannerResponse.
        """
        action = "continue_plan"
        reasoning = "Continuing with current plan"
        
        try:
            # Directly await the async LLM call (we're already in async context)
            action, reasoning = await self._decide_next_action_llm(
                evaluation, quiz_results, unmastered_count, engagement_level
            )
        except Exception as e:
            print(f"[{self.name}] Async decision error: {e}, using fallback")
            import traceback
            traceback.print_exc()
            action = self._decide_next_action_fallback(evaluation, quiz_results, unmastered_count, engagement_level)
            reasoning = self._get_fallback_reasoning(action, quiz_results)
        
        # Store reasoning for use in PlannerResponse
        self.last_decision_reason = reasoning
        return action
    
    async def _reorder_segments_agentic(
        self,
        remaining_segments: List[Dict],
        completed_segments: List[str],
        quiz_results: Dict = None,
        confusion_patterns: List[str] = None
    ) -> List[Dict]:
        """
        🤖 AGENTIC: Dynamically reorder remaining segments based on student progress.
        
        Considers:
        - Prerequisite dependencies
        - Student's demonstrated understanding
        - Confusion patterns requiring different sequencing
        - Cognitive load optimization
        
        Returns: Reordered segment list with reasoning
        """
        try:
            from app.core.agent_thoughts import thoughts_tracker
            
            if not remaining_segments:
                return []
            
            # Build context
            completed_str = f"Completed: {', '.join(completed_segments)}" if completed_segments else "No segments completed yet"
            
            quiz_context = "No quiz data"
            if quiz_results:
                score = quiz_results.get("score_percentage", 0)
                weak_areas = quiz_results.get("unclear_segments", [])
                quiz_context = f"Recent quiz: {score:.0f}% score. Weak areas: {weak_areas[:2] if weak_areas else 'none'}"
            
            confusion_str = "No confusion detected"
            if confusion_patterns:
                confusion_str = f"Confused about: {', '.join(confusion_patterns[:2])}"
            
            # Format remaining segments
            segments_info = []
            for seg in remaining_segments:
                seg_id = seg.get("segment_id", "unknown")
                title = seg.get("title", "Unknown")
                prereqs = seg.get("prerequisites", [])
                difficulty = seg.get("difficulty", "medium")
                segments_info.append(f"- {seg_id}: {title} (prereqs: {prereqs or 'none'}, difficulty: {difficulty})")
            
            segments_str = "\n".join(segments_info)
            
            prompt = f"""You are sequencing learning segments for optimal student understanding.

CONTEXT:
{completed_str}
{quiz_context}
{confusion_str}

REMAINING SEGMENTS:
{segments_str}

TASK: Reorder these segments to maximize learning effectiveness.

CONSIDER:
1. **Prerequisites**: Teach foundational segments before advanced ones
2. **Confusion Repair**: If student is confused about X, address it before building on it
3. **Cognitive Load**: Interleave easy and hard segments to prevent burnout
4. **Logical Flow**: Natural progression of ideas

RULES:
- If student weak on segment A and segment B builds on A → teach A first
- If student confused, prioritize clarifying segments
- If student excelling, challenge with harder segments
- Balance: don't cluster all hard segments together

Return ONLY valid JSON (no code fences):
{{
  "reordered_segments": [
    {{
      "segment_id": "seg_1",
      "title": "Segment Title",
      "original_position": 1,
      "new_position": 1,
      "reasoning": "Why this position"
    }}
  ],
  "sequencing_strategy": "prerequisite-first|confusion-repair|cognitive-balance",
  "overall_reasoning": "2-3 sentences explaining the reordering strategy"
}}

Original order positions are 1, 2, 3, etc."""
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import re
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            if not response_text.startswith('{'):
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            decision = json.loads(response_text)
            
            reordered_data = decision.get("reordered_segments", [])
            strategy = decision.get("sequencing_strategy", "default")
            reasoning = decision.get("overall_reasoning", "")
            
            # Map back to original segment objects
            reordered_segments = []
            for item in reordered_data:
                seg_id = item.get("segment_id")
                # Find original segment
                for seg in remaining_segments:
                    if seg.get("segment_id") == seg_id:
                        reordered_segments.append(seg)
                        break
            
            # Fallback: if parsing failed, keep original order
            if not reordered_segments:
                reordered_segments = remaining_segments
                reasoning = "Fallback: kept original order"
            
            thoughts_tracker.add(
                "Planner (Sequencing)",
                f"Reordered {len(reordered_segments)} segments using '{strategy}' strategy: {reasoning}",
                "🔄",
                {"strategy": strategy, "count": len(reordered_segments)}
            )
            
            print(f"[{self.name}] 🤖 Segment Reordering: {strategy} - {reasoning}")
            
            return reordered_segments
            
        except Exception as e:
            print(f"[{self.name}] Agentic segment reordering failed: {e}, keeping original order")
            return remaining_segments
    
    async def _decide_quiz_timing_agentic(
        self,
        topic: str,
        segments_since_last_quiz: List[str],
        time_since_last_quiz_minutes: int,
        confusion_level: str,
        cognitive_load: str
    ) -> tuple[bool, str]:
        """
        🤖 AGENTIC: Decide if NOW is the right time to insert a quiz.
        
        Considers:
        - Cognitive load (how much learned since last assessment)
        - Forgetting curve (time elapsed)
        - Confusion signals (student struggling = quiz earlier to diagnose)
        - Mastery signals (student excelling = can wait longer)
        
        Returns: (should_quiz_now: bool, reasoning: str)
        """
        try:
            from app.core.agent_thoughts import thoughts_tracker
            
            # Build context
            segments_count = len(segments_since_last_quiz)
            segments_str = f"{segments_count} segment(s): {', '.join(segments_since_last_quiz[:3])}"
            
            time_str = "Just started"
            if time_since_last_quiz_minutes > 0:
                if time_since_last_quiz_minutes < 60:
                    time_str = f"{time_since_last_quiz_minutes} minutes ago"
                else:
                    hours = time_since_last_quiz_minutes // 60
                    time_str = f"{hours} hour(s) ago"
            
            prompt = f"""Decide if NOW is the optimal time to give a quiz for "{topic}".

CURRENT STATE:
- Segments since last quiz: {segments_str}
- Time since last quiz: {time_str}
- Student confusion level: {confusion_level}
- Cognitive load: {cognitive_load}

QUIZ TIMING PRINCIPLES:
1. **Cognitive Load**: Quiz after 2-4 segments (don't overwhelm, don't wait too long)
2. **Forgetting Curve**: Quiz while content is fresh (optimal: 10-30 min after learning)
3. **Confusion Signals**: If confused → quiz EARLIER to diagnose gaps
4. **Mastery Signals**: If mastering easily → can wait longer (fewer interruptions)
5. **Spaced Repetition**: Balance assessment frequency with learning flow

DECISION FACTORS:
- Too Early: < 2 segments or < 5 minutes → let them learn more first
- Optimal: 2-3 segments AND 10-30 min elapsed AND moderate confusion
- Urgent: High confusion → quiz NOW to diagnose
- Can Wait: Low confusion + only 1 segment → let them continue

Return ONLY valid JSON (no code fences):
{{
  "quiz_now": true|false,
  "reasoning": "2-3 sentences explaining why now is optimal or why to wait",
  "optimal_segments_before_quiz": 3,
  "estimated_minutes_to_next_quiz": 15
}}"""
            
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            import re
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            if not response_text.startswith('{'):
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if match:
                    response_text = match.group(0)
            
            decision = json.loads(response_text)
            
            quiz_now = decision.get("quiz_now", False)
            reasoning = decision.get("reasoning", "No reasoning provided")
            
            thoughts_tracker.add(
                "Planner (Quiz Timing)",
                f"Quiz timing decision: {'QUIZ NOW' if quiz_now else 'CONTINUE LEARNING'} - {reasoning}",
                "⏰",
                {"decision": quiz_now, "segments_count": segments_count}
            )
            
            print(f"[{self.name}] 🤖 Quiz Timing: {'QUIZ NOW' if quiz_now else 'CONTINUE'} - {reasoning}")
            
            return (quiz_now, reasoning)
            
        except Exception as e:
            print(f"[{self.name}] Agentic quiz timing failed: {e}, using default logic")
            # Fallback: quiz after 3 segments or if high confusion
            should_quiz = len(segments_since_last_quiz) >= 3 or confusion_level == "high"
            reasoning = f"Rule fallback: {len(segments_since_last_quiz)} segments, {confusion_level} confusion"
            return (should_quiz, reasoning)
    
    def _get_student_profile(self) -> Dict:
        """Get current student profile"""
        try:
            profile = self.db.get_topic_proficiency()
            # TODO: Add successful strategies from feedback loop (Phase 3)
            return profile
        except:
            return {}
    
    def _get_past_remediation_attempts(self, topic: str, segment_id: str) -> List[Dict]:
        """Get history of remediation attempts for this segment"""
        attempts = []
        
        # Look through completed steps for past remediation attempts
        for step in self.completed_steps:
            if (step.get("action") == "study_segment" and 
                step.get("topic") == topic and
                step.get("segment_id") == segment_id and
                "agentic_strategy" in step):
                
                attempts.append({
                    "strategy": step["agentic_strategy"].get("approach_name", "unknown"),
                    "diagnosis": step["agentic_strategy"].get("diagnosis", {}),
                    "score": step.get("quiz_score"),  # If available
                    "timestamp": step.get("timestamp")
                })
        
        return attempts
    
    def _format_profile_for_llm(self, profile: Dict) -> str:
        """Format student profile for LLM"""
        if not profile:
            return "New student, no prior history"
        
        lines = []
        for topic, data in profile.items():
            if isinstance(data, dict):
                accuracy = data.get("accuracy", 0.0)
                attempts = data.get("attempts", 0)
                strength = data.get("strength", "unknown")
                lines.append(f"- {topic}: {accuracy:.1%} accuracy, {attempts} attempts, {strength}")
        
        return "\n".join(lines) if lines else "New student, no prior history"
    
    def _validate_plan(self, plan: List[Dict], reference: Any) -> List[Dict]:
        """Validate and clean plan"""
        validated = []
        
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                continue
            
            # Ensure required fields
            validated_step = {
                "step_id": step.get("step_id", f"step_{i+1}"),
                "action": step.get("action", "study_topic"),
                "topic": step.get("topic", "Unknown Topic"),
                "concept_id": step.get("concept_id", f"concept_{i+1}"),
                "difficulty": step.get("difficulty", "medium"),
                "est_minutes": step.get("est_minutes", 10),
                "why_assigned": step.get("why_assigned", "Practice and reinforcement"),
                "prerequisites": step.get("prerequisites", [])
            }
            
            validated.append(validated_step)
        
        return validated[:5]  # Limit to 5 steps
    
    def _create_fallback_plan(self, concepts: List[Dict], student_choice: str) -> List[Dict]:
        """Create fallback plan if LLM fails"""
        plan = []
        
        if student_choice == "from_beginning":
            # For beginners: alternate study and practice for each concept
            for i, concept in enumerate(concepts[:3]):
                # Study step
                plan.append({
                    "step_id": f"study_{i+1}",
                    "action": "study_topic",
                    "topic": concept.get('label', 'Unknown'),
                    "concept_id": concept.get('concept_id', f'concept_{i}'),
                    "difficulty": "easy",
                    "est_minutes": 10,
                    "why_assigned": f"Learn {concept.get('label', 'this topic')} from basics",
                    "prerequisites": []
                })
                # Practice step  
                plan.append({
                    "step_id": f"practice_{i+1}",
                    "action": "practice_quiz",
                    "topic": concept.get('label', 'Unknown'),
                    "concept_id": concept.get('concept_id', f'concept_{i}'),
                    "difficulty": "easy",
                    "est_minutes": 5,
                    "why_assigned": f"Practice {concept.get('label', 'this topic')} to reinforce learning",
                    "prerequisites": []
                })
        else:
            # For diagnostic/choose_topics: quizzes to assess
            for i, concept in enumerate(concepts[:3]):
                plan.append({
                    "step_id": f"step_{i+1}",
                    "action": "practice_quiz",
                    "topic": concept.get('label', 'Unknown'),
                    "concept_id": concept.get('concept_id', f'concept_{i}'),
                    "difficulty": "medium",
                    "est_minutes": 5,
                    "why_assigned": f"Assess your understanding of {concept.get('label', 'this topic')}",
                    "prerequisites": []
                })
        
        return plan
    
    def _create_review_plan(self, previous_concepts: List[Dict]) -> List[Dict]:
        """Create review plan for previously learned concepts"""
        plan = []
        
        # Review last 2-3 concepts
        for i, concept in enumerate(previous_concepts[-3:]):
            plan.append({
                "step_id": f"review_{i+1}",
                "action": "practice_quiz",
                "topic": concept.get('topic', 'Review'),
                "concept_id": concept.get('concept_id', f'review_{i+1}'),
                "difficulty": "medium",
                "est_minutes": 5,
                "why_assigned": "Review and reinforce previous learning",
                "prerequisites": []
            })
        
        return plan
    
    def _get_next_uncompleted_concept(self) -> Optional[Dict]:
        """Find the next concept from the original list that hasn't been completed"""
        print(f"[{self.name}] 🔍 Finding next concept | Total: {len(self.all_concepts)}, Completed: {self.completed_topics}")
        
        for i, concept in enumerate(self.all_concepts):
            topic_name = concept.get("label", "")
            concept_id = concept.get("concept_id", "")
            print(f"[{self.name}]   Concept {i+1}: '{topic_name}' (ID: {concept_id}) - {'COMPLETED' if topic_name in self.completed_topics else 'AVAILABLE'}")
            
            if topic_name and topic_name not in self.completed_topics:
                print(f"[{self.name}] ✅ Selected: '{topic_name}'")
                return concept
        
        print(f"[{self.name}] ❌ No uncompleted concepts found")
        return None
    
    async def _create_plan_for_concept(self, concept: Dict) -> List[Dict]:
        """Create a study plan for a specific concept with segments"""
        topic = concept.get("label", "Unknown Topic")
        concept_id = concept.get("concept_id", "unknown")
        
        plan = []
        
        # Check if we have segment plan in memory for this topic
        segment_plan = []
        if self.memory:
            try:
                segment_plan = self.memory.get_segment_plan(topic)
                if segment_plan:
                    print(f"[{self.name}] Found {len(segment_plan)} segments for topic '{topic}' in memory")
            except Exception as e:
                print(f"[{self.name}] Could not retrieve segment plan: {e}")
        
        if segment_plan:
            # Create plan steps from segment plan
            for seg in segment_plan:
                segment_id = seg.get("segment_id", "seg_1")
                segment_title = seg.get("title", topic)
                plan.append({
                    "step_id": f"study_{segment_id}",
                    "action": "study_segment",
                    "topic": topic,
                    "concept_id": concept_id,
                    "segment_id": segment_id,
                    "segment_title": segment_title,
                    "difficulty": "medium",
                    "est_minutes": seg.get("est_minutes", 10),
                    "why_assigned": f"Learning {segment_title}"
                })
        else:
            # Fallback: create a single study step for the whole topic
            print(f"[{self.name}] No segment plan found, creating single study step for '{topic}'")
            plan.append({
                "step_id": f"study_{concept_id}",
                "action": "study_segment",
                "topic": topic,
                "concept_id": concept_id,
                "segment_id": "full_topic",
                "segment_title": topic,
                "difficulty": "medium",
                "est_minutes": 15,
                "why_assigned": f"Learning {topic}"
            })
        
        return plan
    
    def reset_state(self):
        """Reset planner state"""
        self.current_plan = None
        self.completed_steps = []
        self.concept_attempts = {}
        self.all_concepts = []
        self.completed_topics = set()

    def _current_index_or_end(self) -> int:
        try:
            return max(len(self.completed_steps), 0)
        except Exception:
            return 0


