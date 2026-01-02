"""
Diagnostic Agent - Analyzes student confusion patterns and identifies root causes
"""
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
import json
import os
from app.core.config import settings


class DiagnosticAgent:
    """
    Analyzes quiz performance and student behavior to diagnose learning gaps.
    Provides detailed understanding of WHY a student is struggling.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.name = "DiagnosticAgent"
        self.llm = llm or ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.3,  # Lower temp for more consistent analysis
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    async def analyze_confusion(
        self,
        quiz_results: Dict,
        segment_id: str,
        topic: str,
        student_questions: List[str] = None,
        past_attempts: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Deep analysis of student confusion patterns.
        
        Returns:
        {
            "root_cause": "specific reason for failure",
            "missing_prerequisite": "concept student lacks",
            "cognitive_gap": "type of understanding issue",
            "confusion_patterns": ["pattern1", "pattern2"],
            "recommended_approach": {
                "style": "visual/analogy/step-by-step/conceptual",
                "start_point": "where to begin explanation",
                "avoid": ["what not to use"],
                "emphasize": ["what to focus on"]
            }
        }
        """
        print(f"[{self.name}] 🔍 Diagnosing confusion for {segment_id}")
        
        # Extract relevant info
        score = quiz_results.get("score", 0)
        incorrect_questions = quiz_results.get("incorrect_questions", [])
        unclear_segments = quiz_results.get("unclear_segments", [])
        segment_performance = quiz_results.get("segment_performance", {})
        
        # Build context
        questions_context = "\n".join(student_questions[:5]) if student_questions else "No questions asked"
        
        attempts_context = "First attempt"
        if past_attempts:
            attempts_context = "\n".join([
                f"- Attempt {i+1}: {a.get('score', 0)}% with strategy '{a.get('strategy', 'unknown')}'"
                for i, a in enumerate(past_attempts[-3:])
            ])
        
        diagnostic_prompt = f"""You are an expert educational diagnostician analyzing why a student is struggling.

STUDENT PERFORMANCE:
Topic: {topic}
Segment: {segment_id}
Quiz Score: {score}%
Unclear Segments: {unclear_segments}

QUESTIONS THEY GOT WRONG:
{json.dumps(incorrect_questions, indent=2)}

SEGMENT PERFORMANCE BREAKDOWN:
{json.dumps(segment_performance, indent=2)}

STUDENT'S QUESTIONS DURING LEARNING:
{questions_context}

PAST REMEDIATION ATTEMPTS:
{attempts_context}

TASK: Provide a deep diagnostic analysis.

Analyze:
1. ROOT CAUSE - What is the SPECIFIC reason they're struggling? (not just "didn't understand")
   - Is it conceptual confusion? Missing prerequisite? Too abstract? Wrong mental model?
   
2. MISSING PREREQUISITE - What foundational concept do they lack?

3. COGNITIVE GAP - What type of understanding issue is this?
   - abstract_to_concrete: Can't map theory to practice
   - concept_confusion: Mixing up similar ideas
   - prerequisite_missing: Lacks foundation
   - application_struggle: Understands theory but can't apply
   
4. CONFUSION PATTERNS - Specific mistakes they're making

5. RECOMMENDED APPROACH - How to teach THIS student
   - Teaching style: visual/analogy/step-by-step/conceptual/code-based
   - Start point: Where to begin (familiar concept, basic principle, etc)
   - Avoid: What NOT to do (don't use X, avoid Y)
   - Emphasize: What to focus on heavily

Return ONLY valid JSON (no markdown, no code fences):
{{
  "root_cause": "specific detailed reason",
  "missing_prerequisite": "concept name or 'none'",
  "cognitive_gap": "abstract_to_concrete|concept_confusion|prerequisite_missing|application_struggle",
  "confusion_patterns": ["pattern1", "pattern2"],
  "recommended_approach": {{
    "style": "visual|analogy|step-by-step|conceptual|code-based",
    "start_point": "where to begin explanation",
    "avoid": ["thing1", "thing2"],
    "emphasize": ["focus1", "focus2"]
  }}
}}

Be specific and actionable. Think like an expert tutor diagnosing a student."""

        try:
            response = await self.llm.ainvoke(diagnostic_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            diagnosis = self._parse_diagnosis(response_text)
            
            print(f"[{self.name}] ✅ Diagnosis: {diagnosis.get('root_cause', 'unknown')}")
            print(f"[{self.name}] 📋 Recommended: {diagnosis.get('recommended_approach', {}).get('style', 'unknown')} approach")
            
            # Log diagnostic insights
            from app.core.agent_thoughts import thoughts_tracker
            root_cause = diagnosis.get('root_cause', 'Unknown')[:80]
            approach_style = diagnosis.get('recommended_approach', {}).get('style', 'balanced')
            thoughts_tracker.add(
                "Diagnostics",
                f"Root cause: {root_cause}. Recommending {approach_style} teaching approach",
                "🔍",
                {"diagnosis": diagnosis}
            )
            
            return diagnosis
            
        except Exception as e:
            print(f"[{self.name}] ❌ Diagnosis failed: {e}")
            # Fallback diagnosis
            return self._fallback_diagnosis(score, segment_id)
    
    def _parse_diagnosis(self, response_text: str) -> Dict:
        """Parse LLM response to extract diagnosis JSON"""
        try:
            # Strip markdown code fences if present
            if "```json" in response_text:
                start = response_text.index("```json") + 7
                end = response_text.rindex("```")
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.index("```") + 3
                end = response_text.rindex("```")
                response_text = response_text[start:end].strip()
            
            # Find JSON object
            if '{' in response_text:
                start = response_text.index('{')
                end = response_text.rindex('}') + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
            
            raise ValueError("No JSON found in response")
            
        except Exception as e:
            print(f"[{self.name}] Parse error: {e}")
            raise
    
    def _fallback_diagnosis(self, score: float, segment_id: str) -> Dict:
        """Fallback diagnosis when LLM fails"""
        print(f"[{self.name}] Using fallback diagnosis")
        
        if score < 30:
            style = "step-by-step"
            gap = "prerequisite_missing"
        elif score < 60:
            style = "analogy"
            gap = "abstract_to_concrete"
        else:
            style = "conceptual"
            gap = "application_struggle"
        
        return {
            "root_cause": f"Student scored {score}% on {segment_id}, indicating comprehension issues",
            "missing_prerequisite": "foundational understanding",
            "cognitive_gap": gap,
            "confusion_patterns": ["general comprehension difficulty"],
            "recommended_approach": {
                "style": style,
                "start_point": "basic concepts with clear examples",
                "avoid": ["complex terminology", "abstract explanations"],
                "emphasize": ["concrete examples", "step-by-step reasoning"]
            }
        }

