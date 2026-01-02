"""
Strategy Generator Agent - Creates custom teaching approaches based on diagnosis
"""
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
import json
import os
from app.core.config import settings


class StrategyGenerator:
    """
    Generates custom teaching strategies based on diagnostic analysis.
    Creates tailored prompts instead of using templates.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.name = "StrategyGenerator"
        self.llm = llm or ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=0.7,  # Higher temp for creative teaching approaches
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    async def generate_strategy(
        self,
        diagnosis: Dict,
        segment_id: str,
        topic: str,
        student_profile: Dict = None
    ) -> Dict[str, Any]:
        """
        Generate a custom teaching strategy (not a template!).
        
        Returns:
        {
            "approach_name": "descriptive name",
            "teaching_hook": "attention-grabbing opening",
            "structure": ["step1", "step2", ...],
            "custom_prompt": "detailed instruction for content generator",
            "success_criteria": ["criterion1", "criterion2"]
        }
        """
        print(f"[{self.name}] 🎯 Generating strategy for {segment_id}")
        
        # Extract diagnosis insights
        root_cause = diagnosis.get("root_cause", "comprehension issue")
        cognitive_gap = diagnosis.get("cognitive_gap", "unknown")
        recommended = diagnosis.get("recommended_approach", {})
        style = recommended.get("style", "balanced")
        start_point = recommended.get("start_point", "fundamentals")
        avoid_list = recommended.get("avoid", [])
        emphasize_list = recommended.get("emphasize", [])
        
        # Student profile context
        learning_style = student_profile.get("preferred_learning_style", "balanced") if student_profile else "balanced"
        past_successes = student_profile.get("successful_strategies", []) if student_profile else []
        
        past_context = ""
        if past_successes:
            past_context = f"\nPAST SUCCESSES FOR THIS STUDENT:\n" + "\n".join([
                f"- {s.get('strategy', 'unknown')}: {s.get('outcome', 'worked well')}"
                for s in past_successes[:3]
            ])
        
        strategy_prompt = f"""You are an expert educator creating a CUSTOM teaching strategy (not a template!).

DIAGNOSIS:
Root Cause: {root_cause}
Cognitive Gap: {cognitive_gap}
Missing Prerequisite: {diagnosis.get('missing_prerequisite', 'none')}
Confusion Patterns: {', '.join(diagnosis.get('confusion_patterns', []))}

RECOMMENDED TEACHING APPROACH:
Style: {style}
Start From: {start_point}
Must Avoid: {', '.join(avoid_list) if avoid_list else 'nothing specific'}
Must Emphasize: {', '.join(emphasize_list) if emphasize_list else 'clarity'}

STUDENT PROFILE:
Learning Style: {learning_style}
{past_context}

TASK: Create a completely CUSTOM teaching strategy for re-teaching "{segment_id}" in "{topic}".

Think like an expert tutor adapting to THIS specific student's needs. Be creative!

Generate:
1. APPROACH NAME - Descriptive name for this strategy (e.g., "Visual Spatial Analogy", "Code-First Walkthrough")

2. TEACHING HOOK - Opening sentence to grab attention and frame the concept
   (connect to something familiar or address the confusion directly)

3. STRUCTURE - Step-by-step teaching flow (4-6 steps)
   Each step should build on previous, address the specific confusion

4. CUSTOM PROMPT - COMPREHENSIVE instruction for content generator (300-500 words)
   Write like you're giving DETAILED instructions to another expert teacher.
   
   MUST INCLUDE:
   - Opening approach: How to start (question? story? problem?)
   - Core explanation strategy: Exact method to explain the concept
   - SPECIFIC examples to use (name them explicitly, e.g., "Use the email spam filter example...")
   - SPECIFIC analogies (e.g., "Liken neural networks to a team of specialists...")
   - Key comparisons/contrasts (e.g., "Contrast supervised vs unsupervised by comparing...")
   - Visual/spatial hints (e.g., "Describe it as layers stacked vertically...")
   - Common pitfalls to address (e.g., "Clarify that X is NOT the same as Y because...")
   - Checkpoint questions to embed (e.g., "Ask: 'Can you see why this matters?'")
   - Closing reinforcement (e.g., "End by summarizing the 3 key takeaways...")
   
   This prompt will be used DIRECTLY to generate teaching content. Be exhaustive!

5. SUCCESS CRITERIA - How to know if this worked (2-3 measurable outcomes)

6. EXERCISES - 2-3 specific exercise questions to test understanding
   Each should target the specific confusion area identified in diagnosis.

Return ONLY valid JSON (no markdown, no code fences):
{{
  "approach_name": "descriptive name",
  "teaching_hook": "one sentence hook",
  "structure": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "custom_prompt": "COMPREHENSIVE 300-500 word instruction covering: opening, core strategy, specific examples (name them!), specific analogies (describe them!), comparisons, visual hints, pitfalls to address, checkpoint questions, and closing. Be EXTREMELY detailed - this is the blueprint for teaching.",
  "success_criteria": [
    "Student can explain X",
    "Student can distinguish Y from Z"
  ],
  "targeted_exercises": [
    {{"question": "specific question targeting the confusion", "difficulty": "easy|medium|hard", "why": "tests if they understood X"}},
    {{"question": "another question", "difficulty": "medium", "why": "verifies they can apply Y"}}
  ]
}}

Make it engaging, tailored, and effective for THIS student's specific confusion!"""

        try:
            response = await self.llm.ainvoke(strategy_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            strategy = self._parse_strategy(response_text)
            
            print(f"[{self.name}] ✅ Strategy: {strategy.get('approach_name', 'Custom Approach')}")
            print(f"[{self.name}] 🎣 Hook: {strategy.get('teaching_hook', '')[:60]}...")
            
            # Log strategy creation
            from app.core.agent_thoughts import thoughts_tracker
            approach_name = strategy.get('approach_name', 'Custom Approach')
            hook = strategy.get('teaching_hook', '')[:60]
            thoughts_tracker.add(
                "Strategy",
                f"Created '{approach_name}' strategy: {hook}...",
                "🎯",
                {"strategy": strategy.get('approach_name'), "hook": hook}
            )
            
            return strategy
            
        except Exception as e:
            print(f"[{self.name}] ❌ Strategy generation failed: {e}")
            # Fallback strategy
            return self._fallback_strategy(diagnosis, segment_id, topic)
    
    def _parse_strategy(self, response_text: str) -> Dict:
        """Parse LLM response to extract strategy JSON"""
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
    
    def _fallback_strategy(self, diagnosis: Dict, segment_id: str, topic: str) -> Dict:
        """Fallback strategy when LLM fails"""
        print(f"[{self.name}] Using fallback strategy")
        
        style = diagnosis.get("recommended_approach", {}).get("style", "step-by-step")
        
        return {
            "approach_name": f"{style.capitalize()} Re-teaching",
            "teaching_hook": f"Let's revisit {segment_id} with a clearer approach.",
            "structure": [
                "Review the core concept simply",
                "Provide concrete examples",
                "Address common confusions",
                "Practice with applications"
            ],
            "custom_prompt": f"""Re-teach {segment_id} in {topic} using a {style} approach.

Start with fundamentals and build up gradually.
Use concrete examples throughout.
Address the student's confusion about this concept.
Make it clear and accessible.

Focus on practical understanding over theory.
Include multiple examples to illustrate each point.
Connect to real-world applications where relevant.""",
            "success_criteria": [
                f"Student can explain {segment_id} clearly",
                f"Student can apply {segment_id} in practice"
            ]
        }

