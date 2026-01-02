"""
Learning Pattern Analyzer - Extracts insights from student interactions
"""
from typing import List, Dict, Any, Optional
import re

class LearningPatternAnalyzer:
    """Analyzes student behavior to identify confusion patterns and learning style"""
    
    @staticmethod
    def detect_confusion_type(question: str, recent_qa: List[Dict]) -> Dict[str, Any]:
        """
        Analyze question to detect confusion type and patterns
        
        Returns:
            {
                "confusion_type": "concept" | "application" | "connection" | "vague",
                "is_repeat": bool,
                "suggested_approach": str,
                "keywords": List[str]
            }
        """
        q_lower = question.lower().strip()
        
        # Vague/general confusion signals
        vague_patterns = [
            "don't understand", "confused", "explain", "what does this mean",
            "can you clarify", "i'm lost", "help", "not clear"
        ]
        
        # Concept confusion (definitional)
        concept_patterns = [
            "what is", "what are", "define", "meaning of", "difference between"
        ]
        
        # Application confusion (how-to)
        application_patterns = [
            "how to", "how do", "how can", "when to use", "steps", "process"
        ]
        
        # Connection confusion (relationships)
        connection_patterns = [
            "why", "relate", "connect", "difference", "compare", "versus"
        ]
        
        confusion_type = "vague"
        if any(p in q_lower for p in concept_patterns):
            confusion_type = "concept"
        elif any(p in q_lower for p in application_patterns):
            confusion_type = "application"
        elif any(p in q_lower for p in connection_patterns):
            confusion_type = "connection"
        elif any(p in q_lower for p in vague_patterns):
            confusion_type = "vague"
        
        # Check for repeat questions (similar to recent ones)
        is_repeat = False
        if recent_qa:
            for qa in recent_qa[:3]:
                prev_q = qa.get('q', '').lower()
                # Simple similarity check
                common_words = set(q_lower.split()) & set(prev_q.split())
                if len(common_words) >= 3 or any(word in prev_q for word in q_lower.split() if len(word) > 4):
                    is_repeat = True
                    break
        
        # Extract key terms (nouns/concepts)
        keywords = [w for w in q_lower.split() if len(w) > 4 and w not in [
            'what', 'when', 'where', 'which', 'understand', 'explain', 'mean'
        ]][:5]
        
        # Suggest approach based on type
        approach_map = {
            "concept": "Provide clear definition with concrete example",
            "application": "Show step-by-step process with practical example",
            "connection": "Explain relationships and build conceptual bridge",
            "vague": "Recap current segment with simplified explanation"
        }
        
        return {
            "confusion_type": confusion_type,
            "is_repeat": is_repeat,
            "suggested_approach": approach_map[confusion_type],
            "keywords": keywords,
            "signal_strength": "high" if is_repeat else "medium" if confusion_type != "vague" else "low"
        }
    
    @staticmethod
    def analyze_quiz_patterns(quiz_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze quiz history to identify learning patterns
        
        Returns:
            {
                "trending": "improving" | "stable" | "declining",
                "struggle_areas": List[str],
                "mastered_areas": List[str],
                "recommendation": str
            }
        """
        if not quiz_results or len(quiz_results) < 2:
            return {
                "trending": "stable",
                "struggle_areas": [],
                "mastered_areas": [],
                "recommendation": "Continue learning"
            }
        
        # Extract scores
        scores = [r.get("score_percentage", 0) / 100.0 for r in quiz_results[:5]]
        
        # Detect trend
        if len(scores) >= 3:
            recent_avg = sum(scores[:2]) / 2
            older_avg = sum(scores[2:]) / len(scores[2:])
            
            if recent_avg > older_avg + 0.15:
                trending = "improving"
            elif recent_avg < older_avg - 0.15:
                trending = "declining"
            else:
                trending = "stable"
        else:
            trending = "stable"
        
        # Identify struggle areas from unclear segments
        struggle_areas = []
        for result in quiz_results[:3]:
            unclear = result.get("unclear_segments", [])
            struggle_areas.extend(unclear)
        
        # Count occurrences
        from collections import Counter
        struggle_counts = Counter(struggle_areas)
        struggle_areas = [seg for seg, cnt in struggle_counts.most_common(3)]
        
        # Mastered areas (high scores)
        mastered_areas = [
            r.get("topic", "") for r in quiz_results[:3]
            if r.get("score_percentage", 0) >= 80
        ][:3]
        
        # Generate recommendation
        if trending == "declining":
            rec = "Review fundamentals before advancing"
        elif trending == "improving":
            rec = "Good progress! Consider increasing difficulty"
        elif struggle_areas:
            rec = f"Focus on {struggle_areas[0]} with targeted practice"
        else:
            rec = "Steady progress, continue current pace"
        
        return {
            "trending": trending,
            "struggle_areas": struggle_areas,
            "mastered_areas": mastered_areas,
            "recommendation": rec
        }
    
    @staticmethod
    def extract_engagement_profile(memory: Any, topic: str) -> Dict[str, Any]:
        """
        SIMPLIFIED: Extract student engagement profile from memory (no difficulty rating)
        
        Returns:
            {
                "engagement_level": "high" | "medium" | "low",
                "preferred_learning_style": str,
                "needs": List[str]
            }
        """
        try:
            # Get interaction data
            recent_qa = memory.get_recent_qa(k=5)
            unmastered = memory.get_unmastered_objectives(topic)
            recent_quizzes = memory.get_recent_quiz_results(topic, k=2)
            
            # Engagement level based on question frequency
            qa_count = len(recent_qa)
            if qa_count >= 3:
                engagement = "high"
            elif qa_count >= 1:
                engagement = "medium"
            else:
                engagement = "low"
            
            # Learning style inference from question types
            if qa_count >= 2:
                how_to_count = sum(1 for qa in recent_qa if 'how' in qa.get('q', '').lower())
                why_count = sum(1 for qa in recent_qa if 'why' in qa.get('q', '').lower())
                
                if how_to_count > why_count:
                    style = "practical/applied"
                elif why_count > how_to_count:
                    style = "conceptual/theoretical"
                else:
                    style = "balanced"
            else:
                style = "balanced"  # No assumptions if not enough data
            
            # Identify needs from quiz performance and objectives
            needs = []
            if recent_quizzes:
                avg_score = sum(q.get("score_percentage", 0) for q in recent_quizzes) / len(recent_quizzes)
                if avg_score < 60:
                    needs.append("More examples and practice")
            
            if len(unmastered) > 3:
                needs.append("Focused practice on specific objectives")
            
            if engagement == "high" and recent_quizzes and recent_quizzes[0].get("score_percentage", 100) < 50:
                needs.append("Different teaching approach - current method not working")
            
            return {
                "engagement_level": engagement,
                "preferred_learning_style": style,
                "needs": needs
            }
        
        except Exception as e:
            return {
                "engagement_level": "medium",
                "preferred_learning_style": "balanced",
                "needs": []
            }

