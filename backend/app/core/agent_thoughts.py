"""
Agent Thoughts Tracker - Capture and expose AI decision-making in real-time
"""
from typing import List, Dict, Any
from datetime import datetime
from contextvars import ContextVar

# Context variable for async-safe per-request storage
_thoughts_context: ContextVar[List[Dict[str, Any]]] = ContextVar('thoughts', default=[])

class AgentThoughtsTracker:
    """
    Async-safe singleton to capture agent thoughts/decisions during a request.
    Uses contextvars for proper async context isolation.
    """
    
    @classmethod
    def clear(cls):
        """Clear thoughts for current request"""
        _thoughts_context.set([])
    
    @classmethod
    def add(cls, agent: str, thought: str, emoji: str = "🤔", metadata: Dict[str, Any] = None):
        """Add a thought from an agent"""
        thoughts = _thoughts_context.get()
        
        new_thought = {
            "agent": agent,
            "thought": thought,
            "emoji": emoji,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        thoughts.append(new_thought)
        _thoughts_context.set(thoughts)
        
        # Also print to console for debugging
        print(f"[{agent}] {emoji} {thought}")
    
    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        """Get all thoughts for current request"""
        return _thoughts_context.get()
    
    @classmethod
    def get_summary(cls) -> str:
        """Get a summary of all agent thoughts"""
        thoughts = cls.get_all()
        if not thoughts:
            return "No agent thoughts recorded"
        
        summary_parts = []
        for t in thoughts:
            summary_parts.append(f"{t['emoji']} {t['agent']}: {t['thought']}")
        
        return " → ".join(summary_parts)


# Global instance
thoughts_tracker = AgentThoughtsTracker()

