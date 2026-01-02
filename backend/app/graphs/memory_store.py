"""
LangGraph-based memory store for cross-thread learning context
Stores taught summaries with semantic search capabilities
"""
from typing import Dict, List, Any, Optional
from langgraph.checkpoint.memory import MemorySaver
from app.core.vectorstore import vectorstore
import json
import time

class AdaptiveMemoryStore:
    """
    Lightweight memory store for adaptive learning.
    Uses LangGraph's MemorySaver for thread state + our Chroma for semantic search.
    """
    
    def __init__(self):
        # LangGraph checkpointer for thread state
        self.checkpointer = MemorySaver()
        
        # Chroma for semantic summary storage (reuse existing taught_summaries collection)
        self.vector_store = vectorstore
    
    def store_taught_summary(
        self, 
        user_id: str, 
        topic: str, 
        segment_id: str, 
        summary: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store a taught segment summary for later retrieval.
        Namespaced by user_id for cross-session learning.
        """
        doc_id = f"{user_id}:{topic}:{segment_id}"
        
        full_metadata = {
            "user_id": user_id,
            "topic": topic,
            "segment_id": segment_id,
            "timestamp": time.time(),
            **(metadata or {})
        }
        
        try:
            self.vector_store.upsert_taught_summary(
                _id=doc_id,
                text=summary,
                metadata=full_metadata
            )
            print(f"[MEMORY_STORE] Stored summary for {topic}:{segment_id}")
        except Exception as e:
            print(f"[MEMORY_STORE] Error storing summary: {e}")
    
    def retrieve_related_summaries(
        self,
        user_id: str,
        query: str,
        topic: Optional[str] = None,
        k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k related taught summaries using semantic search.
        Useful for context when teaching new segments.
        """
        try:
            # Build query with topic if provided
            search_query = f"{topic} {query}" if topic else query
            
            results = self.vector_store.query_taught_summaries(search_query, k=k)
            
            summaries = []
            if results and 'documents' in results:
                docs = results['documents'][0] if results['documents'] else []
                metas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else []
                
                for doc, meta in zip(docs, metas):
                    # Filter by user_id if metadata available
                    if meta and meta.get('user_id') == user_id:
                        summaries.append({
                            "summary": doc,
                            "topic": meta.get("topic", ""),
                            "segment_id": meta.get("segment_id", ""),
                            "timestamp": meta.get("timestamp", 0)
                        })
            
            print(f"[MEMORY_STORE] Retrieved {len(summaries)} related summaries for query: {query[:50]}...")
            return summaries
        except Exception as e:
            print(f"[MEMORY_STORE] Error retrieving summaries: {e}")
            return []
    
    def store_proficiency(self, user_id: str, topic: str, score: float):
        """
        Store topic proficiency score.
        Uses simple key-value in metadata for now.
        """
        doc_id = f"{user_id}:proficiency:{topic}"
        summary_text = f"Proficiency for {topic}: {score:.2f}"
        
        self.vector_store.upsert_taught_summary(
            _id=doc_id,
            text=summary_text,
            metadata={
                "user_id": user_id,
                "topic": topic,
                "proficiency": score,
                "type": "proficiency",
                "timestamp": time.time()
            }
        )
    
    def get_proficiency(self, user_id: str, topic: str) -> Optional[float]:
        """
        Retrieve proficiency score for a topic.
        """
        # Note: This requires a lookup by exact ID, which Chroma doesn't support directly.
        # For now, we'll use the adaptive_state proficiency dict.
        # This is a placeholder for future enhancement.
        return None
    
    def clear_session(self, user_id: str, session_id: str):
        """
        Clear thread-specific state (for testing/reset).
        """
        # LangGraph MemorySaver doesn't have explicit delete, but state expires automatically
        print(f"[MEMORY_STORE] Session {session_id} state will auto-expire")


# Singleton instance
adaptive_memory = AdaptiveMemoryStore()
