"""
Redis-backed session memory for coherent, segment-based learning
"""

import redis
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SessionMemory:
    """
    Redis-backed memory for learning sessions.
    Stores taught content, segments, progress, and context.
    """
    
    def __init__(self, redis_client: redis.Redis, session_id: str):
        self.redis = redis_client
        self.session_id = session_id
        self.prefix = f"session:{session_id}"
        self.default_ttl = 86400  # 24 hours
    
    # ===== Taught Content (for quiz grounding) =====
    
    def store_taught_segment(self, topic: str, segment_id: str, content: str) -> bool:
        """Store content that was taught for a specific segment"""
        try:
            key = f"{self.prefix}:taught:{topic}:{segment_id}"
            self.redis.setex(key, self.default_ttl, content)
            logger.info(f"Stored taught content: {topic}/{segment_id} ({len(content)} chars)")
            return True
        except Exception as e:
            logger.error(f"Failed to store taught segment: {e}")
            return False

    def store_taught_segment_json(self, topic: str, segment_id: str, obj: Dict[str, Any]) -> bool:
        """Store structured taught segment JSON (full_text, summary, excerpts, sources, ts)."""
        try:
            key = f"{self.prefix}:taught_json:{topic}:{segment_id}"
            # Attach timestamp if not present
            if "ts" not in obj:
                obj["ts"] = datetime.now().isoformat()
            self.redis.setex(key, self.default_ttl, json.dumps(obj))
            return True
        except Exception as e:
            logger.error(f"Failed to store taught segment JSON: {e}")
            return False
    
    def get_taught_segment(self, topic: str, segment_id: str) -> str:
        """Get content for a specific taught segment"""
        try:
            key = f"{self.prefix}:taught:{topic}:{segment_id}"
            content = self.redis.get(key)
            return content or ""
        except Exception as e:
            logger.error(f"Failed to get taught segment: {e}")
            return ""
    
    def get_taught_segment_json(self, topic: str, segment_id: str) -> Optional[Dict[str, Any]]:
        """Get structured taught segment JSON for a specific segment"""
        try:
            key = f"{self.prefix}:taught_json:{topic}:{segment_id}"
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get taught segment JSON: {e}")
            return None
    
    def get_all_taught_segments(self, topic: str) -> str:
        """Get all taught content for a topic (for cumulative quizzes)"""
        try:
            pattern = f"{self.prefix}:taught:{topic}:*"
            keys = sorted(self.redis.keys(pattern))
            contents = []
            for key in keys:
                content = self.redis.get(key)
                if content:
                    contents.append(content)
            return "\n\n---\n\n".join(contents)
        except Exception as e:
            logger.error(f"Failed to get all taught segments: {e}")
            return ""

    def get_taught_segments_for_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get structured taught segments JSON for a topic, most recent first."""
        try:
            pattern = f"{self.prefix}:taught_json:{topic}:*"
            keys = sorted(self.redis.keys(pattern), reverse=True)
            segments = []
            for key in keys:
                data = self.redis.get(key)
                if data:
                    try:
                        segments.append(json.loads(data))
                    except Exception:
                        continue
            return segments
        except Exception as e:
            logger.error(f"Failed to get taught segments JSON: {e}")
            return []
    
    # ===== Segment Plans (topic decomposition) =====
    
    def store_segment_plan(self, topic: str, segments: List[Dict]) -> bool:
        """Store how a topic is broken into segments"""
        try:
            key = f"{self.prefix}:segments:{topic}"
            self.redis.setex(key, self.default_ttl, json.dumps(segments))
            logger.info(f"Stored segment plan for {topic}: {len(segments)} segments")
            return True
        except Exception as e:
            logger.error(f"Failed to store segment plan: {e}")
            return False
    
    def get_segment_plan(self, topic: str) -> List[Dict]:
        """Get segment breakdown for a topic"""
        try:
            key = f"{self.prefix}:segments:{topic}"
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return []
        except Exception as e:
            logger.error(f"Failed to get segment plan: {e}")
            return []
    
    def get_next_segment(self, topic: str) -> Optional[Dict]:
        """Get the next untaught segment for a topic"""
        try:
            segments = self.get_segment_plan(topic)
            progress = self.get_topic_progress(topic)
            # Diagnostics: show what memory believes
            try:
                seg_ids = [s.get("segment_id") for s in segments]
                print(f"[MEMORY] get_next_segment('{topic}') | segments={len(segments)} first={seg_ids[:2]} progress_keys={list(progress.keys())}")
                if progress:
                    sample = {k: v.get("status") for k, v in list(progress.items())[:3]}
                    print(f"[MEMORY] progress sample: {sample}")
            except Exception:
                pass
            
            for segment in segments:
                segment_id = segment.get("segment_id")
                if segment_id not in progress or progress[segment_id].get("status") != "completed":
                    return segment
            return None
        except Exception as e:
            logger.error(f"Failed to get next segment: {e}")
            return None
    
    # ===== Progress Tracking =====
    
    def mark_segment_started(self, topic: str, segment_id: str) -> bool:
        """Mark segment as in progress"""
        try:
            key = f"{self.prefix}:progress:{topic}"
            progress = self._get_progress(key)
            progress[segment_id] = {
                "status": "in_progress",
                "started_at": datetime.now().isoformat()
            }
            self.redis.setex(key, self.default_ttl, json.dumps(progress))
            return True
        except Exception as e:
            logger.error(f"Failed to mark segment started: {e}")
            return False
    
    def mark_segment_completed(self, topic: str, segment_id: str, 
                              quiz_score: float = None, time_spent: int = None) -> bool:
        """Mark segment as completed with optional quiz score"""
        try:
            key = f"{self.prefix}:progress:{topic}"
            progress = self._get_progress(key)
            
            if segment_id not in progress:
                progress[segment_id] = {}
            
            progress[segment_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "quiz_score": quiz_score,
                "time_spent_minutes": time_spent
            })
            
            self.redis.setex(key, self.default_ttl, json.dumps(progress))
            logger.info(f"Marked segment completed: {topic}/{segment_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark segment completed: {e}")
            return False
    
    def get_topic_progress(self, topic: str) -> Dict:
        """Get completion status of all segments in a topic"""
        try:
            key = f"{self.prefix}:progress:{topic}"
            return self._get_progress(key)
        except Exception as e:
            logger.error(f"Failed to get topic progress: {e}")
            return {}
    
    # ===== Short-term Session Summaries (last-k) =====
    
    def push_session_summary(self, summary: Dict, k: int = 3) -> bool:
        """Push a concise session summary (e.g., per segment) and trim to last-k."""
        try:
            key = f"{self.prefix}:last_summaries"
            self.redis.lpush(key, json.dumps(summary))
            # Keep only last k
            self.redis.ltrim(key, 0, max(k - 1, 0))
            # Reset TTL on list container via expire
            self.redis.expire(key, self.default_ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to push session summary: {e}")
            return False

    def get_last_session_summaries(self, k: int = 3) -> List[Dict]:
        """Get up to last-k concise session summaries (most recent first)."""
        try:
            key = f"{self.prefix}:last_summaries"
            items = self.redis.lrange(key, 0, max(k - 1, 0)) or []
            summaries = []
            for it in items:
                try:
                    summaries.append(json.loads(it))
                except Exception:
                    # Skip malformed entries
                    continue
            return summaries
        except Exception as e:
            logger.error(f"Failed to get last session summaries: {e}")
            return []

    # ===== Session Context (for planner) =====
    
    def update_context(self, updates: Dict) -> bool:
        """Update session context (current topic/segment, student profile, etc)"""
        try:
            key = f"{self.prefix}:context"
            context = self._get_context(key)
            context.update(updates)
            context["updated_at"] = datetime.now().isoformat()
            self.redis.setex(key, self.default_ttl, json.dumps(context))
            return True
        except Exception as e:
            logger.error(f"Failed to update context: {e}")
            return False
    
    def get_context(self) -> Dict:
        """Get full session context for planner decisions"""
        try:
            key = f"{self.prefix}:context"
            return self._get_context(key)
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return {}
    
    # ===== Student Interactions =====
    
    def add_student_question(self, topic: str, segment_id: str, question: str) -> bool:
        """Track questions asked during a segment"""
        try:
            key = f"{self.prefix}:questions:{topic}:{segment_id}"
            questions = json.loads(self.redis.get(key) or '[]')
            questions.append({
                "question": question,
                "timestamp": datetime.now().isoformat()
            })
            self.redis.setex(key, self.default_ttl, json.dumps(questions))
            return True
        except Exception as e:
            logger.error(f"Failed to add student question: {e}")
            return False

    def add_exercise_interaction(self, topic: str, segment_id: str, payload: Dict[str, Any]) -> bool:
        """Store a single exercise interaction: {prompt, student_answer, correct?, ts}."""
        try:
            key = f"{self.prefix}:exercises:{topic}:{segment_id}"
            interactions = json.loads(self.redis.get(key) or '[]')
            interactions.append({
                **payload,
                "timestamp": datetime.now().isoformat()
            })
            self.redis.setex(key, self.default_ttl, json.dumps(interactions))
            return True
        except Exception as e:
            logger.error(f"Failed to add exercise interaction: {e}")
            return False

    # ===== Quiz Results (topic-level) =====
    
    def append_quiz_result(self, topic: str, result: Dict[str, Any]) -> bool:
        """Append a quiz result for a topic (keep small cap)."""
        try:
            key = f"{self.prefix}:quiz:{topic}:results"
            self.redis.lpush(key, json.dumps(result))
            # Keep last 10 results
            self.redis.ltrim(key, 0, 9)
            self.redis.expire(key, self.default_ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to append quiz result: {e}")
            return False

    def get_recent_quiz_results(self, topic: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get recent quiz results for a topic."""
        try:
            key = f"{self.prefix}:quiz:{topic}:results"
            items = self.redis.lrange(key, 0, max(k - 1, 0)) or []
            results = []
            for it in items:
                try:
                    results.append(json.loads(it))
                except Exception:
                    continue
            return results
        except Exception as e:
            logger.error(f"Failed to get recent quiz results: {e}")
            return []

    # ===== Q&A Conversation History (NEW) =====
    
    def append_qa_exchange(self, question: str, answer: str) -> bool:
        """Store Q&A exchange in conversation history (last 5)"""
        try:
            import time
            key = f"{self.prefix}:qa_history"
            exchange = {
                "q": question,
                "a": answer[:200],  # Truncate to summary length
                "ts": time.time()
            }
            self.redis.lpush(key, json.dumps(exchange))
            self.redis.ltrim(key, 0, 4)  # Keep last 5
            self.redis.expire(key, self.default_ttl)
            logger.info(f"Stored Q&A exchange")
            return True
        except Exception as e:
            logger.error(f"Failed to store Q&A: {e}")
            return False
    
    def get_recent_qa(self, k: int = 3) -> List[Dict[str, str]]:
        """Get last k Q&A exchanges"""
        try:
            key = f"{self.prefix}:qa_history"
            exchanges = []
            raw_list = self.redis.lrange(key, 0, k - 1)
            for item in raw_list:
                try:
                    exchanges.append(json.loads(item))
                except Exception:
                    continue
            return exchanges
        except Exception as e:
            logger.error(f"Failed to get recent Q&A: {e}")
            return []
    
    def store_qa(self, question: str, answer: str, topic: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Store Q&A exchange with optional metadata (for exercise assessments, etc.)"""
        try:
            import time
            key = f"{self.prefix}:qa_detailed"
            exchange = {
                "q": question,
                "a": answer[:500],  # Store more detail than basic qa_history
                "topic": topic,
                "metadata": metadata or {},
                "ts": time.time()
            }
            self.redis.lpush(key, json.dumps(exchange))
            self.redis.ltrim(key, 0, 9)  # Keep last 10 detailed exchanges
            self.redis.expire(key, self.default_ttl)
            logger.info(f"Stored detailed Q&A exchange for {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to store detailed Q&A: {e}")
            return False
    
    def get_weak_exercises(self, topic: str, segment_id: str) -> List[Dict[str, Any]]:
        """Get exercises student got wrong for a segment"""
        try:
            key = f"{self.prefix}:exercises:{topic}:{segment_id}"
            data = self.redis.get(key)
            interactions = json.loads(data) if data else []
            weak = [ex for ex in interactions if ex.get("correct") is False]
            return weak
        except Exception as e:
            logger.error(f"Failed to get weak exercises: {e}")
            return []

    # ===== Episodic Summary =====
    
    def set_episodic_summary(self, text: str, meta: Optional[Dict[str, Any]] = None) -> bool:
        """Store the latest episodic summary text with optional metadata."""
        try:
            key = f"{self.prefix}:episodic_summary"
            payload = {"text": text, "ts": datetime.now().isoformat(), "meta": meta or {}}
            self.redis.setex(key, self.default_ttl, json.dumps(payload))
            return True
        except Exception as e:
            logger.error(f"Failed to set episodic summary: {e}")
            return False

    def get_episodic_summary(self) -> Dict[str, Any]:
        """Get the latest episodic summary payload {text, ts, meta}."""
        try:
            key = f"{self.prefix}:episodic_summary"
            data = self.redis.get(key)
            return json.loads(data) if data else {}
        except Exception as e:
            logger.error(f"Failed to get episodic summary: {e}")
            return {}
    
    # ===== Session Metadata =====
    
    def init_session(self, metadata: Dict, clear_existing: bool = True) -> bool:
        """
        Initialize session with metadata.
        
        Args:
            metadata: Session metadata to store
            clear_existing: If True, clear any existing data for this session ID first
        """
        try:
            # Clear existing session data to ensure fresh start
            if clear_existing:
                self.clear_session()
            
            key = f"{self.prefix}:meta"
            metadata.update({
                "created_at": datetime.now().isoformat(),
                "session_id": self.session_id
            })
            self.redis.setex(key, self.default_ttl, json.dumps(metadata))
            logger.info(f"Initialized session {self.session_id} (cleared_existing={clear_existing})")
            return True
        except Exception as e:
            logger.error(f"Failed to init session: {e}")
            return False
    
    def clear_session(self) -> bool:
        """Clear all data for this session"""
        try:
            pattern = f"{self.prefix}:*"
            keys = self.redis.keys(pattern)
            if keys:
                deleted = self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} keys for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False
    
    def get_session_metadata(self) -> Dict:
        """Get session metadata"""
        try:
            key = f"{self.prefix}:meta"
            data = self.redis.get(key)
            return json.loads(data) if data else {}
        except Exception as e:
            logger.error(f"Failed to get session metadata: {e}")
            return {}
    
    # ===== Exam Context & Learning Goals =====
    
    def set_exam_context(self, exam_type: str, exam_details: Dict = None) -> bool:
        """
        Store exam-focused learning context.
        exam_type: "JEE", "UPSC", "SAT", "GRE", "general", etc.
        exam_details: {
            "exam_name": "JEE Main 2025",
            "focus_areas": ["Physics", "Math"],
            "difficulty_preference": "high_technical_depth" | "broad_coverage"
        }
        """
        try:
            key = f"{self.prefix}:exam_context"
            context = {
                "exam_type": exam_type,
                "exam_details": exam_details or {},
                "set_at": datetime.now().isoformat()
            }
            self.redis.setex(key, self.default_ttl, json.dumps(context))
            logger.info(f"Set exam context: {exam_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to set exam context: {e}")
            return False
    
    def get_exam_context(self) -> Dict:
        """Get exam context for tailoring content"""
        try:
            key = f"{self.prefix}:exam_context"
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            # Default to general learning if not set
            return {"exam_type": "general", "exam_details": {}}
        except Exception as e:
            logger.error(f"Failed to get exam context: {e}")
            return {"exam_type": "general", "exam_details": {}}
    
    # ===== Difficulty Ratings (Student Confidence) =====
    
    def store_difficulty_rating(self, topic: str, segment_id: str, rating: int) -> bool:
        """
        Store student's difficulty rating for a segment.
        rating: 1-5 where 1 = too hard/uncomfortable, 5 = too easy/boring
        """
        try:
            key = f"{self.prefix}:difficulty:{topic}:{segment_id}"
            data = {
                "rating": rating,
                "timestamp": datetime.now().isoformat()
            }
            self.redis.setex(key, self.default_ttl, json.dumps(data))
            
            # Also update rolling average for topic
            self._update_avg_difficulty(topic, rating)
            logger.info(f"Stored difficulty rating: {topic}/{segment_id} = {rating}")
            return True
        except Exception as e:
            logger.error(f"Failed to store difficulty rating: {e}")
            return False
    
    def get_difficulty_rating(self, topic: str, segment_id: str) -> Optional[int]:
        """Get difficulty rating for a specific segment"""
        try:
            key = f"{self.prefix}:difficulty:{topic}:{segment_id}"
            data = self.redis.get(key)
            if data:
                return json.loads(data).get("rating")
            return None
        except Exception as e:
            logger.error(f"Failed to get difficulty rating: {e}")
            return None
    
    def get_avg_difficulty_for_topic(self, topic: str) -> float:
        """Get average difficulty rating for a topic"""
        try:
            key = f"{self.prefix}:avg_difficulty:{topic}"
            data = self.redis.get(key)
            if data:
                return float(data)
            return 3.0  # Neutral default
        except Exception as e:
            logger.error(f"Failed to get avg difficulty: {e}")
            return 3.0
    
    def _update_avg_difficulty(self, topic: str, new_rating: int):
        """Update rolling average difficulty for topic"""
        try:
            key = f"{self.prefix}:avg_difficulty:{topic}"
            current_avg = self.get_avg_difficulty_for_topic(topic)
            
            # Simple exponential moving average (weight new rating 30%)
            new_avg = 0.7 * current_avg + 0.3 * new_rating
            self.redis.setex(key, self.default_ttl, str(new_avg))
        except Exception as e:
            logger.error(f"Failed to update avg difficulty: {e}")
    
    # ===== Learning Objectives Mastery Tracking =====
    
    def mark_objective_mastered(self, topic: str, objective: str, confidence: float) -> bool:
        """
        Track mastery of specific learning objectives.
        confidence: 0.0-1.0 based on quiz performance and difficulty rating
        """
        try:
            key = f"{self.prefix}:objectives:{topic}"
            objectives = json.loads(self.redis.get(key) or '{}')
            
            objectives[objective] = {
                "mastered": confidence >= 0.7,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            self.redis.setex(key, self.default_ttl, json.dumps(objectives))
            logger.info(f"Marked objective: {objective[:50]}... = {confidence:.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to mark objective: {e}")
            return False
    
    def get_unmastered_objectives(self, topic: str) -> List[str]:
        """Get objectives student hasn't mastered yet"""
        try:
            key = f"{self.prefix}:objectives:{topic}"
            data = self.redis.get(key)
            if not data:
                return []
            
            objectives = json.loads(data)
            return [
                obj for obj, details in objectives.items()
                if not details.get("mastered", False)
            ]
        except Exception as e:
            logger.error(f"Failed to get unmastered objectives: {e}")
            return []
    
    def get_objectives_summary(self, topic: str) -> Dict:
        """Get summary of all objectives for a topic"""
        try:
            key = f"{self.prefix}:objectives:{topic}"
            data = self.redis.get(key)
            return json.loads(data) if data else {}
        except Exception as e:
            logger.error(f"Failed to get objectives summary: {e}")
            return {}
    
    # ===== Spaced Repetition System (SM-2 Algorithm) =====
    
    def schedule_review(self, topic: str, mastery_score: float) -> bool:
        """
        Schedule next review using simplified SM-2 algorithm.
        mastery_score: 0.0-1.0 (from quiz performance)
        """
        try:
            import time
            
            # Calculate interval based on mastery score
            if mastery_score >= 0.9:
                interval_days = 7  # Strong mastery - review in 1 week
            elif mastery_score >= 0.7:
                interval_days = 3  # Good mastery - review in 3 days
            elif mastery_score >= 0.5:
                interval_days = 1  # Weak mastery - review tomorrow
            else:
                interval_days = 0.25  # Very weak - review in 6 hours
            
            next_review = datetime.now() + timedelta(days=interval_days)
            timestamp = next_review.timestamp()
            
            # Store in sorted set (score = timestamp for time-based retrieval)
            key = f"{self.prefix}:review_schedule"
            self.redis.zadd(key, {topic: timestamp})
            self.redis.expire(key, self.default_ttl)
            
            logger.info(f"Scheduled review for {topic}: {interval_days} days (mastery={mastery_score:.2f})")
            return True
        except Exception as e:
            logger.error(f"Failed to schedule review: {e}")
            return False
    
    def get_due_reviews(self) -> List[str]:
        """Get topics that are due for review"""
        try:
            import time
            key = f"{self.prefix}:review_schedule"
            now = time.time()
            
            # Get all topics with timestamp <= now
            due_topics = self.redis.zrangebyscore(key, 0, now)
            return list(due_topics) if due_topics else []
        except Exception as e:
            logger.error(f"Failed to get due reviews: {e}")
            return []
    
    def remove_review(self, topic: str) -> bool:
        """Remove topic from review schedule"""
        try:
            key = f"{self.prefix}:review_schedule"
            self.redis.zrem(key, topic)
            return True
        except Exception as e:
            logger.error(f"Failed to remove review: {e}")
            return False
    
    def get_next_review_time(self, topic: str) -> Optional[datetime]:
        """Get next scheduled review time for a topic"""
        try:
            key = f"{self.prefix}:review_schedule"
            timestamp = self.redis.zscore(key, topic)
            if timestamp:
                return datetime.fromtimestamp(timestamp)
            return None
        except Exception as e:
            logger.error(f"Failed to get next review time: {e}")
            return None
    
    # ===== Helper Methods =====
    
    def _get_progress(self, key: str) -> Dict:
        """Helper to get progress dict"""
        data = self.redis.get(key)
        return json.loads(data) if data else {}
    
    def _get_context(self, key: str) -> Dict:
        """Helper to get context dict"""
        data = self.redis.get(key)
        return json.loads(data) if data else {}
    
    def clear_session(self) -> bool:
        """Clear all session data (for testing/reset)"""
        try:
            pattern = f"{self.prefix}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
            logger.info(f"Cleared session: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False

# ===== Redis Client Factory =====

def create_redis_client() -> redis.Redis:
    """Create Redis client with fallback"""
    import os
    try:
        client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        # Test connection
        client.ping()
        logger.info("Redis connection successful")
        return client
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        logger.warning("Falling back to in-memory state")
        return None

# Global Redis client (lazy initialization)
_redis_client = None

def get_redis_client() -> Optional[redis.Redis]:
    """Get or create global Redis client"""
    global _redis_client
    if _redis_client is None:
        _redis_client = create_redis_client()
    return _redis_client
