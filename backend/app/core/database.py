# backend/app/core/database.py
import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

class Database:
    def __init__(self, db_path: str = "agentic_tutor.db"):
        self.db_path = db_path
        self.STUDENT_ID = "student_001"  # Single user system
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create topic_proficiency table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_proficiency (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT UNIQUE NOT NULL,
                accuracy REAL DEFAULT 0.0,
                strength TEXT DEFAULT 'new',
                attempts INTEGER DEFAULT 0,
                last_assessed_at TIMESTAMP
            )
        """)
        
        # Create interaction_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interaction_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                interaction_type TEXT NOT NULL,
                payload TEXT,
                response TEXT
            )
        """)
        
        # Create lesson_sessions table for storing complete lesson history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lesson_sessions (
                session_id TEXT PRIMARY KEY,
                document_name TEXT NOT NULL,
                document_type TEXT,
                concepts TEXT,  -- JSON array of concepts
                document_structure TEXT,  -- JSON of hierarchical structure
                study_plan TEXT,  -- JSON array of study plan
                conversation_history TEXT,  -- JSON array of messages
                quiz_results TEXT,  -- JSON array of quiz results
                final_evaluation TEXT,  -- JSON of final evaluation
                exam_context TEXT,  -- JSON of exam context (type, details)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'active'  -- active, completed, abandoned
            )
        """)
        
        # Create lesson_messages table for detailed chat history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lesson_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,  -- 'tutor' or 'student'
                content TEXT NOT NULL,
                sources TEXT,  -- JSON array of sources
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES lesson_sessions(session_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def get_topic_proficiency(self, topic: str = None) -> Dict[str, Any]:
        """Get proficiency for a topic or all topics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if topic:
            cursor.execute("""
                SELECT topic, accuracy, strength, attempts, last_assessed_at
                FROM topic_proficiency
                WHERE topic = ?
            """, (topic,))
        else:
            cursor.execute("""
                SELECT topic, accuracy, strength, attempts, last_assessed_at
                FROM topic_proficiency
                ORDER BY last_assessed_at DESC
            """)
        
        results = cursor.fetchall()
        conn.close()
        
        proficiency = {}
        for row in results:
            proficiency[row[0]] = {
                "accuracy": row[1],
                "strength": row[2],
                "attempts": row[3],
                "last_assessed_at": row[4]
            }
        
        return proficiency
    
    def update_topic_proficiency(self, topic: str, accuracy: float, 
                                attempts: int = None, strength: str = None) -> bool:
        """Update proficiency for a topic with flexible parameters"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Determine strength if not provided
            if strength is None:
                if accuracy < 0.6:
                    strength = "weak"
                elif accuracy < 0.8:
                    strength = "improving"
                else:
                    strength = "strong"
            
            # Get current attempts if not provided
            if attempts is None:
                cursor.execute("""
                    SELECT attempts FROM topic_proficiency WHERE topic = ?
                """, (topic,))
                result = cursor.fetchone()
                current_attempts = result[0] if result else 0
                attempts = current_attempts + 1
            
            # Insert or update proficiency
            cursor.execute("""
                INSERT OR REPLACE INTO topic_proficiency
                (topic, accuracy, strength, attempts, last_assessed_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (topic, accuracy, strength, attempts))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating proficiency: {e}")
            return False
    
    def create_lesson_session(self, session_id: str, document_name: str, 
                             document_type: str, concepts: List[Dict], 
                             study_plan: List[Dict], document_structure: Dict = None,
                             exam_context: Dict = None) -> bool:
        """Create a new lesson session with optional document structure and exam context"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO lesson_sessions 
                (session_id, document_name, document_type, concepts, study_plan, document_structure, exam_context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, document_name, document_type, 
                  json.dumps(concepts), json.dumps(study_plan), 
                  json.dumps(document_structure) if document_structure else None,
                  json.dumps(exam_context) if exam_context else None))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error creating lesson session: {e}")
            return False
    
    def add_lesson_message(self, session_id: str, role: str, content: str, 
                          sources: List[str] = None) -> bool:
        """Add a message to lesson conversation history"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO lesson_messages 
                (session_id, role, content, sources)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, 
                  json.dumps(sources) if sources else None))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding lesson message: {e}")
            return False
    
    def get_lesson_messages(self, session_id: str) -> List[Dict]:
        """Get all messages for a lesson session"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, sources, timestamp
                FROM lesson_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "role": row[0],
                    "content": row[1],
                    "sources": json.loads(row[2]) if row[2] else [],
                    "timestamp": row[3]
                }
                for row in results
            ]
        except Exception as e:
            print(f"Error getting lesson messages: {e}")
            return []
    
    def update_lesson_session(self, session_id: str, **updates) -> bool:
        """Update lesson session with new data"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key in ['conversation_history', 'quiz_results', 'final_evaluation']:
                    set_clauses.append(f"{key} = ?")
                    values.append(json.dumps(value))
                elif key == 'completed_at':
                    set_clauses.append(f"{key} = CURRENT_TIMESTAMP")
                else:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
            
            if set_clauses:
                values.append(session_id)
                query = f"UPDATE lesson_sessions SET {', '.join(set_clauses)} WHERE session_id = ?"
                cursor.execute(query, values)
                conn.commit()
            
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating lesson session: {e}")
            return False
    
    def get_lesson_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific lesson session"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, document_name, document_type, concepts, 
                       study_plan, conversation_history, quiz_results, 
                       final_evaluation, exam_context, created_at, completed_at, status
                FROM lesson_sessions
                WHERE session_id = ?
            """, (session_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "session_id": result[0],
                    "document_name": result[1],
                    "document_type": result[2],
                    "concepts": json.loads(result[3]) if result[3] else [],
                    "study_plan": json.loads(result[4]) if result[4] else [],
                    "conversation_history": json.loads(result[5]) if result[5] else [],
                    "quiz_results": json.loads(result[6]) if result[6] else [],
                    "final_evaluation": json.loads(result[7]) if result[7] else None,
                    "exam_context": json.loads(result[8]) if result[8] else None,
                    "created_at": result[9],
                    "completed_at": result[10],
                    "status": result[11]
                }
            return None
        except Exception as e:
            print(f"Error getting lesson session: {e}")
            return None
    
    def get_all_lesson_sessions(self) -> List[Dict]:
        """Get all lesson sessions for the user"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, document_name, document_type, 
                       created_at, completed_at, status
                FROM lesson_sessions
                ORDER BY created_at DESC
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "session_id": row[0],
                    "document_name": row[1],
                    "document_type": row[2],
                    "created_at": row[3],
                    "completed_at": row[4],
                    "status": row[5]
                }
                for row in results
            ]
        except Exception as e:
            print(f"Error getting lesson sessions: {e}")
            return []
    
    def log_interaction(self, interaction_type: str, payload: Dict, response: Dict) -> bool:
        """Log an interaction"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interaction_logs (interaction_type, payload, response)
                VALUES (?, ?, ?)
            """, (interaction_type, json.dumps(payload), json.dumps(response)))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging interaction: {e}")
            return False
    
    def get_interaction_history(self, limit: int = 10) -> List[Dict]:
        """Get recent interaction history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT interaction_type, payload, response, timestamp
            FROM interaction_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "type": row[0],
                "payload": json.loads(row[1]),
                "response": json.loads(row[2]),
                "timestamp": row[3]
            }
            for row in results
        ]

# Global database instance
db = Database()