#!/usr/bin/env python3
"""
Migration script to add exam_context column to existing lesson_sessions table.
Run this once to update existing databases.
"""

import sqlite3
import os

def migrate_database(db_path: str = "agentic_tutor.db"):
    """Add exam_context column to lesson_sessions table if it doesn't exist"""
    
    print(f"🔄 Migrating database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(lesson_sessions)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "exam_context" in columns:
            print("✅ Column 'exam_context' already exists. No migration needed.")
            conn.close()
            return True
        
        # Add the new column
        print("📝 Adding 'exam_context' column...")
        cursor.execute("""
            ALTER TABLE lesson_sessions 
            ADD COLUMN exam_context TEXT
        """)
        
        conn.commit()
        conn.close()
        
        print("✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    # Migrate main database
    migrate_database("agentic_tutor.db")
    
    # Also migrate backend database if it exists
    if os.path.exists("backend/agentic_tutor.db"):
        migrate_database("backend/agentic_tutor.db")
    
    print("\n✨ Migration complete!")

