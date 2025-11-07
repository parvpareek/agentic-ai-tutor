"""
Comprehensive logging for agent sessions
Logs PDF structure, LLM prompts/responses, and agent actions to file
"""

import json
import os
from datetime import datetime
from typing import Any, Dict
from pathlib import Path

class SessionLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.log_dir = Path("/home/parv/dev/projects/knit/backend/logs")
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"session_{session_id}_{timestamp}.log"
        
        self._write_header()
    
    def _write_header(self):
        """Write session header"""
        with open(self.log_file, 'w') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"AGENT SESSION LOG\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"=" * 80 + "\n\n")
    
    def log_agent_action(self, agent_name: str, action: str, details: str = ""):
        """Log agent action"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{timestamp}] AGENT: {agent_name}\n")
            f.write(f"ACTION: {action}\n")
            if details:
                f.write(f"DETAILS: {details}\n")
            f.write(f"{'='*80}\n")
    
    def log_pdf_structure(self, structure: Dict[str, Any]):
        """Log parsed PDF structure"""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"PDF STRUCTURE\n")
            f.write(f"{'='*80}\n")
            f.write(json.dumps(structure, indent=2))
            f.write(f"\n{'='*80}\n\n")
    
    def log_llm_call(self, agent_name: str, purpose: str, prompt: str, response: str, duration: float = 0):
        """Log LLM prompt and response"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{timestamp}] LLM CALL - {agent_name}\n")
            f.write(f"PURPOSE: {purpose}\n")
            f.write(f"DURATION: {duration:.2f}s\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"PROMPT:\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{prompt}\n")
            f.write(f"{'-'*80}\n\n")
            f.write(f"RESPONSE:\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{response}\n")
            f.write(f"{'-'*80}\n\n")
    
    def log_data(self, label: str, data: Any):
        """Log structured data"""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"DATA: {label}\n")
            f.write(f"{'='*80}\n")
            if isinstance(data, (dict, list)):
                f.write(json.dumps(data, indent=2))
            else:
                f.write(str(data))
            f.write(f"\n{'='*80}\n\n")
    
    def log_error(self, agent_name: str, error: str, traceback: str = ""):
        """Log error"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'!'*80}\n")
            f.write(f"[{timestamp}] ERROR in {agent_name}\n")
            f.write(f"{'!'*80}\n")
            f.write(f"{error}\n")
            if traceback:
                f.write(f"\nTraceback:\n{traceback}\n")
            f.write(f"{'!'*80}\n\n")

# Global logger instance
_current_logger = None

def init_session_logger(session_id: str) -> SessionLogger:
    """Initialize a new session logger"""
    global _current_logger
    _current_logger = SessionLogger(session_id)
    return _current_logger

def get_logger() -> SessionLogger:
    """Get current session logger"""
    global _current_logger
    if _current_logger is None:
        _current_logger = SessionLogger("default")
    return _current_logger

