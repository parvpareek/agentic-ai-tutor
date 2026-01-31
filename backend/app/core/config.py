# backend/app/core/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./tutor.db")
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", "agentic_tutor.db")  # For SQLite fallback
    
    # Vector Store Configuration
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./chroma_db")  # For local fallback
    
    # ChromaDB Cloud Configuration (for ChromaDB Cloud managed service)
    CHROMA_API_KEY: str = os.getenv("CHROMA_API_KEY", "")  # ChromaDB Cloud API key
    CHROMA_TENANT: str = os.getenv("CHROMA_TENANT", "")  # ChromaDB Cloud tenant ID
    CHROMA_DATABASE: str = os.getenv("CHROMA_DATABASE", "")  # ChromaDB Cloud database name
    USE_CLOUD_CHROMA: bool = os.getenv("USE_CLOUD_CHROMA", "false").lower() == "true"  # Toggle cloud vs local
    
    # Application Configuration
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS Configuration
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "https://agentic-ai-tutor-eight.vercel.app/")
    
    # PDF Parser Service Configuration (nlm-ingestor)
    LLMSHERPA_API_URL: str = os.getenv(
        "LLMSHERPA_API_URL", 
        "http://127.0.0.1:5010/api/parseDocument?renderFormat=all&useNewIndentParser=true"
    )
    
    def __init__(self):
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

# Create global settings instance
settings = Settings()

