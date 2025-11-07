# backend/app/api/progress.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.database import db

router = APIRouter()

class ProgressUpdateRequest(BaseModel):
    topic: str
    accuracy: float

@router.get("/")
async def get_student_progress():
    """Get student progress with recommendations"""
    try:
        proficiency = db.get_topic_proficiency()
        
        # Calculate overall progress
        total_topics = len(proficiency)
        if total_topics == 0:
            return {
                "success": True,
                "progress": {
                    "total_topics": 0,
                    "completed_topics": 0,
                    "average_accuracy": 0,
                    "topics": {}
                },
                "recommendations": ["Start learning by uploading a document!"]
            }
        
        completed_topics = sum(1 for data in proficiency.values() 
                             if isinstance(data, dict) and data.get("accuracy", 0) >= 0.8)
        average_accuracy = sum(data.get("accuracy", 0) for data in proficiency.values() 
                             if isinstance(data, dict)) / total_topics
        
        # Generate recommendations
        weak_topics = [topic for topic, data in proficiency.items() 
                      if isinstance(data, dict) and data.get("accuracy", 0) < 0.6]
        strong_topics = [topic for topic, data in proficiency.items() 
                        if isinstance(data, dict) and data.get("accuracy", 0) >= 0.8]
        
        recommendations = []
        if weak_topics:
            recommendations.append(f"Focus on strengthening: {', '.join(weak_topics[:3])}")
        if strong_topics:
            recommendations.append(f"Great progress in: {', '.join(strong_topics[:2])}")
        if not recommendations:
            recommendations.append("Keep practicing to maintain your current level")
        
        return {
            "success": True,
            "progress": {
                "total_topics": total_topics,
                "completed_topics": completed_topics,
                "average_accuracy": round(average_accuracy * 100, 1),
                "topics": proficiency
            },
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update")
async def update_progress(req: ProgressUpdateRequest):
    """Update student progress for a specific topic"""
    try:
        from app.core.database import db
        success = db.update_topic_proficiency(req.topic, req.accuracy)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update progress")
        
        return {
            "success": True,
            "message": f"Updated progress for {req.topic} with accuracy {req.accuracy}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))