# backend/app/tools/tutor_tools.py
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
from app.core.database import db
from app.core.vectorstore import query_top_k
from app.core.config import settings

# Initialize Google Gemini LLM for tool usage
llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    temperature=0.7,
    google_api_key=settings.GOOGLE_API_KEY
)

# ===== Student Profile Tools =====

@tool
def get_student_profile(student_id: str) -> dict:
    """
    Fetch complete student profile from database.
    Returns student info, target exam, level, and proficiency data.
    """
    # Get basic profile (you'll implement this in database.py)
    profile = {
        "student_id": student_id,
        "target_exam": "JEE",  # Default, should come from DB
        "current_level": "intermediate",
        "topics_studied": []
    }
    
    # Get proficiency data
    proficiency = db.get_topic_proficiency()
    profile["topic_proficiency"] = proficiency
    
    return profile

@tool
def update_student_profile(student_id: str, target_exam: str, current_level: str) -> dict:
    """Update student's profile information"""
    # TODO: Implement in database
    return {
        "success": True,
        "student_id": student_id,
        "target_exam": target_exam,
        "current_level": current_level
    }

# ===== Progress Tracking Tools =====

@tool
def get_topic_proficiency(student_id: str, topic: str) -> dict:
    """
    Get proficiency score for a specific topic.
    Returns score (0.0-1.0), strength level, and attempts.
    """
    proficiency = db.get_topic_proficiency(topic)
    
    if not proficiency or topic not in proficiency:
        return {
            "topic": topic,
            "score": 0.0,
            "strength": "new",
            "attempts": 0
        }
    
    topic_data = proficiency[topic]
    return {
        "topic": topic,
        "score": topic_data.get("accuracy", 0.0),
        "strength": topic_data.get("strength", "new"),
        "attempts": topic_data.get("attempts", 0)
    }

@tool
def update_topic_proficiency(student_id: str, topic: str, score: float) -> dict:
    """
    Update proficiency score for a topic after assessment.
    Score should be between 0.0 and 1.0.
    """
    success = db.update_topic_proficiency(topic, score)
    
    return {
        "success": success,
        "topic": topic,
        "new_score": score,
        "updated_at": "now"
    }

@tool
def get_weak_topics(student_id: str) -> List[str]:
    """
    Get list of topics where student has low proficiency (< 0.6).
    Returns list of topic names.
    """
    proficiency = db.get_topic_proficiency()
    weak_topics = [
        topic for topic, data in proficiency.items()
        if data.get("strength") == "weak"
    ]
    return weak_topics

# ===== Content Retrieval Tools =====

@tool
def retrieve_content(topic: str, difficulty: str = "medium") -> dict:
    """
    Retrieve study materials from vector store for a given topic.
    Returns relevant documents and sources.
    """
    try:
        results = query_top_k(topic, k=3)
        documents = results.get('documents', [[]])[0] if results.get('documents') else []
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
        sources = [meta.get('source', 'unknown') for meta in metadatas]
        
        return {
            "topic": topic,
            "documents": documents,
            "sources": sources,
            "count": len(documents)
        }
    except Exception as e:
        return {
            "topic": topic,
            "documents": [],
            "sources": [],
            "error": str(e)
        }

# ===== Quiz Generation Tools =====

class QuizQuestion(BaseModel):
    """Schema for a single quiz question"""
    question_id: str = Field(description="Unique ID for the question")
    question_text: str = Field(description="The question text")
    options: List[str] = Field(description="4 options labeled A, B, C, D")
    correct_answer: str = Field(description="Correct option letter (A/B/C/D)")
    explanation: str = Field(description="Explanation of correct answer")
    difficulty: str = Field(description="easy/medium/hard")
    topic: str = Field(description="Main topic")

class Quiz(BaseModel):
    """Schema for complete quiz"""
    quiz_id: str
    topic: str
    difficulty: str
    questions: List[QuizQuestion]
    total_marks: int

@tool
def generate_quiz_tool(topic: str, difficulty: str = "medium", num_questions: int = 3) -> dict:
    """
    Generate a structured quiz with multiple choice questions.
    Uses LLM to create exam-style questions (JEE/SAT/GRE level).
    """
    parser = JsonOutputParser(pydantic_object=Quiz)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert quiz generator for competitive exams like JEE, SAT, and GRE."),
        ("user", """Generate a {difficulty} difficulty quiz on {topic} with {num_questions} questions.

Requirements:
- Each question must have exactly 4 options labeled A, B, C, D
- Questions should be exam-style (competitive exam level)
- Include clear explanation for the correct answer
- Make questions progressively harder within the quiz
- Focus on conceptual understanding, not just memorization

Topic: {topic}
Difficulty: {difficulty}
Number of questions: {num_questions}

{format_instructions}

Return ONLY valid JSON matching the schema.""")
    ])
    
    try:
        chain = prompt | llm | parser
        
        quiz = chain.invoke({
            "topic": topic,
            "difficulty": difficulty,
            "num_questions": num_questions,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Add quiz_id if not present
        if "quiz_id" not in quiz:
            import uuid
            quiz["quiz_id"] = f"quiz_{uuid.uuid4().hex[:8]}"
        
        # Add total_marks if not present
        if "total_marks" not in quiz:
            quiz["total_marks"] = len(quiz.get("questions", []))
        
        return quiz
    except Exception as e:
        # Fallback to simple quiz
        import uuid
        return {
            "quiz_id": f"quiz_{uuid.uuid4().hex[:8]}",
            "topic": topic,
            "difficulty": difficulty,
            "total_marks": num_questions,
            "questions": [
                {
                    "question_id": f"q{i+1}",
                    "question_text": f"Sample {difficulty} question {i+1} about {topic}",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "A",
                    "explanation": f"This is a placeholder explanation for question {i+1}",
                    "difficulty": difficulty,
                    "topic": topic
                }
                for i in range(num_questions)
            ],
            "error": str(e)
        }

# ===== Enhanced Answer Evaluation Tools =====

class AnswerEvaluation(BaseModel):
    """Schema for answer evaluation"""
    is_correct: bool
    score: float = Field(description="Score from 0.0 to 1.0")
    feedback: str = Field(description="Detailed feedback")
    misconception: Optional[str] = Field(description="Identified misconception if wrong")
    hint: Optional[str] = Field(description="Hint for improvement")
    confidence_level: str = Field(description="high/medium/low confidence in answer")
    learning_objective: str = Field(description="What concept this question tests")

class QuizEvaluation(BaseModel):
    """Schema for complete quiz evaluation"""
    total_score: int
    max_score: int
    percentage: float
    topics_tested: List[str]
    question_breakdown: List[dict]
    misconceptions: List[str]
    strengths: List[str]
    weaknesses: List[str]
    overall_feedback: str
    recommended_focus: str

@tool
def evaluate_answer_tool(
    question: str,
    options: List[str],
    correct_answer: str,
    student_answer: str,
    explanation: str
) -> dict:
    """
    Evaluate student's answer with detailed feedback.
    Identifies misconceptions and provides hints.
    """
    parser = JsonOutputParser(pydantic_object=AnswerEvaluation)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert tutor evaluating student answers. Be constructive and encouraging."),
        ("user", """Evaluate this answer:

Question: {question}
Options: {options}
Correct Answer: {correct_answer}
Student Answer: {student_answer}
Explanation: {explanation}

Provide:
1. Is the answer correct?
2. Partial credit score (0.0 to 1.0, where 1.0 is fully correct)
3. Constructive feedback explaining why it's right/wrong
4. If wrong, identify the specific misconception
5. Provide a helpful hint for improvement
6. Assess confidence level in the answer
7. Identify the learning objective being tested

Be encouraging and focus on learning, not just grading.

{format_instructions}""")
    ])
    
    try:
        chain = prompt | llm | parser
        
        evaluation = chain.invoke({
            "question": question,
            "options": ", ".join(options),
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "explanation": explanation,
            "format_instructions": parser.get_format_instructions()
        })
        
        return evaluation
    except Exception as e:
        # Fallback evaluation
        is_correct = student_answer.upper().strip() == correct_answer.upper().strip()
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "feedback": "Correct!" if is_correct else f"The correct answer is {correct_answer}. {explanation}",
            "misconception": None if is_correct else "Need to review this concept",
            "hint": None if is_correct else "Review the explanation and try again",
            "confidence_level": "high" if is_correct else "low",
            "learning_objective": "Concept understanding",
            "error": str(e)
        }

@tool
def evaluate_quiz_tool(quiz_id: str, student_answers: List[str]) -> dict:
    """
    Evaluate complete quiz and provide comprehensive analysis.
    """
    # This would typically fetch the quiz from database
    # For now, we'll create a mock evaluation
    try:
        # Mock quiz data (in real implementation, fetch from database)
        mock_quiz = {
            "quiz_id": quiz_id,
            "questions": [
                {
                    "question_id": "q1",
                    "correct_answer": "A",
                    "topic": "probability",
                    "difficulty": "medium"
                },
                {
                    "question_id": "q2", 
                    "correct_answer": "B",
                    "topic": "probability",
                    "difficulty": "hard"
                }
            ]
        }
        
        # Calculate scores
        total_score = 0
        max_score = len(mock_quiz["questions"])
        question_breakdown = []
        misconceptions = []
        strengths = []
        weaknesses = []
        topics_tested = set()
        
        for i, question in enumerate(mock_quiz["questions"]):
            student_answer = student_answers[i] if i < len(student_answers) else ""
            is_correct = student_answer.upper().strip() == question["correct_answer"].upper().strip()
            
            if is_correct:
                total_score += 1
                strengths.append(f"Question {i+1} ({question['topic']})")
            else:
                misconceptions.append(f"Question {i+1}: {question['topic']} - {question['difficulty']} level")
                weaknesses.append(f"Question {i+1} ({question['topic']})")
            
            topics_tested.add(question["topic"])
            
            question_breakdown.append({
                "question_id": question["question_id"],
                "correct": is_correct,
                "topic": question["topic"],
                "difficulty": question["difficulty"],
                "student_answer": student_answer,
                "correct_answer": question["correct_answer"]
            })
        
        percentage = (total_score / max_score) * 100
        
        # Generate overall feedback
        if percentage >= 80:
            overall_feedback = "Excellent work! You have a strong understanding of the concepts."
            recommended_focus = "Continue with more advanced topics"
        elif percentage >= 60:
            overall_feedback = "Good progress! You understand the basics but need more practice."
            recommended_focus = "Focus on the weaker areas identified"
        else:
            overall_feedback = "Don't worry! This is a learning opportunity. Let's work on the fundamentals."
            recommended_focus = "Review basic concepts before moving forward"
        
        return {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": percentage,
            "topics_tested": list(topics_tested),
            "question_breakdown": question_breakdown,
            "misconceptions": misconceptions,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "overall_feedback": overall_feedback,
            "recommended_focus": recommended_focus
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "total_score": 0,
            "max_score": 0,
            "percentage": 0.0
        }

@tool
def detect_misconceptions_tool(quiz_results: dict) -> List[str]:
    """
    Analyze quiz results to detect specific misconceptions.
    """
    misconceptions = []
    
    if not quiz_results or "question_breakdown" not in quiz_results:
        return misconceptions
    
    # Analyze patterns in wrong answers
    wrong_answers = [q for q in quiz_results["question_breakdown"] if not q["correct"]]
    
    # Group by topic to identify patterns
    topic_errors = {}
    for q in wrong_answers:
        topic = q.get("topic", "unknown")
        if topic not in topic_errors:
            topic_errors[topic] = []
        topic_errors[topic].append(q)
    
    # Generate misconception descriptions
    for topic, errors in topic_errors.items():
        if len(errors) >= 2:  # Multiple errors in same topic
            misconceptions.append(f"Struggling with {topic} concepts - needs focused review")
        
        # Check for specific error patterns
        for error in errors:
            if error.get("difficulty") == "hard":
                misconceptions.append(f"Difficulty with advanced {topic} problems")
            elif error.get("difficulty") == "easy":
                misconceptions.append(f"Basic {topic} concepts need reinforcement")
    
    return misconceptions

# ===== Study Plan Tools =====

@tool
def create_study_plan_tool(
    student_id: str,
    target_exam: str,
    weak_topics: List[str],
    deadline_days: int = 30
) -> dict:
    """
    Create personalized study plan based on student's weak topics.
    Returns prioritized topics, daily schedule, and milestones.
    """
    # Simple study plan generation
    study_plan = {
        "student_id": student_id,
        "target_exam": target_exam,
        "duration_days": deadline_days,
        "focus_areas": []
    }
    
    # Prioritize weak topics
    for i, topic in enumerate(weak_topics[:3]):  # Focus on top 3 weak topics
        study_plan["focus_areas"].append({
            "topic": topic,
            "priority": "high" if i == 0 else "medium",
            "estimated_days": 10 if i == 0 else 7,
            "daily_practice_minutes": 30,
            "milestone": f"Achieve 70% proficiency in {topic}"
        })
    
    # Add recommendations
    study_plan["recommendations"] = [
        f"Start with {weak_topics[0] if weak_topics else 'foundational topics'}",
        "Practice daily for consistent improvement",
        "Take quizzes to track progress",
        "Review mistakes and learn from them"
    ]
    
    study_plan["next_action"] = f"Begin with {weak_topics[0]}" if weak_topics else "Take diagnostic quiz"
    
    return study_plan
