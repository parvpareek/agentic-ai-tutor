# backend/app/tools/__init__.py
from .tutor_tools import (
    get_student_profile,
    update_student_profile,
    get_topic_proficiency,
    update_topic_proficiency,
    get_weak_topics,
    retrieve_content,
    generate_quiz_tool,
    evaluate_answer_tool,
    create_study_plan_tool
)

__all__ = [
    'get_student_profile',
    'update_student_profile',
    'get_topic_proficiency',
    'update_topic_proficiency',
    'get_weak_topics',
    'retrieve_content',
    'generate_quiz_tool',
    'evaluate_answer_tool',
    'create_study_plan_tool'
]
