const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Concept {
  concept_id: string;
  label: string;
  supporting_chunk_ids: string[];
}

export interface UploadResponse {
  session_id: string;
  concepts: Concept[];
  study_plan: any[];
  next_action: string;
}

export const uploadDocument = async (file: File, studentChoice: string): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('student_choice', studentChoice);
  
  const response = await fetch(`${API_BASE_URL}/simple-tutor/upload-document`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error('Failed to upload document');
  }
  
  return response.json();
};

export const startFromSaved = async (savedSessionId: string, studentChoice: string = 'from_beginning'): Promise<UploadResponse> => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/start-from-saved`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ saved_session_id: savedSessionId, student_choice: studentChoice })
  });
  if (!response.ok) {
    throw new Error('Failed to start from saved session');
  }
  return response.json();
};

export const executeStep = async (
  stepIndex: number | null = null, 
  exerciseAnswers?: string[], 
  lastTopic?: string,
  lastSegmentId?: string
) => {
  const payload: any = { step_index: stepIndex };
  
  // Add exercise answers if provided
  if (exerciseAnswers && exerciseAnswers.length > 0 && lastTopic && lastSegmentId) {
    payload.exercise_answers = exerciseAnswers;
    payload.last_topic = lastTopic;
    payload.last_segment_id = lastSegmentId;
  }
  
  const response = await fetch(`${API_BASE_URL}/simple-tutor/execute-step`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });
  
  if (!response.ok) {
    throw new Error('Failed to execute step');
  }
  
  return response.json();
};

export const askQuestion = async (question: string, topic: string) => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/ask-question`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, topic }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to ask question');
  }
  
  return response.json();
};

export const submitQuiz = async (answers: string[]) => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/submit-quiz-answers`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ answers }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to submit quiz');
  }
  
  return response.json();
};

export const submitExercise = async (payload: { topic: string; segment_id: string; prompt: string; student_answer: string; correct?: boolean }) => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/submit-exercise`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error('Failed to submit exercise');
  }
  return response.json();
};

export const getCurrentState = async () => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/current-state`);
  
  if (!response.ok) {
    throw new Error('Failed to get current state');
  }
  
  return response.json();
};

// Session memory APIs
export interface SessionSummary {
  topic: string;
  segment_id: string;
  summary: string;
  timestamp?: string | number;
  memory_delta?: string;
}

export const getLastSummaries = async (sessionId: string, k: number = 3): Promise<SessionSummary[]> => {
  const response = await fetch(`${API_BASE_URL}/session/${encodeURIComponent(sessionId)}/summaries/lastk?k=${k}`);
  if (!response.ok) {
    throw new Error('Failed to fetch last summaries');
  }
  const data = await response.json();
  return data.summaries || [];
};

export const getTopicQuizResults = async (sessionId: string, topic: string, k: number = 3): Promise<any[]> => {
  const response = await fetch(`${API_BASE_URL}/session/${encodeURIComponent(sessionId)}/topic/${encodeURIComponent(topic)}/quiz_results?k=${k}`);
  if (!response.ok) {
    throw new Error('Failed to fetch topic quiz results');
  }
  const data = await response.json();
  return data.results || [];
};

// Lesson History API functions
export interface LessonSession {
  session_id: string;
  document_name: string;
  document_type: string;
  created_at: string;
  completed_at?: string;
  status: string;
}

export interface LessonMessage {
  role: 'tutor' | 'student';
  content: string;
  sources: string[];
  timestamp: string;
}

export interface DetailedLessonSession extends LessonSession {
  concepts: Concept[];
  study_plan: any[];
  conversation_history: any[];
  quiz_results: any[];
  final_evaluation: any;
  detailed_messages: LessonMessage[];
}

export const getLessonSessions = async (): Promise<LessonSession[]> => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/lesson-sessions`);
  
  if (!response.ok) {
    throw new Error('Failed to get lesson sessions');
  }
  
  const data = await response.json();
  return data.sessions;
};

export const listSavedSessions = async (): Promise<LessonSession[]> => {
  const response = await fetch(`${API_BASE_URL}/session/sessions/saved`);
  if (!response.ok) throw new Error('Failed to list saved sessions');
  const data = await response.json();
  return data.sessions || [];
};

export const getIndexedChunks = async (sessionId: string, k: number = 50) => {
  const response = await fetch(`${API_BASE_URL}/session/document/${encodeURIComponent(sessionId)}/chunks?k=${k}`);
  if (!response.ok) throw new Error('Failed to get indexed chunks');
  return response.json();
};

export const getLessonSession = async (sessionId: string): Promise<DetailedLessonSession> => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/lesson-sessions/${sessionId}`);
  
  if (!response.ok) {
    throw new Error('Failed to get lesson session');
  }
  
  const data = await response.json();
  return data.session;
};

export const getLessonMessages = async (sessionId: string): Promise<LessonMessage[]> => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/lesson-sessions/${sessionId}/messages`);
  
  if (!response.ok) {
    throw new Error('Failed to get lesson messages');
  }
  
  const data = await response.json();
  return data.messages;
};

export const completeLessonSession = async (sessionId: string): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/lesson-sessions/${sessionId}/complete`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    throw new Error('Failed to complete lesson session');
  }
};

export const submitDifficultyRating = async (topic: string, segmentId: string, rating: number): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/simple-tutor/submit-difficulty-rating`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      topic,
      segment_id: segmentId,
      rating,
    }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to submit difficulty rating');
  }
};
