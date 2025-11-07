import { createContext, useContext, useState, ReactNode } from 'react';
import { Concept } from '@/api/tutorApi';

interface Message {
  role: 'tutor' | 'student';
  content: string;
  sources?: string[];
}

interface TutorState {
  sessionId: string | null;
  concepts: Concept[];
  studyPlan: any[];
  currentStep: number;
  currentTopic: string;
  conversationHistory: Message[];
  currentQuiz: any | null;
  quizPending: boolean;
  studentAnswers: string[];
  progress: {
    completed: number;
    total: number;
  };
  lastEvaluation: any | null;
  plannerReason: string | null;
  currentSegmentId?: string | null;
  currentSegmentTitle?: string | null;
}

interface TutorContextType {
  state: TutorState;
  setSessionId: (id: string) => void;
  setConcepts: (concepts: Concept[]) => void;
  setStudyPlan: (plan: any[]) => void;
  setCurrentStep: (step: number) => void;
  setCurrentTopic: (topic: string) => void;
  setCurrentSegment: (segmentId: string | null, segmentTitle: string | null) => void;
  addMessage: (message: Message) => void;
  setCurrentQuiz: (quiz: any) => void;
  setQuizPending: (pending: boolean) => void;
  setStudentAnswers: (answers: string[]) => void;
  setProgress: (progress: { completed: number; total: number }) => void;
  setLastEvaluation: (evaluation: any) => void;
  setPlannerReason: (reason: string | null) => void;
  resetSession: () => void;
}

const TutorContext = createContext<TutorContextType | undefined>(undefined);

const initialState: TutorState = {
  sessionId: null,
  concepts: [],
  studyPlan: [],
  currentStep: 0,
  currentTopic: '',
  conversationHistory: [],
  currentQuiz: null,
  quizPending: false,
  studentAnswers: [],
  progress: { completed: 0, total: 0 },
  lastEvaluation: null,
  plannerReason: null,
};

export const TutorProvider = ({ children }: { children: ReactNode }) => {
  const [state, setState] = useState<TutorState>(initialState);

  const setSessionId = (id: string) => setState(prev => ({ ...prev, sessionId: id }));
  const setConcepts = (concepts: Concept[]) => setState(prev => ({ ...prev, concepts, progress: { ...prev.progress, total: concepts.length } }));
  const setStudyPlan = (plan: any[]) => setState(prev => ({ ...prev, studyPlan: plan }));
  const setCurrentStep = (step: number) => setState(prev => ({ ...prev, currentStep: step }));
  const setCurrentTopic = (topic: string) => setState(prev => ({ ...prev, currentTopic: topic }));
  const setCurrentSegment = (segmentId: string | null, segmentTitle: string | null) => setState(prev => ({ ...prev, currentSegmentId: segmentId, currentSegmentTitle: segmentTitle }));
  const addMessage = (message: Message) => setState(prev => ({ ...prev, conversationHistory: [...prev.conversationHistory, message] }));
  const setCurrentQuiz = (quiz: any) => setState(prev => ({ ...prev, currentQuiz: quiz, quizPending: quiz !== null }));
  const setQuizPending = (pending: boolean) => setState(prev => ({ ...prev, quizPending: pending }));
  const setStudentAnswers = (answers: string[]) => setState(prev => ({ ...prev, studentAnswers: answers }));
  const setProgress = (progress: { completed: number; total: number }) => setState(prev => ({ ...prev, progress }));
  const setLastEvaluation = (evaluation: any) => setState(prev => ({ ...prev, lastEvaluation: evaluation }));
  const setPlannerReason = (reason: string | null) => setState(prev => ({ ...prev, plannerReason: reason }));
  const resetSession = () => setState(initialState);

  return (
    <TutorContext.Provider value={{
      state,
      setSessionId,
      setConcepts,
      setStudyPlan,
      setCurrentStep,
      setCurrentTopic,
      setCurrentSegment,
      addMessage,
      setCurrentQuiz,
      setQuizPending,
      setStudentAnswers,
      setProgress,
      setLastEvaluation,
      setPlannerReason,
      resetSession,
    }}>
      {children}
    </TutorContext.Provider>
  );
};

export const useTutor = () => {
  const context = useContext(TutorContext);
  if (!context) {
    throw new Error('useTutor must be used within TutorProvider');
  }
  return context;
};
