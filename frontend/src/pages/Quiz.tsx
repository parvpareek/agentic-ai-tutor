import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTutor } from '@/context/TutorContext';
import { QuizQuestion } from '@/components/QuizQuestion';
import { QuestionModal } from '@/components/QuestionModal';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { Button } from '@/components/ui/button';
import { submitQuiz } from '@/api/tutorApi';
import { ChevronLeft, ChevronRight, HelpCircle, Send } from 'lucide-react';

export default function Quiz() {
  const navigate = useNavigate();
  const { state, setStudentAnswers, setLastEvaluation, setQuizPending, addMessage, setPlannerReason } = useTutor();
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<string[]>([]);
  const [showHelp, setShowHelp] = useState(false);
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false); // Prevent duplicate submissions

  if (!state.currentQuiz) {
    navigate('/learn');
    return null;
  }

  const quiz = state.currentQuiz;
  const questions = quiz.questions || [];
  const totalQuestions = questions.length;

  const handleAnswerSelect = (answer: string) => {
    const newAnswers = [...answers];
    newAnswers[currentQuestion] = answer;
    setAnswers(newAnswers);
  };

  const canSubmit = answers.length === totalQuestions && answers.every(a => a);

  const handleSubmit = async () => {
    // CRITICAL FIX: Prevent duplicate submissions
    if (submitted || loading) {
      console.log('Submission already in progress or completed');
      return;
    }
    
    setLoading(true);
    setSubmitted(true);
    setStudentAnswers(answers);
    
    try {
      const response = await submitQuiz(answers);
      setLastEvaluation(response);
      setQuizPending(false);
      
      const evaluation = response.evaluation || {};
      const score = evaluation.correct_answers || 0;
      const total = evaluation.total_questions || 0;
      const percentage = total > 0 ? Math.round((score / total) * 100) : 0;
      
      const resultMessage = `
🎯 **Quiz Results**

Score: ${score}/${total} (${percentage}%)

${percentage >= 80 ? '🌟 Excellent work!' : percentage >= 60 ? '👍 Good effort!' : '📚 Keep practicing!'}

${response.planner_reasoning || 'Great job completing the quiz!'}

Ready to continue learning? Click "Next Step" to proceed!
      `.trim();
      
      addMessage({ role: 'tutor', content: resultMessage });
      if (response.planner_reasoning) {
        setPlannerReason(response.planner_reasoning);
      }
      
      // Small delay to show "submitted" feedback before navigating
      setTimeout(() => {
        navigate('/learn');
      }, 800);
    } catch (error) {
      console.error('Failed to submit quiz:', error);
      setQuizPending(false);
      setSubmitted(false); // Allow retry on error
    } finally {
      setLoading(false);
    }
  };

  if (loading || submitted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <LoadingSpinner 
          message={submitted ? "✅ Submitted! Evaluating your answers..." : "Evaluating your answers..."} 
          size="lg" 
        />
      </div>
    );
  }

  const question = questions[currentQuestion];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <div className="max-w-3xl mx-auto px-4 py-12">
        <div className="bg-card rounded-xl p-8 border border-border shadow-sm">
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h1 className="text-2xl font-bold text-card-foreground">
                📝 Quiz: {state.currentTopic}
              </h1>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowHelp(true)}
              >
                <HelpCircle className="mr-2 h-4 w-4" />
                Need Help?
              </Button>
            </div>

            <div className="flex gap-2 mb-6">
              {questions.map((_: any, idx: number) => (
                <div
                  key={idx}
                  className={`h-2 flex-1 rounded-full transition-colors ${
                    answers[idx] 
                      ? 'bg-primary' 
                      : idx === currentQuestion 
                      ? 'bg-primary/30' 
                      : 'bg-secondary'
                  }`}
                />
              ))}
            </div>
          </div>

          <QuizQuestion
            question={question.question}
            options={question.options}
            selectedAnswer={answers[currentQuestion] || null}
            onSelectAnswer={handleAnswerSelect}
            questionNumber={currentQuestion + 1}
            totalQuestions={totalQuestions}
          />

          <div className="flex items-center justify-between mt-8 pt-6 border-t border-border">
            <Button
              variant="outline"
              onClick={() => setCurrentQuestion(prev => Math.max(0, prev - 1))}
              disabled={currentQuestion === 0}
            >
              <ChevronLeft className="mr-2 h-4 w-4" />
              Previous
            </Button>

            {currentQuestion === totalQuestions - 1 ? (
              <Button
                onClick={handleSubmit}
                disabled={!canSubmit || submitted || loading}
                className={submitted ? 'bg-green-600 hover:bg-green-700' : ''}
              >
                {submitted ? '✅ Submitted!' : 'Submit Quiz'}
                <Send className="ml-2 h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={() => setCurrentQuestion(prev => Math.min(totalQuestions - 1, prev + 1))}
              >
                Next
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </div>

      <QuestionModal 
        isOpen={showHelp}
        onClose={() => setShowHelp(false)}
      />
    </div>
  );
}
