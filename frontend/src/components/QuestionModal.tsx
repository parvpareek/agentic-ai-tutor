import { useState } from 'react';
import { X, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { askQuestion } from '@/api/tutorApi';
import { LoadingSpinner } from './LoadingSpinner';
import { useTutor } from '@/context/TutorContext';

interface QuestionModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const QuestionModal = ({ isOpen, onClose }: QuestionModalProps) => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const { state } = useTutor();

  const handleAsk = async () => {
    if (!question.trim()) return;
    
    setLoading(true);
    try {
      const response = await askQuestion(question, state.currentTopic);
      setAnswer(response.answer || 'No answer provided');
    } catch (error) {
      setAnswer('Failed to get answer. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-card rounded-xl shadow-lg max-w-lg w-full p-6 border border-border">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-card-foreground">Ask the Tutor</h3>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="space-y-4">
          <div className="flex gap-2">
            <Input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Type your question..."
              onKeyDown={(e) => e.key === 'Enter' && handleAsk()}
            />
            <Button onClick={handleAsk} disabled={loading || !question.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </div>

          {loading && <LoadingSpinner message="Thinking..." size="sm" />}

          {answer && (
            <div className="bg-muted p-4 rounded-lg">
              <p className="text-sm text-card-foreground whitespace-pre-wrap">{answer}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
