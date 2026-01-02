import { useNavigate } from 'react-router-dom';
import { useTutor } from '@/context/TutorContext';
import { ResultsBreakdown } from '@/components/ResultsBreakdown';
import { Button } from '@/components/ui/button';
import { BookOpen, ArrowRight, RotateCcw, Trophy } from 'lucide-react';

export default function Results() {
  const navigate = useNavigate();
  const { state, setProgress } = useTutor();

  if (!state.lastEvaluation) {
    navigate('/learn');
    return null;
  }

  const evaluation = state.lastEvaluation.evaluation;
  const score = evaluation.score || 0;
  const total = evaluation.total || 1;
  const percentage = (score / total) * 100;

  const performanceTopics = evaluation.breakdown?.map((item: any) => ({
    topic: item.topic,
    status: item.correct ? 'strong' : 'weak'
  })) || [];

  const recommendation = state.lastEvaluation.planner_evaluation;
  const nextAction = state.lastEvaluation.next_action;

  const handleAction = () => {
    if (nextAction === 'move_forward') {
      const newCompleted = state.progress.completed + 1;
      setProgress({ ...state.progress, completed: newCompleted });
      if (newCompleted < state.progress.total) {
        navigate('/learn');
      } else {
        alert('Congratulations! You have completed all concepts!');
        navigate('/');
      }
    } else if (nextAction === 'clarify_concept') {
      navigate('/learn');
    } else {
      navigate('/quiz');
    }
  };

  const getActionButton = () => {
    if (nextAction === 'move_forward') {
      return {
        text: '⏭️ Move to Next Concept',
        icon: <ArrowRight className="ml-2 h-5 w-5" />
      };
    } else if (nextAction === 'clarify_concept') {
      const weakTopic = performanceTopics.find((t: any) => t.status === 'weak')?.topic || 'weak areas';
      return {
        text: `📚 Review ${weakTopic}`,
        icon: <BookOpen className="ml-2 h-5 w-5" />
      };
    }
    return {
      text: '🔄 Continue Practice',
      icon: <RotateCcw className="ml-2 h-5 w-5" />
    };
  };

  const actionButton = getActionButton();

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <div className="max-w-3xl mx-auto px-4 py-12">
        <div className="text-center mb-8">
          <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-4 ${
            percentage >= 80 ? 'bg-success' : percentage >= 60 ? 'bg-warning' : 'bg-destructive'
          }`}>
            <Trophy className="h-10 w-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-foreground mb-2">
            Quiz Complete!
          </h1>
          <p className="text-lg text-muted-foreground">
            {percentage >= 80 
              ? 'Excellent work!' 
              : percentage >= 60 
              ? 'Good effort, keep practicing!' 
              : 'Keep learning, you\'ll get there!'}
          </p>
        </div>

        <div className="bg-card rounded-xl p-8 border border-border shadow-sm mb-6">
          <ResultsBreakdown 
            score={score}
            total={total}
            topics={performanceTopics}
          />
        </div>

        {state.lastEvaluation.planner_reasoning && (
          <div className="bg-primary/5 rounded-xl p-6 border border-primary/20 mb-6">
            <h3 className="font-semibold text-card-foreground mb-3 flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-primary" />
              AI Tutor's Analysis
            </h3>
            <p className="text-card-foreground whitespace-pre-wrap">
              {state.lastEvaluation.planner_reasoning}
            </p>
          </div>
        )}

        {recommendation && (
          <div className="bg-accent/10 rounded-xl p-6 border border-accent/20 mb-6">
            <h3 className="font-semibold text-card-foreground mb-2">
              Recommendation
            </h3>
            <p className="text-card-foreground">{recommendation}</p>
          </div>
        )}

        <div className="space-y-3">
          <Button 
            onClick={handleAction}
            size="lg"
            className="w-full group"
          >
            {actionButton.text}
            {actionButton.icon}
          </Button>

          <Button 
            variant="outline"
            onClick={() => navigate('/quiz')}
            size="lg"
            className="w-full"
          >
            <RotateCcw className="mr-2 h-4 w-4" />
            Retake Quiz
          </Button>
        </div>
      </div>
    </div>
  );
}
