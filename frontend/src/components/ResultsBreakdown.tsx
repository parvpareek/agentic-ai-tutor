import { CheckCircle, XCircle } from 'lucide-react';

interface TopicPerformance {
  topic: string;
  status: 'strong' | 'weak';
}

interface ResultsBreakdownProps {
  score: number;
  total: number;
  topics: TopicPerformance[];
}

export const ResultsBreakdown = ({ score, total, topics }: ResultsBreakdownProps) => {
  const percentage = (score / total) * 100;
  
  return (
    <div className="space-y-6">
      <div className="text-center">
        <div className="inline-flex items-baseline gap-2 mb-2">
          <span className="text-5xl font-bold text-primary">{score}</span>
          <span className="text-2xl text-muted-foreground">/ {total}</span>
        </div>
        <div className="h-3 bg-secondary rounded-full overflow-hidden max-w-xs mx-auto">
          <div 
            className={`h-full transition-all duration-700 ${
              percentage >= 80 ? 'bg-success' : percentage >= 60 ? 'bg-warning' : 'bg-destructive'
            }`}
            style={{ width: `${percentage}%` }}
          />
        </div>
        <p className="text-sm text-muted-foreground mt-2">
          {percentage.toFixed(0)}% correct
        </p>
      </div>

      <div>
        <h3 className="font-semibold text-card-foreground mb-3">Performance Breakdown</h3>
        <div className="space-y-2">
          {topics.map((topic, index) => (
            <div 
              key={index}
              className="flex items-center justify-between p-3 rounded-lg bg-card border border-border"
            >
              <span className="text-card-foreground">{topic.topic}</span>
              <div className="flex items-center gap-2">
                {topic.status === 'strong' ? (
                  <>
                    <CheckCircle className="h-4 w-4 text-success" />
                    <span className="text-sm font-medium text-success">Strong</span>
                  </>
                ) : (
                  <>
                    <XCircle className="h-4 w-4 text-destructive" />
                    <span className="text-sm font-medium text-destructive">Needs Work</span>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
