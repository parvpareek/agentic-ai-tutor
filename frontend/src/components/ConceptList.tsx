import { Check } from 'lucide-react';
import { Concept } from '@/api/tutorApi';

interface ConceptListProps {
  concepts: Concept[];
  completedCount: number;
  currentTopic?: string;
}

export const ConceptList = ({ concepts, completedCount, currentTopic }: ConceptListProps) => {
  return (
    <div className="space-y-3">
      {concepts.map((concept, index) => {
        const isCompleted = index < completedCount;
        // STEP 1 FIX: Highlight based on topic name match, not index
        const isCurrent = currentTopic === concept.label;
        
        return (
          <div
            key={concept.concept_id}
            className={`
              flex items-center gap-3 p-4 rounded-lg border-2 transition-all
              ${isCurrent ? 'border-primary bg-primary/5' : 'border-border'}
              ${isCompleted ? 'bg-success/5' : ''}
            `}
          >
            <div className={`
              flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-semibold
              ${isCompleted ? 'bg-success text-success-foreground' : 'bg-muted text-muted-foreground'}
            `}>
              {isCompleted ? <Check className="h-5 w-5" /> : index + 1}
            </div>
            <div className="flex-1">
              <p className={`font-medium ${isCurrent ? 'text-primary' : 'text-card-foreground'}`}>
                {concept.label}
              </p>
              {isCurrent && (
                <div className="mt-2 text-xs">
                  {Array.isArray((concept as any).learning_segments) && (concept as any).learning_segments.length > 0 ? (
                    <ol className="list-decimal list-inside space-y-0.5">
                      {(concept as any).learning_segments
                        .slice()
                        .sort((a: any, b: any) => (a.order || 0) - (b.order || 0))
                        .map((s: any) => (
                          <li key={s.segment_id || s.title} className="text-muted-foreground">
                            {s.title || s.segment_id}
                          </li>
                        ))}
                    </ol>
                  ) : (
                    <p className="text-primary/70">Currently learning</p>
                  )}
                </div>
              )}
              {isCompleted && !isCurrent && (
                <p className="text-sm text-success/70 mt-0.5">Completed</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
