import { useState } from 'react';
import { ChevronDown, ChevronRight, Brain } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface AgentThought {
  agent: string;
  thought: string;
  emoji: string;
  timestamp: string;
  metadata?: any;
}

interface AgentThoughtsProps {
  thoughts: AgentThought[];
}

export function AgentThoughts({ thoughts }: AgentThoughtsProps) {
  const [isExpanded, setIsExpanded] = useState(true); // Auto-expand to show agent thinking

  if (!thoughts || thoughts.length === 0) return null;

  return (
    <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mb-4">
      <Button
        variant="ghost"
        size="sm"
        className="w-full justify-between text-left p-2 h-auto"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Brain className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium text-primary">
            🧠 AI Thinking ({thoughts.length} thought{thoughts.length > 1 ? 's' : ''})
          </span>
        </div>
        {isExpanded ? (
          <ChevronDown className="h-4 w-4 text-primary" />
        ) : (
          <ChevronRight className="h-4 w-4 text-primary" />
        )}
      </Button>

      {isExpanded && (
        <div className="mt-3 space-y-2">
          {thoughts.map((thought, index) => (
            <div
              key={index}
              className="bg-card border border-border rounded-md p-2.5 text-xs"
            >
              <div className="flex items-start gap-2">
                <span className="text-base">{thought.emoji}</span>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-primary mb-0.5">
                    {thought.agent}
                  </div>
                  <div className="text-muted-foreground">
                    {thought.thought}
                  </div>
                  {thought.metadata && Object.keys(thought.metadata).length > 0 && (
                    <details className="mt-1.5">
                      <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
                        Details
                      </summary>
                      <pre className="mt-1 text-xs bg-muted p-1.5 rounded overflow-x-auto">
                        {JSON.stringify(thought.metadata, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
                <div className="text-xs text-muted-foreground whitespace-nowrap">
                  {new Date(thought.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

