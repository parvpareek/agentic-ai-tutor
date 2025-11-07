import { Bot, User } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface ChatMessageProps {
  role: 'tutor' | 'student';
  content: string;
  sources?: string[];
}

export const ChatMessage = ({ role, content, sources }: ChatMessageProps) => {
  const isTutor = role === 'tutor';
  
  return (
    <div className={`flex gap-3 ${isTutor ? '' : 'flex-row-reverse'}`}>
      <div className={`
        flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
        ${isTutor ? 'bg-primary' : 'bg-secondary'}
      `}>
        {isTutor ? (
          <Bot className="h-5 w-5 text-primary-foreground" />
        ) : (
          <User className="h-5 w-5 text-secondary-foreground" />
        )}
      </div>
      
      <div className={`flex-1 ${isTutor ? '' : 'flex justify-end'}`}>
        <div className={`
          max-w-[85%] p-4 rounded-2xl
          ${isTutor 
            ? 'bg-card border border-border rounded-tl-none' 
            : 'bg-primary text-primary-foreground rounded-tr-none'
          }
        `}>
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
          
          {sources && sources.length > 0 && (
            <div className="mt-3 pt-3 border-t border-border/50">
              <p className="text-xs text-muted-foreground font-medium mb-1">Sources:</p>
              <div className="flex flex-wrap gap-1">
                {sources.map((source, i) => (
                  <span key={i} className="text-xs bg-muted px-2 py-1 rounded">
                    {source}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
