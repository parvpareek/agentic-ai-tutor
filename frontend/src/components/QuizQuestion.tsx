import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';

interface QuizQuestionProps {
  question: string;
  options: string[];
  selectedAnswer: string | null;
  onSelectAnswer: (answer: string) => void;
  questionNumber: number;
  totalQuestions: number;
}

export const QuizQuestion = ({ 
  question, 
  options, 
  selectedAnswer, 
  onSelectAnswer,
  questionNumber,
  totalQuestions 
}: QuizQuestionProps) => {
  const optionLetters = ['A', 'B', 'C', 'D'];

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-sm text-muted-foreground font-medium">
          Question {questionNumber} of {totalQuestions}
        </p>
        <h3 className="text-xl font-semibold text-card-foreground">
          {question}
        </h3>
      </div>

      <RadioGroup value={selectedAnswer || undefined} onValueChange={onSelectAnswer}>
        <div className="space-y-3">
          {options.map((option, index) => {
            const letter = optionLetters[index];
            const isSelected = selectedAnswer === letter;
            
            return (
              <div key={letter} className="relative">
                <RadioGroupItem
                  value={letter}
                  id={`option-${letter}`}
                  className="peer sr-only"
                />
                <Label
                  htmlFor={`option-${letter}`}
                  className={`
                    flex items-start gap-3 p-4 rounded-lg border-2 cursor-pointer
                    transition-all duration-200
                    ${isSelected 
                      ? 'border-primary bg-primary/5' 
                      : 'border-border hover:border-primary/50 hover:bg-muted/30'
                    }
                  `}
                >
                  <div className={`
                    flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
                    font-semibold text-sm transition-colors
                    ${isSelected 
                      ? 'bg-primary text-primary-foreground' 
                      : 'bg-muted text-muted-foreground'
                    }
                  `}>
                    {letter}
                  </div>
                  <span className="flex-1 pt-1 text-card-foreground">{option}</span>
                </Label>
              </div>
            );
          })}
        </div>
      </RadioGroup>
    </div>
  );
};
