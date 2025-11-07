import { AlertCircle, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ErrorBannerProps {
  message: string;
  onRetry?: () => void;
  onDismiss?: () => void;
}

export const ErrorBanner = ({ message, onRetry, onDismiss }: ErrorBannerProps) => {
  return (
    <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-start gap-3">
      <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        <p className="text-sm text-destructive font-medium">{message}</p>
      </div>
      <div className="flex items-center gap-2">
        {onRetry && (
          <Button variant="ghost" size="sm" onClick={onRetry} className="text-destructive hover:text-destructive">
            Retry
          </Button>
        )}
        {onDismiss && (
          <button onClick={onDismiss} className="text-destructive hover:text-destructive/80">
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  );
};
