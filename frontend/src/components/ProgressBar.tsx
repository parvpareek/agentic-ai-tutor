import React from 'react';
import { Check, Circle, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ProgressBarProps {
  total: number;
  completed: number;
  current?: number;
  className?: string;
  showLabels?: boolean;
  variant?: 'default' | 'compact' | 'detailed';
}

interface ProgressItemProps {
  index: number;
  label: string;
  status: 'completed' | 'current' | 'pending';
  isLast?: boolean;
  variant?: 'default' | 'compact' | 'detailed';
}

const ProgressItem: React.FC<ProgressItemProps> = ({ 
  index, 
  label, 
  status, 
  isLast = false, 
  variant = 'default' 
}) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <Check className="h-4 w-4 text-green-600" />;
      case 'current':
        return <Circle className="h-4 w-4 text-blue-600 fill-current" />;
      case 'pending':
        return <Circle className="h-4 w-4 text-gray-300" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'text-green-700 bg-green-50 border-green-200';
      case 'current':
        return 'text-blue-700 bg-blue-50 border-blue-200';
      case 'pending':
        return 'text-gray-500 bg-gray-50 border-gray-200';
    }
  };

  if (variant === 'compact') {
    return (
      <div className="flex items-center">
        <div className={cn(
          "flex items-center justify-center w-6 h-6 rounded-full border-2",
          status === 'completed' && "bg-green-100 border-green-300",
          status === 'current' && "bg-blue-100 border-blue-300",
          status === 'pending' && "bg-gray-100 border-gray-300"
        )}>
          {getStatusIcon()}
        </div>
        {!isLast && (
          <div className={cn(
            "flex-1 h-0.5 mx-2",
            status === 'completed' ? "bg-green-300" : "bg-gray-200"
          )} />
        )}
      </div>
    );
  }

  if (variant === 'detailed') {
    return (
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          <div className={cn(
            "flex items-center justify-center w-8 h-8 rounded-full border-2",
            status === 'completed' && "bg-green-100 border-green-300",
            status === 'current' && "bg-blue-100 border-blue-300",
            status === 'pending' && "bg-gray-100 border-gray-300"
          )}>
            {getStatusIcon()}
          </div>
        </div>
        <div className="flex-1 min-w-0">
          <div className={cn(
            "text-sm font-medium",
            status === 'completed' && "text-green-700",
            status === 'current' && "text-blue-700",
            status === 'pending' && "text-gray-500"
          )}>
            {label}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {status === 'completed' && 'Completed'}
            {status === 'current' && 'In Progress'}
            {status === 'pending' && 'Pending'}
          </div>
        </div>
        {!isLast && (
          <div className="absolute left-4 top-8 w-0.5 h-6 bg-gray-200" />
        )}
      </div>
    );
  }

  // Default variant
  return (
    <div className="flex items-center">
      <div className={cn(
        "flex items-center justify-center w-8 h-8 rounded-full border-2",
        status === 'completed' && "bg-green-100 border-green-300",
        status === 'current' && "bg-blue-100 border-blue-300",
        status === 'pending' && "bg-gray-100 border-gray-300"
      )}>
        {getStatusIcon()}
      </div>
      <div className="ml-3">
        <div className={cn(
          "text-sm font-medium",
          status === 'completed' && "text-green-700",
          status === 'current' && "text-blue-700",
          status === 'pending' && "text-gray-500"
        )}>
          {label}
        </div>
      </div>
      {!isLast && (
        <div className={cn(
          "flex-1 h-0.5 mx-4",
          status === 'completed' ? "bg-green-300" : "bg-gray-200"
        )} />
      )}
    </div>
  );
};

export const ProgressBar: React.FC<ProgressBarProps> = ({
  total,
  completed,
  current = 0,
  className,
  showLabels = true,
  variant = 'default'
}) => {
  // Calculate progress based on current step
  const progressPercentage = total > 0 ? ((current + 1) / total) * 100 : 0;
  const currentIndex = Math.min(current, total - 1);

  // Generate progress items (for non-minimal variants)
  const items = Array.from({ length: total }, (_, index) => {
    let status: 'completed' | 'current' | 'pending' = 'pending';
    
    if (index < current) {
      status = 'completed';
    } else if (index === current) {
      status = 'current';
    }

    return {
      index,
      label: `Step ${index + 1}`,
      status
    };
  });

  if (variant === 'compact') {
    return (
      <div className={cn("flex items-center space-x-1", className)}>
        {items.map((item, index) => (
          <ProgressItem
            key={item.index}
            index={item.index}
            label={item.label}
            status={item.status}
            isLast={index === items.length - 1}
            variant="compact"
          />
        ))}
      </div>
    );
  }

  if (variant === 'detailed') {
    return (
      <div className={cn("space-y-4", className)}>
        {showLabels && (
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-gray-900">Progress</h3>
            <div className="text-sm text-gray-500">
              {completed} of {total} completed
            </div>
          </div>
        )}
        <div className="relative">
          {items.map((item, index) => (
            <div key={item.index} className="relative">
              <ProgressItem
                index={item.index}
                label={item.label}
                status={item.status}
                isLast={index === items.length - 1}
                variant="detailed"
              />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Default variant - Clean and minimal
  return (
    <div className={cn("w-full", className)}>
      {showLabels && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-medium text-muted-foreground">
            Step {current + 1} of {total}
          </span>
          <span className="text-xs text-muted-foreground">
            {Math.round(progressPercentage)}% complete
          </span>
        </div>
      )}
      
      {/* Clean progress bar */}
      <div className="w-full bg-muted rounded-full h-1.5 overflow-hidden">
        <div
          className="bg-primary h-full rounded-full transition-all duration-500 ease-out"
          style={{ width: `${progressPercentage}%` }}
        />
      </div>
    </div>
  );
};

// Specialized concept progress bar
interface ConceptProgressBarProps {
  concepts: Array<{
    concept_id: string;
    label: string;
    status?: 'completed' | 'current' | 'pending';
  }>;
  currentIndex?: number;
  className?: string;
}

export const ConceptProgressBar: React.FC<ConceptProgressBarProps> = ({
  concepts,
  currentIndex = 0,
  className
}) => {
  const items = concepts.map((concept, index) => {
    let status: 'completed' | 'current' | 'pending' = concept.status || 'pending';
    
    if (index < currentIndex) {
      status = 'completed';
    } else if (index === currentIndex) {
      status = 'current';
    }

    return {
      index,
      label: concept.label,
      status
    };
  });

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-semibold text-gray-900">Learning Progress</h3>
        <div className="text-xs text-gray-500">
          {items.filter(item => item.status === 'completed').length} of {items.length} concepts
        </div>
      </div>
      
      <div className="space-y-1">
        {items.map((item, index) => (
          <div key={item.index} className="flex items-center space-x-2">
            <div className={cn(
              "flex items-center justify-center w-5 h-5 rounded-full border",
              item.status === 'completed' && "bg-green-100 border-green-300",
              item.status === 'current' && "bg-blue-100 border-blue-300",
              item.status === 'pending' && "bg-gray-100 border-gray-300"
            )}>
              {item.status === 'completed' && <Check className="h-3 w-3 text-green-600" />}
              {item.status === 'current' && <Circle className="h-3 w-3 text-blue-600 fill-current" />}
              {item.status === 'pending' && <Circle className="h-3 w-3 text-gray-300" />}
            </div>
            <span className={cn(
              "text-sm",
              item.status === 'completed' && "text-green-700",
              item.status === 'current' && "text-blue-700 font-medium",
              item.status === 'pending' && "text-gray-500"
            )}>
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};