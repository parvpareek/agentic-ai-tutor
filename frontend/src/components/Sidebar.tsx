import { Menu, X, BookOpen } from 'lucide-react';
import { useState } from 'react';
import { ConceptList } from './ConceptList';
import { useTutor } from '@/context/TutorContext';
import { Button } from '@/components/ui/button';

export const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { state } = useTutor();

  const content = (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-sidebar-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BookOpen className="h-5 w-5 text-sidebar-primary" />
          <h2 className="font-semibold text-sidebar-foreground">Study Plan</h2>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsOpen(false)}
          className="md:hidden"
        >
          <X className="h-5 w-5" />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {state.concepts.length > 0 ? (
          <ConceptList 
            concepts={state.concepts}
            completedCount={state.progress.completed}
            currentTopic={state.currentTopic}
          />
        ) : (
          <p className="text-sm text-sidebar-foreground/60 text-center mt-8">
            No concepts yet
          </p>
        )}
      </div>
    </div>
  );

  return (
    <>
      {/* Mobile toggle */}
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setIsOpen(true)}
        className="md:hidden fixed top-4 left-4 z-40"
      >
        <Menu className="h-5 w-5" />
      </Button>

      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed md:sticky top-0 left-0 h-screen bg-sidebar border-r border-sidebar-border z-50
        transition-transform duration-300 w-80
        ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        {content}
      </aside>
    </>
  );
};
