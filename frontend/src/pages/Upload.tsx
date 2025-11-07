import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileUpload } from '@/components/FileUpload';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { ErrorBanner } from '@/components/ErrorBanner';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { uploadDocument, startFromSaved, listSavedSessions } from '@/api/tutorApi';
import { useTutor } from '@/context/TutorContext';
import { GraduationCap } from 'lucide-react';

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [context, setContext] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { setSessionId, setConcepts, setStudyPlan, setCurrentTopic } = useTutor();
  const [savedId, setSavedId] = useState('');
  const [savedOptions, setSavedOptions] = useState<{id:string; label:string}[]>([]);

  const handleStart = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      // Always use "from_beginning" mode
      const response = await uploadDocument(file, 'from_beginning');
      setSessionId(response.session_id);
      setConcepts(response.concepts);
      setStudyPlan(response.study_plan);
      
      // Set current topic from the FIRST STEP in study plan, not concepts
      if (response.study_plan && response.study_plan.length > 0) {
        setCurrentTopic(response.study_plan[0].topic);
      } else if (response.concepts.length > 0) {
        setCurrentTopic(response.concepts[0].label);
      }
      
      navigate('/concepts');
    } catch (err) {
      setError('Failed to process document. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleStartFromSaved = async () => {
    if (!savedId.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await startFromSaved(savedId.trim(), 'from_beginning');
      setSessionId(response.session_id);
      setConcepts(response.concepts);
      setStudyPlan(response.study_plan);
      if (response.study_plan && response.study_plan.length > 0) {
        setCurrentTopic(response.study_plan[0].topic);
      } else if (response.concepts.length > 0) {
        setCurrentTopic(response.concepts[0].label);
      }
      navigate('/concepts');
    } catch (err) {
      setError('Failed to start from saved session.');
    } finally {
      setLoading(false);
    }
  };

  // Load saved sessions for dropdown
  useEffect(() => {
    (async () => {
      try {
        const sessions = await listSavedSessions();
        setSavedOptions(sessions.map(s => ({ id: s.session_id, label: `${s.document_name} (${s.session_id})` })));
      } catch {}
    })();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <div className="max-w-3xl mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary rounded-2xl mb-4">
            <GraduationCap className="h-8 w-8 text-primary-foreground" />
          </div>
          <h1 className="text-4xl font-bold text-foreground mb-3">
            Adaptive AI Tutor
          </h1>
          <p className="text-lg text-muted-foreground">
            Upload your study material and let AI create a personalized learning plan
          </p>
        </div>

        <div className="space-y-6">
          {error && <ErrorBanner message={error} onDismiss={() => setError(null)} onRetry={handleStart} />}

          <div className="bg-card rounded-xl p-6 border border-border shadow-sm">
            <Label className="text-base font-semibold text-card-foreground mb-4 block">
              1. Upload Study Material
            </Label>
            <FileUpload 
              onFileSelect={setFile}
              selectedFile={file}
              onClear={() => setFile(null)}
            />
          </div>

          <div className="bg-card rounded-xl p-6 border border-border shadow-sm">
            <Label htmlFor="context" className="text-base font-semibold text-card-foreground mb-2 block">
              2. What are you studying for? (Optional)
            </Label>
            <Input
              id="context"
              placeholder="e.g., Class 9 History Exam, AP Biology Final..."
              value={context}
              onChange={(e) => setContext(e.target.value)}
              className="mt-2"
            />
          </div>

          <div className="bg-card rounded-xl p-6 border border-border shadow-sm">
            <Label className="text-base font-semibold text-card-foreground mb-4 block">
              2b. Or resume from saved document (skip upload/parsing)
            </Label>
            <div className="flex gap-2 items-center">
              <select className="border border-border rounded px-3 py-2 flex-1 bg-background" value={savedId} onChange={e => setSavedId(e.target.value)}>
                <option value="">Select a saved session…</option>
                {savedOptions.map(opt => (
                  <option key={opt.id} value={opt.id}>{opt.label}</option>
                ))}
              </select>
              <Button onClick={handleStartFromSaved} variant="outline" disabled={loading || !savedId.trim()}>Start From Saved</Button>
            </div>
          </div>

          {loading ? (
            <LoadingSpinner message="Analyzing your document..." size="lg" />
          ) : (
            <Button 
              onClick={handleStart}
              disabled={!file}
              size="lg"
              className="w-full"
            >
              Start Learning
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
