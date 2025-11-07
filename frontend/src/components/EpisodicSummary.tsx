import { useEffect, useState } from 'react';
import { useTutor } from '@/context/TutorContext';

type Payload = { text?: string; ts?: string; meta?: any };

export default function EpisodicSummary() {
  const { state } = useTutor();
  const [payload, setPayload] = useState<Payload | null>(null);

  useEffect(() => {
    const run = async () => {
      if (!state.sessionId) return;
      try {
        const base = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const resp = await fetch(`${base}/session/${encodeURIComponent(state.sessionId)}/episodic_summary`);
        if (!resp.ok) return;
        const data = await resp.json();
        setPayload(data.episodic_summary || null);
      } catch (e) {
        // no-op
      }
    };
    run();
  }, [state.sessionId, state.currentStep]);

  if (!payload || !payload.text) return null;

  return (
    <div className="bg-muted/30 border border-border rounded-md p-3">
      <div className="text-sm font-medium mb-1">Session summary</div>
      <div className="text-sm opacity-90">{payload.text}</div>
    </div>
  );
}


