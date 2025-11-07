import { useEffect, useState } from 'react';
import { getLastSummaries, SessionSummary } from '@/api/tutorApi';
import { useTutor } from '@/context/TutorContext';

export default function LastSummaries() {
  const { state } = useTutor();
  const [summaries, setSummaries] = useState<SessionSummary[]>([]);

  useEffect(() => {
    const run = async () => {
      if (!state.sessionId) return;
      try {
        const data = await getLastSummaries(state.sessionId, 3);
        setSummaries(data);
      } catch (e) {
        // no-op
      }
    };
    run();
  }, [state.sessionId, state.currentStep]);

  if (!summaries || summaries.length === 0) return null;

  return (
    <div className="bg-muted/40 border border-border rounded-md p-3">
      <div className="text-sm font-medium mb-2">What we just covered</div>
      <div className="grid gap-2 md:grid-cols-2">
        {summaries.slice(0, 2).map((s, idx) => (
          <div key={idx} className="bg-card text-card-foreground border border-border rounded p-2 text-sm">
            <div className="opacity-70 mb-1">{s.topic} {s.segment_id && `• ${s.segment_id}`}</div>
            <div>{s.summary}</div>
          </div>
        ))}
      </div>
    </div>
  );
}


