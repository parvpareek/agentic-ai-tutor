import React from 'react';

type Segment = {
  segment_id?: string;
  title?: string;
  order?: number;
  estimated_minutes?: number;
  learning_objectives?: string[];
};

type ConceptWithSegments = {
  concept_id: string;
  label: string;
  learning_segments?: Segment[];
};

export function SegmentList({ concepts }: { concepts: ConceptWithSegments[] }) {
  if (!concepts || concepts.length === 0) return null;

  return (
    <div className="space-y-6">
      {concepts.map((c) => (
        <div key={c.concept_id} className="border border-border rounded-lg p-4">
          <div className="font-semibold mb-2 text-card-foreground">{c.label}</div>
          {c.learning_segments && c.learning_segments.length > 0 ? (
            <ol className="list-decimal list-inside space-y-1 text-sm">
              {c.learning_segments
                .slice()
                .sort((a, b) => (a.order || 0) - (b.order || 0))
                .map((s) => (
                  <li key={s.segment_id || s.title} className="flex items-center justify-between">
                    <span>
                      {s.title || s.segment_id}
                      {s.order ? <span className="text-muted-foreground"> (#{s.order})</span> : null}
                    </span>
                    {typeof s.estimated_minutes === 'number' ? (
                      <span className="text-muted-foreground text-xs">{s.estimated_minutes}m</span>
                    ) : null}
                  </li>
                ))}
            </ol>
          ) : (
            <div className="text-sm text-muted-foreground">No segments detected.</div>
          )}
        </div>
      ))}
    </div>
  );
}


