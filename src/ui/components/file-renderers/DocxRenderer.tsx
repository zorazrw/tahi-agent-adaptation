import { useEffect, useRef, useState } from "react";
import { renderAsync } from "docx-preview";

export function ZoomControls({ zoom, onZoomIn, onZoomOut, min, max }: {
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  min: number;
  max: number;
}) {
  return (
    <div className="flex items-center gap-0.5 shrink-0">
      <button
        onClick={onZoomOut}
        disabled={zoom <= min}
        className="p-0.5 rounded hover:bg-surface-tertiary disabled:opacity-30 disabled:cursor-not-allowed text-ink-500 hover:text-ink-700 transition-colors"
        title="Zoom out"
      >
        <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="3" y1="8" x2="13" y2="8" />
        </svg>
      </button>
      <span className="text-[10px] font-medium text-ink-400 w-7 text-center select-none">
        {Math.round(zoom * 100)}%
      </span>
      <button
        onClick={onZoomIn}
        disabled={zoom >= max}
        className="p-0.5 rounded hover:bg-surface-tertiary disabled:opacity-30 disabled:cursor-not-allowed text-ink-500 hover:text-ink-700 transition-colors"
        title="Zoom in"
      >
        <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="3" y1="8" x2="13" y2="8" />
          <line x1="8" y1="3" x2="8" y2="13" />
        </svg>
      </button>
    </div>
  );
}

export function DocxRenderer({ data, zoom }: { data: { kind: "docx"; data: string }; zoom?: number }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    setLoading(true);
    setError(null);

    const binary = atob(data.data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    renderAsync(bytes.buffer, container, undefined, {
      inWrapper: true,
      breakPages: true,
      useBase64URL: true,
    })
      .then(() => setLoading(false))
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });

    return () => {
      container.innerHTML = "";
    };
  }, [data.data]);

  return (
    <div>
      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground p-2">
          <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span>Rendering document…</span>
        </div>
      )}
      {error && <p className="text-sm text-error p-2">{error}</p>}
      <div
        ref={containerRef}
        className="docx-wrapper"
        style={{ zoom: zoom ?? 1 }}
      />
    </div>
  );
}
