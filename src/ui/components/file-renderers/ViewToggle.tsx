import { useState } from "react";

type ViewMode = "preview" | "source";

export function useViewToggle(initial: ViewMode = "preview") {
  return useState<ViewMode>(initial);
}

/** Eye icon (preview) + </> icon (source) toggle — matches Claude artifact style */
export function ViewToggle({
  mode,
  onChange,
}: {
  mode: ViewMode;
  onChange: (mode: ViewMode) => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-ink-900/15 overflow-hidden">
      <button
        onClick={() => onChange("preview")}
        title="Preview"
        className={`px-1.5 py-1 transition-colors ${
          mode === "preview"
            ? "bg-ink-900/10 text-ink-900"
            : "text-ink-400 hover:text-ink-700 hover:bg-ink-900/5"
        }`}
      >
        {/* Eye icon */}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      </button>
      <button
        onClick={() => onChange("source")}
        title="Source"
        className={`px-1.5 py-1 border-l border-ink-900/15 transition-colors ${
          mode === "source"
            ? "bg-ink-900/10 text-ink-900"
            : "text-ink-400 hover:text-ink-700 hover:bg-ink-900/5"
        }`}
      >
        {/* Code icon */}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="16 18 22 12 16 6" />
          <polyline points="8 6 2 12 8 18" />
        </svg>
      </button>
    </div>
  );
}
