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
    <div className="inline-flex rounded-lg border border-ink-900/12 bg-surface-secondary/80 p-0.5 shadow-sm">
      <button
        type="button"
        onClick={() => onChange("preview")}
        title="Preview"
        className={`rounded-md px-2 py-1 transition-colors ${
          mode === "preview"
            ? "bg-white text-ink-900 shadow-sm"
            : "text-ink-500 hover:text-ink-800"
        }`}
      >
        {/* Eye icon */}
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      </button>
      <button
        type="button"
        onClick={() => onChange("source")}
        title="Source"
        className={`rounded-md px-2 py-1 transition-colors ${
          mode === "source"
            ? "bg-white text-ink-900 shadow-sm"
            : "text-ink-500 hover:text-ink-800"
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
