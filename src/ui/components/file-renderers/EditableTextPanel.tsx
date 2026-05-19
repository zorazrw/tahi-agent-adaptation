import { useState, useEffect, useCallback } from "react";
import { SaveIcon, Loader2Icon } from "lucide-react";
import type { PreviewSaveChrome } from "./index";

type EditableTextPanelProps = {
  content: string;
  onSave: (content: string) => Promise<{ success: boolean; error?: string }>;
  onSaved?: () => void;
  onSaveChromeChange?: (chrome: PreviewSaveChrome | null) => void;
  placeholder?: string;
  className?: string;
  /** When true, use a monospace textarea and no wrapping (for code). */
  monospace?: boolean;
  /** When content from server changes (e.g. after reload), sync local state. */
  contentKey?: string | number;
  /** "document" = transparent field inside TextDocumentFrame; "default" = bordered box. */
  variant?: "default" | "document";
};

export function EditableTextPanel({
  content,
  onSave,
  onSaved,
  onSaveChromeChange,
  placeholder = "Enter text…",
  className = "",
  monospace = true,
  contentKey,
  variant = "default",
}: EditableTextPanelProps) {
  const isDocument = variant === "document";
  const [local, setLocal] = useState(content);
  const [baseline, setBaseline] = useState(content);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLocal(content);
    setBaseline(content);
  }, [content, contentKey]);

  const isDirty = local !== baseline;

  const handleSave = useCallback(async () => {
    if (!isDirty || saving) return;
    setError(null);
    setSaving(true);
    try {
      const result = await onSave(local);
      if (result.success) {
        setBaseline(local);
        onSaved?.();
      } else {
        setError(result.error ?? "Failed to save");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [baseline, isDirty, local, onSave, onSaved, saving]);

  useEffect(() => {
    if (!onSaveChromeChange) return;
    if (!isDirty && !saving && !error) {
      onSaveChromeChange(null);
      return;
    }
    onSaveChromeChange({
      save: () => void handleSave(),
      disabled: !isDirty || saving,
      saving,
      error,
    });
    return () => onSaveChromeChange(null);
  }, [error, handleSave, isDirty, onSaveChromeChange, saving]);

  useEffect(() => {
    const onFlush = () => {
      void handleSave();
    };
    window.addEventListener("preview-flush-save", onFlush);
    return () => window.removeEventListener("preview-flush-save", onFlush);
  }, [handleSave]);

  return (
    <div className={`flex flex-col flex-1 min-h-0 ${className}`}>
      {isDirty && !isDocument && (
        <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
          <button
            type="button"
            onClick={() => void handleSave()}
            disabled={saving}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-sm font-medium bg-ink-800 text-white hover:bg-ink-700 disabled:opacity-50"
          >
            {saving ? (
              <Loader2Icon className="size-4 animate-spin" />
            ) : (
              <SaveIcon className="size-4" />
            )}
            {saving ? "Saving…" : "Save"}
          </button>
          {error && <span className="text-sm text-error">{error}</span>}
          <span className="text-xs text-muted-foreground ml-auto">⌘S to save</span>
        </div>
      )}
      {isDirty && isDocument && error && (
        <p className="text-sm text-error mb-2 shrink-0">{error}</p>
      )}
      <textarea
        value={local}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={() => void handleSave()}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
            e.preventDefault();
            void handleSave();
          }
        }}
        placeholder={placeholder}
        className={
          isDocument
            ? `flex-1 min-h-[12rem] w-full resize-none border-0 bg-transparent px-0 py-0 text-[0.9375rem] leading-[1.75] text-ink-800 focus:outline-none focus:ring-0 selection:bg-primary/15 ${
                monospace ? "font-mono whitespace-pre" : "whitespace-pre-wrap"
              }`
            : `flex-1 min-h-0 w-full resize-none rounded-lg border border-ink-900/12 bg-white/90 px-3 py-2.5 text-[0.9375rem] leading-relaxed text-ink-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/25 ${
                monospace ? "font-mono whitespace-pre" : "whitespace-pre-wrap"
              }`
        }
        spellCheck={!monospace}
      />
    </div>
  );
}
