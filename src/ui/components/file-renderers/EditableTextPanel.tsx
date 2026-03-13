import { useState, useEffect } from "react";
import { SaveIcon, Loader2Icon } from "lucide-react";

type EditableTextPanelProps = {
  content: string;
  onSave: (content: string) => Promise<{ success: boolean; error?: string }>;
  onSaved?: () => void;
  placeholder?: string;
  className?: string;
  /** When true, use a monospace textarea and no wrapping (for code). */
  monospace?: boolean;
  /** When content from server changes (e.g. after reload), sync local state. */
  contentKey?: string | number;
};

export function EditableTextPanel({
  content,
  onSave,
  onSaved,
  placeholder = "Enter text…",
  className = "",
  monospace = true,
  contentKey,
}: EditableTextPanelProps) {
  const [local, setLocal] = useState(content);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLocal(content);
  }, [content, contentKey]);

  const isDirty = local !== content;

  const handleSave = async () => {
    setError(null);
    setSaving(true);
    try {
      const result = await onSave(local);
      if (result.success) {
        onSaved?.();
      } else {
        setError(result.error ?? "Failed to save");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className={`flex flex-col flex-1 min-h-0 ${className}`}>
      {isDirty && (
        <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
          <button
            type="button"
            onClick={handleSave}
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
        </div>
      )}
      <textarea
        value={local}
        onChange={(e) => setLocal(e.target.value)}
        placeholder={placeholder}
        className={`flex-1 min-h-0 w-full resize-none rounded border border-ink-900/15 bg-white/80 px-3 py-2 text-sm text-ink-800 focus:outline-none focus:ring-2 focus:ring-ink-500/30 ${
          monospace ? "font-mono whitespace-pre" : "whitespace-pre-wrap"
        }`}
        spellCheck={!monospace}
      />
    </div>
  );
}
