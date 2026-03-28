import { useState, useMemo, useEffect } from "react";
import JsonView from "@uiw/react-json-view";
import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";

type ViewMode = "tree" | "text";

export function JsonRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  onReload,
}: { data: { kind: "json"; content: string } } & EditableRendererProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("tree");
  const parsed = useMemo(() => {
    try {
      return { value: JSON.parse(data.content), error: null };
    } catch (e) {
      return { value: null, error: e instanceof Error ? e.message : String(e) };
    }
  }, [data.content]);
  const canEdit = Boolean(filePath && onReload);

  useEffect(() => {
    if (!canEdit) setViewMode("tree");
  }, [canEdit]);

  if (canEdit && viewMode === "text") {
    return (
      <div className="flex flex-col flex-1 min-h-0">
        <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
          <button
            type="button"
            onClick={() => setViewMode("tree")}
            className="text-xs text-muted-foreground hover:text-ink-700"
          >
            ← Tree
          </button>
          <span className="text-xs text-muted-foreground">Edit as text</span>
        </div>
        <EditableTextPanel
          content={data.content}
          contentKey={data.content}
          monospace
          onSave={async (content) =>
            window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined)
          }
          onSaved={onReload}
        />
      </div>
    );
  }

  if (parsed.error) {
    return (
      <div className="flex flex-col flex-1 min-h-0">
        {canEdit && (
          <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2">
            <button
              type="button"
              onClick={() => setViewMode("text")}
              className="text-xs text-muted-foreground hover:text-ink-700"
            >
              Edit as text
            </button>
          </div>
        )}
        <p className="text-sm text-error mb-2">Invalid JSON: {parsed.error}</p>
        {canEdit ? (
          <EditableTextPanel
            content={data.content}
            contentKey={data.content}
            monospace
            onSave={async (content) =>
              window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined)
            }
            onSaved={onReload}
          />
        ) : (
          <pre className="text-sm text-ink-700 whitespace-pre-wrap font-mono">{data.content}</pre>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {canEdit && (
        <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
          <button
            type="button"
            onClick={() => setViewMode("text")}
            className="text-xs text-muted-foreground hover:text-ink-700"
          >
            Edit as text
          </button>
        </div>
      )}
      <div className="text-sm overflow-auto">
        <JsonView value={parsed.value} collapsed={2} displayDataTypes={false} />
      </div>
    </div>
  );
}
