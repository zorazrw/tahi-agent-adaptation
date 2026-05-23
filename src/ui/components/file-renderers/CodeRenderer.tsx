import { useEffect, useRef } from "react";
import hljs from "highlight.js";
import { ViewToggle, useViewToggle } from "./ViewToggle";
import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";

type Props = { data: { kind: "code"; content: string; language: string } } & EditableRendererProps;

export function CodeRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  onReload,
  onTextSaveChromeChange,
}: Props) {
  const [mode, setMode] = useViewToggle("source");
  const codeRef = useRef<HTMLElement>(null);
  const canEdit = Boolean(filePath && onReload);

  useEffect(() => {
    if (mode === "source" && codeRef.current && !canEdit) {
      codeRef.current.removeAttribute("data-highlighted");
      hljs.highlightElement(codeRef.current);
    }
  }, [mode, data.content, data.language, canEdit]);

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
        <ViewToggle mode={mode} onChange={setMode} />
        <span className="text-xs text-muted-foreground">{data.language}</span>
      </div>
      {mode === "source" ? (
        canEdit ? (
          <EditableTextPanel
            content={data.content}
            contentKey={data.content}
            monospace
            onSave={async (content) =>
              window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined)
            }
            onSaved={onReload}
            onSaveChromeChange={onTextSaveChromeChange}
          />
        ) : (
          <pre className="text-sm overflow-x-auto">
            <code ref={codeRef} className={`language-${data.language}`}>
              {data.content}
            </code>
          </pre>
        )
      ) : (
        <div className="source-view">{data.content}</div>
      )}
    </div>
  );
}
