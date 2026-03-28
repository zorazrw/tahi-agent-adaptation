import { useState, useEffect, useRef } from "react";
import hljs from "highlight.js";
import { ViewToggle, useViewToggle } from "./ViewToggle";
import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";

type Props = { data: { kind: "html"; content: string } } & EditableRendererProps;

export function HtmlRenderer({ data, filePath, cwd, sessionId, onReload }: Props) {
  const [mode, setMode] = useViewToggle("preview");
  const codeRef = useRef<HTMLElement>(null);
  const canEdit = Boolean(filePath && onReload);
  const [editContent, setEditContent] = useState(data.content);

  useEffect(() => {
    setEditContent(data.content);
  }, [data.content]);

  useEffect(() => {
    if (mode === "source" && codeRef.current && !canEdit) {
      codeRef.current.removeAttribute("data-highlighted");
      hljs.highlightElement(codeRef.current);
    }
  }, [mode, data.content, canEdit]);

  const displayContent = canEdit ? editContent : data.content;

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
        <ViewToggle mode={mode} onChange={setMode} />
        <span className="text-xs text-muted-foreground">HTML</span>
      </div>
      {mode === "preview" ? (
        <iframe
          srcDoc={displayContent}
          sandbox="allow-scripts"
          className="flex-1 w-full min-h-0 border-0 rounded bg-white"
          style={{ minHeight: 200 }}
          title="HTML Preview"
        />
      ) : canEdit ? (
        <EditableTextPanel
          content={editContent}
          contentKey={data.content}
          monospace
          onSave={async (content) => {
            setEditContent(content);
            return window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined);
          }}
          onSaved={onReload}
        />
      ) : (
        <pre className="text-sm overflow-x-auto">
          <code ref={codeRef} className="language-html">
            {data.content}
          </code>
        </pre>
      )}
    </div>
  );
}
