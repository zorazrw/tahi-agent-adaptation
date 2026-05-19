import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import hljs from "highlight.js";
import { ViewToggle, useViewToggle } from "./ViewToggle";
import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";

type Props = { data: { kind: "md"; content: string } } & EditableRendererProps;

export function MarkdownRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  onReload,
  onTextSaveChromeChange,
}: Props) {
  const canEdit = Boolean(filePath && onReload);
  const [mode, setMode] = useViewToggle(canEdit ? "source" : "preview");
  const codeRef = useRef<HTMLElement>(null);
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

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
        <ViewToggle mode={mode} onChange={setMode} />
        <span className="text-xs text-muted-foreground">Markdown</span>
      </div>
      {mode === "preview" ? (
        canEdit ? (
          <div className="md-prose flex-1 min-h-0 overflow-auto">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex, rehypeHighlight, rehypeRaw]}
            >
              {editContent}
            </ReactMarkdown>
          </div>
        ) : (
          <div className="md-prose">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex, rehypeHighlight, rehypeRaw]}
            >
              {data.content}
            </ReactMarkdown>
          </div>
        )
      ) : canEdit ? (
        <EditableTextPanel
          content={editContent}
          contentKey={data.content}
          monospace={false}
          onSave={async (content) => {
            setEditContent(content);
            return window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined);
          }}
          onSaved={onReload}
          onSaveChromeChange={onTextSaveChromeChange}
        />
      ) : (
        <pre className="text-sm overflow-x-auto">
          <code ref={codeRef} className="language-markdown">
            {data.content}
          </code>
        </pre>
      )}
    </div>
  );
}
