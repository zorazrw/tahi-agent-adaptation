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
import { TextDocumentFrame } from "./TextDocumentFrame";

type Props = { data: { kind: "md"; content: string } } & EditableRendererProps;

const remarkPlugins = [remarkGfm, remarkMath];
const rehypePlugins = [rehypeKatex, rehypeHighlight, rehypeRaw];

export function MarkdownRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  onReload,
  onTextSaveChromeChange,
}: Props) {
  const canEdit = Boolean(filePath && onReload);
  const [mode, setMode] = useViewToggle("preview");
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

  const toolbar = <ViewToggle mode={mode} onChange={setMode} />;
  const mdContent = canEdit ? editContent : data.content;

  return (
    <TextDocumentFrame toolbar={toolbar} label="Markdown">
      {mode === "preview" ? (
        <div className="md-prose">
          <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins}>
            {mdContent}
          </ReactMarkdown>
        </div>
      ) : canEdit ? (
        <EditableTextPanel
          content={editContent}
          contentKey={data.content}
          monospace={false}
          variant="document"
          onSave={async (content) => {
            setEditContent(content);
            return window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined);
          }}
          onSaved={onReload}
          onSaveChromeChange={onTextSaveChromeChange}
        />
      ) : (
        <pre className="text-sm overflow-x-auto rounded-lg border border-ink-900/10 bg-surface-secondary/80 px-3 py-2.5">
          <code ref={codeRef} className="language-markdown">
            {data.content}
          </code>
        </pre>
      )}
    </TextDocumentFrame>
  );
}
