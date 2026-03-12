import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import hljs from "highlight.js";
import { ViewToggle, useViewToggle } from "./ViewToggle";

export function MarkdownRenderer({ data }: { data: { kind: "md"; content: string } }) {
  const [mode, setMode] = useViewToggle("preview");
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (mode === "source" && codeRef.current) {
      codeRef.current.removeAttribute("data-highlighted");
      hljs.highlightElement(codeRef.current);
    }
  }, [mode, data.content]);

  return (
    <div>
      <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2">
        <ViewToggle mode={mode} onChange={setMode} />
        <span className="text-xs text-muted-foreground">Markdown</span>
      </div>
      {mode === "preview" ? (
        <div className="md-prose">
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex, rehypeHighlight, rehypeRaw]}>
            {data.content}
          </ReactMarkdown>
        </div>
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
