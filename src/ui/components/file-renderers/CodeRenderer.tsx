import { useEffect, useRef } from "react";
import hljs from "highlight.js";
import { ViewToggle, useViewToggle } from "./ViewToggle";

export function CodeRenderer({ data }: { data: { kind: "code"; content: string; language: string } }) {
  const [mode, setMode] = useViewToggle("source");
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (mode === "source" && codeRef.current) {
      codeRef.current.removeAttribute("data-highlighted");
      hljs.highlightElement(codeRef.current);
    }
  }, [mode, data.content, data.language]);

  return (
    <div>
      <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2">
        <ViewToggle mode={mode} onChange={setMode} />
        <span className="text-xs text-muted-foreground">{data.language}</span>
      </div>
      {mode === "source" ? (
        <pre className="text-sm overflow-x-auto">
          <code ref={codeRef} className={`language-${data.language}`}>
            {data.content}
          </code>
        </pre>
      ) : (
        <div className="source-view">{data.content}</div>
      )}
    </div>
  );
}
