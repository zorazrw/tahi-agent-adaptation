import { useEffect, useRef } from "react";
import hljs from "highlight.js";
import { ViewToggle, useViewToggle } from "./ViewToggle";

export function HtmlRenderer({ data }: { data: { kind: "html"; content: string } }) {
  const [mode, setMode] = useViewToggle("preview");
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (mode === "source" && codeRef.current) {
      codeRef.current.removeAttribute("data-highlighted");
      hljs.highlightElement(codeRef.current);
    }
  }, [mode, data.content]);

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="shrink-0 flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2">
        <ViewToggle mode={mode} onChange={setMode} />
        <span className="text-xs text-muted-foreground">HTML</span>
      </div>
      <div className="flex-1 min-h-0 overflow-auto">
        {mode === "preview" ? (
          <iframe
            srcDoc={data.content}
            sandbox="allow-scripts"
            className="w-full h-full border-0 rounded bg-white"
            title="HTML Preview"
          />
        ) : (
          <pre className="text-sm overflow-auto">
            <code ref={codeRef} className="language-html">
              {data.content}
            </code>
          </pre>
        )}
      </div>
    </div>
  );
}
