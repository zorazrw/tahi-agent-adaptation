import { useEffect, useState } from "react";
import type { StreamMessage } from "../types";

const FILE_TOOL_NAMES = new Set(["Read", "Write", "Edit"]);
const PREVIEW_EXTENSIONS = [".txt", ".xlsx", ".xls", ".docx", ".jpg", ".jpeg", ".png"];

function pathHasPreviewExt(path: string): boolean {
  const lower = path.toLowerCase();
  return PREVIEW_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

/** From the current chat session, find the latest file referred by the agent (tool_use Read/Write/Edit) that we can preview (.txt, .xlsx, .xls, .docx, .jpg, .png). */
export function getLatestPreviewFileRef(messages: StreamMessage[]): string | null {
  let latest: string | null = null;
  for (const msg of messages) {
    if (msg.type !== "assistant" || !("message" in msg)) continue;
    const content = (msg as { message?: { content?: unknown[] } }).message?.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      const b = block as { type?: string; name?: string; input?: { file_path?: string } };
      if (b?.type !== "tool_use" || !FILE_TOOL_NAMES.has(b.name ?? "")) continue;
      const path = b.input?.file_path;
      if (typeof path === "string" && pathHasPreviewExt(path)) {
        latest = path;
      }
    }
  }
  return latest;
}

type PreviewFileResult =
  | { kind: "txt"; content: string }
  | { kind: "xlsx"; data: unknown[][] }
  | { kind: "docx"; html: string }
  | { kind: "image"; dataUrl: string }
  | { error: string };

type FilePreviewProps = {
  filePath: string | null;
  cwd?: string | null;
};

export function FilePreview({ filePath, cwd }: FilePreviewProps) {
  const [result, setResult] = useState<PreviewFileResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!filePath) {
      setResult(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setResult(null);
    window.electron
      .previewFile(filePath, cwd ?? undefined)
      .then((res) => setResult(res))
      .catch((err) => setResult({ error: err instanceof Error ? err.message : String(err) }))
      .finally(() => setLoading(false));
  }, [filePath, cwd]);

  if (!filePath) return null;

  return (
    <div className="flex flex-col h-full min-h-0 rounded-lg border border-ink-900/10 bg-surface-secondary">
      <div className="shrink-0 px-3 py-2 border-b border-ink-900/10">
        <span className="text-xs font-medium text-muted truncate block" title={filePath}>
          {filePath}
        </span>
      </div>
      <div className="flex-1 min-h-0 overflow-auto px-3 py-2">
        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted">
            <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span>Loading…</span>
          </div>
        )}
        {result && "error" in result && !loading && (
          <p className="text-sm text-error">{result.error}</p>
        )}
        {result && "kind" in result && result.kind === "txt" && !loading && (
          <pre className="text-sm text-ink-700 whitespace-pre-wrap break-words font-mono">
            {result.content}
          </pre>
        )}
        {result && "kind" in result && result.kind === "xlsx" && !loading && (
          <div className="overflow-auto">
            <table className="text-sm text-ink-700 border-collapse border border-ink-900/20">
              <tbody>
                {result.data.map((row: unknown[], i: number) => (
                  <tr key={i}>
                    {(Array.isArray(row) ? row : []).map((cell, j) => (
                      <td
                        key={j}
                        className="border border-ink-900/20 px-2 py-1.5 align-top"
                      >
                        {cell != null ? String(cell) : ""}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {result && "kind" in result && result.kind === "docx" && !loading && (
          <div
            className="file-preview-docx text-sm text-ink-700 prose prose-sm max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1"
            dangerouslySetInnerHTML={{ __html: result.html }}
          />
        )}
        {result && "kind" in result && result.kind === "image" && !loading && (
          <div className="flex items-center justify-center min-h-[120px]">
            <img
              src={result.dataUrl}
              alt="Preview"
              className="max-w-full max-h-full object-contain rounded"
            />
          </div>
        )}
      </div>
    </div>
  );
}
