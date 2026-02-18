import { useEffect, useState } from "react";
import type { StreamMessage } from "../types";
import { getRenderer } from "./file-renderers";

const FILE_TOOL_NAMES = new Set(["Read", "Write", "Edit"]);
const PREVIEW_EXTENSIONS = [
  // Documents
  ".txt", ".md", ".csv", ".tsv", ".json",
  ".xlsx", ".xls", ".docx", ".pdf",
  // Web
  ".html", ".htm",
  // Images
  ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg",
  // Media
  ".mp4", ".webm", ".mp3", ".wav", ".ogg",
  // Code
  ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
  ".py", ".rb", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
  ".css", ".scss", ".less", ".php", ".swift", ".kt",
  ".sh", ".bash", ".zsh",
  ".yaml", ".yml", ".toml", ".xml", ".sql",
  ".r", ".lua", ".dart", ".scala", ".ex", ".exs", ".hs", ".ml",
];

function pathHasPreviewExt(path: string): boolean {
  const lower = path.toLowerCase();
  return PREVIEW_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

/** From the current chat session, find the latest file referred by the agent (tool_use Read/Write/Edit) that we can preview (.txt, .xlsx, .xls, .docx, .jpg, .png, .pdf). */
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

/** First previewable file from the given step's output files list (used for step-scoped preview). */
export function getPreviewFileForStep(
  outputFiles: string[][] | undefined,
  selectedStepIndex: number
): string | null {
  const files = outputFiles?.[selectedStepIndex];
  if (!files?.length) return null;
  const path = files.find((p) => pathHasPreviewExt(p));
  return path ?? null;
}

type PreviewFileResult =
  | { kind: "txt"; content: string }
  | { kind: "xlsx"; data: unknown[][] }
  | { kind: "docx"; html: string }
  | { kind: "image"; dataUrl: string }
  | { kind: "pdf"; data: string }
  | { kind: "md"; content: string }
  | { kind: "code"; content: string; language: string }
  | { kind: "csv"; content: string }
  | { kind: "json"; content: string }
  | { kind: "html"; content: string }
  | { kind: "video"; dataUrl: string }
  | { kind: "audio"; dataUrl: string }
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

  const Renderer = result && "kind" in result ? getRenderer(result.kind) : null;

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
        {Renderer && !loading && <Renderer data={result} />}
      </div>
    </div>
  );
}
