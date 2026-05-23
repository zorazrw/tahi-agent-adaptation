import { useCallback, useEffect, useState } from "react";
import type { StreamMessage } from "../types";
import { CopyIcon, FolderOpenIcon, Loader2Icon, RefreshCcwIcon, SaveIcon } from "lucide-react";
import { getRenderer, type PreviewSaveChrome } from "./file-renderers";
import { ZoomControls } from "./file-renderers/DocxRenderer";

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

export function pathHasPreviewExt(path: string): boolean {
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

/** First previewable file from a node's output files list. */
export function getPreviewFileForNode(
  outputFiles: string[] | undefined
): string | null {
  if (!outputFiles?.length) return null;
  const path = outputFiles.find((p) => pathHasPreviewExt(p));
  return path ?? null;
}

/** @deprecated kept for backward compat - use getPreviewFileForNode */
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
  | { kind: "xlsx"; sheets: { name: string; html: string }[] }
  | { kind: "docx"; data: string }
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

function isFileNotFoundError(error: string): boolean {
  return /ENOENT|no such file or directory/i.test(error);
}

/** Get plain-text copyable content from preview result, or null if not copyable. */
function getCopyableContent(result: PreviewFileResult | null): string | null {
  if (!result || "error" in result) return null;
  switch (result.kind) {
    case "txt":
    case "md":
    case "code":
    case "json":
    case "csv":
    case "html":
      return result.content ?? null;
    case "docx":
      return result.data ?? null;
    default:
      return null;
  }
}

type FilePreviewProps = {
  filePath: string | null;
  cwd?: string | null;
  sessionId?: string | null;
  stepCompleted?: boolean;
};

const ZOOM_STEP = 0.1;
const ZOOM_MIN = 0.3;
const ZOOM_MAX = 2.0;

export function FilePreview({ filePath, cwd, sessionId, stepCompleted }: FilePreviewProps) {
  const [result, setResult] = useState<PreviewFileResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [zoom, setZoom] = useState(0.6);
  const [copyFeedback, setCopyFeedback] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [previewSaveChrome, setPreviewSaveChrome] = useState<PreviewSaveChrome | null>(null);

  const onPreviewSaveChromeChange = useCallback((chrome: PreviewSaveChrome | null) => {
    setPreviewSaveChrome(chrome);
  }, []);

  useEffect(() => {
    setPreviewSaveChrome(null);
  }, [filePath]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "s") return;
      if (!previewSaveChrome || previewSaveChrome.disabled) return;
      e.preventDefault();
      previewSaveChrome.save();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [previewSaveChrome]);

  const handleCopyFileContent = () => {
    const text = getCopyableContent(result);
    if (text == null) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 1500);
    });
  };

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
  }, [filePath, cwd, stepCompleted, refreshKey]);

  if (!filePath) {
    return (
      <div className="flex flex-col items-center justify-center h-full rounded-lg border border-dashed border-ink-900/15 bg-surface-secondary/50 text-center px-6">
        <svg viewBox="0 0 24 24" className="h-10 w-10 text-ink-400 mb-3" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
        </svg>
        <p className="text-sm text-muted-foreground">Output files will appear here when the step runs.</p>
      </div>
    );
  }

  const isNotFound = result && "error" in result && isFileNotFoundError(result.error);
  const Renderer = result && "kind" in result ? getRenderer(result.kind) : null;
  const showZoom = result && "kind" in result && result.kind === "docx";
  const isTextualPreview =
    result && "kind" in result && (result.kind === "txt" || result.kind === "md");

  return (
    <div
      className={`flex h-full min-h-0 flex-1 flex-col overflow-hidden border bg-surface-secondary/60 ${
        isTextualPreview ? "rounded-xl border-ink-900/8 shadow-card" : "rounded-lg border-ink-900/10"
      }`}
    >
      <div className="px-3 py-1.5 border-b border-ink-900/10 flex items-center gap-2 shrink-0">
        <span className="text-xs font-medium text-muted-foreground truncate flex-1" title={filePath}>
          {filePath}
        </span>
        {previewSaveChrome && (
          <button
            type="button"
            onClick={() => previewSaveChrome.save()}
            disabled={previewSaveChrome.disabled}
            className="shrink-0 p-1 rounded hover:bg-ink-900/5 text-muted-foreground hover:text-ink-700 transition-colors disabled:opacity-50 disabled:pointer-events-auto"
            aria-label={previewSaveChrome.saving ? "Saving preview changes" : "Save preview changes"}
            title={
              previewSaveChrome.error
                ? previewSaveChrome.error
                : previewSaveChrome.disabled && !previewSaveChrome.saving
                  ? "No preview changes to save yet"
                  : "Save changes to file (Ctrl or ⌘+S)"
            }
          >
            {previewSaveChrome.saving ? (
              <Loader2Icon className="size-4 animate-spin" aria-hidden />
            ) : (
              <SaveIcon className="size-4" aria-hidden />
            )}
          </button>
        )}
        <button
          onClick={() => setRefreshKey((k) => k + 1)}
          className="shrink-0 p-1 rounded hover:bg-ink-900/5 text-muted-foreground hover:text-ink-700 transition-colors"
          aria-label="Refresh preview"
          title="Refresh preview"
        >
          <RefreshCcwIcon className="size-4" />
        </button>
        {getCopyableContent(result) != null && (
          <button
            onClick={handleCopyFileContent}
            className="shrink-0 p-1 rounded hover:bg-ink-900/5 text-muted-foreground hover:text-ink-700 transition-colors"
            aria-label={copyFeedback ? "Copied" : "Copy file content"}
            title={copyFeedback ? "Copied!" : "Copy file content"}
          >
            <CopyIcon className="size-4" />
          </button>
        )}
        <button
          onClick={() => window.electron.showItemInFolder(filePath, cwd ?? undefined)}
          className="shrink-0 p-1 rounded hover:bg-ink-900/5 text-muted-foreground hover:text-ink-700 transition-colors"
          aria-label="Open in Finder"
          title="Open in Finder"
        >
          <FolderOpenIcon className="size-4" />
        </button>
        {showZoom && (
          <ZoomControls
            zoom={zoom}
            onZoomIn={() => setZoom((z) => Math.min(ZOOM_MAX, +(z + ZOOM_STEP).toFixed(1)))}
            onZoomOut={() => setZoom((z) => Math.max(ZOOM_MIN, +(z - ZOOM_STEP).toFixed(1)))}
            min={ZOOM_MIN}
            max={ZOOM_MAX}
          />
        )}
      </div>
      <div className="flex-1 flex flex-col min-h-0 min-w-0 overflow-hidden px-3 py-2">
        {loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span>Loading…</span>
          </div>
        )}
        {isNotFound && !loading && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <svg className="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 6v6l4 2" />
            </svg>
            <span>File will be generated when this step runs</span>
          </div>
        )}
        {result && "error" in result && !isNotFound && !loading && (
          <p className="text-sm text-error">{result.error}</p>
        )}
        {Renderer && !loading && (
          <div className="flex-1 flex flex-col min-h-0 min-w-0 h-full">
            <Renderer
              data={result}
              zoom={showZoom ? zoom : undefined}
              filePath={filePath}
              cwd={cwd ?? undefined}
              sessionId={sessionId ?? undefined}
              onReload={() => setRefreshKey((k) => k + 1)}
              onHtmlVisualSaveChromeChange={onPreviewSaveChromeChange}
              onTextSaveChromeChange={onPreviewSaveChromeChange}
            />
          </div>
        )}
      </div>
    </div>
  );
}
