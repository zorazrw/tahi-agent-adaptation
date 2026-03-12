import { FileIcon, ImageIcon, EyeIcon } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import type { NodeCompletedMessage, WorkflowNode } from "../types";
import { pathHasPreviewExt } from "./FilePreview";

const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"];

function isImageFile(path: string): boolean {
  const lower = path.toLowerCase();
  return IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function findNode(tree: WorkflowNode[], id: string): WorkflowNode | undefined {
  for (const node of tree) {
    if (node.id === id) return node;
    const found = findNode(node.children, id);
    if (found) return found;
  }
  return undefined;
}

export function NodeOutputSnippet({ message }: { message: NodeCompletedMessage }) {
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const sessions = useAppStore((s) => s.sessions);
  const setSelectedNodeId = useAppStore((s) => s.setSelectedNodeId);
  const setPreviewPanelOpen = useAppStore((s) => s.setPreviewPanelOpen);

  const session = activeSessionId ? sessions[activeSessionId] : undefined;
  const tree = session?.workflowTree ?? [];
  const node = findNode(tree, message.nodeId);
  const outputFiles = node?.outputFiles ?? [];
  const previewableFiles = outputFiles.filter(pathHasPreviewExt);

  if (previewableFiles.length === 0) {
    return (
      <div className="flex items-center gap-2 py-2 mt-3">
        <div className="h-px flex-1 bg-ink-900/10" />
        <span className="text-xs text-muted-foreground flex items-center gap-1.5">
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M5 12l4 4L19 6" />
          </svg>
          {message.nodeLabel} completed
        </span>
        <div className="h-px flex-1 bg-ink-900/10" />
      </div>
    );
  }

  const handleClick = () => {
    setSelectedNodeId(message.nodeId);
    setPreviewPanelOpen(true);
  };

  return (
    <button
      onClick={handleClick}
      className="w-full mt-3 flex items-start gap-3 rounded-xl border border-ink-900/10 bg-surface-secondary/80 px-4 py-3 text-left hover:border-primary/30 hover:bg-primary/5 transition-colors group"
    >
      <div className="shrink-0 mt-0.5 flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
        {previewableFiles.some(isImageFile) ? (
          <ImageIcon className="size-4" />
        ) : (
          <FileIcon className="size-4" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-ink-800">
            {message.nodeLabel}
          </span>
        </div>
        <div className="mt-1 flex flex-wrap gap-1.5">
          {previewableFiles.slice(0, 3).map((file) => (
            <span
              key={file}
              className="inline-flex items-center gap-1 rounded-md bg-ink-900/5 px-2 py-0.5 text-xs text-muted-foreground"
            >
              {isImageFile(file) ? <ImageIcon className="size-3" /> : <FileIcon className="size-3" />}
              {file.split("/").pop()}
            </span>
          ))}
          {previewableFiles.length > 3 && (
            <span className="text-xs text-muted-foreground">+{previewableFiles.length - 3} more</span>
          )}
        </div>
      </div>
      <div className="shrink-0 mt-0.5 p-1 text-muted-foreground group-hover:text-primary transition-colors">
        <EyeIcon className="size-4" />
      </div>
    </button>
  );
}
