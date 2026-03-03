import type { FC } from "react";
import { FolderOpenIcon, XIcon } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { getPreviewFileForStep } from "./FilePreview";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

export const PreviewPanelHeader: FC = () => {
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const sessions = useAppStore((s) => s.sessions);
  const previewStepIndex = useAppStore((s) => s.previewStepIndex);
  const setPreviewStepIndex = useAppStore((s) => s.setPreviewStepIndex);
  const setPreviewPanelOpen = useAppStore((s) => s.setPreviewPanelOpen);

  const session = activeSessionId ? sessions[activeSessionId] : undefined;
  const steps = session?.steps ?? [];
  const outputFiles = session?.outputFiles;
  const cwd = session?.cwd;

  // Only show steps that have output files
  const stepsWithFiles = steps
    .map((label, idx) => ({ label, idx }))
    .filter(({ idx }) => (outputFiles?.[idx]?.length ?? 0) > 0);

  const currentFile = getPreviewFileForStep(outputFiles, previewStepIndex);

  return (
    <div className="shrink-0 flex items-center gap-2 px-3 py-2 border-b border-ink-900/10">
      {stepsWithFiles.length > 1 ? (
        <Select
          value={String(previewStepIndex)}
          onValueChange={(v) => setPreviewStepIndex(Number(v))}
        >
          <SelectTrigger size="sm" className="h-7 text-xs min-w-0 max-w-[1000px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent position="popper" side="bottom">
            {stepsWithFiles.map(({ label, idx }) => (
              <SelectItem key={idx} value={String(idx)}>
                <span className="truncate">Step {idx + 1}: {label}</span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <span className="text-xs font-medium text-ink-700 truncate">
          {stepsWithFiles.length === 1
            ? `Step ${stepsWithFiles[0].idx + 1}: ${stepsWithFiles[0].label}`
            : "Preview"}
        </span>
      )}

      {currentFile && (
        <span className="text-xs text-muted-foreground truncate ml-auto mr-2" title={currentFile}>
          {currentFile.split("/").pop()}
        </span>
      )}

      {currentFile && (
        <button
          onClick={() => window.electron.showItemInFolder(currentFile, cwd)}
          className="shrink-0 p-1 rounded hover:bg-ink-900/5 text-muted-foreground hover:text-ink-700 transition-colors"
          aria-label="Open in Finder"
          title="Open in Finder"
        >
          <FolderOpenIcon className="size-4" />
        </button>
      )}

      <button
        onClick={() => setPreviewPanelOpen(false)}
        className="shrink-0 p-1 rounded hover:bg-ink-900/5 text-muted-foreground hover:text-ink-700 transition-colors"
        aria-label="Close preview panel"
      >
        <XIcon className="size-4" />
      </button>
    </div>
  );
};
