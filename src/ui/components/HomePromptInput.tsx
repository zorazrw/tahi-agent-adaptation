import { useCallback, useEffect, useRef, useState } from "react";
import type { ClientEvent } from "../types";
import { useAppStore } from "../store/useAppStore";
import { readStoredAutoInduction } from "../lib/auto-induction";

const DEFAULT_ALLOWED_TOOLS = "Read,Edit,Bash";
const MAX_ROWS = 8;
const LINE_HEIGHT = 21;
const MAX_HEIGHT = MAX_ROWS * LINE_HEIGHT;

function fileTypeLabel(name: string): string {
  const dot = name.lastIndexOf(".");
  if (dot === -1) return "FILE";
  return name.slice(dot + 1).toUpperCase();
}

interface HomePromptInputProps {
  sendEvent: (event: ClientEvent) => void;
}

export function HomePromptInput({ sendEvent }: HomePromptInputProps) {
  const prompt = useAppStore((s) => s.prompt);
  const setPrompt = useAppStore((s) => s.setPrompt);
  const cwd = useAppStore((s) => s.cwd);
  const setCwd = useAppStore((s) => s.setCwd);
  const pendingStart = useAppStore((s) => s.pendingStart);
  const setPendingStart = useAppStore((s) => s.setPendingStart);
  const setGlobalError = useAppStore((s) => s.setGlobalError);
  const attachedFiles = useAppStore((s) => s.attachedFiles);
  const setAttachedFiles = useAppStore((s) => s.setAttachedFiles);
  const tempCwd = useAppStore((s) => s.tempCwd);
  const setTempCwd = useAppStore((s) => s.setTempCwd);
  const expertiseTaskCategory = useAppStore((s) => s.expertiseTaskCategory);
  const setExpertiseTaskCategory = useAppStore((s) => s.setExpertiseTaskCategory);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isCopying, setIsCopying] = useState(false);
  const dragCounterRef = useRef(0);

  // Resize textarea when prompt is set externally (e.g. expertise example picker)
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const scrollHeight = el.scrollHeight;
    if (scrollHeight > MAX_HEIGHT) {
      el.style.height = `${MAX_HEIGHT}px`;
      el.style.overflowY = "auto";
    } else {
      el.style.height = `${Math.max(scrollHeight, LINE_HEIGHT * 2)}px`;
      el.style.overflowY = "hidden";
    }
  }, [prompt]);

  // Prevent Electron from navigating to dropped files (must be global)
  useEffect(() => {
    const preventNav = (e: DragEvent) => { e.preventDefault(); e.stopPropagation(); };
    document.addEventListener("dragover", preventNav);
    document.addEventListener("drop", preventNav);
    return () => {
      document.removeEventListener("dragover", preventNav);
      document.removeEventListener("drop", preventNav);
    };
  }, []);

  const ensureCwd = useCallback(async (): Promise<string> => {
    if (cwd.trim()) return cwd.trim();
    if (tempCwd) return tempCwd;
    const dir = await window.electron.createTempSessionDir();
    setTempCwd(dir);
    return dir;
  }, [cwd, tempCwd, setTempCwd]);

  const handleAttachFiles = useCallback(async (filePaths: string[]) => {
    if (filePaths.length === 0) return;
    setIsCopying(true);
    try {
      const targetDir = await ensureCwd();
      const names = await window.electron.copyFilesToDir(filePaths, targetDir);
      // Read from store directly to avoid stale closure
      const current = useAppStore.getState().attachedFiles;
      setAttachedFiles([...current, ...names]);
    } catch (err) {
      console.error("Failed to attach files:", err);
      setGlobalError(`Failed to attach files: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsCopying(false);
    }
  }, [ensureCwd, setAttachedFiles, setGlobalError]);

  const handleSelectDirectory = useCallback(async () => {
    const result = await window.electron.selectDirectory();
    if (result) {
      setCwd(result);
      setTempCwd(null);
    }
  }, [setCwd, setTempCwd]);

  const handleClearCwd = useCallback(() => {
    setCwd("");
    setTempCwd(null);
  }, [setCwd, setTempCwd]);

  const handleRemoveFile = useCallback((index: number) => {
    const current = useAppStore.getState().attachedFiles;
    setAttachedFiles(current.filter((_, i) => i !== index));
  }, [setAttachedFiles]);

  const handleSend = useCallback(async () => {
    if (!prompt.trim() || pendingStart) return;
    let title = "";
    try {
      setPendingStart(true);
      title = await window.electron.generateSessionTitle(prompt);
    } catch {
      setPendingStart(false);
      setGlobalError("Failed to get session title.");
      return;
    }
    const sessionCwd = cwd.trim() || tempCwd || undefined;
    const fileRefs = attachedFiles.length > 0
      ? "The following files have been copied into the working directory (cwd). Use relative paths to access them:\n" +
        attachedFiles.map((f) => `- ${f}`).join("\n") + "\n\n"
      : "";
    sendEvent({
      type: "session.start",
      payload: {
        title,
        prompt: fileRefs + prompt,
        cwd: sessionCwd,
        allowedTools: DEFAULT_ALLOWED_TOOLS,
        autoContextInduction: readStoredAutoInduction(),
        ...(expertiseTaskCategory ? { expertiseTask: expertiseTaskCategory } : {}),
      },
    });
    setPrompt("");
    setAttachedFiles([]);
    setTempCwd(null);
    setExpertiseTaskCategory(null);
  }, [prompt, pendingStart, cwd, tempCwd, attachedFiles, expertiseTaskCategory, sendEvent, setPendingStart, setGlobalError, setPrompt, setAttachedFiles, setTempCwd, setExpertiseTaskCategory]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key !== "Enter" || e.shiftKey) return;
    e.preventDefault();
    handleSend();
  };

  const handleInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const target = e.currentTarget;
    target.style.height = "auto";
    const scrollHeight = target.scrollHeight;
    if (scrollHeight > MAX_HEIGHT) {
      target.style.height = `${MAX_HEIGHT}px`;
      target.style.overflowY = "auto";
    } else {
      target.style.height = `${scrollHeight}px`;
      target.style.overflowY = "hidden";
    }
  };

  // Drag-drop handlers (counter avoids flicker from child elements)
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current++;
    if (dragCounterRef.current === 1) setIsDragging(true);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current--;
    if (dragCounterRef.current === 0) setIsDragging(false);
  };

  // Listen for file paths resolved by the preload drop interceptor
  useEffect(() => {
    const handler = (e: Event) => {
      const paths = (e as CustomEvent<string[]>).detail;
      if (paths?.length) handleAttachFiles(paths);
    };
    window.addEventListener("electron-drop-paths", handler);
    return () => window.removeEventListener("electron-drop-paths", handler);
  }, [handleAttachFiles]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current = 0;
    setIsDragging(false);
    // File paths are resolved by the preload drop interceptor and delivered
    // via the "electron-drop-paths" custom event (see useEffect above).
    // No need to call getPathForFile here.
  };

  return (
    <div className="w-full max-w-2xl mx-auto relative">
      <div
        className={`rounded-2xl border bg-surface shadow-card transition-all duration-150 ${isDragging ? "border-primary/50 shadow-[0_0_0_3px_rgba(217,119,87,0.15)]" : "border-ink-900/10"}`}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* File cards — above textarea */}
        {attachedFiles.length > 0 && (
          <div className="px-4 pt-4 pb-1 flex flex-wrap gap-3">
            {attachedFiles.map((name, i) => (
              <div
                key={i}
                className="relative group flex flex-col justify-between w-[180px] h-[120px] rounded-xl border border-ink-900/10 bg-surface-secondary p-3"
              >
                <button
                  className="absolute top-1.5 right-1.5 rounded-full p-1 bg-ink-900/5 text-ink-400 opacity-0 group-hover:opacity-100 hover:bg-ink-900/15 hover:text-ink-700 transition-all"
                  onClick={() => handleRemoveFile(i)}
                  aria-label={`Remove ${name}`}
                >
                  <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
                <span className="text-sm text-ink-700 font-medium leading-snug line-clamp-2 break-all pr-5">{name}</span>
                <span className="inline-flex self-start rounded-md bg-ink-900/8 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-ink-500">
                  {fileTypeLabel(name)}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Copying indicator */}
        {isCopying && (
          <div className="px-4 pt-3 flex items-center gap-2 text-xs text-muted-foreground">
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 animate-spin" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
              <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" />
            </svg>
            <span>Copying files...</span>
          </div>
        )}

        {/* Textarea */}
        <div className="px-4 pt-4 pb-1">
          <textarea
            ref={textareaRef}
            rows={2}
            className="w-full resize-none bg-transparent text-sm text-ink-800 placeholder:text-muted-foreground focus:outline-none"
            placeholder="Ask anything"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            disabled={pendingStart}
          />
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-2 px-3 py-1 border-t border-ink-900/5">
          {/* Folder picker */}
          <button
            type="button"
            className="inline-flex h-8 min-h-8 items-center gap-1 rounded-lg px-2 text-sm leading-none text-muted-foreground hover:text-ink-700 hover:bg-ink-900/5 transition-colors"
            onClick={handleSelectDirectory}
            title="Select working folder"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
            </svg>
            {cwd ? (
              <span className="flex items-center gap-1 max-w-[180px] min-w-0">
                <span className="truncate">{cwd.split("/").filter(Boolean).slice(-2).join("/")}</span>
                <button
                  type="button"
                  className="shrink-0 rounded-full p-0 hover:bg-ink-900/10"
                  onClick={(e) => { e.stopPropagation(); handleClearCwd(); }}
                  aria-label="Clear folder"
                >
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              </span>
            ) : (
              <>
                <span>Work in a folder</span>
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="m6 9 6 6 6-6" />
                </svg>
              </>
            )}
          </button>

          <div className="flex-1" />

          {/* Send button */}
          <button
            type="button"
            className="inline-flex h-8 min-w-8 items-center justify-center rounded-full bg-primary px-2.5 text-white shadow-soft hover:bg-primary-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleSend}
            disabled={!prompt.trim() || pendingStart}
            aria-label={pendingStart ? "Starting…" : "Start task"}
          >
            {pendingStart ? (
              <svg viewBox="0 0 24 24" className="h-[1.125rem] w-[1.125rem] animate-spin" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" className="h-[1.125rem] w-[1.125rem]" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl border-2 border-dashed border-primary/40 bg-primary/5 pointer-events-none">
          <p className="text-sm font-medium text-primary">Drop files here</p>
        </div>
      )}
    </div>
  );
}
