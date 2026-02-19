import { useEffect, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { Spinner } from "./Spinner";

interface StartSessionModalProps {
  cwd: string;
  prompt: string;
  pendingStart: boolean;
  onCwdChange: (value: string) => void;
  onPromptChange: (value: string) => void;
  onStart: () => void;
  onClose: () => void;
}

export function StartSessionModal({
  cwd,
  prompt,
  pendingStart,
  onCwdChange,
  onPromptChange,
  onStart,
  onClose
}: StartSessionModalProps) {
  const [recentCwds, setRecentCwds] = useState<string[]>([]);

  useEffect(() => {
    window.electron.getRecentCwds().then(setRecentCwds).catch(console.error);
  }, []);

  const handleSelectDirectory = async () => {
    const result = await window.electron.selectDirectory();
    if (result) onCwdChange(result);
  };

  return (
    <Dialog.Root open onOpenChange={(open) => { if (!open) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-ink-900/20 backdrop-blur-sm" />
        <Dialog.Content className="fixed inset-0 z-50 flex items-center justify-center px-4 py-8">
          <div className="w-full max-w-lg rounded-2xl border border-ink-900/5 bg-surface p-6 shadow-elevated">
            <div className="flex items-center justify-between">
              <Dialog.Title className="text-base font-semibold text-ink-800">Start Session</Dialog.Title>
              <Dialog.Close asChild>
                <button className="rounded-full p-1.5 text-muted hover:bg-surface-tertiary hover:text-ink-700 transition-colors" aria-label="Close">
                  <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              </Dialog.Close>
            </div>
            <p className="mt-2 text-sm text-muted">Create a new session to start interacting with agent.</p>
            <div className="mt-5 grid gap-4">
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted">Working Directory</span>
                <div className="flex gap-2">
                  <input
                    className="flex-1 rounded-xl border border-ink-900/10 bg-surface-secondary px-4 py-2.5 text-sm text-ink-800 placeholder:text-muted-light focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/20 transition-colors"
                    placeholder="/path/to/project"
                    value={cwd}
                    onChange={(e) => onCwdChange(e.target.value)}
                    required
                  />
                  <button
                    type="button"
                    onClick={handleSelectDirectory}
                    className="rounded-xl border border-ink-900/10 bg-surface px-3 py-2 text-sm text-ink-700 hover:bg-surface-tertiary transition-colors"
                  >
                    Browse...
                  </button>
                </div>
                {recentCwds.length > 0 && (
                  <div className="mt-2 grid gap-2 w-full">
                    <div className="text-[11px] font-medium uppercase tracking-wide text-muted-light">Recent</div>
                    <div className="flex flex-wrap gap-2 w-full min-w-0">
                      {recentCwds.map((path) => (
                        <button
                          key={path}
                          type="button"
                          className={`truncate rounded-full border px-3 py-1.5 text-xs transition-colors whitespace-nowrap ${cwd === path ? "border-accent/60 bg-accent/10 text-ink-800" : "border-ink-900/10 bg-white text-muted hover:border-ink-900/20 hover:text-ink-700"}`}
                          onClick={() => onCwdChange(path)}
                          title={path}
                        >
                          {path}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </label>
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted">Prompt</span>
                <textarea
                  rows={4}
                  className="rounded-xl border border-ink-900/10 bg-surface-secondary p-3 text-sm text-ink-800 placeholder:text-muted-light focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/20 transition-colors resize-none"
                  placeholder="Describe the task you want agent to handle..."
                  value={prompt}
                  onChange={(e) => onPromptChange(e.target.value)}
                />
              </label>
              <button
                className="flex flex-col items-center rounded-full bg-accent px-5 py-3 text-sm font-medium text-white shadow-soft hover:bg-accent-hover transition-colors disabled:cursor-not-allowed disabled:opacity-50"
                onClick={onStart}
                disabled={pendingStart || !cwd.trim() || !prompt.trim()}
              >
                {pendingStart ? (
                  <Spinner />
                ) : "Start Session"}
              </button>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
