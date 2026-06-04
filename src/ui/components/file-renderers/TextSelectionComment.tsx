import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { MessageSquarePlusIcon } from "lucide-react";
import { formatQuotedSelectionMessage } from "../../lib/format-quoted-message";

type ToolbarState = {
  top: number;
  left: number;
};

type Props = {
  children: ReactNode;
  filePath?: string;
  onSendComment?: (prompt: string) => void | Promise<void>;
};

export function TextSelectionComment({
  children,
  filePath,
  onSendComment,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const quotedTextRef = useRef("");
  const [toolbar, setToolbar] = useState<ToolbarState | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [comment, setComment] = useState("");
  const [sending, setSending] = useState(false);
  const commentRef = useRef<HTMLTextAreaElement>(null);

  const enabled = Boolean(onSendComment);

  const clear = useCallback(() => {
    setToolbar(null);
    setFormOpen(false);
    setComment("");
    quotedTextRef.current = "";
  }, []);

  const updateFromSelection = useCallback(() => {
    if (!enabled) return;
    const container = containerRef.current;
    const sel = window.getSelection();
    if (!container || !sel || sel.isCollapsed || sel.rangeCount === 0) {
      if (!formOpen) clear();
      return;
    }
    const range = sel.getRangeAt(0);
    if (!container.contains(range.commonAncestorContainer)) {
      if (!formOpen) clear();
      return;
    }
    const text = sel.toString();
    if (!text.trim()) {
      if (!formOpen) clear();
      return;
    }
    const rect = range.getBoundingClientRect();
    quotedTextRef.current = text;
    setToolbar({
      top: rect.bottom + 6,
      left: rect.left + rect.width / 2,
    });
  }, [clear, enabled, formOpen]);

  useEffect(() => {
    if (!formOpen) return;
    const id = requestAnimationFrame(() => commentRef.current?.focus());
    return () => cancelAnimationFrame(id);
  }, [formOpen]);

  useEffect(() => {
    if (!enabled) clear();
  }, [clear, enabled]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") clear();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clear]);

  const openForm = () => {
    if (!quotedTextRef.current.trim()) return;
    setFormOpen(true);
  };

  const handleSubmit = async () => {
    if (!onSendComment || sending) return;
    const quote = quotedTextRef.current;
    const body = comment.trim();
    if (!quote.trim() || !body) return;
    setSending(true);
    try {
      await onSendComment(formatQuotedSelectionMessage(quote, body, filePath));
      clear();
      window.getSelection()?.removeAllRanges();
    } finally {
      setSending(false);
    }
  };

  const floatingUi =
    enabled && (toolbar || formOpen)
      ? createPortal(
          <div className="pointer-events-none fixed inset-0 z-[80]">
            {toolbar && !formOpen && (
              <div
                className="pointer-events-auto absolute -translate-x-1/2"
                style={{ top: toolbar.top, left: toolbar.left }}
              >
                <button
                  type="button"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={openForm}
                  className="inline-flex items-center gap-1.5 rounded-full border border-ink-900/12 bg-white px-3 py-1.5 text-xs font-medium text-ink-800 shadow-[0_2px_10px_rgba(26,25,21,0.12)] hover:border-primary/30 hover:bg-primary/5 transition-colors"
                >
                  <MessageSquarePlusIcon className="size-3.5 shrink-0 text-primary" aria-hidden />
                  Comment
                </button>
              </div>
            )}
            {formOpen && toolbar && (
              <div
                className="pointer-events-auto absolute w-[min(20rem,calc(100vw-2rem))] -translate-x-1/2 rounded-xl border border-ink-900/12 bg-white p-3 shadow-[0_4px_20px_rgba(26,25,21,0.14)]"
                style={{ top: toolbar.top, left: toolbar.left }}
                onMouseDown={(e) => e.stopPropagation()}
              >
                <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Comment on selection
                </p>
                <blockquote className="mb-2 max-h-24 overflow-y-auto rounded-md border-l-2 border-primary/40 bg-ink-900/[0.03] px-2.5 py-1.5 text-xs text-ink-700 whitespace-pre-wrap break-words">
                  {quotedTextRef.current.trim()}
                </blockquote>
                <textarea
                  ref={commentRef}
                  rows={3}
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="What should the agent change or address?"
                  className="mb-2 w-full resize-none rounded-lg border border-ink-900/12 bg-white px-2.5 py-2 text-sm text-ink-900 placeholder:text-ink-500 focus:outline-none focus:ring-2 focus:ring-primary/20"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void handleSubmit();
                    }
                  }}
                />
                <div className="flex items-center justify-end gap-2">
                  <button
                    type="button"
                    onClick={clear}
                    className="rounded-md px-2.5 py-1 text-xs font-medium text-muted-foreground hover:bg-ink-900/5 hover:text-ink-800"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleSubmit()}
                    disabled={sending || !comment.trim()}
                    className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-white hover:bg-primary-hover disabled:opacity-50"
                  >
                    {sending ? "Sending…" : "Send"}
                  </button>
                </div>
              </div>
            )}
          </div>,
          document.body
        )
      : null;

  return (
    <div ref={containerRef} className="flex min-h-0 flex-1 flex-col" onMouseUp={updateFromSelection}>
      {children}
      {floatingUi}
    </div>
  );
}
