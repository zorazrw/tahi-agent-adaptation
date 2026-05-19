import type { ReactNode } from "react";

type TextDocumentFrameProps = {
  toolbar?: ReactNode;
  label?: string;
  children: ReactNode;
  className?: string;
  /** When false, children fill the frame without the inner reading column padding. */
  padded?: boolean;
};

/** Paper-like reading surface for .txt / .md preview and source editing. */
export function TextDocumentFrame({
  toolbar,
  label,
  children,
  className = "",
  padded = true,
}: TextDocumentFrameProps) {
  return (
    <div className={`flex h-full min-h-0 flex-1 flex-col ${className}`}>
      {(toolbar || label) && (
        <div className="flex items-center gap-2 pb-2.5 mb-2.5 shrink-0 border-b border-ink-900/8">
          {toolbar}
          {label ? (
            <span className="ml-auto text-[11px] font-medium uppercase tracking-[0.06em] text-muted-foreground">
              {label}
            </span>
          ) : null}
        </div>
      )}
      <div className="text-doc-surface flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-ink-900/10 bg-white shadow-[0_1px_2px_rgba(26,25,21,0.05),0_6px_20px_rgba(26,25,21,0.06)]">
        {padded ? (
          <div className="text-doc-body text-doc-content flex min-h-0 flex-1 flex-col p-7 sm:p-10">
            {children}
          </div>
        ) : (
          children
        )}
      </div>
    </div>
  );
}

export function PlainTextReadingView({ content }: { content: string }) {
  return (
    <article className="text-doc-plain whitespace-pre-wrap break-words selection:bg-primary/15">
      {content}
    </article>
  );
}
