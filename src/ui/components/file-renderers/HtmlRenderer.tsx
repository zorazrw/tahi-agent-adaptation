import { useState, useEffect, useRef, useCallback } from "react";
import hljs from "highlight.js";
import { SaveIcon, Loader2Icon } from "lucide-react";
import { ViewToggle, useViewToggle } from "./ViewToggle";
import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";
import {
  serializeIframeDocument,
  attachHtmlTextEdit,
  attachHtmlLayoutDrag,
} from "./html-preview-edit";

type Props = { data: { kind: "html"; content: string } } & EditableRendererProps;

type HtmlVisualTool = "none" | "text" | "layout";

export function HtmlRenderer({ data, filePath, cwd, sessionId, onReload }: Props) {
  const [mode, setMode] = useViewToggle("preview");
  const codeRef = useRef<HTMLElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const previewCleanupRef = useRef<(() => void) | null>(null);
  const editContentRef = useRef(data.content);

  const canEdit = Boolean(filePath && onReload);
  const [editContent, setEditContent] = useState(data.content);
  const [visualTool, setVisualTool] = useState<HtmlVisualTool>("none");
  const [visualDirty, setVisualDirty] = useState(false);
  const [visualSaving, setVisualSaving] = useState(false);
  const [visualSaveError, setVisualSaveError] = useState<string | null>(null);

  const visualDirtyRef = useRef(visualDirty);
  const visualSavingRef = useRef(visualSaving);
  visualDirtyRef.current = visualDirty;
  visualSavingRef.current = visualSaving;

  const displayContent = canEdit ? editContent : data.content;

  useEffect(() => {
    setEditContent(data.content);
  }, [data.content]);

  useEffect(() => {
    editContentRef.current = editContent;
  }, [editContent]);

  useEffect(() => {
    if (mode === "source" && codeRef.current && !canEdit) {
      codeRef.current.removeAttribute("data-highlighted");
      hljs.highlightElement(codeRef.current);
    }
  }, [mode, data.content, canEdit]);

  const runPreviewCleanup = useCallback(() => {
    previewCleanupRef.current?.();
    previewCleanupRef.current = null;
  }, []);

  const markVisualDirtyFromDoc = useCallback(() => {
    const doc = iframeRef.current?.contentDocument;
    if (!doc?.documentElement) return;
    setVisualDirty(serializeIframeDocument(doc) !== editContentRef.current);
  }, []);

  const handleSaveVisual = useCallback(async () => {
    const doc = iframeRef.current?.contentDocument;
    if (!doc || !filePath) return;
    const html = serializeIframeDocument(doc);
    setVisualSaveError(null);
    setVisualSaving(true);
    try {
      const result = await window.electron.writeFile(filePath, cwd ?? undefined, html, sessionId ?? undefined);
      if (result.success) {
        setEditContent(html);
        editContentRef.current = html;
        setVisualDirty(false);
        onReload?.();
      } else {
        setVisualSaveError(result.error ?? "Failed to save");
      }
    } catch (e) {
      setVisualSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setVisualSaving(false);
    }
  }, [filePath, cwd, sessionId, onReload]);

  const handleSaveVisualRef = useRef(handleSaveVisual);
  handleSaveVisualRef.current = handleSaveVisual;

  const onVisualSaveHotkey = useCallback((e: KeyboardEvent) => {
    if (e.key?.toLowerCase() !== "s") return;
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    if (visualSavingRef.current || !visualDirtyRef.current) return;
    e.preventDefault();
    void handleSaveVisualRef.current();
  }, []);

  const attachPreviewTool = useCallback(
    (tool: "text" | "layout") => {
      const doc = iframeRef.current?.contentDocument;
      if (!doc?.documentElement) return;

      runPreviewCleanup();

      const onChange = () => markVisualDirtyFromDoc();
      previewCleanupRef.current =
        tool === "text"
          ? attachHtmlTextEdit(doc, onChange, onVisualSaveHotkey)
          : attachHtmlLayoutDrag(doc, onChange, onVisualSaveHotkey);
      markVisualDirtyFromDoc();
    },
    [markVisualDirtyFromDoc, runPreviewCleanup, onVisualSaveHotkey]
  );

  useEffect(() => {
    return () => runPreviewCleanup();
  }, [runPreviewCleanup]);

  useEffect(() => {
    if (mode !== "preview" || visualTool === "none" || !canEdit) {
      runPreviewCleanup();
      setVisualDirty(false);
      return;
    }
    const doc = iframeRef.current?.contentDocument;
    if (doc?.readyState === "complete" && doc.documentElement) {
      attachPreviewTool(visualTool);
    }
  }, [mode, visualTool, canEdit, attachPreviewTool, runPreviewCleanup]);

  const handleIframeLoad = useCallback(() => {
    if (mode !== "preview" || visualTool === "none" || !canEdit) {
      runPreviewCleanup();
      setVisualDirty(false);
      return;
    }
    attachPreviewTool(visualTool);
  }, [mode, visualTool, canEdit, attachPreviewTool, runPreviewCleanup]);

  useEffect(() => {
    if (visualTool === "none") setVisualSaveError(null);
  }, [visualTool]);

  /** Re-sync dirty after interactions outside the iframe (toolbar, splitter); layout + text both need this for Save. */
  useEffect(() => {
    if ((visualTool !== "layout" && visualTool !== "text") || mode !== "preview" || !canEdit) return;
    const sync = () => markVisualDirtyFromDoc();
    window.addEventListener("pointerup", sync, true);
    window.addEventListener("pointercancel", sync, true);
    return () => {
      window.removeEventListener("pointerup", sync, true);
      window.removeEventListener("pointercancel", sync, true);
    };
  }, [visualTool, mode, canEdit, markVisualDirtyFromDoc]);

  /** After enabling Text mode, focus moves to the toolbar button; move focus into the iframe so typing/caret work. */
  useEffect(() => {
    if (visualTool !== "text" || mode !== "preview" || !canEdit) return;
    const id = window.setTimeout(() => {
      iframeRef.current?.contentWindow?.focus();
    }, 0);
    return () => clearTimeout(id);
  }, [visualTool, mode, canEdit]);

  /** Ctrl/Cmd+S from the shell (toolbar focused); iframe focused is handled inside attachHtmlTextEdit / attachHtmlLayoutDrag. */
  useEffect(() => {
    if (mode !== "preview" || visualTool === "none" || !canEdit) return;
    const onKey = (e: KeyboardEvent) => onVisualSaveHotkey(e);
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [mode, visualTool, canEdit, onVisualSaveHotkey]);

  /** `allow-same-origin` must apply from the first paint when the file is saveable; otherwise the iframe document is opaque and parent JS never gets a usable `contentDocument`, so Edit text / Move never attach. */
  const previewSandbox = canEdit ? "allow-scripts allow-same-origin" : "allow-scripts";

  const toolBtn = (id: HtmlVisualTool, label: string) => (
    <button
      type="button"
      onClick={() => setVisualTool((t) => (t === id ? "none" : id))}
      className={`rounded-md px-2 py-1 text-xs font-medium transition-colors ${
        visualTool === id
          ? "bg-ink-800 text-white"
          : "bg-ink-900/5 text-ink-600 hover:bg-ink-900/10"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex flex-wrap items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
        <ViewToggle mode={mode} onChange={setMode} />
        <span className="text-xs text-muted-foreground">HTML</span>
        {mode === "preview" && canEdit && (
          <>
            <span className="text-ink-900/15 text-xs" aria-hidden>
              |
            </span>
            <span className="text-xs text-muted-foreground shrink-0">In preview:</span>
            {toolBtn("text", "Text")}
            {toolBtn("layout", "Move")}
          </>
        )}
        {mode === "preview" && visualTool !== "none" && canEdit && (
          <>
            <span className="text-ink-900/15 text-xs" aria-hidden>
              |
            </span>
            <button
              type="button"
              onClick={handleSaveVisual}
              disabled={visualSaving || !visualDirty}
              title={
                visualDirty
                  ? "Save preview changes to file (Ctrl or ⌘+S)"
                  : "No changes to save yet (edit or drag first)"
              }
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium bg-primary text-white hover:bg-primary/90 disabled:opacity-50 disabled:pointer-events-auto"
            >
              {visualSaving ? <Loader2Icon className="size-3.5 animate-spin" /> : <SaveIcon className="size-3.5" />}
              {visualSaving ? "Saving…" : "Save"}
            </button>
            {visualSaveError && <span className="text-xs text-error">{visualSaveError}</span>}
          </>
        )}
      </div>
      {mode === "preview" ? (
        <iframe
          ref={iframeRef}
          srcDoc={displayContent}
          sandbox={previewSandbox}
          onLoad={handleIframeLoad}
          tabIndex={canEdit ? 0 : undefined}
          className="flex-1 w-full min-h-0 border-0 rounded bg-white"
          style={{ minHeight: 200 }}
          title="HTML Preview"
        />
      ) : canEdit ? (
        <EditableTextPanel
          content={editContent}
          contentKey={data.content}
          monospace
          onSave={async (content) => {
            setEditContent(content);
            return window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined);
          }}
          onSaved={onReload}
        />
      ) : (
        <pre className="text-sm overflow-x-auto">
          <code ref={codeRef} className="language-html">
            {data.content}
          </code>
        </pre>
      )}
    </div>
  );
}
