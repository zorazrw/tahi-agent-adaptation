import { useState, useEffect, useRef, useCallback } from "react";
import hljs from "highlight.js";
import {
  SaveIcon,
  Loader2Icon,
  ShapesIcon,
  AlignCenterHorizontal,
  TextAlignStart,
  TextAlignCenter,
  TextAlignEnd,
  AlignVerticalJustifyStart,
  AlignVerticalJustifyCenter,
  AlignVerticalJustifyEnd,
} from "lucide-react";
import { ViewToggle, useViewToggle } from "./ViewToggle";
import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";
import {
  serializeIframeDocument,
  attachHtmlTextEdit,
  attachHtmlLayoutDrag,
  insertPreviewShape,
  applyPreviewTextAlignment,
  isDeleteOrBackspaceKey,
  tryDeleteSelectedPreviewOrLayoutBlock,
  type PreviewShapeKind,
  type PreviewTextAlignH,
  type PreviewTextAlignV,
} from "./html-preview-edit";

type Props = { data: { kind: "html"; content: string } } & EditableRendererProps;

type HtmlVisualTool = "none" | "text" | "layout";

const previewToolbarMenuItemBtn =
  "flex size-9 items-center justify-center rounded text-ink-800 hover:bg-ink-900/8";
const previewToolbarPopoverTriggerBtn =
  "inline-flex items-center justify-center rounded-md p-1.5 text-xs font-medium bg-ink-900/5 text-ink-600 hover:bg-ink-900/10";

/** Tiny glyphs for the Move-mode picker (ink palette, matches inserted shapes). */
function PreviewShapeMenuGlyph({ kind }: { kind: PreviewShapeKind }) {
  const common = "shrink-0 block text-ink-600";
  if (kind === "rectangle") {
    return (
      <svg className={common} width="28" height="28" viewBox="0 0 28 28" aria-hidden>
        <rect
          x="5"
          y="8"
          width="18"
          height="12"
          rx="2"
          fill="currentColor"
          fillOpacity={0.07}
          stroke="currentColor"
          strokeWidth="1.5"
        />
      </svg>
    );
  }
  if (kind === "circle") {
    return (
      <svg className={common} width="28" height="28" viewBox="0 0 28 28" aria-hidden>
        <circle
          cx="14"
          cy="14"
          r="6.75"
          fill="currentColor"
          fillOpacity={0.07}
          stroke="currentColor"
          strokeWidth="1.5"
        />
      </svg>
    );
  }
  return (
    <svg className={common} width="28" height="28" viewBox="0 0 28 28" aria-hidden>
      <line
        x1="8"
        y1="19"
        x2="20"
        y2="9"
        stroke="currentColor"
        strokeWidth="2.25"
        strokeLinecap="round"
      />
    </svg>
  );
}

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
  const [shapeMenuOpen, setShapeMenuOpen] = useState(false);
  const shapeMenuRef = useRef<HTMLDivElement>(null);
  const [alignMenuOpen, setAlignMenuOpen] = useState(false);
  const alignMenuRef = useRef<HTMLDivElement>(null);

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
    const dirty = serializeIframeDocument(doc) !== editContentRef.current;
    visualDirtyRef.current = dirty;
    setVisualDirty(dirty);
  }, []);

  /** Sync ref immediately on keystroke so ⌘/Ctrl+S works before React re-renders (hotkey reads the ref). */
  const markVisualLikelyDirty = useCallback(() => {
    visualDirtyRef.current = true;
    setVisualDirty(true);
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
        visualDirtyRef.current = false;
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

  const handleInsertShape = useCallback(
    (kind: PreviewShapeKind) => {
      const doc = iframeRef.current?.contentDocument;
      if (!doc?.body) return;
      insertPreviewShape(doc, kind);
      markVisualDirtyFromDoc();
      setShapeMenuOpen(false);
    },
    [markVisualDirtyFromDoc]
  );

  const applyTextAlignH = useCallback(
    (value: PreviewTextAlignH) => {
      const doc = iframeRef.current?.contentDocument;
      if (!applyPreviewTextAlignment(doc, "h", value)) return;
      markVisualDirtyFromDoc();
      setAlignMenuOpen(false);
      iframeRef.current?.contentWindow?.focus();
    },
    [markVisualDirtyFromDoc]
  );

  const applyTextAlignV = useCallback(
    (value: PreviewTextAlignV) => {
      const doc = iframeRef.current?.contentDocument;
      if (!applyPreviewTextAlignment(doc, "v", value)) return;
      markVisualDirtyFromDoc();
      setAlignMenuOpen(false);
      iframeRef.current?.contentWindow?.focus();
    },
    [markVisualDirtyFromDoc]
  );

  useEffect(() => {
    if (!shapeMenuOpen && !alignMenuOpen) return;
    const onDocDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (shapeMenuOpen && shapeMenuRef.current && !shapeMenuRef.current.contains(t)) {
        setShapeMenuOpen(false);
      }
      if (alignMenuOpen && alignMenuRef.current && !alignMenuRef.current.contains(t)) {
        setAlignMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocDown, true);
    return () => document.removeEventListener("mousedown", onDocDown, true);
  }, [shapeMenuOpen, alignMenuOpen]);

  useEffect(() => {
    if (visualTool !== "layout") setShapeMenuOpen(false);
    if (visualTool !== "text") setAlignMenuOpen(false);
  }, [visualTool]);

  const attachPreviewTool = useCallback(
    (tool: "text" | "layout") => {
      const doc = iframeRef.current?.contentDocument;
      if (!doc?.documentElement) return;

      runPreviewCleanup();

      const onChange = () => markVisualDirtyFromDoc();
      previewCleanupRef.current =
        tool === "text"
          ? attachHtmlTextEdit(doc, onChange, onVisualSaveHotkey, markVisualLikelyDirty)
          : attachHtmlLayoutDrag(doc, onChange, onVisualSaveHotkey);
      markVisualDirtyFromDoc();
    },
    [markVisualDirtyFromDoc, markVisualLikelyDirty, runPreviewCleanup, onVisualSaveHotkey]
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

  /**
   * Delete / Backspace while a shape is selected: host window often has focus (toolbar), so the iframe
   * never sees the key. Handle here and mirror dirty state.
   */
  useEffect(() => {
    if (mode !== "preview" || visualTool !== "layout" || !canEdit) return;
    const onHostKey = (e: KeyboardEvent) => {
      if (!isDeleteOrBackspaceKey(e)) return;
      const t = e.target;
      if (t instanceof Element && t.closest?.("input,textarea,[contenteditable=true],select")) return;
      const idoc = iframeRef.current?.contentDocument;
      if (!tryDeleteSelectedPreviewOrLayoutBlock(idoc)) return;
      e.preventDefault();
      e.stopPropagation();
      markVisualLikelyDirty();
      markVisualDirtyFromDoc();
    };
    window.addEventListener("keydown", onHostKey, true);
    return () => window.removeEventListener("keydown", onHostKey, true);
  }, [mode, visualTool, canEdit, markVisualDirtyFromDoc, markVisualLikelyDirty]);

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
      className={`rounded-md px-2 py-1 text-xs font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ink-900/20 focus-visible:ring-offset-2 focus-visible:ring-offset-white ${
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
        <div className="flex flex-wrap items-center gap-2 min-w-0 flex-1">
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
        </div>
        {mode === "preview" && visualTool !== "none" && canEdit && (
          <div className="flex items-center gap-2 shrink-0">
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
            {visualTool === "text" && (
              <div className="relative" ref={alignMenuRef}>
                <button
                  type="button"
                  onClick={() => setAlignMenuOpen((o) => !o)}
                  title="Text alignment"
                  aria-expanded={alignMenuOpen}
                  aria-haspopup="menu"
                  className={previewToolbarPopoverTriggerBtn}
                >
                  <AlignCenterHorizontal className="size-4" aria-hidden />
                  <span className="sr-only">Text alignment</span>
                </button>
                {alignMenuOpen && (
                  <div
                    role="menu"
                    className="absolute right-0 top-full z-50 mt-1 flex flex-col gap-1.5 rounded-md border border-ink-900/15 bg-white p-1.5 shadow-md"
                  >
                    <div className="flex items-center gap-0.5" role="group" aria-label="Horizontal align">
                      {(
                        [
                          ["left", TextAlignStart, "Align left"],
                          ["center", TextAlignCenter, "Align center"],
                          ["right", TextAlignEnd, "Align right"],
                        ] as const
                      ).map(([val, Icon, title]) => (
                        <button
                          key={val}
                          type="button"
                          role="menuitem"
                          title={title}
                          aria-label={title}
                          className={previewToolbarMenuItemBtn}
                          onClick={() => applyTextAlignH(val)}
                        >
                          <Icon className="size-4 shrink-0" aria-hidden />
                        </button>
                      ))}
                    </div>
                    <div className="h-px bg-ink-900/10" aria-hidden />
                    <div className="flex items-center gap-0.5" role="group" aria-label="Vertical align block">
                      {(
                        [
                          ["start", AlignVerticalJustifyStart, "Top"],
                          ["middle", AlignVerticalJustifyCenter, "Middle"],
                          ["end", AlignVerticalJustifyEnd, "Bottom"],
                        ] as const
                      ).map(([val, Icon, title]) => (
                        <button
                          key={val}
                          type="button"
                          role="menuitem"
                          title={title}
                          aria-label={`Vertical ${title.toLowerCase()}`}
                          className={previewToolbarMenuItemBtn}
                          onClick={() => applyTextAlignV(val)}
                        >
                          <Icon className="size-4 shrink-0" aria-hidden />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            {visualTool === "layout" && (
              <div className="relative" ref={shapeMenuRef}>
                <button
                  type="button"
                  onClick={() => setShapeMenuOpen((o) => !o)}
                  title="Insert shape"
                  aria-expanded={shapeMenuOpen}
                  aria-haspopup="menu"
                  className={previewToolbarPopoverTriggerBtn}
                >
                  <ShapesIcon className="size-4" aria-hidden />
                  <span className="sr-only">Insert shape</span>
                </button>
                {shapeMenuOpen && (
                  <div
                    role="menu"
                    className="absolute right-0 top-full z-50 mt-1 flex items-center gap-0.5 rounded-md border border-ink-900/15 bg-white p-1 shadow-md"
                  >
                    {(["rectangle", "circle", "line"] as const).map((kind) => (
                      <button
                        key={kind}
                        type="button"
                        role="menuitem"
                        title={
                          kind === "rectangle" ? "Rectangle" : kind === "circle" ? "Circle" : "Line"
                        }
                        aria-label={
                          kind === "rectangle"
                            ? "Insert rectangle"
                            : kind === "circle"
                              ? "Insert circle"
                              : "Insert line"
                        }
                        className={previewToolbarMenuItemBtn}
                        onClick={() => handleInsertShape(kind)}
                      >
                        <PreviewShapeMenuGlyph kind={kind} />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
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
