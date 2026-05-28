import { useState, useEffect, useRef, useCallback } from "react";
import hljs from "highlight.js";
import {
  ShapesIcon,
  Palette,
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
  applyPreviewTextColor,
  snapshotPreviewTextColorSelection,
  isDeleteOrBackspaceKey,
  tryDeleteSelectedPreviewOrLayoutBlock,
  type PreviewShapeKind,
  type PreviewTextAlignH,
  type PreviewTextAlignV,
} from "./html-preview-edit";

type Props = { data: { kind: "html"; content: string } } & EditableRendererProps;

type HtmlVisualTool = "none" | "text" | "layout";

type ContentSize = { w: number; h: number };

/** Cheap fingerprint so iframe remounts when disk content changes (srcDoc-only updates are unreliable in Electron). */
function hashPreviewContent(content: string): number {
  let h = 0;
  for (let i = 0; i < content.length; i++) {
    h = (Math.imul(31, h) + content.charCodeAt(i)) | 0;
  }
  return h;
}

function previewIframeKey(reloadKey: number | undefined, content: string): string {
  return `${reloadKey ?? 0}:${content.length}:${hashPreviewContent(content)}`;
}

function readIframeContentSize(doc: Document): ContentSize {
  const root = doc.documentElement;
  const body = doc.body;
  return {
    w: Math.max(root.scrollWidth, body?.scrollWidth ?? 0, root.offsetWidth, body?.offsetWidth ?? 0, 1),
    h: Math.max(root.scrollHeight, body?.scrollHeight ?? 0, root.offsetHeight, body?.offsetHeight ?? 0, 1),
  };
}

const previewToolbarMenuItemBtn =
  "flex size-9 items-center justify-center rounded text-ink-800 hover:bg-ink-900/8";
const previewToolbarPopoverTriggerBtn =
  "inline-flex items-center justify-center rounded-md p-1.5 text-xs font-medium bg-ink-900/5 text-ink-600 hover:bg-ink-900/10";

/** Fallback value for the hidden ``custom`` color input. */
const PREVIEW_DEFAULT_TEXT_COLOR = "#000000";

/** One hue family per column (left→right); each column is light→dark top→bottom. */
const PREVIEW_TEXT_COLOR_COLUMNS: readonly { label: string; shades: readonly string[] }[] = [
  {
    label: "Red",
    shades: ["#ffe4e6", "#fecdd3", "#fda4af", "#fb7185", "#e11d48", "#9f1239", "#450a0a"],
  },
  {
    label: "Orange",
    shades: ["#ffedd5", "#fed7aa", "#fdba74", "#fb923c", "#ea580c", "#c2410c", "#7c2d12"],
  },
  {
    label: "Yellow",
    shades: ["#fefce8", "#fef9c3", "#fef08a", "#facc15", "#ca8a04", "#a16207", "#713f12"],
  },
  {
    label: "Green",
    shades: ["#f0fdf4", "#dcfce7", "#86efac", "#22c55e", "#16a34a", "#166534", "#052e16"],
  },
  {
    label: "Blue",
    shades: ["#eff6ff", "#dbeafe", "#93c5fd", "#3b82f6", "#2563eb", "#1e3a8a", "#172554"],
  },
  {
    label: "Purple",
    shades: ["#faf5ff", "#f3e8ff", "#d8b4fe", "#a855f7", "#7e22ce", "#581c87", "#3b0764"],
  },
  {
    label: "Gray",
    shades: ["#ffffff", "#f3f4f6", "#d1d5db", "#9ca3af", "#6b7280", "#374151", "#030712"],
  },
];

/** Tiny glyphs for the Shape-mode picker (ink palette, matches inserted shapes). */
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

export function HtmlRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  reloadKey,
  onReload,
  onHtmlVisualSaveChromeChange,
  onTextSaveChromeChange,
}: Props) {
  const [mode, setMode] = useViewToggle("preview");
  const codeRef = useRef<HTMLElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const iframeViewportRef = useRef<HTMLDivElement>(null);
  const previewCleanupRef = useRef<(() => void) | null>(null);
  /** ResizeObserver / timers on the iframe document so fit-to-view updates after async layout (e.g. Chart.js). */
  const previewLayoutFitCleanupRef = useRef<(() => void) | null>(null);
  const editContentRef = useRef(data.content);
  const fitRafRef = useRef<number | null>(null);

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
  const [colorMenuOpen, setColorMenuOpen] = useState(false);
  const colorMenuRef = useRef<HTMLDivElement>(null);
  const previewTextColorCustomInputRef = useRef<HTMLInputElement>(null);
  const [previewTextColor, setPreviewTextColor] = useState(PREVIEW_DEFAULT_TEXT_COLOR);
  const [fitScale, setFitScale] = useState(1);
  const [contentSize, setContentSize] = useState<ContentSize | null>(null);
  const contentSizeRef = useRef<ContentSize | null>(null);

  const visualDirtyRef = useRef(visualDirty);
  const visualSavingRef = useRef(visualSaving);
  visualDirtyRef.current = visualDirty;
  visualSavingRef.current = visualSaving;

  const displayContent = canEdit ? editContent : data.content;
  const iframeKey = previewIframeKey(reloadKey, displayContent);

  useEffect(() => {
    setEditContent(data.content);
    editContentRef.current = data.content;
    visualDirtyRef.current = false;
    setVisualDirty(false);
    setVisualSaveError(null);
  }, [data.content, reloadKey]);

  useEffect(() => {
    const onDiscard = () => {
      setEditContent(data.content);
      editContentRef.current = data.content;
      visualDirtyRef.current = false;
      setVisualDirty(false);
      setVisualSaveError(null);
    };
    window.addEventListener("preview-reload-discard", onDiscard);
    return () => window.removeEventListener("preview-reload-discard", onDiscard);
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

  /**
   * Fit rendered HTML into the preview viewport. Content size is measured once (or when
   * the document layout changes), then only scale is recomputed on viewport resize.
   * Sizing the iframe as ``100/fitScale%`` caused a feedback loop: a smaller viewport
   * lowered scale → wider iframe layout → larger scrollWidth → lower scale again.
   */
  const recomputeFitScaleFromViewport = useCallback(
    (size: ContentSize) => {
      const viewport = iframeViewportRef.current;
      if (!viewport || mode !== "preview") return;
      const vw = Math.max(1, viewport.clientWidth);
      const vh = Math.max(1, viewport.clientHeight);
      const next = Math.min(1, vw / size.w, vh / size.h);
      setFitScale(Number.isFinite(next) ? Math.max(0.05, +next.toFixed(4)) : 1);
    },
    [mode]
  );

  const remeasureContentAndFit = useCallback(() => {
    const frame = iframeRef.current;
    const viewport = iframeViewportRef.current;
    const doc = frame?.contentDocument;
    if (!frame || !viewport || !doc?.documentElement || mode !== "preview") {
      setFitScale(1);
      return;
    }
    const size = readIframeContentSize(doc);
    contentSizeRef.current = size;
    setContentSize(size);
    recomputeFitScaleFromViewport(size);
  }, [mode, recomputeFitScaleFromViewport]);

  const scheduleRemeasureContentAndFit = useCallback(() => {
    if (fitRafRef.current != null) {
      cancelAnimationFrame(fitRafRef.current);
    }
    fitRafRef.current = requestAnimationFrame(() => {
      fitRafRef.current = null;
      remeasureContentAndFit();
    });
  }, [remeasureContentAndFit]);

  const scheduleViewportFit = useCallback(() => {
    if (fitRafRef.current != null) {
      cancelAnimationFrame(fitRafRef.current);
    }
    fitRafRef.current = requestAnimationFrame(() => {
      fitRafRef.current = null;
      const size = contentSizeRef.current;
      if (size) recomputeFitScaleFromViewport(size);
      else remeasureContentAndFit();
    });
  }, [remeasureContentAndFit, recomputeFitScaleFromViewport]);

  const markVisualDirtyFromDoc = useCallback(() => {
    const doc = iframeRef.current?.contentDocument;
    if (!doc?.documentElement) return;
    const dirty = serializeIframeDocument(doc) !== editContentRef.current;
    visualDirtyRef.current = dirty;
    setVisualDirty(dirty);
    scheduleRemeasureContentAndFit();
  }, [scheduleRemeasureContentAndFit]);

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

  const applyPreviewTextColorHex = useCallback(
    (hex: string) => {
      setPreviewTextColor(hex);
      const doc = iframeRef.current?.contentDocument;
      if (!applyPreviewTextColor(doc, hex)) return;
      markVisualDirtyFromDoc();
      iframeRef.current?.contentWindow?.focus();
    },
    [markVisualDirtyFromDoc]
  );

  useEffect(() => {
    if (!shapeMenuOpen && !alignMenuOpen && !colorMenuOpen) return;
    const onDocDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (shapeMenuOpen && shapeMenuRef.current && !shapeMenuRef.current.contains(t)) {
        setShapeMenuOpen(false);
      }
      if (alignMenuOpen && alignMenuRef.current && !alignMenuRef.current.contains(t)) {
        setAlignMenuOpen(false);
      }
      if (colorMenuOpen && colorMenuRef.current && !colorMenuRef.current.contains(t)) {
        setColorMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocDown, true);
    return () => document.removeEventListener("mousedown", onDocDown, true);
  }, [shapeMenuOpen, alignMenuOpen, colorMenuOpen]);

  useEffect(() => {
    if (visualTool !== "layout") setShapeMenuOpen(false);
    if (visualTool !== "text") {
      setAlignMenuOpen(false);
      setColorMenuOpen(false);
    }
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
    return () => {
      previewLayoutFitCleanupRef.current?.();
      previewLayoutFitCleanupRef.current = null;
    };
  }, [mode, displayContent]);

  useEffect(() => {
    contentSizeRef.current = null;
    setContentSize(null);
    setFitScale(1);
  }, [displayContent, reloadKey]);

  useEffect(() => {
    if (mode !== "preview") return;
    const viewport = iframeViewportRef.current;
    if (!viewport) return;
    const ro = new ResizeObserver(() => {
      scheduleViewportFit();
    });
    ro.observe(viewport);
    return () => ro.disconnect();
  }, [mode, scheduleViewportFit]);

  useEffect(() => {
    if (mode !== "preview") {
      setFitScale(1);
      return;
    }
    scheduleRemeasureContentAndFit();
  }, [mode, displayContent, scheduleRemeasureContentAndFit]);

  useEffect(() => {
    return () => {
      if (fitRafRef.current != null) {
        cancelAnimationFrame(fitRafRef.current);
      }
    };
  }, []);

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
    previewLayoutFitCleanupRef.current?.();
    previewLayoutFitCleanupRef.current = null;

    scheduleRemeasureContentAndFit();

    const doc = iframeRef.current?.contentDocument;
    if (doc?.documentElement && mode === "preview") {
      const ro = new ResizeObserver(() => scheduleRemeasureContentAndFit());
      ro.observe(doc.documentElement);
      if (doc.body) ro.observe(doc.body);
      const bump = () => scheduleRemeasureContentAndFit();
      const t1 = window.setTimeout(bump, 0);
      const t2 = window.setTimeout(bump, 120);
      const t3 = window.setTimeout(bump, 400);
      previewLayoutFitCleanupRef.current = () => {
        ro.disconnect();
        window.clearTimeout(t1);
        window.clearTimeout(t2);
        window.clearTimeout(t3);
      };
    }

    if (mode !== "preview" || visualTool === "none" || !canEdit) {
      runPreviewCleanup();
      setVisualDirty(false);
      return;
    }
    attachPreviewTool(visualTool);
  }, [mode, visualTool, canEdit, attachPreviewTool, runPreviewCleanup, scheduleRemeasureContentAndFit]);

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

  useEffect(() => {
    if (!onHtmlVisualSaveChromeChange) return;
    return () => onHtmlVisualSaveChromeChange(null);
  }, [onHtmlVisualSaveChromeChange]);

  useEffect(() => {
    if (!onHtmlVisualSaveChromeChange) return;
    if (mode !== "preview" || visualTool === "none" || !canEdit) {
      onHtmlVisualSaveChromeChange(null);
      return;
    }
    onHtmlVisualSaveChromeChange({
      save: () => {
        void handleSaveVisual();
      },
      disabled: visualSaving || !visualDirty,
      saving: visualSaving,
      error: visualSaveError,
    });
  }, [
    mode,
    visualTool,
    canEdit,
    visualSaving,
    visualDirty,
    visualSaveError,
    onHtmlVisualSaveChromeChange,
    handleSaveVisual,
  ]);

  /**
   * ``allow-same-origin`` is required for (a) saveable files so Text/Shape can attach, and (b) all previews so the host
   * can read ``contentDocument`` for fit-to-view and observe layout after async scripts (Chart.js, etc.). ``allow-scripts``
   * alone makes the iframe opaque in Chromium, breaking scale math and inner ``ResizeObserver``s.
   */
  const previewSandbox = "allow-scripts allow-same-origin";

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
              {toolBtn("text", "Text")}
              {toolBtn("layout", "Shape")}
            </>
          )}
        </div>
        {mode === "preview" && visualTool !== "none" && canEdit && (
          <div className="flex items-center gap-2 shrink-0">
            {visualTool === "text" && (
              <div className="relative inline-flex" ref={colorMenuRef}>
                <input
                  ref={previewTextColorCustomInputRef}
                  type="color"
                  value={previewTextColor}
                  onChange={(e) => {
                    const hex = e.target.value;
                    applyPreviewTextColorHex(hex);
                    setColorMenuOpen(false);
                  }}
                  className="absolute h-px w-px opacity-0 -z-10"
                  tabIndex={-1}
                  aria-hidden
                />
                <button
                  type="button"
                  onPointerDownCapture={() => {
                    const doc = iframeRef.current?.contentDocument;
                    snapshotPreviewTextColorSelection(doc);
                  }}
                  onClick={() => setColorMenuOpen((o) => !o)}
                  title="Text color (select text first)"
                  aria-expanded={colorMenuOpen}
                  aria-haspopup="menu"
                  className={previewToolbarPopoverTriggerBtn}
                >
                  <Palette className="size-4" aria-hidden />
                  <span className="sr-only">Text color</span>
                </button>
                {colorMenuOpen && (
                  <div
                    role="menu"
                    className="absolute right-0 top-full z-50 mt-1 w-max max-w-[calc(100vw-2rem)] rounded-md border border-ink-900/15 bg-white p-2 shadow-md"
                  >
                    <div
                      className="flex gap-1.5"
                      role="group"
                      aria-label="Preset text colors by hue (columns), light to dark (top to bottom)"
                    >
                      {PREVIEW_TEXT_COLOR_COLUMNS.map((col) => (
                        <div
                          key={col.label}
                          className="flex flex-col gap-1"
                          role="group"
                          aria-label={`${col.label} shades`}
                        >
                          {col.shades.map((hex) => (
                            <button
                              key={`${col.label}-${hex}`}
                              type="button"
                              role="menuitem"
                              title={`${col.label}: ${hex}`}
                              aria-label={`Apply ${col.label} ${hex}`}
                              className="size-7 shrink-0 rounded border border-ink-900/20 shadow-sm outline-none ring-offset-2 hover:scale-105 focus-visible:ring-2 focus-visible:ring-ink-900/25 transition-transform"
                              style={{ backgroundColor: hex }}
                              onClick={() => {
                                applyPreviewTextColorHex(hex);
                                setColorMenuOpen(false);
                              }}
                            />
                          ))}
                        </div>
                      ))}
                    </div>
                    <div className="mt-2 border-t border-ink-900/10 pt-2">
                      <button
                        type="button"
                        role="menuitem"
                        className="w-full rounded px-2 py-1.5 text-left text-xs font-medium text-ink-700 hover:bg-ink-900/8"
                        onClick={() => {
                          setColorMenuOpen(false);
                          previewTextColorCustomInputRef.current?.click();
                        }}
                      >
                        Custom color…
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
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
        <div ref={iframeViewportRef} className="flex-1 min-h-0 w-full overflow-hidden rounded bg-white">
          <div
            style={
              contentSize
                ? {
                    transform: `scale(${fitScale})`,
                    transformOrigin: "top left",
                    width: contentSize.w,
                    height: contentSize.h,
                  }
                : undefined
            }
          >
            <iframe
              key={iframeKey}
              ref={iframeRef}
              srcDoc={displayContent}
              sandbox={previewSandbox}
              onLoad={handleIframeLoad}
              tabIndex={canEdit ? 0 : undefined}
              className="block border-0 bg-white"
              style={{
                minHeight: 200,
                width: contentSize?.w ?? "100%",
                height: contentSize?.h ?? "100%",
              }}
              title="HTML Preview"
            />
          </div>
        </div>
      ) : canEdit ? (
        <EditableTextPanel
          content={editContent}
          contentKey={iframeKey}
          monospace
          onSave={async (content) => {
            setEditContent(content);
            return window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined);
          }}
          onSaved={onReload}
          onSaveChromeChange={onTextSaveChromeChange ?? onHtmlVisualSaveChromeChange}
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
