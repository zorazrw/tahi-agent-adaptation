/** Helpers for in-preview HTML editing (text + drag-layout) inside a same-origin iframe. */

export type HtmlDocCleanup = () => void;

const MOVE_STYLE_ID = "__ac-html-move-injected";

/** User-placed shapes in Move mode (serialized with the saved HTML). */
export type PreviewShapeKind = "rectangle" | "circle" | "line";

const PREVIEW_SHAPE_CLASS = "ac-preview-shape";
const PREVIEW_SHAPE_INNER_CLASS = "ac-preview-shape-inner";
const PREVIEW_SHAPE_CHROME_CLASS = "ac-preview-shape-chrome";
const PREVIEW_SHAPE_SELECTED_CLASS = "ac-preview-shape--selected";
const LAYOUT_BLOCK_SELECTED_CLASS = "ac-preview-layout-selected";

const NON_DELETABLE_LAYOUT_TAGS = new Set([
  "HTML",
  "BODY",
  "HEAD",
  "SCRIPT",
  "STYLE",
  "LINK",
  "META",
  "BASE",
  "TITLE",
  "NOSCRIPT",
  "TEMPLATE",
]);

/** App ink tokens (see ``index.css``); iframe has no Tailwind, so use fixed neutrals. */
const SHAPE_INK_STROKE = "#666661";
const SHAPE_INK_FILL = "rgba(102, 102, 97, 0.08)";

function getShapeInner(wrap: HTMLElement): HTMLElement | null {
  return wrap.querySelector(`:scope > .${PREVIEW_SHAPE_INNER_CLASS}`) as HTMLElement | null;
}

function getLineBar(inner: HTMLElement): HTMLElement | null {
  const ch = inner.firstElementChild;
  return ch instanceof HTMLElement && ch.tagName === "DIV" ? ch : null;
}

function getShapeBodyElement(wrap: HTMLElement): HTMLElement | null {
  const named = getShapeInner(wrap);
  if (named) return named;
  for (const c of wrap.children) {
    if (!(c instanceof HTMLElement)) continue;
    if (c.classList.contains(PREVIEW_SHAPE_CHROME_CLASS)) continue;
    return c;
  }
  return null;
}

/**
 * SE corner resize zone (viewport coords). Slop scales with shape size so tiny shapes are not 100% “resize”.
 */
function isPointerInSeResizeZone(inner: HTMLElement, clientX: number, clientY: number): boolean {
  const r = inner.getBoundingClientRect();
  const w = Math.max(1, r.width);
  const h = Math.max(1, r.height);
  const sx = Math.min(40, Math.max(16, w * 0.28));
  const sy = Math.min(40, Math.max(16, h * 0.28));
  return (
    clientX >= r.right - sx &&
    clientX <= r.right + 12 &&
    clientY >= r.bottom - sy &&
    clientY <= r.bottom + 12
  );
}

/**
 * Remove ephemeral shape UI from a detached DOM subtree (used when serializing a clone).
 */
function stripEphemeralPreviewUiFromSubtree(root: ParentNode): void {
  root.querySelectorAll(`.${PREVIEW_SHAPE_CHROME_CLASS}`).forEach((el) => el.remove());
  root.querySelectorAll(`.${PREVIEW_SHAPE_SELECTED_CLASS}`).forEach((el) =>
    el.classList.remove(PREVIEW_SHAPE_SELECTED_CLASS)
  );
  root.querySelectorAll(`.${LAYOUT_BLOCK_SELECTED_CLASS}`).forEach((el) =>
    el.classList.remove(LAYOUT_BLOCK_SELECTED_CLASS)
  );
}

function isDeletableLayoutBlock(el: HTMLElement): boolean {
  const tag = el.tagName.toUpperCase();
  if (NON_DELETABLE_LAYOUT_TAGS.has(tag)) return false;
  if (!el.parentElement) return false;
  return true;
}

function clearLayoutBlockSelection(doc: Document): void {
  doc.querySelectorAll(`.${LAYOUT_BLOCK_SELECTED_CLASS}`).forEach((n) => n.classList.remove(LAYOUT_BLOCK_SELECTED_CLASS));
}

function selectLayoutBlock(doc: Document, el: HTMLElement): void {
  if (!isDeletableLayoutBlock(el)) return;
  clearLayoutBlockSelection(doc);
  el.classList.add(LAYOUT_BLOCK_SELECTED_CLASS);
}

/**
 * Delete controls + selection from the live document (e.g. when leaving Move mode).
 */
export function stripPreviewShapeEditorUi(doc: Document): void {
  stripEphemeralPreviewUiFromSubtree(doc);
}

/** Forward-delete / Backspace—covers layout labels and ``code`` for embedded iframes / macOS. */
export function isDeleteOrBackspaceKey(e: KeyboardEvent): boolean {
  return (
    e.key === "Delete" ||
    e.key === "Backspace" ||
    e.code === "Delete" ||
    e.code === "Backspace"
  );
}

/** If a preview shape is selected, remove it and clear selection. */
export function tryDeleteSelectedPreviewShape(doc: Document | null | undefined): boolean {
  if (!doc) return false;
  const sel = doc.querySelector(`.${PREVIEW_SHAPE_SELECTED_CLASS}`) as HTMLElement | null;
  if (!sel?.classList.contains(PREVIEW_SHAPE_CLASS)) return false;
  sel.remove();
  selectPreviewShape(doc, null);
  return true;
}

function tryDeleteSelectedLayoutBlock(doc: Document | null | undefined): boolean {
  if (!doc) return false;
  const el = doc.querySelector(`.${LAYOUT_BLOCK_SELECTED_CLASS}`) as HTMLElement | null;
  if (!el || !isDeletableLayoutBlock(el)) return false;
  el.remove();
  clearLayoutBlockSelection(doc);
  return true;
}

/** Shape removal first, then LM layout block marked for delete in Move mode. */
export function tryDeleteSelectedPreviewOrLayoutBlock(doc: Document | null | undefined): boolean {
  if (tryDeleteSelectedPreviewShape(doc)) return true;
  return tryDeleteSelectedLayoutBlock(doc);
}

/**
 * Attach one delete control + SE resize handle per shape (idempotent).
 */
export function mountPreviewShapeChrome(doc: Document, wrap: HTMLElement): void {
  let inner = getShapeInner(wrap);
  if (!inner) {
    const first = wrap.firstElementChild;
    if (!(first instanceof HTMLElement)) return;
    if (first.classList.contains(PREVIEW_SHAPE_CHROME_CLASS)) return;
    inner = first;
    inner.classList.add(PREVIEW_SHAPE_INNER_CLASS);
  }

  inner.style.position = "relative";
  inner.style.overflow = "visible";
  wrap.style.overflow = "visible";

  if (!wrap.querySelector(".ac-preview-shape-delete")) {
    const del = doc.createElement("button");
    del.type = "button";
    del.className = `${PREVIEW_SHAPE_CHROME_CLASS} ac-preview-shape-delete`;
    del.setAttribute("aria-label", "Remove shape");
    del.textContent = "×";
    del.style.cssText = [
      "position:absolute",
      "top:-10px",
      "right:-10px",
      "width:22px",
      "height:22px",
      "padding:0",
      "line-height:20px",
      "font-size:16px",
      "font-weight:600",
      "border-radius:9999px",
      `border:1px solid ${SHAPE_INK_STROKE}`,
      "background:#fff",
      "color:#2d2d2a",
      "cursor:pointer",
      "z-index:10004",
      "box-sizing:border-box",
    ].join(";");
    wrap.appendChild(del);
  }

  if (!inner.querySelector(".ac-preview-shape-handle")) {
    const handle = doc.createElement("div");
    handle.className = `${PREVIEW_SHAPE_CHROME_CLASS} ac-preview-shape-handle ac-preview-shape-handle--se`;
    handle.dataset.handle = "se";
    /** Inside inner so parent ``overflow:hidden`` on body/slides does not clip; sits on the corner. */
    handle.style.cssText = [
      "position:absolute",
      "right:2px",
      "bottom:2px",
      "width:16px",
      "height:16px",
      "box-sizing:border-box",
      `border:1px solid ${SHAPE_INK_STROKE}`,
      "background:#fff",
      "z-index:5",
      "cursor:nwse-resize",
      "touch-action:none",
      "box-shadow:0 0 0 1px rgba(255,255,255,0.9)",
    ].join(";");
    inner.appendChild(handle);
  }
}

/** After load / new insert: ensure every shape has interaction chrome. */
export function mountAllPreviewShapeChrome(doc: Document): void {
  doc.querySelectorAll(`.${PREVIEW_SHAPE_CLASS}`).forEach((el) => {
    if (el instanceof HTMLElement) mountPreviewShapeChrome(doc, el);
  });
}

function selectPreviewShape(doc: Document, wrap: HTMLElement | null, opts?: { focus?: boolean }): void {
  clearLayoutBlockSelection(doc);
  const doFocus = opts?.focus !== false;
  doc.querySelectorAll(`.${PREVIEW_SHAPE_CLASS}`).forEach((el) => {
    if (!(el instanceof HTMLElement)) return;
    el.classList.remove(PREVIEW_SHAPE_SELECTED_CLASS);
    el.removeAttribute("tabindex");
  });
  if (wrap) {
    wrap.classList.add(PREVIEW_SHAPE_SELECTED_CLASS);
    if (doFocus) {
      wrap.setAttribute("tabindex", "-1");
      try {
        wrap.focus({ preventScroll: true });
      } catch {
        /* ignore */
      }
    }
  }
}

export function insertPreviewShape(doc: Document, kind: PreviewShapeKind): void {
  const body = doc.body;
  if (!body) return;

  const n = body.querySelectorAll(`.${PREVIEW_SHAPE_CLASS}`).length;
  const left = 40 + (n % 6) * 28;
  const top = 40 + (n % 6) * 28;

  const wrap = doc.createElement("div");
  wrap.className = PREVIEW_SHAPE_CLASS;
  wrap.setAttribute("data-ac-shape", kind);
  wrap.style.cssText = [
    "position:fixed",
    `left:${left}px`,
    `top:${top}px`,
    "z-index:10000",
    "pointer-events:auto",
    "box-sizing:border-box",
    "cursor:grab",
    "overflow:visible",
  ].join(";");

  const inner = doc.createElement("div");
  inner.className = PREVIEW_SHAPE_INNER_CLASS;
  inner.style.boxSizing = "border-box";
  inner.style.position = "relative";
  inner.style.overflow = "visible";

  if (kind === "rectangle") {
    inner.style.width = "120px";
    inner.style.height = "72px";
    inner.style.border = `1px solid ${SHAPE_INK_STROKE}`;
    inner.style.background = SHAPE_INK_FILL;
    inner.style.borderRadius = "4px";
  } else if (kind === "circle") {
    inner.style.width = "80px";
    inner.style.height = "80px";
    inner.style.border = `1px solid ${SHAPE_INK_STROKE}`;
    inner.style.background = SHAPE_INK_FILL;
    inner.style.borderRadius = "50%";
  } else {
    inner.style.width = "140px";
    inner.style.minHeight = "20px";
    inner.style.height = "20px";
    inner.style.display = "flex";
    inner.style.alignItems = "center";
    const bar = doc.createElement("div");
    bar.style.height = "4px";
    bar.style.width = "100%";
    bar.style.borderRadius = "2px";
    bar.style.background = SHAPE_INK_STROKE;
    bar.style.flexShrink = "0";
    inner.appendChild(bar);
    wrap.appendChild(inner);
    body.appendChild(wrap);
    mountPreviewShapeChrome(doc, wrap);
    return;
  }

  wrap.appendChild(inner);
  body.appendChild(wrap);
  mountPreviewShapeChrome(doc, wrap);
}

export function serializeIframeDocument(doc: Document): string {
  const liveRoot = doc.documentElement;
  if (!liveRoot) return "";
  const cloneRoot = liveRoot.cloneNode(true) as HTMLElement;
  stripEphemeralPreviewUiFromSubtree(cloneRoot);
  cloneRoot.querySelector(`#${MOVE_STYLE_ID}`)?.remove();

  const dt = doc.doctype;
  let preamble = "";
  if (dt) {
    const pub = dt.publicId ? ` PUBLIC "${dt.publicId}"` : "";
    const sys = dt.systemId ? ` "${dt.systemId}"` : "";
    preamble = `<!DOCTYPE ${dt.name}${pub}${sys}>\n`;
  }
  return preamble + cloneRoot.outerHTML;
}

function resolveDragTarget(doc: Document, target: EventTarget | null): HTMLElement | null {
  if (!target || !(target as Node)) return null;
  let el: HTMLElement | null =
    (target as Node).nodeType === Node.ELEMENT_NODE
      ? (target as HTMLElement)
      : (target as Text).parentElement;
  while (el && (el === doc.body || el === doc.documentElement)) {
    el = el.parentElement;
  }
  if (!el || el === doc.body || el === doc.documentElement) return null;
  const tag = el.tagName.toUpperCase();
  if (tag === "SCRIPT" || tag === "STYLE" || tag === "HEAD") return null;
  return el;
}

/** Phrasing / line content — dragging these only moved text, not the surrounding “textbox” shell. */
const PHRASING_TAGS = new Set([
  "SPAN", "STRONG", "EM", "B", "I", "U", "S", "SMALL", "MARK", "CODE",
  "SUB", "SUP", "TIME", "CITE", "LABEL", "ABBR", "DFN", "KBD", "SAMP", "VAR",
  "FONT", "A", "BR", "BDI", "BDO", "Q",
]);

/** Keep drags on the control itself, not an outer card. */
const INTERACTIVE_ROOT_TAGS = new Set([
  "INPUT", "TEXTAREA", "SELECT", "BUTTON", "OPTION", "OPTGROUP", "VIDEO", "AUDIO", "IFRAME", "OBJECT", "IMG", "SVG", "CANVAS",
]);

const WRAPPER_PARENT_TAGS = new Set([
  "DIV", "SECTION", "ARTICLE", "ASIDE", "MAIN", "HEADER", "FOOTER", "NAV", "BLOCKQUOTE", "FIGURE",
]);

/**
 * Walk up from the hit target so Move drags a layout box (card, panel), not only inner text nodes.
 */
function promoteToLayoutDragRoot(doc: Document, start: HTMLElement): HTMLElement {
  const view = doc.defaultView;
  if (!view) return start;
  let cur: HTMLElement = start;

  for (;;) {
    const parentEl: HTMLElement | null = cur.parentElement;
    if (!parentEl || parentEl === doc.body || parentEl === doc.documentElement) break;

    const tagU = cur.tagName.toUpperCase();
    if (INTERACTIVE_ROOT_TAGS.has(tagU)) break;

    const cs = view.getComputedStyle(cur);
    const disp = cs.display;
    if (PHRASING_TAGS.has(tagU) || disp === "inline" || disp === "contents") {
      cur = parentEl;
      continue;
    }
    if (tagU === "P" || /^H[1-6]$/.test(tagU)) {
      cur = parentEl;
      continue;
    }
    break;
  }

  // Single-child chains: <div class="card"><div class="inner"><p>…</p></div></div> → drag the card.
  for (;;) {
    const p: HTMLElement | null = cur.parentElement;
    if (!p || p === doc.body || p === doc.documentElement) break;
    const tagP = p.tagName.toUpperCase();
    if (!WRAPPER_PARENT_TAGS.has(tagP)) break;
    if (p.childElementCount !== 1 || p.firstElementChild !== cur) break;
    cur = p;
  }

  return cur;
}

function parseTranslate(el: HTMLElement): { x: number; y: number } {
  const t = el.style.transform;
  const m = t.match(/translate\(\s*(-?[\d.]+)px\s*,\s*(-?[\d.]+)px\s*\)/);
  if (m) return { x: parseFloat(m[1]), y: parseFloat(m[2]) };
  return { x: 0, y: 0 };
}

export type PreviewTextAlignH = "left" | "center" | "right";
export type PreviewTextAlignV = "start" | "middle" | "end";

const TEXT_ALIGN_BLOCK_TAGS = new Set([
  "P",
  "DIV",
  "SECTION",
  "ARTICLE",
  "ASIDE",
  "MAIN",
  "HEADER",
  "FOOTER",
  "NAV",
  "BLOCKQUOTE",
  "FIGURE",
  "LI",
  "TD",
  "TH",
  "H1",
  "H2",
  "H3",
  "H4",
  "H5",
  "H6",
]);

function getDesignModeBlockElement(doc: Document): HTMLElement | null {
  const sel = doc.getSelection();
  if (!sel || sel.rangeCount === 0) return doc.body;
  let node: Node | null = sel.anchorNode;
  if (!node) return doc.body;
  if (node.nodeType === Node.TEXT_NODE) node = node.parentNode;
  let el = node as HTMLElement | null;
  while (el) {
    if (el === doc.body || el === doc.documentElement) return doc.body;
    const tag = el.tagName?.toUpperCase() ?? "";
    if (TEXT_ALIGN_BLOCK_TAGS.has(tag)) return el;
    el = el.parentElement;
  }
  return doc.body;
}

/**
 * Apply alignment in preview Text mode (``designMode``). Horizontal uses execCommand; vertical
 * uses flex on the current block (not ``body``).
 */
export function applyPreviewTextAlignment(
  doc: Document | null | undefined,
  axis: "h",
  value: PreviewTextAlignH
): boolean;
export function applyPreviewTextAlignment(
  doc: Document | null | undefined,
  axis: "v",
  value: PreviewTextAlignV
): boolean;
export function applyPreviewTextAlignment(
  doc: Document | null | undefined,
  axis: "h" | "v",
  value: PreviewTextAlignH | PreviewTextAlignV
): boolean {
  if (!doc || doc.designMode !== "on" || !doc.body) return false;
  try {
    doc.body.focus();
    if (axis === "h") {
      const cmd =
        value === "left"
          ? "justifyLeft"
          : value === "center"
            ? "justifyCenter"
            : "justifyRight";
      return doc.execCommand(cmd, false);
    }
    const v = value as PreviewTextAlignV;
    const block = getDesignModeBlockElement(doc);
    if (!block || block === doc.body || block === doc.documentElement) return false;
    const jc = v === "start" ? "flex-start" : v === "middle" ? "center" : "flex-end";
    block.style.display = "flex";
    block.style.flexDirection = "column";
    block.style.justifyContent = jc;
    if (!block.style.minHeight) block.style.minHeight = "2.5em";
    return true;
  } catch {
    return false;
  }
}

export function attachHtmlTextEdit(
  doc: Document,
  onChange: () => void,
  onSaveShortcut?: (e: KeyboardEvent) => void,
  /** Fires synchronously on user edit so Save / Ctrl+S can run before the next React render (refs stay in sync). */
  onUserInput?: () => void
): HtmlDocCleanup {
  if (!doc.body) return () => {};

  doc.querySelectorAll("[contenteditable]").forEach((el) => {
    if (el.getAttribute("contenteditable")?.toLowerCase() === "false") {
      el.removeAttribute("contenteditable");
    }
  });

  /**
   * `body.contentEditable` is fragile for full HTML in srcdoc (focus/host quirks, nested blocks).
   * `designMode` makes the whole document act as one editable surface, matching browser edit expectations.
   */
  doc.designMode = "on";
  try {
    doc.execCommand("defaultParagraphSeparator", false, "p");
  } catch {
    /* unsupported in some environments */
  }

  let dirtyFlush = 0;
  const flushDirty = () => {
    globalThis.clearTimeout(dirtyFlush);
    dirtyFlush = globalThis.setTimeout(() => onChange(), 0);
  };

  const onUserEdit = () => {
    onUserInput?.();
    flushDirty();
  };

  const obs = new MutationObserver(() => flushDirty());
  if (doc.documentElement) {
    obs.observe(doc.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      characterData: true,
    });
  }

  doc.addEventListener("input", onUserEdit);
  doc.addEventListener("keyup", onUserEdit);
  /** IME (e.g. CJK): `input` may be incomplete until composition ends. */
  doc.addEventListener("compositionend", onUserEdit);

  const win = doc.defaultView;
  requestAnimationFrame(() => {
    win?.focus();
  });

  const onSaveKey = (e: KeyboardEvent) => {
    onSaveShortcut?.(e);
  };
  if (onSaveShortcut) {
    doc.addEventListener("keydown", onSaveKey, true);
  }

  return () => {
    globalThis.clearTimeout(dirtyFlush);
    obs.disconnect();
    doc.removeEventListener("input", onUserEdit);
    doc.removeEventListener("keyup", onUserEdit);
    doc.removeEventListener("compositionend", onUserEdit);
    if (onSaveShortcut) {
      doc.removeEventListener("keydown", onSaveKey, true);
    }
    doc.designMode = "off";
  };
}

type LayoutDragState =
  | {
      mode: "move";
      el: HTMLElement;
      startX: number;
      startY: number;
      baseX: number;
      baseY: number;
    }
  | {
      mode: "resize";
      wrap: HTMLElement;
      inner: HTMLElement;
      kind: PreviewShapeKind;
      startX: number;
      startY: number;
      startW: number;
      startH: number;
      startBarH: number;
    };

export function attachHtmlLayoutDrag(
  doc: Document,
  onChange: () => void,
  onSaveShortcut?: (e: KeyboardEvent) => void
): HtmlDocCleanup {
  const body = doc.body;
  if (!body) return () => {};

  body.contentEditable = "false";
  body.classList.add("layout-mode");

  const style = doc.createElement("style");
  style.id = MOVE_STYLE_ID;
  style.textContent = `
    body.layout-mode { user-select: none !important; }
    body.layout-mode *:not(.ac-preview-shape-handle):not(.ac-preview-shape-delete) {
      cursor: grab !important;
    }
    body.layout-mode *:not(.ac-preview-shape-handle):not(.ac-preview-shape-delete):active {
      cursor: grabbing !important;
    }
    body.layout-mode .ac-preview-shape-handle {
      cursor: nwse-resize !important;
      touch-action: none !important;
    }
    body.layout-mode .ac-preview-shape-delete {
      cursor: pointer !important;
    }
    body.layout-mode .${PREVIEW_SHAPE_SELECTED_CLASS} {
      outline: 1px solid ${SHAPE_INK_STROKE};
      outline-offset: 1px;
    }
    body.layout-mode .${LAYOUT_BLOCK_SELECTED_CLASS} {
      outline: 1px dashed ${SHAPE_INK_STROKE};
      outline-offset: 2px;
    }
    body.layout-mode .${PREVIEW_SHAPE_INNER_CLASS} {
      position: relative !important;
      overflow: visible !important;
    }
  `;
  doc.head?.appendChild(style);

  mountAllPreviewShapeChrome(doc);

  let dragging: LayoutDragState | null = null;

  const endDrag = () => {
    if (!dragging) return;
    dragging = null;
    onChange();
  };

  const onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    const target = e.target as Node;
    const el = target instanceof Element ? target : target.parentElement;
    if (!el) return;

    const delBtn = el.closest(".ac-preview-shape-delete");
    if (delBtn) {
      const wrap = delBtn.closest(`.${PREVIEW_SHAPE_CLASS}`);
      if (wrap) {
        e.preventDefault();
        e.stopPropagation();
        wrap.remove();
        selectPreviewShape(doc, null);
        onChange();
      }
      return;
    }

    const shapeWrap = el.closest(`.${PREVIEW_SHAPE_CLASS}`) as HTMLElement | null;
    if (shapeWrap) {
      e.preventDefault();
      e.stopPropagation();
      const inner = getShapeBodyElement(shapeWrap);
      if (!inner) return;

      const handleEl = el.closest(".ac-preview-shape-handle") as HTMLElement | null;
      const onResizeHandle = Boolean(handleEl);
      const inCornerZone = !onResizeHandle && isPointerInSeResizeZone(inner, e.clientX, e.clientY);

      if (!inner.classList.contains(PREVIEW_SHAPE_INNER_CLASS)) inner.classList.add(PREVIEW_SHAPE_INNER_CLASS);

      if (onResizeHandle || inCornerZone) {
        const kind = (shapeWrap.getAttribute("data-ac-shape") ?? "rectangle") as PreviewShapeKind;
        const bar = kind === "line" ? getLineBar(inner) : null;
        const cs = doc.defaultView?.getComputedStyle(inner);
        const startW = parseFloat(inner.style.width) || parseFloat(cs?.width ?? "0") || 40;
        const startH = parseFloat(inner.style.height) || parseFloat(cs?.height ?? "0") || 40;
        const startBarH = bar
          ? parseFloat(bar.style.height) || parseFloat(doc.defaultView?.getComputedStyle(bar).height ?? "4") || 4
          : 4;
        selectPreviewShape(doc, shapeWrap, { focus: false });
        dragging = {
          mode: "resize",
          wrap: shapeWrap,
          inner,
          kind,
          startX: e.clientX,
          startY: e.clientY,
          startW,
          startH,
          startBarH,
        };
        const captureEl = handleEl ?? inner;
        try {
          captureEl.setPointerCapture(e.pointerId);
        } catch {
          try {
            shapeWrap.setPointerCapture(e.pointerId);
          } catch {
            /* ignore */
          }
        }
        return;
      }

      selectPreviewShape(doc, shapeWrap);
      const { x, y } = parseTranslate(shapeWrap);
      dragging = {
        mode: "move",
        el: shapeWrap,
        startX: e.clientX,
        startY: e.clientY,
        baseX: x,
        baseY: y,
      };
      const pos = doc.defaultView?.getComputedStyle(shapeWrap).position;
      if (pos === "static") shapeWrap.style.position = "relative";
      try {
        shapeWrap.setPointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      return;
    }

    selectPreviewShape(doc, null);

    const hit = resolveDragTarget(doc, e.target);
    if (!hit) {
      clearLayoutBlockSelection(doc);
      return;
    }
    const he = promoteToLayoutDragRoot(doc, hit);
    e.preventDefault();
    e.stopPropagation();
    selectLayoutBlock(doc, he);
    const { x, y } = parseTranslate(he);
    dragging = { mode: "move", el: he, startX: e.clientX, startY: e.clientY, baseX: x, baseY: y };
    const pos = doc.defaultView?.getComputedStyle(he).position;
    if (pos === "static") he.style.position = "relative";
    try {
      he.setPointerCapture(e.pointerId);
    } catch {
      /* ignore if element cannot capture */
    }
  };

  const onPointerMove = (e: PointerEvent) => {
    if (!dragging) return;
    e.preventDefault();
    if (dragging.mode === "move") {
      const dx = e.clientX - dragging.startX;
      const dy = e.clientY - dragging.startY;
      dragging.el.style.transform = `translate(${dragging.baseX + dx}px, ${dragging.baseY + dy}px)`;
      return;
    }
    const dx = e.clientX - dragging.startX;
    const dy = e.clientY - dragging.startY;
    const { inner, kind, startW, startH, startBarH } = dragging;
    if (kind === "rectangle") {
      inner.style.width = `${Math.max(32, Math.round(startW + dx))}px`;
      inner.style.height = `${Math.max(24, Math.round(startH + dy))}px`;
    } else if (kind === "circle") {
      const s = Math.max(28, Math.round(startW + (dx + dy) / 2));
      inner.style.width = `${s}px`;
      inner.style.height = `${s}px`;
    } else {
      inner.style.width = `${Math.max(48, Math.round(startW + dx))}px`;
      const bar = getLineBar(inner);
      const bh = Math.max(2, Math.round(startBarH + dy));
      inner.style.minHeight = `${Math.max(12, bh + 8)}px`;
      inner.style.height = `${Math.max(12, bh + 8)}px`;
      if (bar) {
        bar.style.height = `${bh}px`;
      }
    }
  };

  const onPointerUp = () => {
    endDrag();
  };

  /**
   * Iframe ``blur`` often fires while resizing (pointer still captured) when the OS shifts
   * focus or the pointer leaves the frame. Clearing ``dragging`` there stops ``pointermove``
   * from applying width/height — resize appears broken.
   */
  const onWinBlur = () => {
    if (dragging?.mode === "resize") return;
    endDrag();
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (!isDeleteOrBackspaceKey(e)) return;
    const t = e.target;
    if (t instanceof Element && t.closest?.("input,textarea,[contenteditable=true],select")) return;
    if (!tryDeleteSelectedPreviewOrLayoutBlock(doc)) return;
    e.preventDefault();
    e.stopPropagation();
    onChange();
  };

  const win = doc.defaultView;

  doc.addEventListener("pointerdown", onPointerDown, true);
  /** Deliver move/up on the iframe window so pointer capture + leaving the frame still updates / ends the gesture. */
  win?.addEventListener("pointermove", onPointerMove, true);
  win?.addEventListener("pointerup", onPointerUp, true);
  win?.addEventListener("pointercancel", onPointerUp, true);
  win?.addEventListener("blur", onWinBlur);

  /** DOM inserts (e.g. shapes) + live inline `style` updates (transform, box sizes). */
  let dirtyFlush = 0;
  const flushDirty = () => {
    (win ?? window).clearTimeout(dirtyFlush);
    dirtyFlush = (win ?? window).setTimeout(() => {
      onChange();
    }, 0);
  };
  const attrObs = new MutationObserver(() => flushDirty());
  if (doc.documentElement) {
    attrObs.observe(doc.documentElement, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["style", "class"],
    });
  }

  const onSaveKey = (e: KeyboardEvent) => {
    onSaveShortcut?.(e);
  };
  if (onSaveShortcut) {
    doc.addEventListener("keydown", onSaveKey, true);
  }

  /** ``document`` does not always receive keys when focus is unclear; the iframe window does. */
  win?.addEventListener("keydown", onKeyDown, true);

  return () => {
    (win ?? window).clearTimeout(dirtyFlush);
    attrObs.disconnect();
    if (onSaveShortcut) {
      doc.removeEventListener("keydown", onSaveKey, true);
    }
    win?.removeEventListener("keydown", onKeyDown, true);
    doc.removeEventListener("pointerdown", onPointerDown, true);
    win?.removeEventListener("pointermove", onPointerMove, true);
    win?.removeEventListener("pointerup", onPointerUp, true);
    win?.removeEventListener("pointercancel", onPointerUp, true);
    win?.removeEventListener("blur", onWinBlur);
    stripPreviewShapeEditorUi(doc);
    body.classList.remove("layout-mode");
    style.remove();
  };
}
