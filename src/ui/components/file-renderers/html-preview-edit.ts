/** Helpers for in-preview HTML editing (text + drag-layout) inside a same-origin iframe. */

export type HtmlDocCleanup = () => void;

const MOVE_STYLE_ID = "__ac-html-move-injected";

export function serializeIframeDocument(doc: Document): string {
  const injected = doc.getElementById(MOVE_STYLE_ID);
  const parent = injected?.parentNode ?? null;
  const next = injected?.nextSibling ?? null;
  injected?.remove();
  try {
    const dt = doc.doctype;
    let preamble = "";
    if (dt) {
      const pub = dt.publicId ? ` PUBLIC "${dt.publicId}"` : "";
      const sys = dt.systemId ? ` "${dt.systemId}"` : "";
      preamble = `<!DOCTYPE ${dt.name}${pub}${sys}>\n`;
    }
    return preamble + (doc.documentElement?.outerHTML ?? "");
  } finally {
    if (injected && parent) {
      parent.insertBefore(injected, next);
    }
  }
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
    body.layout-mode * { cursor: grab !important; }
    body.layout-mode *:active { cursor: grabbing !important; }
    body.layout-mode { user-select: none !important; }
  `;
  doc.head?.appendChild(style);

  let dragging: {
    el: HTMLElement;
    startX: number;
    startY: number;
    baseX: number;
    baseY: number;
  } | null = null;

  const endDrag = () => {
    if (!dragging) return;
    dragging = null;
    onChange();
  };

  const onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    const hit = resolveDragTarget(doc, e.target);
    if (!hit) return;
    const he = promoteToLayoutDragRoot(doc, hit);
    e.preventDefault();
    e.stopPropagation();
    const { x, y } = parseTranslate(he);
    dragging = { el: he, startX: e.clientX, startY: e.clientY, baseX: x, baseY: y };
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
    const dx = e.clientX - dragging.startX;
    const dy = e.clientY - dragging.startY;
    dragging.el.style.transform = `translate(${dragging.baseX + dx}px, ${dragging.baseY + dy}px)`;
  };

  const onPointerUp = () => {
    endDrag();
  };

  doc.addEventListener("pointerdown", onPointerDown, true);
  doc.addEventListener("pointermove", onPointerMove, true);
  doc.addEventListener("pointerup", onPointerUp, true);
  doc.addEventListener("pointercancel", onPointerUp, true);

  const win = doc.defaultView;
  win?.addEventListener("blur", endDrag);

  /** Live inline `style` updates (transform, position) don’t always deliver pointerup inside the iframe in Electron when releasing outside; observe attributes instead. */
  let dirtyFlush = 0;
  const flushDirty = () => {
    window.clearTimeout(dirtyFlush);
    dirtyFlush = window.setTimeout(() => {
      onChange();
    }, 0);
  };
  const attrObs = new MutationObserver(() => flushDirty());
  if (doc.documentElement) {
    attrObs.observe(doc.documentElement, {
      subtree: true,
      attributes: true,
      attributeFilter: ["style"],
    });
  }

  const onSaveKey = (e: KeyboardEvent) => {
    onSaveShortcut?.(e);
  };
  if (onSaveShortcut) {
    doc.addEventListener("keydown", onSaveKey, true);
  }

  return () => {
    window.clearTimeout(dirtyFlush);
    attrObs.disconnect();
    if (onSaveShortcut) {
      doc.removeEventListener("keydown", onSaveKey, true);
    }
    doc.removeEventListener("pointerdown", onPointerDown, true);
    doc.removeEventListener("pointermove", onPointerMove, true);
    doc.removeEventListener("pointerup", onPointerUp, true);
    doc.removeEventListener("pointercancel", onPointerUp, true);
    win?.removeEventListener("blur", endDrag);
    body.classList.remove("layout-mode");
    style.remove();
  };
}
