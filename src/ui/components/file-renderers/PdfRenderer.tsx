import { useState, useRef, useEffect, useCallback } from "react";
import * as pdfjsLib from "pdfjs-dist";

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.mjs",
  import.meta.url
).toString();

const MIN_SCALE = 0.25;
const MAX_SCALE = 3;
const ZOOM_STEP = 0.25;

export function PdfRenderer({ data }: { data: { kind: "pdf"; data: string } }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [numPages, setNumPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [scale, setScale] = useState(1);
  const [loading, setLoading] = useState(true);
  const pdfDocRef = useRef<pdfjsLib.PDFDocumentProxy | null>(null);

  const clampScale = (s: number) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));

  // Load the PDF document once
  useEffect(() => {
    let cancelled = false;

    const raw = atob(data.data);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

    pdfjsLib.getDocument({ data: bytes }).promise.then((doc) => {
      if (cancelled) return;
      pdfDocRef.current = doc;
      setNumPages(doc.numPages);
      setCurrentPage(1);
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [data.data]);

  // Render the current page at the current scale
  useEffect(() => {
    const doc = pdfDocRef.current;
    if (!doc || !canvasRef.current) return;
    let cancelled = false;

    doc.getPage(currentPage).then((page) => {
      if (cancelled || !canvasRef.current) return;
      const viewport = page.getViewport({ scale });
      const canvas = canvasRef.current;
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      page.render({ canvas, viewport });
    });

    return () => {
      cancelled = true;
    };
  }, [currentPage, scale, numPages]);

  const prevPage = useCallback(() => setCurrentPage((p) => Math.max(1, p - 1)), []);
  const nextPage = useCallback(() => setCurrentPage((p) => Math.min(numPages, p + 1)), [numPages]);
  const zoomIn = () => setScale((s) => clampScale(s + ZOOM_STEP));
  const zoomOut = () => setScale((s) => clampScale(s - ZOOM_STEP));

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
        <span>Loading PDF…</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Toolbar */}
      <div className="shrink-0 flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2">
        {/* Page navigation */}
        <button
          onClick={prevPage}
          disabled={currentPage <= 1}
          className="px-2 py-0.5 text-xs rounded border border-ink-900/20 hover:bg-ink-900/5 disabled:opacity-40"
        >
          ‹ Prev
        </button>
        <span className="text-xs text-muted-foreground">
          {currentPage} / {numPages}
        </span>
        <button
          onClick={nextPage}
          disabled={currentPage >= numPages}
          className="px-2 py-0.5 text-xs rounded border border-ink-900/20 hover:bg-ink-900/5 disabled:opacity-40"
        >
          Next ›
        </button>

        <div className="w-px h-4 bg-ink-900/20 mx-1" />

        {/* Zoom controls */}
        <button
          onClick={zoomOut}
          disabled={scale <= MIN_SCALE}
          className="px-2 py-0.5 text-xs rounded border border-ink-900/20 hover:bg-ink-900/5 disabled:opacity-40"
        >
          −
        </button>
        <span className="text-xs text-muted-foreground w-12 text-center">{Math.round(scale * 100)}%</span>
        <button
          onClick={zoomIn}
          disabled={scale >= MAX_SCALE}
          className="px-2 py-0.5 text-xs rounded border border-ink-900/20 hover:bg-ink-900/5 disabled:opacity-40"
        >
          +
        </button>
      </div>

      {/* Canvas area */}
      <div className="flex-1 min-h-0 overflow-auto flex justify-center">
        <canvas ref={canvasRef} className="max-w-full" />
      </div>
    </div>
  );
}
