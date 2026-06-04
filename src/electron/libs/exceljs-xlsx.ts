/**
 * ExcelJS <-> neutral XlsxModel bridge. Runs in the Electron MAIN process
 * (the renderer is sandboxed, so all .xlsx byte I/O lives here).
 *
 *   fileToModel(buf)   .xlsx bytes -> XlsxModel        (read, full styles)
 *   modelToBytes(model) XlsxModel  -> .xlsx bytes      (write, full styles)
 *   modelToText(model)  XlsxModel  -> TSV              (semantic capture text)
 *
 * Unlike the SheetJS community build, ExcelJS reads AND writes fonts, fills,
 * borders, alignment, number formats, merges and frozen panes, so styling
 * round-trips. ExcelJS has no formula calculator; the renderer's Univer engine
 * recomputes, and we persist the computed value as each formula's cached result.
 */
import ExcelJS from "exceljs";
import type {
  XlsxModel,
  XlsxSheet,
  XlsxCell,
  XlsxMerge,
  XlsxBorders,
  XlsxBorderSide,
  XlsxHAlign,
  XlsxVAlign,
} from "../../lib/xlsx-model.js";

/* helpers */

/** ExcelJS colors are AARRGGBB; emit `#RRGGBB`. */
function argbToHex(argb?: string): string | undefined {
  if (!argb || typeof argb !== "string" || argb.length < 6) return undefined;
  return "#" + argb.slice(-6).toUpperCase();
}

/** `#RRGGBB` -> `FFRRGGBB` (opaque ARGB for ExcelJS). */
function hexToArgb(hex?: string): string | undefined {
  if (!hex) return undefined;
  const h = hex.replace("#", "");
  if (h.length === 6) return "FF" + h.toUpperCase();
  if (h.length === 8) return h.toUpperCase();
  return undefined;
}

function colLettersToIdx(letters: string): number {
  let n = 0;
  for (const ch of letters) n = n * 26 + (ch.charCodeAt(0) - 64);
  return n - 1;
}

function parseA1Range(s: string): XlsxMerge | null {
  const m = /^([A-Z]+)(\d+):([A-Z]+)(\d+)$/.exec(s);
  if (!m) return null;
  return {
    sr: Number(m[2]) - 1,
    sc: colLettersToIdx(m[1]),
    er: Number(m[4]) - 1,
    ec: colLettersToIdx(m[3]),
  };
}

/** Excel "character" column width <-> approximate pixels. */
const widthToPx = (w: number) => Math.round(w * 7 + 5);
const pxToWidth = (px: number) => Math.max(1, (px - 5) / 7);

function normalizeResult(
  r: unknown
): string | number | boolean | null | undefined {
  if (r == null) return undefined;
  if (typeof r === "number" || typeof r === "boolean" || typeof r === "string") return r;
  if (r instanceof Date) return r.toISOString();
  if (typeof r === "object" && "error" in (r as object)) return String((r as { error: unknown }).error);
  return undefined;
}

/* READ: .xlsx -> XlsxModel */

function readCellStyle(cell: ExcelJS.Cell, out: XlsxCell): void {
  if (cell.numFmt && cell.numFmt !== "General") out.numFmt = cell.numFmt;

  const font = cell.font;
  if (font) {
    if (font.bold) out.bold = true;
    if (font.italic) out.italic = true;
    if (font.underline) out.underline = true;
    if (typeof font.size === "number") out.size = font.size;
    if (font.name) out.fontName = font.name;
    const fc = argbToHex(font.color?.argb);
    if (fc) out.color = fc;
  }

  const fill = cell.fill;
  if (fill && fill.type === "pattern" && fill.pattern === "solid") {
    const bg = argbToHex(fill.fgColor?.argb);
    if (bg) out.fill = bg;
  }

  const al = cell.alignment;
  if (al) {
    if (al.wrapText) out.wrap = true;
    if (al.horizontal && ["left", "center", "right", "justify"].includes(al.horizontal))
      out.hAlign = al.horizontal as XlsxHAlign;
    if (al.vertical) {
      const v = al.vertical === "middle" ? "middle" : al.vertical === "top" ? "top" : al.vertical === "bottom" ? "bottom" : undefined;
      if (v) out.vAlign = v as XlsxVAlign;
    }
  }

  const b = cell.border;
  if (b) {
    const side = (s?: Partial<ExcelJS.Border>): XlsxBorderSide | undefined => {
      if (!s?.style) return undefined;
      return { style: String(s.style), color: argbToHex(s.color?.argb) };
    };
    const borders: XlsxBorders = {};
    const t = side(b.top), bo = side(b.bottom), l = side(b.left), r = side(b.right);
    if (t) borders.top = t;
    if (bo) borders.bottom = bo;
    if (l) borders.left = l;
    if (r) borders.right = r;
    if (Object.keys(borders).length) out.border = borders;
  }
}

export async function fileToModel(buf: Buffer | ArrayBuffer | Uint8Array, fileName = "workbook.xlsx"): Promise<XlsxModel> {
  const wb = new ExcelJS.Workbook();
  // ExcelJS accepts a Node Buffer; coerce ArrayBuffer/Uint8Array.
  const data = Buffer.isBuffer(buf) ? buf : Buffer.from(buf as ArrayBuffer);
  // ExcelJS bundles its own (narrower) Buffer type that clashes with @types/node's
  // Buffer<ArrayBufferLike>; the value is correct, so cast past the type mismatch.
  await wb.xlsx.load(data as unknown as ArrayBuffer);

  const sheets: XlsxSheet[] = [];
  wb.eachSheet((ws) => {
    const cells: XlsxCell[] = [];
    let maxR = 0;
    let maxC = 0;

    ws.eachRow({ includeEmpty: false }, (row, rowNumber) => {
      row.eachCell({ includeEmpty: false }, (cell, colNumber) => {
        const r = rowNumber - 1;
        const c = colNumber - 1;
        const out: XlsxCell = { r, c };

        if (cell.type === ExcelJS.ValueType.Formula) {
          // `cell.formula` resolves shared formulas to a string; result is cached.
          out.f = cell.formula;
          const v = normalizeResult((cell.value as { result?: unknown })?.result);
          if (v !== undefined) out.v = v;
        } else {
          const val = cell.value;
          if (val != null) {
            if (typeof val === "object") {
              if ("richText" in val) out.v = (val.richText as Array<{ text: string }>).map((t) => t.text).join("");
              else if ("hyperlink" in val) out.v = String((val as { text?: string }).text ?? (val as { hyperlink?: string }).hyperlink ?? "");
              else if (val instanceof Date) out.v = cell.text;
              else out.v = cell.text;
            } else {
              out.v = val as string | number | boolean;
            }
          }
        }

        readCellStyle(cell, out);

        // keep only cells that carry value, formula, or any style
        const hasContent = out.v !== undefined || out.f !== undefined;
        const hasStyle = out.numFmt || out.bold || out.italic || out.underline || out.size || out.fontName || out.color || out.fill || out.wrap || out.hAlign || out.vAlign || out.border;
        if (!hasContent && !hasStyle) return;

        cells.push(out);
        if (r > maxR) maxR = r;
        if (c > maxC) maxC = c;
      });
    });

    // merges
    const mergeStrings: string[] = (ws.model as { merges?: string[] }).merges ?? [];
    const merges: XlsxMerge[] = mergeStrings.map(parseA1Range).filter((m): m is XlsxMerge => m != null);
    for (const m of merges) {
      if (m.er > maxR) maxR = m.er;
      if (m.ec > maxC) maxC = m.ec;
    }

    // frozen panes
    let freeze: { xSplit: number; ySplit: number } | undefined;
    const view = ws.views?.[0];
    if (view && view.state === "frozen") {
      const xSplit = Number(view.xSplit) || 0;
      const ySplit = Number(view.ySplit) || 0;
      if (xSplit || ySplit) freeze = { xSplit, ySplit };
    }

    // column widths (best-effort)
    const colWidths: Record<number, number> = {};
    ws.columns?.forEach((col, i) => {
      if (col && typeof col.width === "number") colWidths[i] = widthToPx(col.width);
    });

    sheets.push({
      name: ws.name,
      rowCount: Math.max(maxR + 1, 50),
      colCount: Math.max(maxC + 1, 20),
      cells,
      merges,
      freeze,
      colWidths: Object.keys(colWidths).length ? colWidths : undefined,
    });
  });

  return { fileName, sheets };
}

/* WRITE: XlsxModel -> .xlsx bytes */

export async function modelToBytes(model: XlsxModel): Promise<Buffer> {
  const wb = new ExcelJS.Workbook();
  wb.creator = "agent-cowork";

  for (const sheet of model.sheets) {
    const ws = wb.addWorksheet(sheet.name);

    for (const cell of sheet.cells) {
      const xc = ws.getCell(cell.r + 1, cell.c + 1);

      if (cell.f) {
        xc.value = cell.v !== undefined && cell.v !== null
          ? ({ formula: cell.f, result: cell.v as ExcelJS.CellValue } as ExcelJS.CellFormulaValue)
          : ({ formula: cell.f } as ExcelJS.CellFormulaValue);
      } else if (cell.v !== undefined && cell.v !== null) {
        xc.value = cell.v as ExcelJS.CellValue;
      }

      if (cell.numFmt) xc.numFmt = cell.numFmt;

      if (cell.bold || cell.italic || cell.underline || cell.size || cell.fontName || cell.color) {
        xc.font = {
          bold: cell.bold || undefined,
          italic: cell.italic || undefined,
          underline: cell.underline || undefined,
          size: cell.size,
          name: cell.fontName,
          color: cell.color ? { argb: hexToArgb(cell.color) } : undefined,
        };
      }

      if (cell.fill) {
        xc.fill = { type: "pattern", pattern: "solid", fgColor: { argb: hexToArgb(cell.fill) } };
      }

      if (cell.wrap || cell.hAlign || cell.vAlign) {
        xc.alignment = {
          wrapText: cell.wrap || undefined,
          horizontal: cell.hAlign,
          vertical: cell.vAlign,
        };
      }

      if (cell.border) {
        const toBorder = (s?: XlsxBorderSide): Partial<ExcelJS.Border> | undefined =>
          s?.style ? { style: s.style as ExcelJS.BorderStyle, color: s.color ? { argb: hexToArgb(s.color) } : undefined } : undefined;
        xc.border = {
          top: toBorder(cell.border.top),
          bottom: toBorder(cell.border.bottom),
          left: toBorder(cell.border.left),
          right: toBorder(cell.border.right),
        };
      }
    }

    // merges (after values; ExcelJS keeps the top-left value)
    for (const m of sheet.merges) {
      try {
        ws.mergeCells(m.sr + 1, m.sc + 1, m.er + 1, m.ec + 1);
      } catch {
        /* overlapping/duplicate merge: skip */
      }
    }

    // frozen panes
    if (sheet.freeze && (sheet.freeze.xSplit || sheet.freeze.ySplit)) {
      ws.views = [{ state: "frozen", xSplit: sheet.freeze.xSplit, ySplit: sheet.freeze.ySplit }];
    }

    // column widths
    if (sheet.colWidths) {
      for (const [c, px] of Object.entries(sheet.colWidths)) {
        ws.getColumn(Number(c) + 1).width = pxToWidth(px);
      }
    }
  }

  const out = await wb.xlsx.writeBuffer();
  return Buffer.isBuffer(out) ? out : Buffer.from(out as ArrayBuffer);
}

/* CAPTURE: XlsxModel -> TSV text */

/**
 * Deterministic textual rendering of the workbook. Recorded as the `file_edit`
 * snapshot content so the existing text-diff verifier pipeline
 * (`gatherHumanFileEditDiffs`) yields cell-level deltas across saves, and so
 * the agent sees readable cell content instead of an opaque binary blob.
 */
export function modelToText(model: XlsxModel): string {
  const parts: string[] = [];
  for (const sheet of model.sheets) {
    parts.push(`# Sheet: ${sheet.name}`);
    // build a dense grid over the used range
    let maxR = 0;
    let maxC = 0;
    for (const cell of sheet.cells) {
      if (cell.r > maxR) maxR = cell.r;
      if (cell.c > maxC) maxC = cell.c;
    }
    const grid: string[][] = Array.from({ length: maxR + 1 }, () => Array.from({ length: maxC + 1 }, () => ""));
    for (const cell of sheet.cells) {
      const display = cell.f ? `=${cell.f}` : cell.v == null ? "" : String(cell.v);
      grid[cell.r][cell.c] = display;
    }
    for (const row of grid) parts.push(row.join("\t"));
    parts.push("");
  }
  return parts.join("\n");
}
