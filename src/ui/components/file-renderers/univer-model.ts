/**
 * Neutral XlsxModel <-> Univer IWorkbookData. This is the renderer-side half of
 * the bridge (the MAIN process never imports @univerjs). It owns all the
 * Univer-specific enum mapping.
 *
 *   modelToWorkbookData(model)        -> snapshot for univerAPI.createWorkbook
 *   workbookDataToModel(snapshot, fn) -> XlsxModel to hand back to MAIN (ExcelJS)
 */
import {
  BooleanNumber,
  WrapStrategy,
  HorizontalAlign,
  VerticalAlign,
  BorderStyleTypes,
  LocaleType,
} from "@univerjs/core";
import type {
  IWorkbookData,
  IWorksheetData,
  ICellData,
  IStyleData,
  IBorderData,
} from "@univerjs/core";
import type {
  XlsxModel,
  XlsxCell,
  XlsxBorderSide,
  XlsxHAlign,
  XlsxVAlign,
} from "../../../lib/xlsx-model";

/* helpers */

export function colToLetters(col: number): string {
  let s = "";
  let n = col;
  do {
    s = String.fromCharCode(65 + (n % 26)) + s;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return s;
}

function hashStr(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}

const H_TO_UNIVER: Record<XlsxHAlign, HorizontalAlign> = {
  left: HorizontalAlign.LEFT,
  center: HorizontalAlign.CENTER,
  right: HorizontalAlign.RIGHT,
  justify: HorizontalAlign.JUSTIFIED,
};
const V_TO_UNIVER: Record<XlsxVAlign, VerticalAlign> = {
  top: VerticalAlign.TOP,
  middle: VerticalAlign.MIDDLE,
  bottom: VerticalAlign.BOTTOM,
};
const H_FROM_UNIVER: Record<number, XlsxHAlign> = {
  [HorizontalAlign.LEFT]: "left",
  [HorizontalAlign.CENTER]: "center",
  [HorizontalAlign.RIGHT]: "right",
  [HorizontalAlign.JUSTIFIED]: "justify",
};
const V_FROM_UNIVER: Record<number, XlsxVAlign> = {
  [VerticalAlign.TOP]: "top",
  [VerticalAlign.MIDDLE]: "middle",
  [VerticalAlign.BOTTOM]: "bottom",
};

const BORDER_TO_UNIVER: Record<string, BorderStyleTypes> = {
  thin: BorderStyleTypes.THIN,
  hair: BorderStyleTypes.HAIR,
  dotted: BorderStyleTypes.DOTTED,
  dashed: BorderStyleTypes.DASHED,
  dashDot: BorderStyleTypes.DASH_DOT,
  dashDotDot: BorderStyleTypes.DASH_DOT_DOT,
  double: BorderStyleTypes.DOUBLE,
  medium: BorderStyleTypes.MEDIUM,
  mediumDashed: BorderStyleTypes.MEDIUM_DASHED,
  thick: BorderStyleTypes.THICK,
};
const BORDER_FROM_UNIVER: Record<number, string> = {
  [BorderStyleTypes.THIN]: "thin",
  [BorderStyleTypes.HAIR]: "hair",
  [BorderStyleTypes.DOTTED]: "dotted",
  [BorderStyleTypes.DASHED]: "dashed",
  [BorderStyleTypes.DASH_DOT]: "dashDot",
  [BorderStyleTypes.DASH_DOT_DOT]: "dashDotDot",
  [BorderStyleTypes.DOUBLE]: "double",
  [BorderStyleTypes.MEDIUM]: "medium",
  [BorderStyleTypes.MEDIUM_DASHED]: "mediumDashed",
  [BorderStyleTypes.THICK]: "thick",
};

/* IMPORT: XlsxModel -> IWorkbookData */

function cellToStyle(cell: XlsxCell): IStyleData | null {
  const s: IStyleData = {};
  let touched = false;
  if (cell.numFmt) { s.n = { pattern: cell.numFmt }; touched = true; }
  if (cell.bold) { s.bl = BooleanNumber.TRUE; touched = true; }
  if (cell.italic) { s.it = BooleanNumber.TRUE; touched = true; }
  if (cell.underline) { s.ul = { s: BooleanNumber.TRUE }; touched = true; }
  if (typeof cell.size === "number") { s.fs = cell.size; touched = true; }
  if (cell.fontName) { s.ff = cell.fontName; touched = true; }
  if (cell.color) { s.cl = { rgb: cell.color }; touched = true; }
  if (cell.fill) { s.bg = { rgb: cell.fill }; touched = true; }
  if (cell.wrap) { s.tb = WrapStrategy.WRAP; touched = true; }
  if (cell.hAlign) { s.ht = H_TO_UNIVER[cell.hAlign]; touched = true; }
  if (cell.vAlign) { s.vt = V_TO_UNIVER[cell.vAlign]; touched = true; }
  if (cell.border) {
    const bd: IBorderData = {};
    const side = (b?: XlsxBorderSide) =>
      b?.style
        ? { s: BORDER_TO_UNIVER[b.style] ?? BorderStyleTypes.THIN, cl: { rgb: b.color ?? "#000000" } }
        : undefined;
    const t = side(cell.border.top), b = side(cell.border.bottom), l = side(cell.border.left), r = side(cell.border.right);
    if (t) bd.t = t;
    if (b) bd.b = b;
    if (l) bd.l = l;
    if (r) bd.r = r;
    if (Object.keys(bd).length) { s.bd = bd; touched = true; }
  }
  return touched ? s : null;
}

export function modelToWorkbookData(model: XlsxModel): IWorkbookData {
  const styles: Record<string, IStyleData> = {};
  const sheets: Record<string, IWorksheetData> = {};
  const sheetOrder: string[] = [];

  model.sheets.forEach((sheet, idx) => {
    const sheetId = `sheet-${idx + 1}`;
    sheetOrder.push(sheetId);

    const cellData: Record<number, Record<number, ICellData>> = {};
    for (const cell of sheet.cells) {
      const cd: ICellData = {};
      if (cell.f) cd.f = cell.f.startsWith("=") ? cell.f : `=${cell.f}`;
      if (cell.v !== undefined && cell.v !== null) cd.v = cell.v as ICellData["v"];
      const st = cellToStyle(cell);
      if (st) {
        const id = "s_" + hashStr(JSON.stringify(st));
        styles[id] = st;
        cd.s = id;
      }
      if (cd.f === undefined && cd.v === undefined && cd.s === undefined) continue;
      (cellData[cell.r] ||= {})[cell.c] = cd;
    }

    const columnData: Record<number, { w: number }> = {};
    if (sheet.colWidths) {
      for (const [c, px] of Object.entries(sheet.colWidths)) columnData[Number(c)] = { w: px };
    }

    const ySplit = sheet.freeze?.ySplit ?? 0;
    const xSplit = sheet.freeze?.xSplit ?? 0;

    sheets[sheetId] = {
      id: sheetId,
      name: sheet.name,
      tabColor: "",
      hidden: BooleanNumber.FALSE,
      rowCount: Math.max(sheet.rowCount, 50),
      columnCount: Math.max(sheet.colCount, 20),
      zoomRatio: 1,
      scrollTop: 0,
      scrollLeft: 0,
      defaultColumnWidth: 88,
      defaultRowHeight: 24,
      mergeData: sheet.merges.map((m) => ({
        startRow: m.sr,
        startColumn: m.sc,
        endRow: m.er,
        endColumn: m.ec,
      })),
      cellData,
      rowData: {},
      columnData,
      showGridlines: BooleanNumber.TRUE,
      rowHeader: { width: 46, hidden: BooleanNumber.FALSE },
      columnHeader: { height: 20, hidden: BooleanNumber.FALSE },
      freeze: { xSplit, ySplit, startRow: ySplit, startColumn: xSplit },
      selections: ["A1"],
      rightToLeft: BooleanNumber.FALSE,
    } as unknown as IWorksheetData;
  });

  return {
    id: "wb-" + hashStr(model.fileName + model.sheets.length),
    name: model.fileName,
    appVersion: "0.24.0",
    locale: LocaleType.EN_US,
    sheetOrder,
    styles,
    sheets,
    rev: 1,
  } as IWorkbookData;
}

/* EXPORT: IWorkbookData -> XlsxModel */

function styleToCellFields(style: IStyleData | undefined, cell: XlsxCell): void {
  if (!style) return;
  if (style.n?.pattern) cell.numFmt = style.n.pattern;
  if (style.bl === BooleanNumber.TRUE) cell.bold = true;
  if (style.it === BooleanNumber.TRUE) cell.italic = true;
  if (style.ul?.s === BooleanNumber.TRUE) cell.underline = true;
  if (typeof style.fs === "number") cell.size = style.fs;
  if (style.ff) cell.fontName = style.ff;
  if (style.cl?.rgb) cell.color = normHex(style.cl.rgb);
  if (style.bg?.rgb) cell.fill = normHex(style.bg.rgb);
  if (style.tb === WrapStrategy.WRAP) cell.wrap = true;
  if (style.ht != null && H_FROM_UNIVER[style.ht]) cell.hAlign = H_FROM_UNIVER[style.ht];
  if (style.vt != null && V_FROM_UNIVER[style.vt]) cell.vAlign = V_FROM_UNIVER[style.vt];
  if (style.bd) {
    // Univer's Nullable<IBorderStyleData> widens awkwardly; read each side loosely.
    const side = (b: unknown): XlsxBorderSide | undefined => {
      const bb = b as { s?: number; cl?: { rgb?: string } } | null | undefined;
      return bb && bb.s != null
        ? { style: BORDER_FROM_UNIVER[bb.s] ?? "thin", color: bb.cl?.rgb ? normHex(bb.cl.rgb) : undefined }
        : undefined;
    };
    const sb = style.bd as { t?: unknown; b?: unknown; l?: unknown; r?: unknown };
    const t = side(sb.t), b = side(sb.b), l = side(sb.l), r = side(sb.r);
    const border: XlsxCell["border"] = {};
    if (t) border.top = t;
    if (b) border.bottom = b;
    if (l) border.left = l;
    if (r) border.right = r;
    if (Object.keys(border).length) cell.border = border;
  }
}

function normHex(rgb: string): string {
  const h = rgb.replace("#", "");
  return "#" + h.slice(-6).toUpperCase();
}

export function workbookDataToModel(snapshot: IWorkbookData, fileName: string): XlsxModel {
  const styles = (snapshot.styles ?? {}) as Record<string, IStyleData>;
  const sheets = (snapshot.sheetOrder ?? []).map((sheetId) => {
    const ws = snapshot.sheets[sheetId];
    const cells: XlsxCell[] = [];
    let maxR = 0;
    let maxC = 0;
    const cd = (ws.cellData ?? {}) as unknown as Record<string, Record<string, ICellData>>;
    for (const rKey of Object.keys(cd)) {
      const r = Number(rKey);
      for (const cKey of Object.keys(cd[rKey] ?? {})) {
        const c = Number(cKey);
        const src = cd[rKey][cKey];
        if (!src) continue;
        const cell: XlsxCell = { r, c };
        if (src.f != null && src.f !== "") cell.f = String(src.f).replace(/^=/, "");
        if (src.v !== undefined && src.v !== null && src.v !== "") cell.v = src.v as XlsxCell["v"];
        const st = typeof src.s === "string" ? styles[src.s] : (src.s as IStyleData | undefined);
        styleToCellFields(st, cell);
        const hasContent = cell.v !== undefined || cell.f !== undefined;
        const hasStyle = cell.numFmt || cell.bold || cell.italic || cell.underline || cell.size || cell.fontName || cell.color || cell.fill || cell.wrap || cell.hAlign || cell.vAlign || cell.border;
        if (!hasContent && !hasStyle) continue;
        cells.push(cell);
        if (r > maxR) maxR = r;
        if (c > maxC) maxC = c;
      }
    }

    const merges = (ws.mergeData ?? []).map((m) => ({
      sr: m.startRow, sc: m.startColumn, er: m.endRow, ec: m.endColumn,
    }));
    for (const m of merges) { if (m.er > maxR) maxR = m.er; if (m.ec > maxC) maxC = m.ec; }

    const fz = (ws as { freeze?: { xSplit?: number; ySplit?: number } }).freeze;
    const freeze = fz && (fz.xSplit || fz.ySplit) ? { xSplit: fz.xSplit ?? 0, ySplit: fz.ySplit ?? 0 } : undefined;

    const colWidths: Record<number, number> = {};
    const colData = (ws.columnData ?? {}) as unknown as Record<string, { w?: number }>;
    for (const c of Object.keys(colData)) {
      if (typeof colData[c]?.w === "number") colWidths[Number(c)] = colData[c].w as number;
    }

    return {
      name: ws.name ?? sheetId,
      rowCount: Math.max(maxR + 1, 1),
      colCount: Math.max(maxC + 1, 1),
      cells,
      merges,
      freeze,
      colWidths: Object.keys(colWidths).length ? colWidths : undefined,
    };
  });

  return { fileName, sheets };
}
