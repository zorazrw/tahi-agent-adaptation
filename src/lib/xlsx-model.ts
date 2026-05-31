/**
 * Neutral, plain-JSON spreadsheet model: the IPC payload between the Electron
 * MAIN process (which talks ExcelJS) and the RENDERER (which talks Univer).
 *
 * Why a neutral model instead of passing a Univer snapshot directly:
 *   - The renderer is sandboxed (no Node/fs), so ExcelJS must run in MAIN.
 *   - MAIN must stay free of any @univerjs dependency (bundle + version churn),
 *     so it never touches Univer enums/types. It only produces/consumes this
 *     plain shape.
 *   - The renderer owns the Univer-specific mapping (the real enums) in
 *     `univer-model.ts`.
 *
 * Everything here is JSON-serializable (crosses the contextBridge cleanly).
 * Coordinates are 0-based (row `r`, column `c`) to match Univer's cell matrix.
 */

export type XlsxHAlign = "left" | "center" | "right" | "justify";
export type XlsxVAlign = "top" | "middle" | "bottom";

/** One border edge. `style` uses ExcelJS border-style names (thin/medium/thick/etc.). */
export interface XlsxBorderSide {
  style?: string;
  /** `#RRGGBB` */
  color?: string;
}

export interface XlsxBorders {
  top?: XlsxBorderSide;
  bottom?: XlsxBorderSide;
  left?: XlsxBorderSide;
  right?: XlsxBorderSide;
}

export interface XlsxCell {
  r: number;
  c: number;
  /** Literal value (kept as-is for numbers/strings/booleans). Dates arrive as display text. */
  v?: string | number | boolean | null;
  /** Formula WITHOUT the leading "=" (e.g. `SUM(B4:B8)`). */
  f?: string;
  /** Number-format code (e.g. `"$"#,##0`, `0.0%`). */
  numFmt?: string;
  bold?: boolean;
  italic?: boolean;
  underline?: boolean;
  /** Font size in points. */
  size?: number;
  fontName?: string;
  /** Font color `#RRGGBB`. */
  color?: string;
  /** Solid fill color `#RRGGBB`. */
  fill?: string;
  wrap?: boolean;
  hAlign?: XlsxHAlign;
  vAlign?: XlsxVAlign;
  border?: XlsxBorders;
}

export interface XlsxMerge {
  /** start row/col (0-based, inclusive) */
  sr: number;
  sc: number;
  /** end row/col (0-based, inclusive) */
  er: number;
  ec: number;
}

export interface XlsxSheet {
  name: string;
  rowCount: number;
  colCount: number;
  cells: XlsxCell[];
  merges: XlsxMerge[];
  /** Frozen panes: number of frozen columns / rows from the top-left. */
  freeze?: { xSplit: number; ySplit: number };
  /** Column index (0-based) -> width in px (best-effort). */
  colWidths?: Record<number, number>;
  /** Row index (0-based) -> height in px (best-effort). */
  rowHeights?: Record<number, number>;
}

export interface XlsxModel {
  /** Original file basename, for display + the export filename. */
  fileName: string;
  sheets: XlsxSheet[];
}
