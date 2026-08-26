import type { ComponentType } from "react";
import { TextRenderer } from "./TextRenderer";
import { SpreadsheetRenderer } from "./SpreadsheetRenderer";
import { XlsHtmlRenderer } from "./XlsHtmlRenderer";
import { DocxRenderer } from "./DocxRenderer";
import { ImageRenderer } from "./ImageRenderer";
import { SvgRenderer } from "./SvgRenderer";
import { PdfRenderer } from "./PdfRenderer";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { CodeRenderer } from "./CodeRenderer";
import { CsvRenderer } from "./CsvRenderer";
import { JsonRenderer } from "./JsonRenderer";
import { HtmlRenderer } from "./HtmlRenderer";
import { VideoRenderer } from "./VideoRenderer";
import { AudioRenderer } from "./AudioRenderer";

/** Preview header Save affordance (HTML visual edit or text/source editors). */
export type PreviewSaveChrome = {
  save: () => void;
  disabled: boolean;
  saving: boolean;
  error: string | null;
};

/** @deprecated use PreviewSaveChrome */
export type HtmlVisualSaveChrome = PreviewSaveChrome;

export type EditableRendererProps = {
  filePath?: string;
  cwd?: string | null;
  /** When set, successful saves record a ``file_edit`` message for this task session. */
  sessionId?: string | null;
  onReload?: () => void;
  /** HTML preview only: fired when visual edit (Text/Shape) save affordance should appear or clear. */
  onHtmlVisualSaveChromeChange?: (chrome: PreviewSaveChrome | null) => void;
  /** Markdown/code/text source editors: header Save + records ``file_edit`` on save. */
  onTextSaveChromeChange?: (chrome: PreviewSaveChrome | null) => void;
  /** Bumps when the user hits Refresh so renderers can force-remount (e.g. HTML iframe). */
  reloadKey?: number;
};

type RendererComponent = ComponentType<
  // Renderer data is discriminated by `kind`; the registry keeps the shared chrome props typed.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  { data: any; zoom?: number } & EditableRendererProps
>;

const renderers: Record<string, RendererComponent> = {
  txt: TextRenderer,
  xlsx: SpreadsheetRenderer,
  xls: XlsHtmlRenderer,
  docx: DocxRenderer,
  image: ImageRenderer,
  svg: SvgRenderer,
  pdf: PdfRenderer,
  md: MarkdownRenderer,
  code: CodeRenderer,
  csv: CsvRenderer,
  json: JsonRenderer,
  html: HtmlRenderer,
  video: VideoRenderer,
  audio: AudioRenderer,
};

export function getRenderer(kind: string): RendererComponent | null {
  return renderers[kind] ?? null;
}
