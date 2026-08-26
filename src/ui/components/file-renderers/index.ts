import type { ComponentType } from "react";
import type { EditableRendererProps } from "./types";
export type { EditableRendererProps, HtmlVisualSaveChrome, PreviewSaveChrome } from "./types";
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
