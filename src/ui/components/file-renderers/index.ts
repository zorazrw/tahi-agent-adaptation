import type { ComponentType } from "react";
import { TextRenderer } from "./TextRenderer";
import { SpreadsheetRenderer } from "./SpreadsheetRenderer";
import { DocxRenderer } from "./DocxRenderer";
import { ImageRenderer } from "./ImageRenderer";
import { PdfRenderer } from "./PdfRenderer";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { CodeRenderer } from "./CodeRenderer";
import { CsvRenderer } from "./CsvRenderer";
import { JsonRenderer } from "./JsonRenderer";
import { HtmlRenderer } from "./HtmlRenderer";
import { VideoRenderer } from "./VideoRenderer";
import { AudioRenderer } from "./AudioRenderer";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type RendererComponent = ComponentType<{ data: any }>;

const renderers: Record<string, RendererComponent> = {
  txt: TextRenderer,
  xlsx: SpreadsheetRenderer,
  docx: DocxRenderer,
  image: ImageRenderer,
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
