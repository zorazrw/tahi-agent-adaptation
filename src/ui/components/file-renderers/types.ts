export type PreviewSaveChrome = {
  save: () => void;
  disabled: boolean;
  saving: boolean;
  error: string | null;
};

export type HtmlVisualSaveChrome = PreviewSaveChrome;

export type EditableRendererProps = {
  filePath?: string;
  cwd?: string | null;
  sessionId?: string | null;
  onReload?: () => void;
  onHtmlVisualSaveChromeChange?: (chrome: PreviewSaveChrome | null) => void;
  onTextSaveChromeChange?: (chrome: PreviewSaveChrome | null) => void;
  reloadKey?: number;
};
