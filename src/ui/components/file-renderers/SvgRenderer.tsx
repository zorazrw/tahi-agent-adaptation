import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2Icon } from "lucide-react";
import { validateSvg } from "../../../lib/svg-tools";
import type { EditableRendererProps, PreviewSaveChrome } from "./index";

type Props = { data: { kind: "svg"; content: string } } & EditableRendererProps;

type EditorMessage = {
  type?: unknown;
  channel?: unknown;
  svg?: unknown;
};

export function SvgRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  onReload,
  onTextSaveChromeChange,
}: Props) {
  const frameRef = useRef<HTMLIFrameElement>(null);
  const svgRef = useRef(data.content);
  const baselineRef = useRef(data.content);
  const [ready, setReady] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const channel = useMemo(() => crypto.randomUUID(), []);
  const canEdit = Boolean(filePath && onReload);

  const sendSvg = useCallback((svg: string) => {
    frameRef.current?.contentWindow?.postMessage(
      { type: "agent-cowork:set-svg", channel, svg },
      "*"
    );
  }, [channel]);

  const save = useCallback(async () => {
    if (!filePath || !onReload || saving || !dirty) return;
    setSaving(true);
    setError(null);
    try {
      const content = validateSvg(svgRef.current);
      const result = await window.electron.writeFile(
        filePath,
        cwd ?? undefined,
        content,
        sessionId ?? undefined
      );
      if (!result.success) throw new Error(result.error || "Could not save SVG");
      setDirty(false);
      onReload();
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : String(saveError));
    } finally {
      setSaving(false);
    }
  }, [cwd, dirty, filePath, onReload, saving, sessionId]);

  useEffect(() => {
    svgRef.current = data.content;
    baselineRef.current = data.content;
    setDirty(false);
    setError(null);
    if (ready) sendSvg(data.content);
  }, [data.content, ready, sendSvg]);

  useEffect(() => {
    const receiveMessage = (event: MessageEvent<EditorMessage>) => {
      if (event.source !== frameRef.current?.contentWindow || event.data?.channel !== channel) return;
      if (event.data.type === "agent-cowork:svg-ready") {
        setReady(true);
        return;
      }
      if (event.data.type === "agent-cowork:svg-loaded" && typeof event.data.svg === "string") {
        try {
          const loadedSvg = validateSvg(event.data.svg);
          svgRef.current = loadedSvg;
          baselineRef.current = loadedSvg;
          setDirty(false);
          setError(null);
        } catch (loadError) {
          setError(loadError instanceof Error ? loadError.message : String(loadError));
        }
        return;
      }
      if (event.data.type !== "agent-cowork:svg-changed" || typeof event.data.svg !== "string") return;
      try {
        const nextSvg = validateSvg(event.data.svg);
        svgRef.current = nextSvg;
        setDirty(nextSvg !== baselineRef.current);
        setError(null);
      } catch (changeError) {
        setError(changeError instanceof Error ? changeError.message : String(changeError));
      }
    };

    window.addEventListener("message", receiveMessage);
    return () => window.removeEventListener("message", receiveMessage);
  }, [channel, data.content, sendSvg]);

  useEffect(() => {
    if (!onTextSaveChromeChange) return;
    const chrome: PreviewSaveChrome = {
      save: () => void save(),
      disabled: !canEdit || !dirty || saving,
      saving,
      error,
    };
    onTextSaveChromeChange(chrome);
    return () => onTextSaveChromeChange(null);
  }, [canEdit, dirty, error, onTextSaveChromeChange, save, saving]);

  useEffect(() => {
    const flushSave = () => void save();
    window.addEventListener("preview-flush-save", flushSave);
    return () => window.removeEventListener("preview-flush-save", flushSave);
  }, [save]);

  return (
    <div className="relative flex min-h-0 flex-1 overflow-hidden bg-[#d8d1c3]">
      <iframe
        ref={frameRef}
        src={`./svgedit/index.html#${encodeURIComponent(channel)}`}
        title="Interactive SVG editor"
        className="h-full w-full border-0 bg-[#d8d1c3]"
      />
      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center gap-2 bg-[#d8d1c3] text-xs font-medium text-ink-600">
          <Loader2Icon className="size-4 animate-spin" />
          Loading SVG editor
        </div>
      )}
      {error && (
        <div className="absolute bottom-3 left-1/2 max-w-[calc(100%-1.5rem)] -translate-x-1/2 rounded-md border border-error/20 bg-white/95 px-3 py-2 text-xs text-error shadow-md">
          {error}
        </div>
      )}
    </div>
  );
}
