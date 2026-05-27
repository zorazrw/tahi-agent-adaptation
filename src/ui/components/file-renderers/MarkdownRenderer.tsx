import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import { ViewToggle, useViewToggle } from "./ViewToggle";
import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";
import { PlainTextReadingView, TextDocumentFrame } from "./TextDocumentFrame";

type Props = { data: { kind: "md"; content: string } } & EditableRendererProps;

const remarkPlugins = [remarkGfm, remarkMath];
const rehypePlugins = [rehypeKatex, rehypeHighlight, rehypeRaw];

export function MarkdownRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  onReload,
  onTextSaveChromeChange,
}: Props) {
  const canEdit = Boolean(filePath && onReload);
  const [mode, setMode] = useViewToggle("preview");
  /** Single draft shared by preview + source so toggling modes stays in sync before save. */
  const [draft, setDraft] = useState(data.content);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const isDirty = canEdit && draft !== data.content;

  useEffect(() => {
    setDraft(data.content);
  }, [data.content]);

  const saveDraft = useCallback(async (content: string) => {
    if (!canEdit || !filePath) return { success: false, error: "File is not editable" };
    setSaveError(null);
    setSaving(true);
    try {
      const res = await window.electron.writeFile(filePath, cwd ?? undefined, content, sessionId ?? undefined);
      if (res.success) {
        onReload?.();
      } else {
        setSaveError(res.error ?? "Failed to save");
      }
      return res;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setSaveError(msg);
      return { success: false, error: msg };
    } finally {
      setSaving(false);
    }
  }, [canEdit, filePath, cwd, sessionId, onReload]);

  useEffect(() => {
    if (!onTextSaveChromeChange || !canEdit) return;
    onTextSaveChromeChange({
      save: () => {
        void saveDraft(draft);
      },
      disabled: saving || !isDirty,
      saving,
      error: saveError,
    });
    return () => onTextSaveChromeChange(null);
  }, [onTextSaveChromeChange, canEdit, saveDraft, draft, saving, isDirty, saveError]);

  const toolbar = <ViewToggle mode={mode} onChange={setMode} />;

  return (
    <TextDocumentFrame toolbar={toolbar} label="Markdown">
      {mode === "preview" ? (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="md-prose">
            <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins}>
              {draft}
            </ReactMarkdown>
          </div>
        </div>
      ) : canEdit ? (
        <EditableTextPanel
          content={draft}
          contentKey={data.content}
          monospace={false}
          variant="document"
          onContentChange={setDraft}
          onSave={saveDraft}
        />
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto">
          <PlainTextReadingView content={draft} />
        </div>
      )}
    </TextDocumentFrame>
  );
}
