import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";
import { PlainTextReadingView, TextDocumentFrame } from "./TextDocumentFrame";

type Props = { data: { kind: "txt"; content: string } } & EditableRendererProps;

export function TextRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  onReload,
  onTextSaveChromeChange,
}: Props) {
  const canEdit = Boolean(filePath && onReload);

  if (canEdit) {
    return (
      <TextDocumentFrame label="Plain text">
        <EditableTextPanel
          content={data.content}
          contentKey={data.content}
          monospace
          variant="document"
          onSave={async (content) =>
            window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined)
          }
          onSaved={onReload}
          onSaveChromeChange={onTextSaveChromeChange}
        />
      </TextDocumentFrame>
    );
  }

  return (
    <TextDocumentFrame label="Plain text">
      <PlainTextReadingView content={data.content} />
    </TextDocumentFrame>
  );
}
