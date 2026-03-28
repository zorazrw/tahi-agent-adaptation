import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";

type Props = { data: { kind: "txt"; content: string } } & EditableRendererProps;

export function TextRenderer({ data, filePath, cwd, sessionId, onReload }: Props) {
  const canEdit = Boolean(filePath && onReload);

  if (canEdit) {
    return (
      <EditableTextPanel
        content={data.content}
        contentKey={data.content}
        monospace
        onSave={async (content) =>
          window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined)
        }
        onSaved={onReload}
      />
    );
  }

  return (
    <pre className="text-sm text-ink-700 whitespace-pre-wrap break-words font-mono">
      {data.content}
    </pre>
  );
}
