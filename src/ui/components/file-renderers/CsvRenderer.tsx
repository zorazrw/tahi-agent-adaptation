import { useState, useMemo, useEffect } from "react";
import type { EditableRendererProps } from "./index";
import { EditableTextPanel } from "./EditableTextPanel";

function parseCsv(content: string): string[][] {
  const lines = content.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) return [];

  // Auto-detect delimiter: if tabs are more common than commas, use tab
  const firstLine = lines[0]!;
  const delimiter = (firstLine.split("\t").length > firstLine.split(",").length) ? "\t" : ",";

  return lines.map((line) => {
    const cells: string[] = [];
    let current = "";
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const ch = line[i]!;
      if (inQuotes) {
        if (ch === '"' && line[i + 1] === '"') {
          current += '"';
          i++;
        } else if (ch === '"') {
          inQuotes = false;
        } else {
          current += ch;
        }
      } else {
        if (ch === '"') {
          inQuotes = true;
        } else if (ch === delimiter) {
          cells.push(current);
          current = "";
        } else {
          current += ch;
        }
      }
    }
    cells.push(current);
    return cells;
  });
}

type ViewMode = "table" | "text";

export function CsvRenderer({
  data,
  filePath,
  cwd,
  sessionId,
  onReload,
  onTextSaveChromeChange,
}: { data: { kind: "csv"; content: string } } & EditableRendererProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("table");
  const rows = useMemo(() => parseCsv(data.content), [data.content]);
  const canEdit = Boolean(filePath && onReload);

  useEffect(() => {
    if (!canEdit) setViewMode("table");
  }, [canEdit]);

  if (canEdit && viewMode === "text") {
    return (
      <div className="flex flex-col flex-1 min-h-0">
        <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
          <button
            type="button"
            onClick={() => setViewMode("table")}
            className="text-xs text-muted-foreground hover:text-ink-700"
          >
            ← Table
          </button>
          <span className="text-xs text-muted-foreground">Edit as text</span>
        </div>
        <EditableTextPanel
          content={data.content}
          contentKey={data.content}
          monospace
          onSave={async (content) =>
            window.electron.writeFile(filePath!, cwd ?? undefined, content, sessionId ?? undefined)
          }
          onSaved={onReload}
          onSaveChromeChange={onTextSaveChromeChange}
        />
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="flex flex-col flex-1 min-h-0">
        {canEdit && (
          <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2">
            <button
              type="button"
              onClick={() => setViewMode("text")}
              className="text-xs text-muted-foreground hover:text-ink-700"
            >
              Edit as text
            </button>
          </div>
        )}
        <p className="text-sm text-muted-foreground">Empty file</p>
      </div>
    );
  }

  const header = rows[0]!;
  const body = rows.slice(1);

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {canEdit && (
        <div className="flex items-center gap-2 pb-2 border-b border-ink-900/10 mb-2 shrink-0">
          <button
            type="button"
            onClick={() => setViewMode("text")}
            className="text-xs text-muted-foreground hover:text-ink-700"
          >
            Edit as text
          </button>
        </div>
      )}
      <div className="overflow-auto">
        <table className="text-sm text-ink-700 border-collapse border border-ink-900/20">
          <thead>
            <tr>
              {header.map((cell, j) => (
                <th
                  key={j}
                  className="border border-ink-900/20 px-2 py-1.5 text-left font-semibold bg-ink-900/5"
                >
                  {cell}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j} className="border border-ink-900/20 px-2 py-1.5 align-top">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
