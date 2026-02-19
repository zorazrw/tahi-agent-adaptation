import { useMemo } from "react";

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

export function CsvRenderer({ data }: { data: { kind: "csv"; content: string } }) {
  const rows = useMemo(() => parseCsv(data.content), [data.content]);

  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">Empty file</p>;
  }

  const header = rows[0]!;
  const body = rows.slice(1);

  return (
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
  );
}
