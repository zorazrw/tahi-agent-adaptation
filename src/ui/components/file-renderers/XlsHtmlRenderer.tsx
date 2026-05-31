import { useState } from "react";

type SheetData = { name: string; html: string };

/**
 * Read-only HTML rendering for legacy `.xls` (and as a fallback when ExcelJS
 * cannot parse an `.xlsx`). `.xlsx` uses the editable Univer renderer instead.
 */
export function XlsHtmlRenderer({ data }: { data: { kind: "xls"; sheets: SheetData[] } }) {
  const [activeIndex, setActiveIndex] = useState(0);
  const sheets = data.sheets;
  const activeSheet = sheets[activeIndex];

  if (!activeSheet) {
    return <p className="text-sm text-muted-foreground p-2">No sheets found</p>;
  }

  return (
    <div className="spreadsheet-preview flex flex-col flex-1 overflow-hidden">
      {sheets.length > 1 && (
        <div className="flex gap-0.5 border-b border-ink-900/10 px-1 pt-1 overflow-x-auto shrink-0">
          {sheets.map((sheet, i) => (
            <button
              key={i}
              onClick={() => setActiveIndex(i)}
              className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors whitespace-nowrap ${
                i === activeIndex
                  ? "bg-surface text-ink-900 border border-b-0 border-ink-900/10"
                  : "text-ink-500 hover:text-ink-700 hover:bg-surface-secondary/50"
              }`}
            >
              {sheet.name}
            </button>
          ))}
        </div>
      )}
      <div
        className="flex-1 overflow-auto"
        dangerouslySetInnerHTML={{ __html: activeSheet.html }}
      />
    </div>
  );
}
