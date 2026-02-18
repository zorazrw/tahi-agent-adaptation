export function SpreadsheetRenderer({ data }: { data: { kind: "xlsx"; data: unknown[][] } }) {
  return (
    <div className="overflow-auto">
      <table className="text-sm text-ink-700 border-collapse border border-ink-900/20">
        <tbody>
          {data.data.map((row: unknown[], i: number) => (
            <tr key={i}>
              {(Array.isArray(row) ? row : []).map((cell, j) => (
                <td
                  key={j}
                  className="border border-ink-900/20 px-2 py-1.5 align-top"
                >
                  {cell != null ? String(cell) : ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
