import { useMemo } from "react";
import JsonView from "@uiw/react-json-view";

export function JsonRenderer({ data }: { data: { kind: "json"; content: string } }) {
  const parsed = useMemo(() => {
    try {
      return { value: JSON.parse(data.content), error: null };
    } catch (e) {
      return { value: null, error: e instanceof Error ? e.message : String(e) };
    }
  }, [data.content]);

  if (parsed.error) {
    return (
      <div>
        <p className="text-sm text-error mb-2">Invalid JSON: {parsed.error}</p>
        <pre className="text-sm text-ink-700 whitespace-pre-wrap font-mono">{data.content}</pre>
      </div>
    );
  }

  return (
    <div className="text-sm">
      <JsonView value={parsed.value} collapsed={2} displayDataTypes={false} />
    </div>
  );
}
