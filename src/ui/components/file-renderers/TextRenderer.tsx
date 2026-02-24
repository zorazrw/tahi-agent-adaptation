export function TextRenderer({ data }: { data: { kind: "txt"; content: string } }) {
  return (
    <pre className="text-sm text-ink-700 whitespace-pre-wrap break-words font-mono">
      {data.content}
    </pre>
  );
}
