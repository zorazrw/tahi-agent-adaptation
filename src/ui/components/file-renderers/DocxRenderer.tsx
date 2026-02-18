export function DocxRenderer({ data }: { data: { kind: "docx"; html: string } }) {
  return (
    <div
      className="file-preview-docx text-sm text-ink-700 prose prose-sm max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-ol:my-1"
      dangerouslySetInnerHTML={{ __html: data.html }}
    />
  );
}
