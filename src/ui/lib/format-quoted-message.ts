/** User message for LM export as message("…") — quoted span plus comment. */
export function formatQuotedSelectionMessage(
  quotedText: string,
  comment: string,
  filePath?: string
): string {
  const quote = quotedText.trim();
  const body = comment.trim();
  if (!quote) return body;
  if (!body) return quote;
  const fileNote = filePath ? ` (from ${filePath})` : "";
  return `Quote${fileNote}:\n${quote}\n\n${body}`;
}
