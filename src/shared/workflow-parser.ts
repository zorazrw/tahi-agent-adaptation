/** Parse numbered steps from LLM text; stop at OUTPUT FILES or VERIFIERS. */
export function parseNumberedSteps(text: string): string[] {
  if (!text || typeof text !== "string") return [];
  const outputFilesStart = text.search(/\nOUTPUT FILES:\s*\n/i);
  const verifiersStart = text.search(/\nVERIFIERS:\s*\n/i);
  const end = [outputFilesStart, verifiersStart].filter((i) => i >= 0);
  const workflowEnd = end.length ? Math.min(...end) : text.length;
  const workflowText = text.slice(0, workflowEnd);
  const lines = workflowText.split(/\n/).map((s) => s.trim()).filter(Boolean);
  const steps: string[] = [];
  for (const line of lines) {
    const match = line.match(/^\s*\d+[.)]\s*(.+)$/);
    if (match) steps.push(match[1].trim());
  }
  return steps;
}

/** Parse OUTPUT FILES block into per-step file paths (string[][]). Expects "OUTPUT FILES:" then "Step N: path1, path2" lines. */
export function parseOutputFilesBlock(text: string, stepCount: number): string[][] {
  const result: string[][] = Array.from({ length: stepCount }, () => []);
  const outputFilesIdx = text.search(/\nOUTPUT FILES:\s*\n/i);
  if (outputFilesIdx < 0) return result;
  const verifiersIdx = text.search(/\nVERIFIERS:\s*\n/i);
  const blockEnd = verifiersIdx >= 0 ? verifiersIdx : text.length;
  const block = text.slice(outputFilesIdx, blockEnd);
  const stepLineRegex = /^Step\s*(\d+)\s*:\s*(.+)$/gm;
  let match: RegExpExecArray | null;
  while ((match = stepLineRegex.exec(block)) !== null) {
    const stepNum = parseInt(match[1], 10);
    const pathsStr = match[2].trim();
    const idx = stepNum - 1;
    if (idx >= 0 && idx < stepCount && pathsStr) {
      const paths = pathsStr.split(/[,;]/).map((p) => p.trim()).filter(Boolean);
      result[idx] = paths;
    }
  }
  return result;
}

function parseBulletCriteria(content: string): string[] {
  const lines = content.split(/\n/).map((s) => s.trim()).filter(Boolean);
  const result: string[] = [];
  for (const line of lines) {
    const m = line.match(/^[-*•]\s+(.+)$/) || line.match(/^\d+[.)]\s+(.+)$/);
    if (m) result.push(m[1].trim());
    else if (line) result.push(line);
  }
  return result;
}

/** Parse VERIFIERS block into per-step criteria (string[][]). Expects "VERIFIERS:" then "Step N:" sections with bullet lines. */
export function parseVerifiersBlock(text: string, stepCount: number): string[][] {
  const verifiers: string[][] = Array.from({ length: stepCount }, () => []);
  const verifiersIdx = text.search(/\nVERIFIERS:\s*\n/i);
  if (verifiersIdx < 0) return verifiers;
  const block = text.slice(verifiersIdx);
  const stepRegex = /^Step\s*(\d+)\s*:?\s*\n/gm;
  let match: RegExpExecArray | null = null;
  let lastStepNum = 0;
  let lastEnd = 0;
  while ((match = stepRegex.exec(block)) !== null) {
    const stepNum = parseInt(match[1], 10);
    if (lastStepNum > 0) {
      const content = block.slice(lastEnd, match.index);
      const criteria = parseBulletCriteria(content);
      const idx = lastStepNum - 1;
      if (idx >= 0 && idx < stepCount) verifiers[idx] = criteria;
    }
    lastStepNum = stepNum;
    lastEnd = match.index + match[0].length;
  }
  if (lastStepNum > 0) {
    const content = block.slice(lastEnd);
    const criteria = parseBulletCriteria(content);
    const idx = lastStepNum - 1;
    if (idx >= 0 && idx < stepCount) verifiers[idx] = criteria;
  }
  return verifiers;
}

/** Extract the text before the OUTPUT FILES section (preamble + numbered steps as markdown). */
export function extractPreWorkflowText(text: string): string {
  const outputFilesIdx = text.search(/\nOUTPUT FILES:\s*\n/i);
  if (outputFilesIdx < 0) return text;
  return text.slice(0, outputFilesIdx).trimEnd();
}

/** Check whether text contains the OUTPUT FILES / VERIFIERS workflow pattern. */
export function hasWorkflowPattern(text: string): boolean {
  return /\nOUTPUT FILES:\s*\n/i.test(text);
}
