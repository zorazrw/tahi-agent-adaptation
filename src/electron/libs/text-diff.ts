type LineOp = { type: "equal" | "del" | "add"; line: string };

function computeLineOps(origLines: string[], currLines: string[]): LineOp[] {
  const n = origLines.length;
  const m = currLines.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      if (origLines[i] === currLines[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const ops: LineOp[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (origLines[i] === currLines[j]) {
      ops.push({ type: "equal", line: origLines[i] });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: "del", line: origLines[i] });
      i++;
    } else {
      ops.push({ type: "add", line: currLines[j] });
      j++;
    }
  }
  while (i < n) {
    ops.push({ type: "del", line: origLines[i] });
    i++;
  }
  while (j < m) {
    ops.push({ type: "add", line: currLines[j] });
    j++;
  }
  return ops;
}

function truncSnippet(text: string, max = 160): string {
  const t = text.trim();
  if (t.length <= max) return t;
  return t.slice(0, max - 3).trimEnd() + "...";
}

function wordChangeSnippets(beforeLine: string, afterLine: string): string {
  const wa = beforeLine.split(/\s+/).filter(Boolean);
  const wb = afterLine.split(/\s+/).filter(Boolean);
  if (wa.join(" ") === wb.join(" ")) return "(same tokens)";
  const parts: string[] = [];
  let ai = 0;
  let bi = 0;
  while (ai < wa.length || bi < wb.length) {
    if (ai < wa.length && bi < wb.length && wa[ai] === wb[bi]) {
      ai++;
      bi++;
      continue;
    }
    const nextA = wa.indexOf(wb[bi] ?? "", ai);
    const nextB = wb.indexOf(wa[ai] ?? "", bi);
    if (bi < wb.length && (nextA === -1 || (nextB !== -1 && nextB - bi <= nextA - ai))) {
      parts.push(`+ ${truncSnippet(wb[bi], 100)}`);
      bi++;
    } else if (ai < wa.length) {
      parts.push(`- ${truncSnippet(wa[ai], 100)}`);
      ai++;
    } else {
      break;
    }
    if (parts.length >= 6) break;
  }
  return parts.length > 0 ? parts.join(", ") : `${truncSnippet(beforeLine, 100)} → ${truncSnippet(afterLine, 100)}`;
}

/**
 * Line-aligned compact diff: only changed line groups, with short token snippets (not a full unified diff).
 */
export function compactFileEditAnnotation(
  before: string,
  after: string,
  pathDisp: string,
  maxChunks = 8
): string {
  if (before === after) return "";
  const bl = before.split(/\r?\n/).map((l) => l.replace(/\r$/, ""));
  const al = after.split(/\r?\n/).map((l) => l.replace(/\r$/, ""));
  const ops = computeLineOps(bl, al);
  const parts: string[] = [];

  for (let k = 0; k < ops.length; k++) {
    if (ops[k].type === "equal") continue;
    const delLines: string[] = [];
    const addLines: string[] = [];
    while (k < ops.length && ops[k].type !== "equal") {
      if (ops[k].type === "del") delLines.push(ops[k].line);
      else if (ops[k].type === "add") addLines.push(ops[k].line);
      k++;
    }
    k--;

    const bJoin = delLines.join(" ");
    const aJoin = addLines.join(" ");
    if (delLines.length > 0 && addLines.length > 0) {
      const ws = wordChangeSnippets(bJoin, aJoin);
      if (ws !== "(same tokens)") parts.push(`• ${truncSnippet(ws, 700)}`);
    } else if (delLines.length > 0) {
      parts.push(`• removed: ${truncSnippet(bJoin, 200)}`);
    } else if (addLines.length > 0) {
      parts.push(`• added: ${truncSnippet(aJoin, 200)}`);
    }
    if (parts.length >= maxChunks) break;
  }

  if (parts.length === 0) return "";
  let body = parts.join("\n");
  const maxOut = 8000;
  if (body.length > maxOut) {
    body = body.slice(0, maxOut - 50).trimEnd() + `\n… (truncated, path=${pathDisp})`;
  }
  return `path=${pathDisp}\n${body}`;
}

/** Compact line-based diff between two texts (changed hunks + small context). */
export function buildTextDiff(
  original: string,
  current: string,
  maxHunks = 8,
  contextLines = 1
): string {
  const origLines = original.split(/\r?\n/);
  const currLines = current.split(/\r?\n/);
  const ops = computeLineOps(origLines, currLines);

  const hunks: { start: number; end: number }[] = [];
  for (let k = 0; k < ops.length; k++) {
    if (ops[k].type === "equal") continue;
    const start = Math.max(0, k - contextLines);
    let end = Math.min(ops.length - 1, k + contextLines);
    while (end + 1 < ops.length && ops[end + 1].type !== "equal") end++;
    if (hunks.length > 0 && start <= hunks[hunks.length - 1].end + 1) {
      hunks[hunks.length - 1].end = Math.max(hunks[hunks.length - 1].end, end);
    } else {
      hunks.push({ start, end });
    }
  }

  if (hunks.length === 0) return "";

  const lines: string[] = [];
  const limitedHunks = hunks.slice(0, maxHunks);
  for (let h = 0; h < limitedHunks.length; h++) {
    const { start: s, end: e } = limitedHunks[h];
    if (h > 0) lines.push("...");
    for (let k = s; k <= e; k++) {
      const op = ops[k];
      if (op.type === "equal") lines.push(`  ${op.line}`);
      else if (op.type === "del") lines.push(`- ${op.line}`);
      else lines.push(`+ ${op.line}`);
    }
  }
  return lines.join("\n");
}
