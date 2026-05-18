/** Compact line-based diff between two texts (changed hunks + small context). */
export function buildTextDiff(
  original: string,
  current: string,
  maxHunks = 8,
  contextLines = 1
): string {
  const origLines = original.split(/\r?\n/);
  const currLines = current.split(/\r?\n/);
  const n = origLines.length;
  const m = currLines.length;

  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      if (origLines[i] === currLines[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  type Op = { type: "equal" | "del" | "add"; line: string; i: number; j: number };
  const ops: Op[] = [];
  let i = 0,
    j = 0;
  while (i < n && j < m) {
    if (origLines[i] === currLines[j]) {
      ops.push({ type: "equal", line: origLines[i], i, j });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: "del", line: origLines[i], i, j });
      i++;
    } else {
      ops.push({ type: "add", line: currLines[j], i, j });
      j++;
    }
  }
  while (i < n) {
    ops.push({ type: "del", line: origLines[i], i, j });
    i++;
  }
  while (j < m) {
    ops.push({ type: "add", line: currLines[j], i, j });
    j++;
  }

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
      if (op.type === "equal") {
        lines.push(`  ${op.line}`);
      } else if (op.type === "del") {
        lines.push(`- ${op.line}`);
      } else {
        lines.push(`+ ${op.line}`);
      }
    }
  }
  return lines.join("\n");
}
