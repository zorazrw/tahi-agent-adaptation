type SnapshotWorkflowNode = {
  id?: string;
  description?: string;
  verifiers?: Array<{ criterion?: string; status?: string }>;
  children?: SnapshotWorkflowNode[];
};

type SnapshotRow = {
  message: { type?: string };
  snapshot?: { workflow?: SnapshotWorkflowNode[] } | null;
};

function oneLineField(val: unknown): string {
  if (val == null) return "";
  return String(val).replace(/\s+/g, " ").trim();
}

export function findNodeInSnapshotWorkflow(
  wf: SnapshotWorkflowNode[] | null | undefined,
  nodeId: string
): SnapshotWorkflowNode | null {
  if (!wf?.length || !nodeId.trim()) return null;
  const stack = [...wf];
  while (stack.length > 0) {
    const n = stack.shift()!;
    if (n.id === nodeId) return n;
    if (n.children?.length) stack.push(...n.children);
  }
  return null;
}

/** Verifier lines for one workflow node only. */
export function flattenNodeVerifierLines(node: SnapshotWorkflowNode | null | undefined): string[] {
  if (!node) return [];
  return (node.verifiers ?? []).map((v) => {
    const crit = oneLineField(v.criterion);
    const st = oneLineField(v.status);
    return crit ? `${crit}: ${st}` : `(empty criterion): ${st}`;
  });
}

/** Preorder flatten: one line per verifier as ``{criterion}: {status}`` (all nodes). */
export function flattenWorkflowVerifierLines(
  wf: SnapshotWorkflowNode[] | null | undefined
): string[] {
  const out: string[] = [];
  function walk(nodes: SnapshotWorkflowNode[] | undefined) {
    if (!nodes?.length) return;
    for (const n of nodes) {
      out.push(...flattenNodeVerifierLines(n));
      walk(n.children);
    }
  }
  walk(wf ?? []);
  return out;
}

function verifierLineCriterionKey(line: string): string {
  const s = line.replace(/[\n\r]+$/, "");
  const idx = s.lastIndexOf(": ");
  return idx === -1 ? s : s.slice(0, idx);
}

function verifierLineStatusTail(line: string): string {
  const s = line.replace(/[\n\r]+$/, "");
  const idx = s.lastIndexOf(": ");
  return idx === -1 ? "" : s.slice(idx + 2);
}

function truncateAnnotation(text: string, max: number): string {
  const s = text.trim();
  if (s.length <= max) return s;
  return s.slice(0, max - 1).trimEnd() + "…";
}

/** Diff flattened verifier lines by criterion key (matches export ``_compact_verifier_lines_annotation``). */
export function compactVerifierLinesAnnotation(
  beforeLines: string[],
  afterLines: string[],
  pathDisp = "verifiers",
  maxOut = 8000
): string {
  const bl = beforeLines.map((l) => l.replace(/[\n\r]+$/, ""));
  const al = afterLines.map((l) => l.replace(/[\n\r]+$/, ""));
  if (bl.join("\n") === al.join("\n")) {
    return `(no textual change) path=${pathDisp}`;
  }

  const kb = new Map(bl.map((l) => [verifierLineCriterionKey(l), verifierLineStatusTail(l)]));
  const ka = new Map(al.map((l) => [verifierLineCriterionKey(l), verifierLineStatusTail(l)]));
  const keysB = new Set(kb.keys());
  const keysA = new Set(ka.keys());
  const parts: string[] = [];

  for (const k of [...keysB].filter((x) => keysA.has(x)).sort()) {
    const stb = kb.get(k) ?? "";
    const sta = ka.get(k) ?? "";
    if (stb === sta) continue;
    parts.push(`• ${truncateAnnotation(k, 520)}: ${stb} → ${sta}`);
  }
  for (const k of [...keysB].filter((x) => !keysA.has(x)).sort()) {
    parts.push(`• removed: ${truncateAnnotation(`${k}: ${kb.get(k) ?? ""}`, 220)}`);
  }
  for (const k of [...keysA].filter((x) => !keysB.has(x)).sort()) {
    parts.push(`• added: ${truncateAnnotation(`${k}: ${ka.get(k) ?? ""}`, 220)}`);
  }

  if (parts.length === 0) {
    return `(no localized changes detected) path=${pathDisp}`;
  }

  let body = parts.join("\n");
  if (body.length > maxOut) {
    body = body.slice(0, maxOut - 50).trimEnd() + `\n… (annotation truncated, path=${pathDisp})`;
  }
  return `path=${pathDisp}\n${body}`;
}

function isNoOpVerifierAnnotation(annotation: string): boolean {
  return (
    annotation.startsWith("(no textual change) path=") ||
    annotation.startsWith("(no localized changes detected) path=")
  );
}

function isAgentRoundEnd(message: { type?: string }): boolean {
  return message.type === "run_result" || message.type === "result";
}

function lastUserPromptIndex(rows: SnapshotRow[]): number {
  for (let i = rows.length - 1; i >= 0; i--) {
    if (rows[i].message.type === "user_prompt") return i;
  }
  return rows.length;
}

/** Index of the first message after the last completed agent round (before the latest user_prompt). */
export function findHumanEditsWindowStart(rows: SnapshotRow[]): number {
  const end = lastUserPromptIndex(rows);
  for (let i = end - 1; i >= 0; i--) {
    if (isAgentRoundEnd(rows[i].message)) return i + 1;
  }
  return 0;
}

/** Exclusive end index for human edits (the latest ``user_prompt``, not included). */
export function findHumanEditsWindowEnd(rows: SnapshotRow[]): number {
  const idx = lastUserPromptIndex(rows);
  return idx < rows.length ? idx : rows.length;
}

function hasHumanVerifierEditInWindow(rows: SnapshotRow[], editFrom: number, editTo: number): boolean {
  const end = Math.min(editTo, rows.length);
  for (let i = Math.max(0, editFrom); i < end; i++) {
    if (rows[i].message.type === "edit_verifier") return true;
  }
  return false;
}

/**
 * Verifier diff for the active workflow node only, and only when the human issued
 * ``edit_verifier`` this round. Compares that node's verifiers at the last
 * ``run_result`` vs the ``user_prompt`` snapshot (endpoints of the human-edit window).
 */
export function gatherVerifierEditDiffSinceAgentRound(
  rows: SnapshotRow[],
  editFrom: number,
  editTo: number,
  nodeId: string | undefined
): string[] {
  const nid = nodeId?.trim();
  if (!nid) return [];
  if (!hasHumanVerifierEditInWindow(rows, editFrom, editTo)) return [];

  const beforeIdx = editFrom > 0 && isAgentRoundEnd(rows[editFrom - 1].message) ? editFrom - 1 : -1;
  const afterIdx = editTo < rows.length && rows[editTo].message.type === "user_prompt" ? editTo : -1;
  if (beforeIdx < 0 || afterIdx < 0) return [];

  const beforeNode = findNodeInSnapshotWorkflow(rows[beforeIdx].snapshot?.workflow ?? null, nid);
  const afterNode = findNodeInSnapshotWorkflow(rows[afterIdx].snapshot?.workflow ?? null, nid);
  const pathDisp = afterNode?.description?.trim() || beforeNode?.description?.trim() || nid;

  const annotation = compactVerifierLinesAnnotation(
    flattenNodeVerifierLines(beforeNode),
    flattenNodeVerifierLines(afterNode),
    `verifiers (${pathDisp})`
  );
  if (!annotation.trim() || isNoOpVerifierAnnotation(annotation)) return [];
  return [annotation];
}

/** Append human file and verifier edit diffs to a follow-up user message for task execution. */
export function appendHumanEditsToContinuePrompt(
  userPrompt: string,
  fileDiffs: string[],
  verifierDiffs: string[]
): string {
  const trimmed = userPrompt.trim();
  if (!trimmed) return trimmed;
  if (fileDiffs.length === 0 && verifierDiffs.length === 0) return trimmed;

  const sections: string[] = [
    trimmed,
    "",
    "---",
    "",
    "The human made edits since your last response. Treat the current on-disk files and verifier criteria as authoritative.",
  ];

  if (fileDiffs.length > 0) {
    sections.push(
      "",
      "Human file edits (localized line changes):",
      ...fileDiffs
    );
  }

  if (verifierDiffs.length > 0) {
    sections.push("", "Human verifier edits:", ...verifierDiffs);
  }

  return sections.join("\n");
}
