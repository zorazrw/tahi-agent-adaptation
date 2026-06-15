import type { StreamMessage } from "../types.js";
import type { ExportEnvironmentSnapshot } from "./message-state-snapshot.js";
import { snapshotFileUtf8Content } from "./message-state-snapshot.js";
import { buildTextDiff, compactFileEditAnnotation } from "./text-diff.js";

export type MessageRowWithSnapshot = {
  message: StreamMessage;
  snapshot: ExportEnvironmentSnapshot | null;
};

/** Line diffs for each human ``file_edit`` (before/after from consecutive message snapshots). */
export function gatherHumanFileEditDiffs(
  rows: MessageRowWithSnapshot[],
  cwd?: string,
  fromIndex = 0
): string[] {
  const out: string[] = [];
  for (let i = Math.max(0, fromIndex); i < rows.length; i++) {
    const { message, snapshot } = rows[i];
    if (message.type !== "file_edit") continue;
    const path = String(message.path ?? "").trim();
    if (!path) continue;

    const after = snapshotFileUtf8Content(snapshot, path, cwd) ?? "";
    let before: string | null = null;
    for (let j = i - 1; j >= 0; j--) {
      const prior = snapshotFileUtf8Content(rows[j].snapshot, path, cwd);
      if (prior != null) {
        before = prior;
        break;
      }
    }
    const beforeText = before ?? "";
    if (beforeText === after) continue;

    const diff = buildTextDiff(beforeText, after);
    if (!diff.trim()) continue;
    out.push(`File: ${path}\n${diff}`);
  }
  return out;
}

function isAgentRoundEnd(message: StreamMessage): boolean {
  return message.type === "run_result" || message.type === "result";
}

function agentRoundSnapshotIndex(rows: MessageRowWithSnapshot[], editFrom: number): number {
  if (editFrom > 0 && isAgentRoundEnd(rows[editFrom - 1].message)) return editFrom - 1;
  return -1;
}

function messageSendSnapshotIndex(rows: MessageRowWithSnapshot[], editTo: number): number {
  if (editTo < rows.length && rows[editTo].message.type === "user_prompt") return editTo;
  return -1;
}

function resolveAfterFileContent(
  rows: MessageRowWithSnapshot[],
  path: string,
  cwd: string | undefined,
  editFrom: number,
  editTo: number,
  afterIdx: number
): string {
  if (afterIdx >= 0) {
    const fromPrompt = snapshotFileUtf8Content(rows[afterIdx].snapshot, path, cwd);
    if (fromPrompt != null) return fromPrompt;
  }
  for (let i = editTo - 1; i >= editFrom; i--) {
    if (rows[i].message.type !== "file_edit") continue;
    const fromEdit = snapshotFileUtf8Content(rows[i].snapshot, path, cwd);
    if (fromEdit != null) return fromEdit;
  }
  return "";
}

function collectChangedFilePaths(
  rows: MessageRowWithSnapshot[],
  beforeIdx: number,
  afterIdx: number,
  editFrom: number,
  editTo: number,
  cwd?: string
): string[] {
  const paths = new Set<string>();
  for (const idx of [beforeIdx, afterIdx]) {
    if (idx < 0) continue;
    for (const entry of rows[idx].snapshot?.file ?? []) {
      const p = String(entry.path ?? "").trim();
      if (p) paths.add(p);
    }
  }
  const end = Math.min(editTo, rows.length);
  for (let i = Math.max(0, editFrom); i < end; i++) {
    const message = rows[i].message;
    if (message.type !== "file_edit") continue;
    const p = String(message.path ?? "").trim();
    if (p) paths.add(p);
  }
  return [...paths].filter((path) => {
    const before = beforeIdx >= 0 ? snapshotFileUtf8Content(rows[beforeIdx].snapshot, path, cwd) ?? "" : "";
    const after = resolveAfterFileContent(rows, path, cwd, editFrom, editTo, afterIdx);
    return before !== after;
  });
}

/**
 * One compact diff per changed file: after the last agent ``run_result`` → when the human sends the message.
 */
export function gatherFileEditDiffsSinceAgentRound(
  rows: MessageRowWithSnapshot[],
  cwd: string | undefined,
  editFrom: number,
  editTo: number
): string[] {
  const beforeIdx = agentRoundSnapshotIndex(rows, editFrom);
  const afterIdx = messageSendSnapshotIndex(rows, editTo);
  if (beforeIdx < 0 || afterIdx < 0) return [];

  const out: string[] = [];
  for (const path of collectChangedFilePaths(rows, beforeIdx, afterIdx, editFrom, editTo, cwd)) {
    const before = snapshotFileUtf8Content(rows[beforeIdx].snapshot, path, cwd) ?? "";
    const after = resolveAfterFileContent(rows, path, cwd, editFrom, editTo, afterIdx);
    const annotation = compactFileEditAnnotation(before, after, path);
    if (annotation.trim()) out.push(annotation);
  }
  return out;
}
