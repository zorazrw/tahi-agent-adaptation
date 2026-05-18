import type { StreamMessage } from "../types.js";
import type { ExportEnvironmentSnapshot } from "./message-state-snapshot.js";
import { snapshotFileUtf8Content } from "./message-state-snapshot.js";
import { buildTextDiff } from "./text-diff.js";

export type MessageRowWithSnapshot = {
  message: StreamMessage;
  snapshot: ExportEnvironmentSnapshot | null;
};

/** Line diffs for each human ``file_edit`` (before/after from consecutive message snapshots). */
export function gatherHumanFileEditDiffs(
  rows: MessageRowWithSnapshot[],
  cwd?: string
): string[] {
  const out: string[] = [];
  for (let i = 0; i < rows.length; i++) {
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
