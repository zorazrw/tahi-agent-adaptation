import {
  estimateTokens,
  getLatestCompactionEntry,
  type SessionEntry,
  type SessionManager,
} from "@mariozechner/pi-coding-agent";

/** Max agent tool calls kept in LLM context during task execution (node solve / follow-ups). */
export const EXECUTION_CONTEXT_MAX_ACTIONS = 8;

function countToolCallsInEntry(entry: SessionEntry): number {
  if (entry.type !== "message") return 0;
  const msg = entry.message;
  if (msg.role !== "assistant") return 0;
  const content = msg.content;
  if (!Array.isArray(content)) return 0;
  let n = 0;
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const type = (block as { type?: string }).type;
    if (type === "toolCall" || type === "tool_use") n += 1;
  }
  return n;
}

function totalToolActionsOnPath(path: SessionEntry[]): number {
  return path.reduce((sum, entry) => sum + countToolCallsInEntry(entry), 0);
}

/** Entry id of the oldest assistant message to keep when retaining the last ``maxActions`` tool calls. */
export function findFirstKeptEntryIdForLastActions(
  path: SessionEntry[],
  maxActions: number
): string | null {
  if (maxActions <= 0) return null;
  let seen = 0;
  for (let i = path.length - 1; i >= 0; i--) {
    const calls = countToolCallsInEntry(path[i]);
    if (calls === 0) continue;
    seen += calls;
    if (seen >= maxActions) {
      return path[i].id;
    }
  }
  return null;
}

/**
 * Drop older agent tool history from the Pi session context via compaction, keeping only the
 * most recent ``maxActions`` tool calls. No-op when the path already has fewer actions.
 */
export function trimSessionToLastAgentActions(
  sessionManager: SessionManager,
  maxActions = EXECUTION_CONTEXT_MAX_ACTIONS
): boolean {
  const leafId = sessionManager.getLeafId();
  if (!leafId) return false;

  const path = sessionManager.getBranch(leafId);
  if (totalToolActionsOnPath(path) <= maxActions) return false;

  const firstKeptEntryId = findFirstKeptEntryIdForLastActions(path, maxActions);
  if (!firstKeptEntryId) return false;

  const existing = getLatestCompactionEntry(path);
  if (existing?.firstKeptEntryId === firstKeptEntryId) return false;

  const omitted = totalToolActionsOnPath(path) - maxActions;
  const messages = sessionManager.buildSessionContext().messages;
  const tokensBefore = messages.reduce((sum, msg) => sum + estimateTokens(msg), 0);
  sessionManager.appendCompaction(
    `Earlier agent actions (${omitted} tool call(s)) are omitted from context. Continue from the last ${maxActions} actions.`,
    firstKeptEntryId,
    tokensBefore
  );
  return true;
}
