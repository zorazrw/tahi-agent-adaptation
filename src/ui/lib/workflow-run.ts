import type { WorkflowNode } from "../types";

/**
 * Next workflow node id in depth-first order, matching main-process triggerNodeSolve:
 * nodes above verification depth with children delegate to descendants; otherwise this node runs.
 */
export function findNextRunnableWorkflowNodeId(
  tree: WorkflowNode[],
  verificationDepth: number
): string | null {
  for (const node of tree) {
    if (node.status === "completed") continue;
    if (node.children.length > 0 && node.depth < verificationDepth) {
      const nested = findNextRunnableWorkflowNodeId(node.children, verificationDepth);
      if (nested) return nested;
      continue;
    }
    return node.id;
  }
  return null;
}

const pendingAutoAdvanceSessions = new Set<string>();

export function markPendingWorkflowAutoAdvance(sessionId: string) {
  pendingAutoAdvanceSessions.add(sessionId);
}

/** Call after session.status. Returns true if we should enqueue the next step. */
export function consumePendingWorkflowAutoAdvance(
  sessionId: string,
  status: "idle" | "running" | "completed" | "error"
): boolean {
  if (status === "error" || status === "idle") {
    pendingAutoAdvanceSessions.delete(sessionId);
    return false;
  }
  if (status !== "completed") return false;
  if (!pendingAutoAdvanceSessions.has(sessionId)) return false;
  pendingAutoAdvanceSessions.delete(sessionId);
  return true;
}

export function clearPendingWorkflowAutoAdvance(sessionId: string) {
  pendingAutoAdvanceSessions.delete(sessionId);
}

export function clearAllPendingWorkflowAutoAdvance() {
  pendingAutoAdvanceSessions.clear();
}
