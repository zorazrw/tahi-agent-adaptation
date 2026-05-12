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

/**
 * True iff the workflow tree is non-empty and every node in it (recursively)
 * is still `pending` — i.e. the plan was just generated and nothing has been
 * started, completed, or errored yet. Used to detect the "post-initial-planning"
 * moment so prediction can run there too.
 */
export function isWorkflowUntouched(tree: WorkflowNode[]): boolean {
  if (tree.length === 0) return false;
  const visit = (nodes: WorkflowNode[]): boolean => {
    for (const n of nodes) {
      if (n.status !== "pending") return false;
      if (n.children.length > 0 && !visit(n.children)) return false;
    }
    return true;
  };
  return visit(tree);
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
