import type { WorkflowNode } from "../types.js";

function workflowSkeleton(nodes: WorkflowNode[]): unknown {
  return nodes.map((node) => ({
    id: node.id,
    description: node.description,
    outputFiles: [...(node.outputFiles ?? [])],
    children: workflowSkeleton(node.children ?? []),
  }));
}

function flattenNodes(nodes: WorkflowNode[], out: WorkflowNode[] = []): WorkflowNode[] {
  for (const n of nodes) {
    out.push(n);
    flattenNodes(n.children ?? [], out);
  }
  return out;
}

/** Stable comparison of verifier lines and marks per node (order within each node preserved). */
function verifierPayload(tree: WorkflowNode[]): unknown {
  const flat = flattenNodes(tree);
  flat.sort((a, b) => a.id.localeCompare(b.id));
  return flat.map((n) => ({
    id: n.id,
    verifiers: [...(n.verifiers ?? [])],
    verifierMarks: [...(n.verifierMarks ?? [])],
  }));
}

/**
 * Classify a user-driven workflow tree update from the renderer (sidebar Progress region).
 *
 * - ``workflow``: structure / descriptions / output file paths / child order changed (sidebar renames included).
 * - ``verifier``: only verifier lines or marks changed (same skeleton).
 *
 * IPC records ``edit_workflow`` (full workflow + files) when ``workflow`` is true, or ``edit_verifier``
 * (``environment.verifier`` + files) when only ``verifier`` is true (criterion text or check/cross status).
 */
export function classifyUserWorkflowTreeEdit(
  oldTree: WorkflowNode[] | undefined,
  newTree: WorkflowNode[]
): { workflow: boolean; verifier: boolean } {
  const oldNorm = oldTree ?? [];
  const skOld = JSON.stringify(workflowSkeleton(oldNorm));
  const skNew = JSON.stringify(workflowSkeleton(newTree));
  const workflowEdit = skOld !== skNew;
  const verifierEdit = JSON.stringify(verifierPayload(oldNorm)) !== JSON.stringify(verifierPayload(newTree));
  return {
    workflow: workflowEdit,
    verifier: verifierEdit && !workflowEdit,
  };
}
