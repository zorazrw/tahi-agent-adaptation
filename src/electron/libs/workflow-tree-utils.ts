import type { VerifierMark } from "../types.js";

export type NodeStatus = "pending" | "running" | "completed" | "error";

export type WorkflowNode = {
  id: string;
  description: string;
  outputFiles: string[];
  verifiers: string[];
  verifierMarks: VerifierMark[];
  children: WorkflowNode[];
  status: NodeStatus;
  depth: number;
  resumePoint?: { uuid: string; claudeSessionId: string };
};

/** Find a node by id in the tree. */
export function findNodeById(tree: WorkflowNode[], id: string): WorkflowNode | undefined {
  for (const node of tree) {
    if (node.id === id) return node;
    const found = findNodeById(node.children, id);
    if (found) return found;
  }
  return undefined;
}

/** Find the parent of a node by id. */
export function findParentNode(tree: WorkflowNode[], id: string): WorkflowNode | undefined {
  for (const node of tree) {
    for (const child of node.children) {
      if (child.id === id) return node;
    }
    const found = findParentNode(node.children, id);
    if (found) return found;
  }
  return undefined;
}

/** Get the next incomplete child of a parent node. */
export function getNextIncompleteChild(parent: WorkflowNode): WorkflowNode | undefined {
  return parent.children.find(c => c.status !== "completed");
}

/** Check if a node and all its descendants are completed. */
export function isNodeFullyComplete(node: WorkflowNode): boolean {
  if (node.children.length === 0) return node.status === "completed";
  return node.children.every(isNodeFullyComplete);
}

/** Determine whether children below verification depth should auto-advance. */
export function shouldAutoAdvance(parent: WorkflowNode, verificationDepth: number): boolean {
  // Children auto-run if they are below the verification depth
  return parent.children.length > 0 && parent.children[0].depth > verificationDepth;
}

/** Get the maximum depth in the tree. */
export function getMaxDepth(tree: WorkflowNode[]): number {
  let max = 0;
  for (const node of tree) {
    max = Math.max(max, node.depth);
    if (node.children.length > 0) {
      max = Math.max(max, getMaxDepth(node.children));
    }
  }
  return max;
}

/** Flatten the tree to an array of leaf nodes (depth-first). */
export function flattenToLeaves(tree: WorkflowNode[]): WorkflowNode[] {
  const leaves: WorkflowNode[] = [];
  for (const node of tree) {
    if (node.children.length === 0) {
      leaves.push(node);
    } else {
      leaves.push(...flattenToLeaves(node.children));
    }
  }
  return leaves;
}

/** Flatten the entire tree to an array of all nodes (depth-first). */
export function flattenAll(tree: WorkflowNode[]): WorkflowNode[] {
  const result: WorkflowNode[] = [];
  for (const node of tree) {
    result.push(node);
    result.push(...flattenAll(node.children));
  }
  return result;
}

/** Build the ancestor path description for a node (e.g., "Phase 2 > Task 3"). */
export function getNodePath(tree: WorkflowNode[], nodeId: string): string {
  const path: string[] = [];
  function find(nodes: WorkflowNode[]): boolean {
    for (const node of nodes) {
      path.push(node.description);
      if (node.id === nodeId) return true;
      if (find(node.children)) return true;
      path.pop();
    }
    return false;
  }
  find(tree);
  return path.join(" > ");
}

/** Update a node's status in-place within the tree. Returns true if found. */
export function updateNodeStatus(tree: WorkflowNode[], nodeId: string, status: NodeStatus): boolean {
  const node = findNodeById(tree, nodeId);
  if (!node) return false;
  node.status = status;
  return true;
}

/** Mark a node and all its descendants as completed. */
export function completeNodeAndDescendants(node: WorkflowNode): void {
  node.status = "completed";
  for (const child of node.children) {
    completeNodeAndDescendants(child);
  }
}

/** Reset a node and all its descendants to "pending". */
export function resetNode(node: WorkflowNode): void {
  node.status = "pending";
  node.verifierMarks = node.verifiers.map(() => undefined);
  for (const child of node.children) {
    resetNode(child);
  }
}

/**
 * Hydrate a raw workflow plan input (from the MCP tool) into a WorkflowNode tree.
 * Supports both flat (array of steps) and hierarchical (nodes with children) input.
 */
export type RawWorkflowNode = {
  description: string;
  outputFiles: string[];
  verifiers: string[];
  children?: RawWorkflowNode[];
};

export function hydrateWorkflowTree(input: RawWorkflowNode[], depth = 0): WorkflowNode[] {
  return input.map((raw) => ({
    id: crypto.randomUUID(),
    description: raw.description,
    outputFiles: raw.outputFiles,
    verifiers: raw.verifiers,
    verifierMarks: raw.verifiers.map(() => undefined),
    children: raw.children ? hydrateWorkflowTree(raw.children, depth + 1) : [],
    status: "pending" as NodeStatus,
    depth,
  }));
}

/**
 * Convert legacy flat steps/outputFiles/verificationCriteria/verifierMarks
 * into a single-depth tree (one root, steps as children).
 */
export function migrateFromFlatSteps(
  steps: string[],
  outputFiles: string[][],
  verificationCriteria: string[][],
  verifierMarks: VerifierMark[][],
  completedStepIndices: number[]
): WorkflowNode[] {
  if (steps.length === 0) return [];

  const children: WorkflowNode[] = steps.map((desc, i) => ({
    id: crypto.randomUUID(),
    description: desc,
    outputFiles: outputFiles[i] ?? [],
    verifiers: verificationCriteria[i] ?? [],
    verifierMarks: verifierMarks[i] ?? (verificationCriteria[i] ?? []).map(() => undefined),
    children: [],
    status: completedStepIndices.includes(i) ? "completed" as NodeStatus : "pending" as NodeStatus,
    depth: 1,
  }));

  const root: WorkflowNode = {
    id: crypto.randomUUID(),
    description: "Task",
    outputFiles: [],
    verifiers: [],
    verifierMarks: [],
    children,
    status: children.every(c => c.status === "completed") ? "completed" : "pending",
    depth: 0,
  };

  return [root];
}
