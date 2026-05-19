import { z } from "zod";
import type { ClientEvent, WorkflowNode } from "./runtime-types.js";

/**
 * Schema for actions a user (or a confident prediction) can *execute*.
 *
 * Distinct from `PredictedUserActionType`, which is just a label the LLM
 * emits. An ExecutableAction carries the payload needed to actually dispatch
 * the action — e.g. `edit_workflow` includes a patch to apply to the current
 * workflow tree.
 *
 * `unknown` is intentionally absent: by definition it has no executable payload.
 */

// LLM-emitted verifier marks: strict enum, no transform (so the schema is
// JSON-Schema representable via z.toJSONSchema). DB-side normalization lives
// separately in session-store.
const verifierMark = z.enum(["check", "cross"]).optional();

const workflowNodeSchema: z.ZodType<WorkflowNode> = z.lazy(() =>
  z.object({
    id: z.string(),
    description: z.string(),
    outputFiles: z.array(z.string()),
    verifiers: z.array(z.string()),
    verifierMarks: z.array(verifierMark),
    children: z.array(workflowNodeSchema),
    status: z.enum(["pending", "running", "completed", "error"]),
    depth: z.number(),
    resumePoint: z
      .union([
        z.object({ entryId: z.string() }),
        z.object({ uuid: z.string(), claudeSessionId: z.string() }),
      ])
      .optional(),
    originalOutputs: z
      .array(z.object({ path: z.string(), content: z.string() }))
      .optional(),
  })
);

export const workflowTreeSchema = z.array(workflowNodeSchema);

const nodeStatusSchema = z.enum(["pending", "running", "completed", "error"]);

const workflowPatchNodeSchema = z.object({
  id: z.string(),
  description: z.string(),
  outputFiles: z.array(z.string()).default([]),
  verifiers: z.array(z.string()).default([]),
  verifierMarks: z.array(verifierMark).default([]),
  children: z.array(workflowNodeSchema).default([]),
  status: nodeStatusSchema.default("pending"),
  depth: z.number().optional(),
});

const addWorkflowNodeOperation = z.object({
  op: z.literal("add_node"),
  parentId: z.string().nullable().optional(),
  afterNodeId: z.string().nullable().optional(),
  node: workflowPatchNodeSchema,
});

const updateWorkflowNodeOperation = z.object({
  op: z.literal("update_node"),
  nodeId: z.string(),
  description: z.string().optional(),
  outputFiles: z.array(z.string()).optional(),
  verifiers: z.array(z.string()).optional(),
  verifierMarks: z.array(verifierMark).optional(),
  status: nodeStatusSchema.optional(),
});

const deleteWorkflowNodeOperation = z.object({
  op: z.literal("delete_node"),
  nodeId: z.string(),
});

const moveWorkflowNodeOperation = z.object({
  op: z.literal("move_node"),
  nodeId: z.string(),
  parentId: z.string().nullable().optional(),
  afterNodeId: z.string().nullable().optional(),
});

export const workflowPatchOperationSchema = z.discriminatedUnion("op", [
  addWorkflowNodeOperation,
  updateWorkflowNodeOperation,
  deleteWorkflowNodeOperation,
  moveWorkflowNodeOperation,
]);

export const workflowPatchSchema = z.array(workflowPatchOperationSchema);

const messageAction = z.object({
  type: z.literal("message"),
  text: z.string().min(1),
  verificationNodeId: z.string().optional(),
});

const editWorkflowAction = z.object({
  type: z.literal("edit_workflow"),
  patch: workflowPatchSchema,
});

const editVerifierAction = z.object({
  type: z.literal("edit_verifier"),
  nodeId: z.string(),
  verifiers: z.array(z.string()),
});

const fileEditAction = z.object({
  type: z.literal("file_edit"),
  path: z.string().min(1),
  contents: z.string(),
});

const brainEditAction = z.object({
  type: z.literal("brain_edit"),
  kind: z.enum(["memory", "skill"]),
  sections: z
    .array(
      z.object({
        fileName: z.string().min(1),
        content: z.string(),
      })
    )
    .min(1),
  deletedFileNames: z.array(z.string()).optional(),
});

const stopAction = z.object({
  type: z.literal("stop"),
});

export const executableActionSchema = z.discriminatedUnion("type", [
  messageAction,
  editWorkflowAction,
  editVerifierAction,
  fileEditAction,
  brainEditAction,
  stopAction,
]);

export type ExecutableAction = z.infer<typeof executableActionSchema>;
export type ExecutableActionType = ExecutableAction["type"];
export type WorkflowPatchOperation = z.infer<typeof workflowPatchOperationSchema>;

export type Dispatcher = (event: ClientEvent) => void;

export type ExecuteContext = {
  sessionId: string;
  sendEvent: Dispatcher;
  /** Required for `edit_verifier`: the current workflow tree the patch is merged into. */
  currentWorkflowTree?: WorkflowNode[];
  /** Optional: persist file_edit content. If omitted, only the record event fires. */
  writeFile?: (path: string, contents: string) => Promise<void> | void;
};

function setVerifiersOnTree(
  tree: WorkflowNode[],
  nodeId: string,
  verifiers: string[]
): { tree: WorkflowNode[]; changed: boolean } {
  let changed = false;
  const visit = (nodes: WorkflowNode[]): WorkflowNode[] =>
    nodes.map((n) => {
      if (n.id === nodeId) {
        changed = true;
        return { ...n, verifiers, verifierMarks: verifiers.map(() => undefined) };
      }
      return { ...n, children: visit(n.children) };
    });
  return { tree: visit(tree), changed };
}

function cloneWorkflowTree(tree: WorkflowNode[]): WorkflowNode[] {
  return JSON.parse(JSON.stringify(tree)) as WorkflowNode[];
}

function normalizeVerifierMarks(
  verifiers: string[],
  marks?: Array<WorkflowNode["verifierMarks"][number]>
): WorkflowNode["verifierMarks"] {
  if (!marks) return verifiers.map(() => undefined);
  return verifiers.map((_, index) => marks[index] ?? undefined);
}

function refreshDepths(node: WorkflowNode, depth: number): WorkflowNode {
  return {
    ...node,
    depth,
    verifierMarks: normalizeVerifierMarks(node.verifiers, node.verifierMarks),
    children: (node.children ?? []).map((child) => refreshDepths(child, depth + 1)),
  };
}

function findNode(tree: WorkflowNode[], nodeId: string): WorkflowNode | undefined {
  for (const node of tree) {
    if (node.id === nodeId) return node;
    const found = findNode(node.children, nodeId);
    if (found) return found;
  }
  return undefined;
}

function getChildrenForParent(
  tree: WorkflowNode[],
  parentId?: string | null
): { children: WorkflowNode[]; depth: number } {
  if (!parentId) return { children: tree, depth: 0 };
  const parent = findNode(tree, parentId);
  if (!parent) throw new Error(`edit_workflow patch: parent node ${parentId} not found.`);
  return { children: parent.children, depth: parent.depth + 1 };
}

function insertNode(
  siblings: WorkflowNode[],
  node: WorkflowNode,
  afterNodeId?: string | null
): void {
  if (!afterNodeId) {
    siblings.push(node);
    return;
  }
  const index = siblings.findIndex((item) => item.id === afterNodeId);
  if (index === -1) {
    siblings.push(node);
    return;
  }
  siblings.splice(index + 1, 0, node);
}

function removeNode(tree: WorkflowNode[], nodeId: string): WorkflowNode | undefined {
  for (let i = 0; i < tree.length; i += 1) {
    const node = tree[i];
    if (node.id === nodeId) {
      tree.splice(i, 1);
      return node;
    }
    const removed = removeNode(node.children, nodeId);
    if (removed) return removed;
  }
  return undefined;
}

function updateNode(tree: WorkflowNode[], operation: Extract<WorkflowPatchOperation, { op: "update_node" }>): boolean {
  const node = findNode(tree, operation.nodeId);
  if (!node) return false;
  if (operation.description !== undefined) node.description = operation.description;
  if (operation.outputFiles !== undefined) node.outputFiles = [...operation.outputFiles];
  if (operation.status !== undefined) node.status = operation.status;
  if (operation.verifiers !== undefined) {
    node.verifiers = [...operation.verifiers];
    node.verifierMarks = normalizeVerifierMarks(node.verifiers, operation.verifierMarks);
  } else if (operation.verifierMarks !== undefined) {
    node.verifierMarks = normalizeVerifierMarks(node.verifiers, operation.verifierMarks);
  }
  return true;
}

export function applyWorkflowPatch(
  currentWorkflowTree: WorkflowNode[],
  patch: WorkflowPatchOperation[]
): WorkflowNode[] {
  const tree = cloneWorkflowTree(currentWorkflowTree);
  for (const operation of patch) {
    switch (operation.op) {
      case "add_node": {
        const { children, depth } = getChildrenForParent(tree, operation.parentId);
        const node = refreshDepths(
          {
            ...operation.node,
            depth: operation.node.depth ?? depth,
            outputFiles: [...operation.node.outputFiles],
            verifiers: [...operation.node.verifiers],
            verifierMarks: normalizeVerifierMarks(operation.node.verifiers, operation.node.verifierMarks),
            children: cloneWorkflowTree(operation.node.children),
          },
          depth
        );
        insertNode(children, node, operation.afterNodeId);
        break;
      }
      case "update_node":
        if (!updateNode(tree, operation)) {
          throw new Error(`edit_workflow patch: node ${operation.nodeId} not found.`);
        }
        break;
      case "delete_node":
        if (!removeNode(tree, operation.nodeId)) {
          throw new Error(`edit_workflow patch: node ${operation.nodeId} not found.`);
        }
        break;
      case "move_node": {
        const node = removeNode(tree, operation.nodeId);
        if (!node) throw new Error(`edit_workflow patch: node ${operation.nodeId} not found.`);
        const { children, depth } = getChildrenForParent(tree, operation.parentId);
        insertNode(children, refreshDepths(node, depth), operation.afterNodeId);
        break;
      }
    }
  }
  return tree;
}

/**
 * Translate an ExecutableAction into the actual side-effects.
 *
 * For `stop` this is a no-op — included for exhaustiveness so the consumer
 * doesn't need its own discriminator switch.
 */
export async function executeAction(
  action: ExecutableAction,
  ctx: ExecuteContext
): Promise<void> {
  switch (action.type) {
    case "message":
      ctx.sendEvent({
        type: "session.continue",
        payload: {
          sessionId: ctx.sessionId,
          prompt: action.text,
          ...(action.verificationNodeId
            ? { verificationNodeId: action.verificationNodeId }
            : {}),
        },
      });
      return;

    case "edit_workflow":
      if (!ctx.currentWorkflowTree) {
        throw new Error(
          "edit_workflow requires ExecuteContext.currentWorkflowTree to apply the patch."
        );
      }
      ctx.sendEvent({
        type: "session.updateWorkflowTree",
        payload: {
          sessionId: ctx.sessionId,
          workflowTree: applyWorkflowPatch(ctx.currentWorkflowTree, action.patch),
        },
      });
      return;

    case "edit_verifier": {
      if (!ctx.currentWorkflowTree) {
        throw new Error(
          "edit_verifier requires ExecuteContext.currentWorkflowTree to merge the patch into."
        );
      }
      const { tree, changed } = setVerifiersOnTree(
        ctx.currentWorkflowTree,
        action.nodeId,
        action.verifiers
      );
      if (!changed) {
        throw new Error(`edit_verifier: node ${action.nodeId} not found in workflow tree.`);
      }
      ctx.sendEvent({
        type: "session.updateWorkflowTree",
        payload: { sessionId: ctx.sessionId, workflowTree: tree },
      });
      return;
    }

    case "file_edit":
      if (ctx.writeFile) await ctx.writeFile(action.path, action.contents);
      // Server-side file-edit recording is currently triggered from the
      // electron main process (recordFileEditAfterPreviewSave). UI callers
      // should perform the write and rely on the existing IPC bridge to
      // record the event.
      return;

    case "brain_edit": {
      const requestId = `brain-edit-${Date.now()}`;
      if (action.kind === "memory") {
        ctx.sendEvent({
          type: "memory.write",
          payload: {
            requestId,
            sections: action.sections,
            ...(action.deletedFileNames
              ? { deletedFileNames: action.deletedFileNames }
              : {}),
          },
        });
      } else {
        ctx.sendEvent({
          type: "skills.write",
          payload: {
            requestId,
            sections: action.sections,
            ...(action.deletedFileNames
              ? { deletedFileNames: action.deletedFileNames }
              : {}),
          },
        });
      }
      ctx.sendEvent({
        type: "session.recordBrainEdit",
        payload: { sessionId: ctx.sessionId },
      });
      return;
    }

    case "stop":
      return;
  }
}
