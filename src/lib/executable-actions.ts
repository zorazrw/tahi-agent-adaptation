import { z } from "zod";
import type { ClientEvent, WorkflowNode } from "./runtime-types.js";

/**
 * Schema for actions a user (or a confident prediction) can *execute*.
 *
 * Distinct from `PredictedUserActionType`, which is just a label the LLM
 * emits. An ExecutableAction carries the payload needed to actually dispatch
 * the action — e.g. `edit_workflow` includes the new workflow tree.
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

const messageAction = z.object({
  type: z.literal("message"),
  text: z.string().min(1),
  verificationNodeId: z.string().optional(),
});

const editWorkflowAction = z.object({
  type: z.literal("edit_workflow"),
  workflowTree: workflowTreeSchema,
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
        return { ...n, verifiers };
      }
      return { ...n, children: visit(n.children) };
    });
  return { tree: visit(tree), changed };
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
      ctx.sendEvent({
        type: "session.updateWorkflowTree",
        payload: { sessionId: ctx.sessionId, workflowTree: action.workflowTree },
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
