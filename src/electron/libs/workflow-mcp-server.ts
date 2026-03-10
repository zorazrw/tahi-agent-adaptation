import { createSdkMcpServer, tool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod/v4";
import { hydrateWorkflowTree, type RawWorkflowNode, type WorkflowNode } from "./workflow-tree-utils.js";

/** Recursive Zod schema for workflow nodes with optional children. */
const workflowNodeSchema: z.ZodType<RawWorkflowNode> = z.lazy(() =>
  z.object({
    description: z.string(),
    outputFiles: z.array(z.string()),
    verifiers: z.array(z.string()),
    children: z.array(workflowNodeSchema).optional(),
  })
);

export type WorkflowPlanInput = {
  tasks: RawWorkflowNode[];
};

export function createWorkflowMcpServer(
  onWorkflowPlan: (workflowTree: WorkflowNode[]) => void
) {
  return createSdkMcpServer({
    name: "workflow",
    version: "1.0.0",
    tools: [
      tool(
        "WorkflowPlan",
        "Register a hierarchical workflow plan. Structure as a tree: one root task containing 3-5 phases, each with child tasks. Each node has description, outputFiles, verifiers, and optionally children.",
        { tasks: z.array(workflowNodeSchema) },
        async ({ tasks }) => {
          const tree = hydrateWorkflowTree(tasks);
          onWorkflowPlan(tree);
          return {
            content: [{ type: "text" as const, text: "Workflow plan registered. STOP now. Do not proceed with any step execution. The human operator will trigger each step individually." }],
          };
        }
      ),
    ],
  });
}
