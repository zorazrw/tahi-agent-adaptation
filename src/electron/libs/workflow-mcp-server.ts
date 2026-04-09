import { createSdkMcpServer, tool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod/v4";
import { hydrateWorkflowTree, type RawWorkflowNode, type WorkflowNode } from "./workflow-tree-utils.js";

/** Strip any single wrapper root so level 0 is always the real steps (3–5 main steps). Unwrap recursively until we have multiple roots or a leaf. */
function normalizeRoots(tasks: RawWorkflowNode[]): RawWorkflowNode[] {
  let roots = tasks;
  while (roots.length === 1 && roots[0].children && roots[0].children.length > 0) {
    roots = roots[0].children;
  }
  return roots;
}

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
        "Register a hierarchical workflow plan. Provide 3-5 main steps at the top level (no single wrapper root). Each main step must have a visually verifiable output: set outputFiles to file **names only** (e.g. slide.html, report.md)—no directories or absolute paths—or describe in verifiers what the operator can check. Each node has description, outputFiles, verifiers, and optionally children. For control/detail mode: add children to a main step to break it into detailed sub-steps; the number of sub-steps can vary by complexity. Prefer .md for document-style output; use .txt when markdown does not apply. Step execution resolves these names under the user-selected working directory.",
        { tasks: z.array(workflowNodeSchema) },
        async ({ tasks }) => {
          const roots = normalizeRoots(tasks);
          const tree = hydrateWorkflowTree(roots);
          onWorkflowPlan(tree);
          return {
            content: [{ type: "text" as const, text: "Workflow plan registered. STOP now. Do not proceed with any step execution. The human operator will trigger each step individually." }],
          };
        }
      ),
    ],
  });
}
