import { createSdkMcpServer, tool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod/v4";

const stepSchema = z.object({
  description: z.string(),
  outputFiles: z.array(z.string()),
  verifiers: z.array(z.string()),
});

export type WorkflowPlanInput = {
  steps: Array<{
    description: string;
    outputFiles: string[];
    verifiers: string[];
  }>;
};

/** Convert the nested per-step schema into flat arrays the rest of the app expects. */
export function flattenWorkflowPlan(input: WorkflowPlanInput): {
  steps: string[];
  outputFiles: string[][];
  verificationCriteria: string[][];
} {
  const steps: string[] = [];
  const outputFiles: string[][] = [];
  const verificationCriteria: string[][] = [];
  for (const step of input.steps) {
    steps.push(step.description);
    outputFiles.push(step.outputFiles);
    verificationCriteria.push(step.verifiers);
  }
  return { steps, outputFiles, verificationCriteria };
}

export function createWorkflowMcpServer(
  onWorkflowPlan: (input: WorkflowPlanInput) => void
) {
  return createSdkMcpServer({
    name: "workflow",
    version: "1.0.0",
    tools: [
      tool(
        "WorkflowPlan",
        "Register a structured workflow plan with steps, output files, and verification criteria. Call this once at the start of a task to define the execution plan.",
        { steps: z.array(stepSchema) },
        async ({ steps }) => {
          onWorkflowPlan({ steps });
          return {
            content: [{ type: "text" as const, text: "Workflow plan registered. STOP now. Do not proceed with any step execution. The human operator will trigger each step individually." }],
          };
        }
      ),
    ],
  });
}
