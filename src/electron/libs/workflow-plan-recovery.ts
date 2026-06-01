import type { AgentMessage } from "@mariozechner/pi-agent-core";
import {
  isWorkflowPlanToolName,
  normalizeWorkflowPlanRoots,
  parseWorkflowPlanPayload,
  type WorkflowPlanTaskInput,
} from "../../lib/workflow-plan-parse.js";
import type { ServerEvent, WorkflowNode } from "../types.js";
import type { Session } from "./session-store.js";
import { hydrateWorkflowTree, type RawWorkflowNode } from "./workflow-tree-utils.js";

export { isWorkflowPlanToolName } from "../../lib/workflow-plan-parse.js";

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

function toRawWorkflowNodes(tasks: WorkflowPlanTaskInput[]): RawWorkflowNode[] {
  return tasks.map((t) => ({
    description: t.description,
    outputFiles: t.outputFiles,
    verifiers: t.verifiers,
    children: t.children ? toRawWorkflowNodes(t.children) : undefined,
  }));
}

export function extractTasksFromWorkflowPlanArgs(args: unknown): RawWorkflowNode[] | null {
  const parsed = parseWorkflowPlanPayload(args);
  return parsed ? toRawWorkflowNodes(parsed) : null;
}

export function extractTasksFromAssistantText(text: string): RawWorkflowNode[] | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const direct = parseWorkflowPlanPayload(trimmed);
  if (direct) return toRawWorkflowNodes(direct);
  const fence = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  const raw = (fence ? fence[1] : trimmed).trim();
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start !== -1 && end > start) {
    const sliced = parseWorkflowPlanPayload(raw.slice(start, end + 1));
    if (sliced) return toRawWorkflowNodes(sliced);
  }
  return null;
}

export function registerWorkflowPlanFromTasks(
  session: Session,
  tasks: RawWorkflowNode[],
  onEvent: (event: ServerEvent) => void
): WorkflowNode[] {
  const roots = normalizeWorkflowPlanRoots(tasks);
  const tree = hydrateWorkflowTree(roots);
  onEvent({
    type: "workflow.plan",
    payload: { sessionId: session.id, workflowTree: tree },
  });
  return tree;
}

/**
 * If the model emitted a workflow plan in tool args or JSON text but ``planRegistered`` was not set
 * (e.g. alternate tool names), register the tree from session messages.
 */
export function tryRecoverWorkflowPlanFromMessages(
  messages: AgentMessage[],
  session: Session,
  onEvent: (event: ServerEvent) => void
): boolean {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = asRecord(messages[i]);
    if (!message || message.role !== "assistant") continue;
    const content = Array.isArray(message.content) ? message.content : [];
    for (const block of content) {
      const row = asRecord(block);
      if (!row) continue;

      if (row.type === "toolCall" || row.type === "tool_use") {
        const name = String(row.name ?? "");
        if (!isWorkflowPlanToolName(name)) continue;
        const args =
          row.type === "toolCall"
            ? (row.arguments ?? row.args)
            : (row.input ?? row.arguments);
        const tasks = extractTasksFromWorkflowPlanArgs(args);
        if (tasks) {
          registerWorkflowPlanFromTasks(session, tasks, onEvent);
          return true;
        }
      }

      if (row.type === "text" && "text" in row) {
        const tasks = extractTasksFromAssistantText(String(row.text ?? ""));
        if (tasks) {
          registerWorkflowPlanFromTasks(session, tasks, onEvent);
          return true;
        }
      }
    }
  }
  return false;
}
