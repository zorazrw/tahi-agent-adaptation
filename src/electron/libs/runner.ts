import { query, type SDKMessage, type PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type { ServerEvent } from "../types.js";
import type { Session } from "./session-store.js";

import { getCurrentApiConfig, buildEnvForConfig, getClaudeCodePath} from "./claude-settings.js";
import { getEnhancedEnv } from "./util.js";
import { syncAppSkills } from "./skill-store.js";
import { createWorkflowMcpServer } from "./workflow-mcp-server.js";
import type { WorkflowNode } from "./workflow-tree-utils.js";
import { appendFileSync } from "fs";
import { join } from "path";
import { app } from "electron";
import { readMemoryForPrompt } from "./memory-store.js";

const LOG_PATH = join(app.getPath("userData"), "runner.log");

function log(msg: string) {
  const line = `[${new Date().toISOString()}] ${msg}\n`;
  try { appendFileSync(LOG_PATH, line); } catch { /* ignore */ }
  console.error(line.trimEnd());
}


export type RunnerOptions = {
  prompt: string;
  session: Session;
  /** When true, run only to get a new workflow plan; do not update session.claudeSessionId. */
  regenerateWorkflow?: boolean;
  resumeSessionId?: string;
  /** When rerunning a node, resume the SDK conversation up to this message UUID. */
  resumeSessionAt?: string;
  /**
   * When true, do not emit session.status "completed" on a successful result message.
   * Used for node solves so ipc-handlers can run verifier labeling first, then emit completed.
   */
  suppressSessionStatusOnSuccess?: boolean;
  onEvent: (event: ServerEvent) => void;
  onSessionUpdate?: (updates: Partial<Session>) => void;
};

export type RunnerHandle = {
  abort: () => void;
};

const DEFAULT_CWD = process.cwd();

/** Appended to the user's first message so the model calls the WorkflowPlan MCP tool. */
const WORKFLOW_PLAN_INSTRUCTION = [
  "",
  "IMPORTANT: You MUST call the mcp__workflow__WorkflowPlan tool as your very first action to register a structured plan.",
  "Do NOT write out steps as text. Use the tool with structured JSON input.",
  "Structure: Provide 3-5 main steps at the top level. Do NOT add a single wrapper root that repeats the task.",
  "Each main step (automation / level 0) must have a visually verifiable output: set outputFiles to a path (e.g. report.md, summary.txt) or use verifiers to describe what the operator can check (e.g. file exists, content contains X).",
  "For control mode (detailed view): add optional children to any main step to break it into detailed sub-steps; the number of sub-steps can depend on that step's complexity.",
  "Do NOT add separate validation/verification/testing steps — our system handles verification via verifier criteria on each node.",
  "Keep descriptions short but complete (under 10 words). Each node needs: description, outputFiles, verifiers, and optionally children.",
  "For outputFiles: prefer .md for document-style output so the UI shows markdown preview; use .txt when markdown does not apply.",
  "After calling the tool, STOP. Do NOT execute any steps yourself.",
  "The human operator will trigger each step individually.",
  "",
  "Task instruction:"
].join("\n");

/** Builds the prompt used when re-generating the workflow plan (regenerateWorkflow run). */
export function buildRegenerateWorkflowPrompt(taskSummary: string): string {
  const task = (taskSummary || "Current task").trim();
  return (
    WORKFLOW_PLAN_INSTRUCTION +
    "\nRe-generate the workflow plan for this task. Call the WorkflowPlan tool with a new plan. Do not execute any steps.\n\nTask: " +
    task
  );
}

function buildPromptForQuery(userPrompt: string, isFirstMessage: boolean): string {
  const trimmed = userPrompt.trim();
  if (!trimmed) return trimmed;
  if (!isFirstMessage) return trimmed;
  return WORKFLOW_PLAN_INSTRUCTION + trimmed;
}

/** Builds the resume prompt for executing a single workflow node. */
export function buildPromptForNode(
  nodeDescription: string,
  pathContext: string,
  outputFiles: string[] = [],
  humanEdits?: string
): string {
  const hasMd = outputFiles.some((f) => f.toLowerCase().endsWith(".md"));
  const formatNote = hasMd
    ? "\n\nWhen writing output to .md files, use markdown format (headers, lists, code blocks, etc.) so the file preview shows formatted content."
    : "";
  const filesNote =
    outputFiles.length > 0
      ? "\n\nRelevant output files for this step (these should be treated as the source of truth when refining your work):\n" +
        outputFiles.map((f) => `- ${f}`).join("\n")
      : "";
  const refinementNote =
    "\n\nWhen refining or updating existing outputs, you MUST first call the Read tool to load the latest on-disk contents of any relevant output files, " +
    "then apply edits on top of that version using Edit or Write. Do NOT recreate files from memory or discard existing content, and do NOT revert to older model-only drafts.";
  const editsNote = humanEdits
    ? "\n\nThe human has manually edited your previous outputs. For each file below you will see:\n" +
      "(1) the original model-written output, and\n" +
      "(2) the current version after human edits.\n\n" +
      "You MUST treat version (2) as the authoritative base text and ONLY apply further changes on top of it. " +
      "Never discard or overwrite the human-edited version, and never recreate content purely from memory of earlier drafts.\n\n" +
      "Human-edited files (showing original vs current):\n" +
      humanEdits
    : "";
  return `Proceed with: ${pathContext}\n\nTask: ${nodeDescription}${filesNote}${formatNote}${refinementNote}${editsNote}`;
}

export async function runClaude(options: RunnerOptions): Promise<RunnerHandle> {
  const {
    prompt,
    session,
    regenerateWorkflow,
    resumeSessionId,
    resumeSessionAt,
    suppressSessionStatusOnSuccess,
    onEvent,
    onSessionUpdate,
  } = options;
  const abortController = new AbortController();
  const isFirstMessage = resumeSessionId == null;
  const basePrompt = regenerateWorkflow ? prompt : buildPromptForQuery(prompt, isFirstMessage);
  const memoryPrefix = readMemoryForPrompt();
  const promptToSend = memoryPrefix ? memoryPrefix + basePrompt : basePrompt;

  // Emit the fully constructed prompt being sent to the LM for debugging/inspection in the UI.
  onEvent({
    type: "session.effectivePrompt",
    payload: { sessionId: session.id, prompt: promptToSend }
  });

  const sendMessage = (message: SDKMessage) => {
    onEvent({
      type: "stream.message",
      payload: { sessionId: session.id, message }
    });
  };

  const sendPermissionRequest = (toolUseId: string, toolName: string, input: unknown) => {
    onEvent({
      type: "permission.request",
      payload: { sessionId: session.id, toolUseId, toolName, input }
    });
  };

  // Start the query in the background
  (async () => {
    const savedClaudeSessionId = session.claudeSessionId;

    try {
      const config = getCurrentApiConfig();

      if (!config) {
        onEvent({
          type: "session.status",
          payload: { sessionId: session.id, status: "error", title: session.title, cwd: session.cwd, error: "API configuration not found. Please configure API settings." }
        });
        return;
      }

      syncAppSkills();

      const env = buildEnvForConfig(config);
      const mergedEnv = {
        ...getEnhancedEnv(),
        ...env
      };

      const defaultTools = [
        "Task", "TaskOutput", "Bash", "Glob", "Grep", "ExitPlanMode", "Read", "Edit", "Write",
        "NotebookEdit", "WebFetch", "WebSearch", "KillShell", "AskUserQuestion",
        "Skill", "EnterPlanMode"
      ];
      const codeExecutionTool = "CodeExecution";
      const toolsList = defaultTools.includes(codeExecutionTool)
        ? defaultTools
        : [...defaultTools, codeExecutionTool];

      let planRegistered = false;
      let mcpServers: Record<string, ReturnType<typeof createWorkflowMcpServer>> | undefined;
      if (isFirstMessage) {
        mcpServers = {
          workflow: createWorkflowMcpServer((workflowTree: WorkflowNode[]) => {
            onEvent({
              type: "workflow.plan",
              payload: { sessionId: session.id, workflowTree }
            });
            planRegistered = true;
          })
        };
      }

      log(`[runner] Starting query: resume=${resumeSessionId ?? "none"}, resumeAt=${resumeSessionAt ?? "none"}, prompt="${promptToSend.slice(0, 80)}..."`);

      const q = query({
        prompt: promptToSend,
        options: {
          cwd: session.cwd ?? DEFAULT_CWD,
          settingSources: ["user", "project"],
          maxThinkingTokens: 0,
          resume: resumeSessionId,
          resumeSessionAt,
          abortController,
          env: mergedEnv,
          pathToClaudeCodeExecutable: getClaudeCodePath(),
          permissionMode: "bypassPermissions",
          includePartialMessages: true,
          allowDangerouslySkipPermissions: true,
          tools: toolsList,
          mcpServers,
          stderr: (data: string) => {
            log(`[runner:stderr] ${data.trimEnd()}`);
          },
          canUseTool: async (toolName, input, { signal }) => {
            if (toolName === "AskUserQuestion") {
              const toolUseId = crypto.randomUUID();

              sendPermissionRequest(toolUseId, toolName, input);

              return new Promise<PermissionResult>((resolve) => {
                session.pendingPermissions.set(toolUseId, {
                  toolUseId,
                  toolName,
                  input,
                  resolve: (result) => {
                    session.pendingPermissions.delete(toolUseId);
                    resolve(result as PermissionResult);
                  }
                });

                signal.addEventListener("abort", () => {
                  session.pendingPermissions.delete(toolUseId);
                  resolve({ behavior: "deny", message: "Session aborted" });
                });
              });
            }

            return { behavior: "allow", updatedInput: input };
          }
        }
      });

      let messageCount = 0;
      let gotSuccessResult = false;
      for await (const message of q) {
        messageCount++;
        log(`[runner] msg #${messageCount}: type=${message.type}${
          "subtype" in message ? ` subtype=${(message as any).subtype}` : ""
        }${message.type === "result" ? ` cost=$${(message as any).total_cost_usd?.toFixed(4)}` : ""}`);

        if (message.type === "system" && "subtype" in message && message.subtype === "init") {
          if (!regenerateWorkflow) {
            const sdkSessionId = message.session_id;
            if (sdkSessionId) session.claudeSessionId = sdkSessionId;
          }
        }

        sendMessage(message);

        if (message.type === "result") {
          if (message.subtype === "success") gotSuccessResult = true;
          if (!planRegistered) {
            const status = message.subtype === "success" ? "completed" : "error";
            const skipOkStatus = suppressSessionStatusOnSuccess && message.subtype === "success";
            if (!skipOkStatus) {
              onEvent({
                type: "session.status",
                payload: { sessionId: session.id, status, title: session.title }
              });
            }
          }
        }
      }

      if (regenerateWorkflow) {
        session.claudeSessionId = savedClaudeSessionId;
        if (!planRegistered) {
          onEvent({
            type: "session.status",
            payload: { sessionId: session.id, status: "idle", title: session.title, cwd: session.cwd }
          });
        }
      } else {
        if (gotSuccessResult || planRegistered) {
          onSessionUpdate?.({ claudeSessionId: session.claudeSessionId });
        } else {
          log(`[runner] Run did not succeed; restoring claudeSessionId to ${savedClaudeSessionId}`);
          session.claudeSessionId = savedClaudeSessionId;
        }
      }

      log(`[runner] Query finished. Total messages: ${messageCount}`);
    } catch (error) {
      log(`[runner] Query error: ${error instanceof Error ? error.stack ?? error.message : String(error)}`);

      if (savedClaudeSessionId) {
        session.claudeSessionId = savedClaudeSessionId;
        onSessionUpdate?.({ claudeSessionId: savedClaudeSessionId });
      }

      if ((error as Error).name === "AbortError") {
        return;
      }
      onEvent({
        type: "session.status",
        payload: { sessionId: session.id, status: "error", title: session.title, error: String(error) }
      });
    }
  })();

  return {
    abort: () => abortController.abort()
  };
}
