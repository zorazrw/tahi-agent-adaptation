import { query, type SDKMessage, type PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type { ServerEvent } from "../types.js";
import type { Session } from "./session-store.js";

import { getCurrentApiConfig, buildEnvForConfig, getClaudeCodePath} from "./claude-settings.js";
import { getEnhancedEnv } from "./util.js";
import { syncAppSkills } from "./skill-store.js";
import { createWorkflowMcpServer, flattenWorkflowPlan } from "./workflow-mcp-server.js";
import { appendFileSync } from "fs";
import { join } from "path";
import { app } from "electron";

const LOG_PATH = join(app.getPath("userData"), "runner.log");

function log(msg: string) {
  const line = `[${new Date().toISOString()}] ${msg}\n`;
  try { appendFileSync(LOG_PATH, line); } catch { /* ignore */ }
  console.error(line.trimEnd());
}


export type RunnerOptions = {
  prompt: string;
  session: Session;
  resumeSessionId?: string;
  /** When rerunning a step, resume the SDK conversation up to this message UUID. */
  resumeSessionAt?: string;
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
  "Each step needs: a short description (under 10 words), expected output file(s), and verification criteria.",
  "Aim for 4 steps or fewer. After calling the tool, STOP. Do NOT execute any steps yourself.",
  "The human operator will trigger each step individually.",
  "",
  "Task instruction:"
].join("\n");

function buildPromptForQuery(userPrompt: string, isFirstMessage: boolean): string {
  const trimmed = userPrompt.trim();
  if (!trimmed) return trimmed;
  if (!isFirstMessage) return trimmed;
  return WORKFLOW_PLAN_INSTRUCTION + trimmed;
}

/** Builds the resume prompt for executing a single workflow step. */
export function buildPromptForStep(stepDescription: string, stepIndex: number, totalSteps: number): string {
  const oneBased = stepIndex + 1;
  return `Proceed with step ${oneBased} of ${totalSteps}: ${stepDescription}`;
}

export async function runClaude(options: RunnerOptions): Promise<RunnerHandle> {
  const { prompt, session, resumeSessionId, resumeSessionAt, onEvent, onSessionUpdate } = options;
  const abortController = new AbortController();
  const isFirstMessage = resumeSessionId == null;
  const promptToSend = buildPromptForQuery(prompt, isFirstMessage);

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
    // Save claudeSessionId so we can restore it if the run crashes.
    // Must be outside try/catch so the catch block can access it.
    const savedClaudeSessionId = session.claudeSessionId;

    try {
      // 获取当前配置
      const config = getCurrentApiConfig();

      if (!config) {
        onEvent({
          type: "session.status",
          payload: { sessionId: session.id, status: "error", title: session.title, cwd: session.cwd, error: "API configuration not found. Please configure API settings." }
        });
        return;
      }

      // Sync app-managed skills into ~/.claude/skills/ for SDK discovery
      syncAppSkills();

      // 使用 Anthropic SDK
      const env = buildEnvForConfig(config);
      const mergedEnv = {
        ...getEnhancedEnv(),
        ...env
      };

      // Build tools list: default Claude Code tools + CodeExecution (if not already present)
      const defaultTools = [
        "Task", "TaskOutput", "Bash", "Glob", "Grep", "ExitPlanMode", "Read", "Edit", "Write",
        "NotebookEdit", "WebFetch", "WebSearch", "KillShell", "AskUserQuestion",
        "Skill", "EnterPlanMode"
      ];
      const codeExecutionTool = "CodeExecution";
      const toolsList = defaultTools.includes(codeExecutionTool)
        ? defaultTools
        : [...defaultTools, codeExecutionTool];

      // Only provide the workflow MCP server on the first message (planning phase).
      // The agent is instructed to call WorkflowPlan then stop naturally.
      // On resume (step execution), omit it so the agent focuses on the step.
      let planRegistered = false;
      let mcpServers: Record<string, ReturnType<typeof createWorkflowMcpServer>> | undefined;
      if (isFirstMessage) {
        mcpServers = {
          workflow: createWorkflowMcpServer((input) => {
            const { steps, outputFiles, verificationCriteria } = flattenWorkflowPlan(input);
            onEvent({
              type: "workflow.plan",
              payload: { sessionId: session.id, steps, outputFiles, verificationCriteria }
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
            // For AskUserQuestion, we need to wait for user response
            if (toolName === "AskUserQuestion") {
              const toolUseId = crypto.randomUUID();

              // Send permission request to frontend
              sendPermissionRequest(toolUseId, toolName, input);

              // Create a promise that will be resolved when user responds
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

                // Handle abort
                signal.addEventListener("abort", () => {
                  session.pendingPermissions.delete(toolUseId);
                  resolve({ behavior: "deny", message: "Session aborted" });
                });
              });
            }

            // Auto-approve other tools
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
          const sdkSessionId = message.session_id;
          if (sdkSessionId) {
            // Update locally so this run can reference it, but we'll
            // only persist if the run succeeds (see below).
            session.claudeSessionId = sdkSessionId;
          }
        }

        sendMessage(message);

        // When the agent finishes a turn, update status — but skip if
        // a plan was just registered (the workflow.plan handler already set "idle").
        if (message.type === "result") {
          if (message.subtype === "success") gotSuccessResult = true;
          if (!planRegistered) {
            const status = message.subtype === "success" ? "completed" : "error";
            onEvent({
              type: "session.status",
              payload: { sessionId: session.id, status, title: session.title }
            });
          }
        }
      }

      // Only persist the new claudeSessionId if the run completed successfully.
      // This prevents cascading failures where a crashed run's session ID
      // corrupts subsequent resume attempts.
      if (gotSuccessResult || planRegistered) {
        onSessionUpdate?.({ claudeSessionId: session.claudeSessionId });
      } else {
        log(`[runner] Run did not succeed; restoring claudeSessionId to ${savedClaudeSessionId}`);
        session.claudeSessionId = savedClaudeSessionId;
      }

      log(`[runner] Query finished. Total messages: ${messageCount}`);
    } catch (error) {
      log(`[runner] Query error: ${error instanceof Error ? error.stack ?? error.message : String(error)}`);

      // Restore claudeSessionId so the next retry uses the last-known-good session
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
