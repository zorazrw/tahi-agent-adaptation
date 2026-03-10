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
  /** When rerunning a node, resume the SDK conversation up to this message UUID. */
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
  "Keep it simple. For small tasks use a flat list (no children). For medium tasks use one level of children. Only use 3 levels for genuinely complex tasks.",
  "Do NOT add separate validation/verification/testing steps — our system handles verification natively via verifier criteria on each node.",
  "Keep descriptions short but complete (under 10 words). Each node needs: description, outputFiles, verifiers, and optionally children.",
  "Aim for fewer, meaningful steps rather than many granular ones.",
  "After calling the tool, STOP. Do NOT execute any steps yourself.",
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

/** Builds the resume prompt for executing a single workflow node. */
export function buildPromptForNode(nodeDescription: string, pathContext: string): string {
  return `Proceed with: ${pathContext}\n\nTask: ${nodeDescription}`;
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
          const sdkSessionId = message.session_id;
          if (sdkSessionId) {
            session.claudeSessionId = sdkSessionId;
          }
        }

        sendMessage(message);

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

      if (gotSuccessResult || planRegistered) {
        onSessionUpdate?.({ claudeSessionId: session.claudeSessionId });
      } else {
        log(`[runner] Run did not succeed; restoring claudeSessionId to ${savedClaudeSessionId}`);
        session.claudeSessionId = savedClaudeSessionId;
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
