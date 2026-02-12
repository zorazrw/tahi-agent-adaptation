import { query, type SDKMessage, type PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import type { ServerEvent } from "../types.js";
import type { Session } from "./session-store.js";

import { getCurrentApiConfig, buildEnvForConfig, getClaudeCodePath} from "./claude-settings.js";
import { getEnhancedEnv } from "./util.js";


export type RunnerOptions = {
  prompt: string;
  session: Session;
  resumeSessionId?: string;
  onEvent: (event: ServerEvent) => void;
  onSessionUpdate?: (updates: Partial<Session>) => void;
};

export type RunnerHandle = {
  abort: () => void;
};

const DEFAULT_CWD = process.cwd();

/** Appended to the user's first message so the model produces workflow steps, file names, and verifiers. */
const TODO_LIST_INSTRUCTION = [
  "",
  "For a given task, you must produce exactly three outputs:",
  "  1. Workflow steps (short action/outcome per step)",
  "  2. File name(s) for each step (the expected output file name or path per step)",
  "  3. Verifiers (what to check per step: file exists, content correct, etc.)",
  "",
  "Rules:",
  "- Each step must be within 10 words; ideally use 4 steps or fewer.",
  "- Each step must have clear, tangible file output. In the workflow list describe only the action (e.g. 'Create summary report')—do NOT put file names in the step text.",
  "- Output in this exact format:",
  "",
  "  1. First step (action/outcome only)",
  "  2. Second step",
  "  ...",
  "",
  "OUTPUT FILES:",
  "Step 1: file1.xlsx",
  "Step 2: path/to/file2.png",
  "...",
  "",
  "VERIFIERS:",
  "Step 1:",
  "- Output file exists",
  "- [Optional: main quality check]",
  "Step 2:",
  "- ...",
  "",
  "In OUTPUT FILES you must list the output file name (or path) for each step—one line per step; multiple files on one line separated by commas. In VERIFIERS list only what to check (e.g. 'Output file exists', 'Data is correct'). Use the exact headers OUTPUT FILES:, Step N:, and VERIFIERS:.",
  "",
  "Task instruction:"
].join("\n");

function buildPromptForQuery(userPrompt: string, isFirstMessage: boolean): string {
  const trimmed = userPrompt.trim();
  if (!trimmed) return trimmed;
  if (!isFirstMessage) return trimmed;
  return TODO_LIST_INSTRUCTION + trimmed;
}

/** Builds the user prompt for solving a single workflow step (used for task-solving LLM calls). */
export function buildPromptForStep(stepDescription: string, stepIndex: number, totalSteps: number): string {
  const oneBased = stepIndex + 1;
  return [
    `Execute step ${oneBased} of ${totalSteps} of the workflow. Complete only this sub-task.`,
    "",
    `Step: ${stepDescription}`
  ].join("\n");
}

export async function runClaude(options: RunnerOptions): Promise<RunnerHandle> {
  const { prompt, session, resumeSessionId, onEvent, onSessionUpdate } = options;
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
      
      // 使用 Anthropic SDK
      const env = buildEnvForConfig(config);
      const mergedEnv = {
        ...getEnhancedEnv(),
        ...env
      };
      
      const q = query({
        prompt: promptToSend,
        options: {
          cwd: session.cwd ?? DEFAULT_CWD,
          resume: resumeSessionId,
          abortController,
          env: mergedEnv,
          pathToClaudeCodeExecutable: getClaudeCodePath(),
          permissionMode: "bypassPermissions",
          includePartialMessages: true,
          allowDangerouslySkipPermissions: true,
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

      // Capture session_id from init message
      for await (const message of q) {
        // Extract session_id from system init message
        if (message.type === "system" && "subtype" in message && message.subtype === "init") {
          const sdkSessionId = message.session_id;
          if (sdkSessionId) {
            session.claudeSessionId = sdkSessionId;
            onSessionUpdate?.({ claudeSessionId: sdkSessionId });
          }
        }

        // Send message to frontend
        sendMessage(message);

        // Check for result to update session status
        if (message.type === "result") {
          const status = message.subtype === "success" ? "completed" : "error";
          onEvent({
            type: "session.status",
            payload: { sessionId: session.id, status, title: session.title }
          });
        }
      }

      // Query completed normally
      if (session.status === "running") {
        onEvent({
          type: "session.status",
          payload: { sessionId: session.id, status: "completed", title: session.title }
        });
      }
    } catch (error) {
      if ((error as Error).name === "AbortError") {
        // Session was aborted, don't treat as error
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
