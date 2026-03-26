import { Type } from "@sinclair/typebox";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import {
  createAgentSession,
  createBashTool,
  createEditTool,
  createFindTool,
  createGrepTool,
  createLsTool,
  createReadTool,
  createWriteTool,
  type AgentSessionEvent,
  type ToolDefinition,
} from "@mariozechner/pi-coding-agent";
import type { AppPermissionResult, PiAssistantBlock, ServerEvent, StreamMessage } from "../types.js";
import { createPiManagers, createPiResourceLoader, createPiSessionManager } from "./pi-config.js";
import type { Session } from "./session-store.js";
import { hydrateWorkflowTree, type RawWorkflowNode } from "./workflow-tree-utils.js";

export type RunnerOptions = {
  prompt: string;
  session: Session;
  regenerateWorkflow?: boolean;
  branchEntryId?: string;
  onEvent: (event: ServerEvent) => void;
  onSessionUpdate?: (updates: Partial<Session>) => void;
};

export type RunnerHandle = {
  abort: () => void;
};

const WORKFLOW_PLAN_APPEND_SYSTEM_PROMPT = [
  "IMPORTANT: You MUST call the workflow_plan tool as your very first action to register a structured plan.",
  "Do NOT write out steps as text. Use the tool with structured JSON input.",
  "Structure: Provide 3-5 main steps at the top level. Do NOT add a single wrapper root that repeats the task.",
  "Each main step must have a visually verifiable output: use outputFiles or clear verifiers.",
  "You may add children to break a main step into detailed sub-steps when useful.",
  "Do NOT add separate validation/testing steps. Express checks inside each step's verifiers.",
  "Keep descriptions short but complete. Each node needs description, outputFiles, verifiers, and optional children.",
  "Prefer .md for document-style outputs when markdown preview is useful.",
  "After calling workflow_plan, STOP. Do not execute any steps yourself.",
  "The human operator will trigger each step individually.",
].join("\n");

function normalizeRoots(tasks: RawWorkflowNode[]): RawWorkflowNode[] {
  let roots = tasks;
  while (roots.length === 1 && roots[0].children && roots[0].children.length > 0) {
    roots = roots[0].children;
  }
  return roots;
}

function buildPromptForQuery(userPrompt: string, isFirstMessage: boolean): string {
  const trimmed = userPrompt.trim();
  if (!trimmed) return trimmed;
  return trimmed;
}

export function buildRegenerateWorkflowPrompt(taskSummary: string): string {
  const task = (taskSummary || "Current task").trim();
  return `Re-generate the workflow plan for this task. Call workflow_plan with a new plan. Do not execute any steps.\n\nTask: ${task}`;
}

export function buildPromptForNode(
  nodeDescription: string,
  pathContext: string,
  outputFiles: string[] = [],
  humanEdits?: string,
  /** Session cwd; Read/Write/Edit are resolved under this directory. */
  sessionCwd?: string
): string {
  const cwd = (sessionCwd ?? "").trim();
  const cwdNote = cwd
    ? "\n\nWorking directory for this session (the agent runs with this as cwd). For Read, Write, Edit, and any file paths, use each output file as a path **relative to this directory** (e.g. the basename `report.md` means read/write under this folder). Do not place outputs outside this directory unless the task explicitly requires it.\n" +
      `Working directory: ${cwd}`
    : "\n\nUse paths relative to the session working directory for all Read, Write, and Edit calls.";
  const hasMd = outputFiles.some((f) => f.toLowerCase().endsWith(".md"));
  const formatNote = hasMd
    ? "\n\nWhen writing output to .md files, use markdown format so the file preview renders properly."
    : "";
  const filesNote =
    outputFiles.length > 0
      ? "\n\nRelevant output files for this step:\n" + outputFiles.map((f) => `- ${f}`).join("\n")
      : "";
  const refinementNote =
    "\n\nWhen refining existing outputs, first read the current on-disk contents, then edit on top of that version. Do not recreate files from memory.";
  const editsNote = humanEdits
    ? "\n\nThe human has manually edited previous outputs. Treat the current version as the authoritative base.\n\nHuman-edited files:\n" +
      humanEdits
    : "";
  return `Proceed with: ${pathContext}\n\nTask: ${nodeDescription}${cwdNote}${filesNote}${formatNote}${refinementNote}${editsNote}`;
}

function stringifyToolContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "type" in item && item.type === "text" && "text" in item) {
          return String(item.text ?? "");
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function extractUserPrompt(message: Record<string, unknown>): string {
  const content = message.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (!item || typeof item !== "object") return "";
        if ("type" in item && item.type === "text" && "text" in item) {
          return String(item.text ?? "");
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

function normalizeAssistantBlocks(message: Record<string, unknown>, includeToolUses: boolean): PiAssistantBlock[] {
  const content = Array.isArray(message.content) ? message.content : [];
  const blocks: PiAssistantBlock[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object" || !("type" in block)) continue;
    if (block.type === "text" && "text" in block) {
      blocks.push({ type: "text", text: String(block.text ?? "") });
    } else if (block.type === "thinking" && "thinking" in block) {
      blocks.push({ type: "thinking", thinking: String(block.thinking ?? "") });
    } else if (includeToolUses && block.type === "toolCall") {
      blocks.push({
        type: "tool_use",
        id: String((block as { id?: unknown }).id ?? crypto.randomUUID()),
        name: String((block as { name?: unknown }).name ?? "tool"),
        input: ((block as { arguments?: Record<string, unknown> }).arguments ?? {}) as Record<string, unknown>,
      });
    }
  }
  return blocks;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

function buildCanonicalHistory(
  sessionFile: string | undefined,
  provider: string | undefined,
  model: string | undefined,
  cwd: string | undefined,
  thinkingLevel: string | undefined,
  messages: AgentMessage[],
  runResult?: {
    status: "success" | "error" | "aborted";
    error?: string;
    usage?: {
      input?: number;
      output?: number;
      cacheRead?: number;
      cacheWrite?: number;
      totalTokens?: number;
      cost?: {
        input?: number;
        output?: number;
        cacheRead?: number;
        cacheWrite?: number;
        total?: number;
      };
    };
  }
): StreamMessage[] {
  const history: StreamMessage[] = [
    {
      type: "system_init",
      engine: "pi",
      sessionFile,
      provider,
      model,
      cwd,
      thinkingLevel,
    },
  ];

  for (const rawMessage of messages) {
    const message = asRecord(rawMessage);
    if (!message) continue;
    if (message.role === "user") {
      history.push({
        type: "user_prompt",
        prompt: extractUserPrompt(message),
      });
      continue;
    }

    if (message.role === "assistant") {
      const blocks = normalizeAssistantBlocks(message, true);
      if (blocks.length > 0) {
        history.push({
          type: "assistant",
          engine: "pi",
          id: crypto.randomUUID(),
          blocks,
          provider: typeof message.provider === "string" ? message.provider : provider,
          model: typeof message.model === "string" ? message.model : model,
          stopReason: typeof message.stopReason === "string" ? message.stopReason : undefined,
          timestamp: typeof message.timestamp === "number" ? message.timestamp : undefined,
        });
      }
      continue;
    }

    if (message.role === "toolResult") {
      history.push({
        type: "tool_result",
        engine: "pi",
        toolUseId: String(message.toolCallId ?? ""),
        toolName: String(message.toolName ?? "tool"),
        content: stringifyToolContent(message.content),
        isError: Boolean(message.isError),
        details: message.details,
        timestamp: typeof message.timestamp === "number" ? message.timestamp : undefined,
      });
    }
  }

  if (runResult) {
    history.push({
      type: "run_result",
      engine: "pi",
      status: runResult.status,
      error: runResult.error,
      usage: runResult.usage,
      timestamp: Date.now(),
    });
  }

  return history;
}

function resolveRunStatus(planRegistered: boolean, regenerateWorkflow: boolean | undefined): "idle" | "completed" {
  if (planRegistered || regenerateWorkflow) return "idle";
  return "completed";
}

function createWorkflowPlanTool(session: Session, onEvent: (event: ServerEvent) => void): ToolDefinition {
  const workflowNodeSchema = Type.Recursive((Self) =>
    Type.Object({
      description: Type.String(),
      outputFiles: Type.Array(Type.String()),
      verifiers: Type.Array(Type.String()),
      children: Type.Optional(Type.Array(Self)),
    })
  );

  return {
    name: "workflow_plan",
    label: "Workflow Plan",
    description:
      "Register a hierarchical workflow plan. Provide 3-5 main steps at the top level with description, outputFiles, verifiers, and optional children.",
    promptSnippet: "workflow_plan: register a hierarchical task plan before doing any work",
    parameters: Type.Object({
      tasks: Type.Array(workflowNodeSchema),
    }),
    execute: async (_toolCallId, params) => {
      const roots = normalizeRoots((params as { tasks: RawWorkflowNode[] }).tasks);
      const tree = hydrateWorkflowTree(roots);
      onEvent({
        type: "workflow.plan",
        payload: { sessionId: session.id, workflowTree: tree },
      });
      return {
        content: [
          {
            type: "text",
            text: "Workflow plan registered. Stop now. Do not execute any steps.",
          },
        ],
        details: { workflowTree: tree },
      };
    },
  };
}

function createAskUserQuestionTool(session: Session, onEvent: (event: ServerEvent) => void): ToolDefinition {
  const questionOption = Type.Object({
    label: Type.String(),
    description: Type.Optional(Type.String()),
  });
  const question = Type.Object({
    question: Type.String(),
    header: Type.Optional(Type.String()),
    options: Type.Optional(Type.Array(questionOption)),
    multiSelect: Type.Optional(Type.Boolean()),
  });

  return {
    name: "ask_user_question",
    label: "Ask User Question",
    description: "Ask the operator a structured question and wait for the answer.",
    parameters: Type.Object({
      questions: Type.Array(question),
      answers: Type.Optional(Type.Record(Type.String(), Type.String())),
    }),
    execute: async (_toolCallId, params, signal) => {
      const toolUseId = crypto.randomUUID();
      onEvent({
        type: "permission.request",
        payload: {
          sessionId: session.id,
          toolUseId,
          toolName: "ask_user_question",
          input: params,
        },
      });

      const result = await new Promise<AppPermissionResult>((resolve) => {
        session.pendingPermissions.set(toolUseId, {
          toolUseId,
          toolName: "ask_user_question",
          input: params,
          resolve: (permissionResult) => {
            session.pendingPermissions.delete(toolUseId);
            resolve(permissionResult);
          },
        });

        signal?.addEventListener("abort", () => {
          session.pendingPermissions.delete(toolUseId);
          resolve({ behavior: "deny", message: "Session aborted" });
        });
      });

      if (result.behavior === "deny") {
        return {
          content: [{ type: "text", text: result.message ?? "User declined to answer." }],
          details: { denied: true },
        };
      }

      const updatedInput = (result.updatedInput ?? params) as Record<string, unknown>;
      const answers = (updatedInput.answers ?? {}) as Record<string, string>;
      const summary =
        Object.keys(answers).length > 0
          ? Object.entries(answers)
              .map(([key, value]) => `${key}: ${value}`)
              .join("\n")
          : "User answered the questions.";

      return {
        content: [{ type: "text", text: summary }],
        details: { answers },
      };
    },
  };
}

export async function runClaude(options: RunnerOptions): Promise<RunnerHandle> {
  const { prompt, session, regenerateWorkflow, branchEntryId, onEvent, onSessionUpdate } = options;
  let disposed = false;
  let piSessionAbort: (() => Promise<void>) | undefined;

  (async () => {
    const cwd = session.cwd ?? process.cwd();
    const sessionManager = createPiSessionManager(session.id, cwd, session.piSessionFile);
    if (branchEntryId) {
      sessionManager.branch(branchEntryId);
    }

    const { agentDir, authStorage, modelRegistry, settingsManager } = createPiManagers(cwd);
    const isFirstMessage = !session.piSessionFile;
    const promptToSend = regenerateWorkflow ? prompt : buildPromptForQuery(prompt, isFirstMessage);
    const resourceLoader = await createPiResourceLoader(cwd, {
      appendSystemPrompt: isFirstMessage || regenerateWorkflow ? WORKFLOW_PLAN_APPEND_SYSTEM_PROMPT : undefined,
    });
    let planRegistered = false;
    let lastUsage: Record<string, unknown> | undefined;

    onEvent({
      type: "session.effectivePrompt",
      payload: { sessionId: session.id, prompt: promptToSend },
    });

    const { session: piSession, modelFallbackMessage } = await createAgentSession({
      cwd,
      agentDir,
      authStorage,
      modelRegistry,
      settingsManager,
      resourceLoader,
      sessionManager,
      tools: [
        createReadTool(cwd),
        createBashTool(cwd),
        createEditTool(cwd),
        createWriteTool(cwd),
        createGrepTool(cwd),
        createFindTool(cwd),
        createLsTool(cwd),
      ],
      customTools: [createWorkflowPlanTool(session, onEvent), createAskUserQuestionTool(session, onEvent)],
    });

    piSessionAbort = () => piSession.abort();

    if (modelFallbackMessage && !piSession.model) {
      onEvent({
        type: "session.status",
        payload: {
          sessionId: session.id,
          status: "error",
          title: session.title,
          cwd: session.cwd,
          error: modelFallbackMessage,
        },
      });
      return;
    }

    const persistSessionFile = () => {
      const nextSessionFile = piSession.sessionFile ?? sessionManager.getSessionFile();
      if (nextSessionFile && nextSessionFile !== session.piSessionFile) {
        session.piSessionFile = nextSessionFile;
        onSessionUpdate?.({ engine: "pi", piSessionFile: nextSessionFile });
      }
    };

    persistSessionFile();
    onEvent({
      type: "stream.message",
      payload: {
        sessionId: session.id,
        message: {
          type: "system_init",
          engine: "pi",
          sessionFile: piSession.sessionFile ?? sessionManager.getSessionFile(),
          provider: piSession.model?.provider,
          model: piSession.model?.id,
          cwd,
          thinkingLevel: piSession.thinkingLevel,
        },
      },
    });

    const unsubscribe = piSession.subscribe((event: AgentSessionEvent) => {
      if (disposed) return;
      if (event.type === "tool_execution_start") {
        onEvent({
          type: "stream.message",
          payload: {
            sessionId: session.id,
            message: {
              type: "assistant",
              engine: "pi",
              id: crypto.randomUUID(),
              blocks: [
                {
                  type: "tool_use",
                  id: event.toolCallId,
                  name: event.toolName,
                  input: (event.args ?? {}) as Record<string, unknown>,
                },
              ],
              provider: piSession.model?.provider,
              model: piSession.model?.id,
            },
          },
        });
        if (event.toolName === "workflow_plan") {
          planRegistered = true;
        }
        return;
      }

      if (event.type === "tool_execution_end") {
        onEvent({
          type: "stream.message",
          payload: {
            sessionId: session.id,
            message: {
              type: "tool_result",
              engine: "pi",
              toolUseId: event.toolCallId,
              toolName: event.toolName,
              content: stringifyToolContent((event.result as { content?: unknown }).content),
              isError: Boolean(event.isError),
              details: (event.result as { details?: unknown }).details,
              timestamp: Date.now(),
            },
          },
        });
        return;
      }

      if (event.type === "message_end") {
        const message = asRecord(event.message);
        if (!message) return;
        if (message.role === "assistant") {
          const blocks = normalizeAssistantBlocks(message, false);
          if (blocks.length > 0) {
            onEvent({
              type: "stream.message",
              payload: {
                sessionId: session.id,
                message: {
                  type: "assistant",
                  engine: "pi",
                  id: crypto.randomUUID(),
                  blocks,
                  provider: typeof message.provider === "string" ? message.provider : piSession.model?.provider,
                  model: typeof message.model === "string" ? message.model : piSession.model?.id,
                  stopReason: typeof message.stopReason === "string" ? message.stopReason : undefined,
                  timestamp: typeof message.timestamp === "number" ? message.timestamp : undefined,
                },
              },
            });
          }
          lastUsage = (message.usage ?? undefined) as Record<string, unknown> | undefined;
        }
      }

      if (event.type === "agent_end") {
        const lastAssistant = [...event.messages]
          .reverse()
          .find((message) => typeof message === "object" && message !== null && "role" in message && message.role === "assistant") as
          | Record<string, unknown>
          | undefined;
        lastUsage = (lastAssistant?.usage ?? lastUsage) as Record<string, unknown> | undefined;
      }
    });

    try {
      await piSession.prompt(promptToSend);
      persistSessionFile();

      const runResult = {
        status: "success" as const,
        usage: lastUsage as
          | {
              input?: number;
              output?: number;
              cacheRead?: number;
              cacheWrite?: number;
              totalTokens?: number;
              cost?: {
                input?: number;
                output?: number;
                cacheRead?: number;
                cacheWrite?: number;
                total?: number;
              };
            }
          | undefined,
      };

      onEvent({
        type: "stream.message",
        payload: {
          sessionId: session.id,
          message: {
            type: "run_result",
            engine: "pi",
            status: runResult.status,
            usage: runResult.usage,
            timestamp: Date.now(),
          },
        },
      });

      onEvent({
        type: "session.messagesReset",
        payload: {
          sessionId: session.id,
          messages: buildCanonicalHistory(
            piSession.sessionFile ?? sessionManager.getSessionFile(),
            piSession.model?.provider,
            piSession.model?.id,
            cwd,
            piSession.thinkingLevel,
            sessionManager.buildSessionContext().messages,
            runResult
          ),
        },
      });

      onSessionUpdate?.({
        engine: "pi",
        piSessionFile: piSession.sessionFile ?? sessionManager.getSessionFile(),
      });

      onEvent({
        type: "session.status",
        payload: {
          sessionId: session.id,
          status: resolveRunStatus(planRegistered, regenerateWorkflow),
          title: session.title,
          cwd: session.cwd,
        },
      });
    } catch (error) {
      if ((error as Error).name === "AbortError") {
        onEvent({
          type: "stream.message",
          payload: {
            sessionId: session.id,
            message: {
              type: "run_result",
              engine: "pi",
              status: "aborted",
              timestamp: Date.now(),
            },
          },
        });
        return;
      }

      onEvent({
        type: "stream.message",
        payload: {
          sessionId: session.id,
          message: {
            type: "run_result",
            engine: "pi",
            status: "error",
            error: String(error),
            timestamp: Date.now(),
          },
        },
      });
      onEvent({
        type: "session.status",
        payload: {
          sessionId: session.id,
          status: "error",
          title: session.title,
          cwd: session.cwd,
          error: String(error),
        },
      });
    } finally {
      disposed = true;
      unsubscribe();
      piSession.dispose();
    }
  })();

  return {
    abort: () => {
      piSessionAbort?.().catch(() => {});
    },
  };
}
