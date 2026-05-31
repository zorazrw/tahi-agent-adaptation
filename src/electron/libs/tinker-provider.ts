import { spawn } from "child_process";
import { fileURLToPath } from "url";
import type { ChildProcessWithoutNullStreams } from "child_process";
import {
  calculateCost,
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type ImageContent,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
  type ThinkingContent,
  type Tool,
  type ToolCall,
  type ToolResultMessage,
} from "@mariozechner/pi-ai";
import type { ModelRegistry } from "@mariozechner/pi-coding-agent";
import {
  TINKER_API,
  TINKER_ENV_API_KEY_COMMAND,
  TINKER_PROVIDER,
  buildRegisteredTinkerModels,
  getRegisteredTinkerBaseUrl,
  readStoredTinkerProviderConfig,
} from "./tinker-config.js";
import { emitLlmDebug } from "./llm-debug.js";
import { assistantTextForDisplay } from "../../lib/assistant-display-sanitize.js";

type BridgeTextPart = { type: "text"; text: string };
type BridgeThinkingPart = { type: "thinking"; thinking: string };
type BridgeToolCall = {
  id?: string | null;
  type?: "function";
  function?: {
    name?: string;
    arguments?: string;
  };
};
type BridgeMessage = {
  content?: string | Array<BridgeTextPart | BridgeThinkingPart>;
  tool_calls?: BridgeToolCall[];
  unparsed_tool_calls?: Array<{ raw_text?: string; error?: string }>;
};
type BridgeResult = {
  ok: boolean;
  error?: string;
  renderer_name?: string;
  parse_success?: boolean;
  message?: BridgeMessage;
  usage?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
    totalTokens?: number;
  };
};

type BridgeRequest = {
  provider: {
    base_url?: string;
    api_key?: string;
  };
  model: {
    id: string;
    base_model: string;
    model_path?: string;
    renderer_name?: string;
  };
  options: {
    reasoning?: string;
    max_tokens: number;
    temperature: number;
  };
  context: {
    system_prompt?: string;
    messages: Array<Record<string, unknown>>;
    tools?: Array<Record<string, unknown>>;
  };
};

type BridgeResolveCheckpointRequest = {
  command: "resolve_checkpoint";
  provider: {
    base_url?: string;
    api_key?: string;
  };
  tinker_path: string;
};

type BridgeResolveCheckpointResult =
  | {
      ok: true;
      base_model: string;
    }
  | {
      ok: false;
      error: string;
    };

const bridgeProjectPath = fileURLToPath(new URL("../../../tinker-bridge", import.meta.url));
const TINKER_BRIDGE_IDLE_TIMEOUT_MS = 10 * 60 * 1000;

type BridgePendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

let persistentBridgeChild: ChildProcessWithoutNullStreams | null = null;
let persistentBridgeIdleTimer: NodeJS.Timeout | null = null;
let persistentBridgeBuffer = "";
let persistentBridgePending: BridgePendingRequest[] = [];

function clearPersistentBridgeIdleTimer(): void {
  if (persistentBridgeIdleTimer) {
    clearTimeout(persistentBridgeIdleTimer);
    persistentBridgeIdleTimer = null;
  }
}

function schedulePersistentBridgeIdleShutdown(): void {
  clearPersistentBridgeIdleTimer();
  persistentBridgeIdleTimer = setTimeout(() => {
    shutdownTinkerBridge("idle-timeout");
  }, TINKER_BRIDGE_IDLE_TIMEOUT_MS);
}

function resetPersistentBridgeState(): void {
  clearPersistentBridgeIdleTimer();
  persistentBridgeChild = null;
  persistentBridgeBuffer = "";
  const pending = persistentBridgePending;
  persistentBridgePending = [];
  for (const req of pending) {
    req.reject(new Error("Tinker bridge exited before completing request"));
  }
}

function ensurePersistentBridgeServer(): ChildProcessWithoutNullStreams {
  if (persistentBridgeChild && !persistentBridgeChild.killed) {
    schedulePersistentBridgeIdleShutdown();
    return persistentBridgeChild;
  }

  const child = spawn("uv", ["run", "--project", bridgeProjectPath, "python", "-m", "tinker_bridge", "--serve"], {
    cwd: bridgeProjectPath,
    stdio: ["pipe", "pipe", "pipe"],
  });
  persistentBridgeChild = child;

  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    persistentBridgeBuffer += chunk;
    while (true) {
      const newlineIdx = persistentBridgeBuffer.indexOf("\n");
      if (newlineIdx === -1) break;
      const line = persistentBridgeBuffer.slice(0, newlineIdx).trim();
      persistentBridgeBuffer = persistentBridgeBuffer.slice(newlineIdx + 1);
      if (!line) continue;
      const next = persistentBridgePending.shift();
      if (!next) continue;
      try {
        next.resolve(JSON.parse(line));
      } catch (error) {
        next.reject(
          new Error(
            `Failed to parse Tinker bridge server response: ${error instanceof Error ? error.message : String(error)}`
          )
        );
      }
    }
  });

  child.on("error", (error) => {
    const pending = persistentBridgePending;
    persistentBridgePending = [];
    for (const req of pending) req.reject(error);
    resetPersistentBridgeState();
  });

  child.on("close", (code) => {
    if (code !== 0) {
      const pending = persistentBridgePending;
      persistentBridgePending = [];
      for (const req of pending) {
        req.reject(new Error(`Tinker bridge server exited with code ${code}`));
      }
    }
    resetPersistentBridgeState();
  });

  schedulePersistentBridgeIdleShutdown();
  return child;
}

async function invokeBridgePersistent<T>(payload: unknown, signal?: AbortSignal): Promise<T> {
  if (signal?.aborted) {
    throw new Error("Tinker request aborted");
  }
  const child = ensurePersistentBridgeServer();
  schedulePersistentBridgeIdleShutdown();

  return await new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(new Error("Tinker request aborted"));
    signal?.addEventListener("abort", onAbort, { once: true });

    persistentBridgePending.push({
      resolve: (value) => {
        signal?.removeEventListener("abort", onAbort);
        schedulePersistentBridgeIdleShutdown();
        resolve(value as T);
      },
      reject: (reason) => {
        signal?.removeEventListener("abort", onAbort);
        reject(reason);
      },
    });

    child.stdin.write(`${JSON.stringify(payload)}\n`);
  });
}

export async function ensureTinkerBridgeWarm(configPath: string): Promise<void> {
  const config = readStoredTinkerProviderConfig(configPath);
  if (!config) return;
  ensurePersistentBridgeServer();
  schedulePersistentBridgeIdleShutdown();
}

export function shutdownTinkerBridge(_reason?: "idle-timeout" | "session-deleted" | "settings-changed" | "app-shutdown"): void {
  const child = persistentBridgeChild;
  if (child && !child.killed) {
    child.kill();
  }
  resetPersistentBridgeState();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeSchemaRef(ref: string): string {
  if (ref === "#" || ref.startsWith("#/$defs/")) {
    return ref;
  }
  if (ref.startsWith("#/definitions/")) {
    return ref.replace("#/definitions/", "#/$defs/");
  }
  if (/^[A-Za-z0-9_.-]+$/.test(ref)) {
    return `#/$defs/${ref}`;
  }
  return ref;
}

export function normalizeJsonSchemaForTinker(schema: Record<string, unknown>): Record<string, unknown> {
  const hoistedDefs: Record<string, unknown> = {};

  const visit = (value: unknown, isRoot = false): unknown => {
    if (Array.isArray(value)) {
      return value.map((item) => visit(item));
    }
    if (!isRecord(value)) {
      return value;
    }

    const localId =
      typeof value.$id === "string" && value.$id.trim() && !value.$id.startsWith("#")
        ? value.$id.trim()
        : undefined;
    const next: Record<string, unknown> = {};

    for (const [key, child] of Object.entries(value)) {
      if (key === "$id" && localId) {
        continue;
      }
      if (key === "$ref" && typeof child === "string") {
        next.$ref = normalizeSchemaRef(child);
        continue;
      }
      if ((key === "definitions" || key === "$defs") && isRecord(child)) {
        const defs = Object.fromEntries(
          Object.entries(child).map(([defKey, defValue]) => [defKey, visit(defValue)])
        );
        next.$defs = {
          ...(isRecord(next.$defs) ? next.$defs : {}),
          ...defs,
        };
        continue;
      }
      next[key] = visit(child);
    }

    if (localId && !isRoot) {
      if (!(localId in hoistedDefs)) {
        hoistedDefs[localId] = next;
      }
      return { $ref: `#/$defs/${localId}` };
    }

    return next;
  };

  const normalized = visit(schema, true);
  if (!isRecord(normalized)) {
    return schema;
  }

  if (Object.keys(hoistedDefs).length > 0) {
    normalized.$defs = {
      ...(isRecord(normalized.$defs) ? normalized.$defs : {}),
      ...hoistedDefs,
    };
  }

  return normalized;
}

function contentPartToText(part: TextContent | ImageContent): string {
  if (part.type === "text") {
    return part.text;
  }
  return `[image: ${part.mimeType}]`;
}

function toolResultToText(message: ToolResultMessage): string {
  return message.content.map(contentPartToText).filter(Boolean).join("\n");
}

function assistantContentToBridge(
  content: AssistantMessage["content"],
): { content: Array<BridgeTextPart | BridgeThinkingPart> | string; toolCalls?: BridgeToolCall[] } {
  const parts: Array<BridgeTextPart | BridgeThinkingPart> = [];
  const toolCalls: BridgeToolCall[] = [];

  for (const block of content) {
    if (block.type === "text" && block.text) {
      parts.push({ type: "text", text: block.text });
      continue;
    }
    if (block.type === "thinking" && block.thinking) {
      parts.push({ type: "thinking", thinking: block.thinking });
      continue;
    }
    if (block.type === "toolCall") {
      toolCalls.push({
        id: block.id,
        type: "function",
        function: {
          name: block.name,
          arguments: JSON.stringify(block.arguments ?? {}),
        },
      });
    }
  }

  return {
    content: parts.length > 0 ? parts : "",
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
  };
}

function toolToBridge(tool: Tool): Record<string, unknown> {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: normalizeJsonSchemaForTinker(tool.parameters as Record<string, unknown>),
    },
  };
}

function contextToBridgeMessages(context: Context): Array<Record<string, unknown>> {
  return context.messages.map((message) => {
    if (message.role === "user") {
      if (typeof message.content === "string") {
        return { role: "user", content: message.content };
      }
      return {
        role: "user",
        content: message.content.map((part) =>
          part.type === "text"
            ? { type: "text", text: part.text }
            : { type: "text", text: `[image: ${part.mimeType}]` }
        ),
      };
    }

    if (message.role === "assistant") {
      const assistant = assistantContentToBridge(message.content);
      return {
        role: "assistant",
        content: assistant.content,
        tool_calls: assistant.toolCalls,
      };
    }

    return {
      role: "tool",
      content: toolResultToText(message),
      tool_call_id: message.toolCallId,
      name: message.toolName,
    };
  });
}

function createEmptyUsage() {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function createBaseMessage(model: Model<string>): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: createEmptyUsage(),
    stopReason: "stop",
    timestamp: Date.now(),
  };
}

function parseToolArguments(raw: string | undefined): Record<string, unknown> {
  if (!raw) {
    return {};
  }
  const parsed = JSON.parse(raw) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Tinker returned non-object tool arguments");
  }
  return parsed as Record<string, unknown>;
}

function bridgeMessageToPiContent(message: BridgeMessage | undefined): AssistantMessage["content"] {
  if (!message) {
    return [];
  }

  const blocks: AssistantMessage["content"] = [];
  if (typeof message.content === "string") {
    const text = assistantTextForDisplay(message.content);
    if (text) {
      blocks.push({ type: "text", text });
    }
  } else if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (part.type === "text" && part.text) {
        const text = assistantTextForDisplay(part.text);
        if (!text) continue;
        blocks.push({ type: "text", text });
      } else if (part.type === "thinking" && part.thinking) {
        blocks.push({ type: "thinking", thinking: part.thinking });
      }
    }
  }

  for (const toolCall of message.tool_calls ?? []) {
    const name = toolCall.function?.name?.trim();
    if (!name) {
      continue;
    }
    blocks.push({
      type: "toolCall",
      id: toolCall.id?.trim() || crypto.randomUUID(),
      name,
      arguments: parseToolArguments(toolCall.function?.arguments),
    });
  }

  return blocks;
}

function emitResolvedMessage(stream: AssistantMessageEventStream, message: AssistantMessage): void {
  stream.push({
    type: "start",
    partial: {
      ...message,
      content: [],
    },
  });
}

async function invokeBridge<T>(payload: unknown, signal?: AbortSignal): Promise<T> {
  return await invokeBridgePersistent<T>(payload, signal);
}

function buildBridgeRequest(
  model: Model<string>,
  apiKey: string | undefined,
  context: Context,
  options: SimpleStreamOptions | undefined,
  configPath: string,
): BridgeRequest {
  const config = readStoredTinkerProviderConfig(configPath);
  if (!config || config.model.id !== model.id) {
    throw new Error(`Missing Tinker configuration for model ${model.id}`);
  }

  return {
    provider: {
      base_url: config.baseUrl,
      api_key: apiKey,
    },
    model: {
      id: config.model.id,
      base_model: config.model.baseModel,
      model_path: config.model.modelPath,
      renderer_name: config.model.rendererName,
    },
    options: {
      reasoning: options?.reasoning,
      max_tokens: options?.maxTokens ?? config.model.maxTokens,
      temperature: 0,
    },
    context: {
      system_prompt: context.systemPrompt,
      messages: contextToBridgeMessages(context),
      tools: context.tools?.map(toolToBridge),
    },
  };
}

export function createTinkerStreamSimple(configPath: string) {
  return (model: Model<string>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream => {
    const stream = createAssistantMessageEventStream();
    const output = createBaseMessage(model);

    (async () => {
      let bridgeRequest: BridgeRequest | undefined;

      // Emit start + thinking immediately so the UI shows a loading indicator
      // while the non-streaming bridge call is in flight.
      emitResolvedMessage(stream, output);
      output.content.push({ type: "thinking", thinking: "" });
      stream.push({
        type: "thinking_start",
        contentIndex: 0,
        partial: output,
      });

      try {
        const apiKey = options?.apiKey?.trim();
        if (!apiKey) {
          throw new Error("No Tinker API key configured. Save one in Settings or set TINKER_API_KEY.");
        }

        bridgeRequest = buildBridgeRequest(model, apiKey, context, options, configPath);
        const bridgeResult = await invokeBridge<BridgeResult>(bridgeRequest, options?.signal);
        if (!bridgeResult.ok) {
          throw new Error(bridgeResult.error || "Tinker bridge returned an unknown error");
        }

        const unparsedToolCalls = bridgeResult.message?.unparsed_tool_calls ?? [];
        const parsedToolCalls = bridgeResult.message?.tool_calls ?? [];
        if (unparsedToolCalls.length > 0 && parsedToolCalls.length === 0 && (context.tools?.length ?? 0) > 0) {
          const firstError = unparsedToolCalls[0];
          throw new Error(firstError.error || "Tinker returned an unparsed tool call");
        }

        output.content = bridgeMessageToPiContent(bridgeResult.message);
        output.stopReason =
          parsedToolCalls.length > 0 ? "toolUse" : bridgeResult.parse_success === false ? "length" : "stop";
        output.usage.input = bridgeResult.usage?.input ?? 0;
        output.usage.output = bridgeResult.usage?.output ?? 0;
        output.usage.cacheRead = bridgeResult.usage?.cacheRead ?? 0;
        output.usage.cacheWrite = bridgeResult.usage?.cacheWrite ?? 0;
        output.usage.totalTokens = bridgeResult.usage?.totalTokens ?? output.usage.input + output.usage.output;
        calculateCost(model, output.usage);

        emitLlmDebug({
          title: "Tinker Request/Response",
          provider: model.provider,
          model: model.id,
          request: bridgeRequest,
          response: bridgeResult,
        });

        stream.push({
          type: "done",
          reason: output.stopReason as "stop" | "length" | "toolUse",
          message: output,
        });
        stream.end();
      } catch (error) {
        output.stopReason = options?.signal?.aborted ? "aborted" : "error";
        output.errorMessage = error instanceof Error ? error.message : String(error);
        emitLlmDebug({
          title: "Tinker Request/Response",
          provider: model.provider,
          model: model.id,
          request: bridgeRequest,
          error: output.errorMessage,
        });
        stream.push({
          type: "error",
          reason: output.stopReason,
          error: output,
        });
        stream.end();
      }
    })();

    return stream;
  };
}

export async function resolveTinkerCheckpoint(
  tinkerPath: string,
  apiKey?: string,
  baseUrl?: string,
): Promise<BridgeResolveCheckpointResult> {
  const payload: BridgeResolveCheckpointRequest = {
    command: "resolve_checkpoint",
    provider: {
      api_key: apiKey?.trim() || undefined,
      base_url: baseUrl?.trim() || undefined,
    },
    tinker_path: tinkerPath.trim(),
  };
  const result = await invokeBridge<
    | {
        ok: true;
        base_model?: string;
      }
    | {
        ok: false;
        error?: string;
      }
  >(payload);

  if (result.ok) {
    if (!result.base_model?.trim()) {
      return {
        ok: false,
        error: "Tinker bridge did not return a base model",
      };
    }
    return {
      ok: true,
      base_model: result.base_model,
    };
  }

  return {
    ok: false,
    error: result.error || "Failed to resolve checkpoint",
  };
}

export function registerTinkerProvider(modelRegistry: ModelRegistry, configPath: string): void {
  const config = readStoredTinkerProviderConfig(configPath);
  if (!config) {
    return;
  }

  modelRegistry.registerProvider(TINKER_PROVIDER, {
    api: TINKER_API,
    apiKey: TINKER_ENV_API_KEY_COMMAND,
    baseUrl: getRegisteredTinkerBaseUrl(config),
    models: buildRegisteredTinkerModels(config),
    streamSimple: createTinkerStreamSimple(configPath),
  });
}
