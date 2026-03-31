import { spawn } from "child_process";
import { fileURLToPath } from "url";
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
    if (message.content) {
      blocks.push({ type: "text", text: message.content });
    }
  } else if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (part.type === "text" && part.text) {
        blocks.push({ type: "text", text: part.text });
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
  return await new Promise<T>((resolve, reject) => {
    const child = spawn(
      "uv",
      ["run", "--project", bridgeProjectPath, "python", "-m", "tinker_bridge"],
      {
        cwd: bridgeProjectPath,
        stdio: ["pipe", "pipe", "pipe"],
        signal,
      }
    );

    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      reject(error);
    });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || `Tinker bridge exited with code ${code}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout) as T);
      } catch (error) {
        reject(
          new Error(
            `Failed to parse Tinker bridge response: ${error instanceof Error ? error.message : String(error)}`
          )
        );
      }
    });

    child.stdin.end(JSON.stringify(payload));
  });
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
