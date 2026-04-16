import { AsyncLocalStorage } from "node:async_hooks";
import type { PiLlmDebugMessage } from "../types.js";

type LlmDebugContext = {
  emit?: (message: PiLlmDebugMessage) => void;
  provider?: string;
  model?: string;
};

const storage = new AsyncLocalStorage<LlmDebugContext>();

function redactValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(redactValue);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, entryValue]) => {
        if (key === "api_key" || key === "authorization") {
          return [key, entryValue ? "[redacted]" : entryValue];
        }
        return [key, redactValue(entryValue)];
      })
    );
  }
  return value;
}

export async function runWithLlmDebugContext<T>(
  context: LlmDebugContext,
  fn: () => Promise<T>,
): Promise<T> {
  return await storage.run(context, fn);
}

export function emitLlmDebug(
  message: Omit<PiLlmDebugMessage, "type" | "engine" | "timestamp">,
): void {
  const context = storage.getStore();
  if (!context?.emit) {
    return;
  }

  context.emit({
    type: "llm_debug",
    engine: "pi",
    provider: message.provider ?? context.provider,
    model: message.model ?? context.model,
    title: message.title ?? "LLM Debug",
    request: redactValue(message.request),
    response: redactValue(message.response),
    error: message.error,
    timestamp: Date.now(),
  });
}
