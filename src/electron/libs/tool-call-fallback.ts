/**
 * Fallback parser for models (e.g. Qwen3) that output tool calls as
 * <tool_call>{"name":"...","arguments":{...}}</tool_call> inside the text
 * content rather than through the API's native tool_calls field.
 *
 * This wraps the standard OpenAI-compatible stream functions and post-processes
 * the final message: if no real toolCall blocks were produced but the text
 * contains <tool_call> XML, we extract and promote them to proper toolCall
 * blocks so the agent can execute them.
 */

import {
  createAssistantMessageEventStream,
  streamSimpleOpenAICompletions,
  streamSimpleOpenAIResponses,
  type Api,
  type AssistantMessageEventStream,
  type Context,
  type Model,
  type SimpleStreamOptions,
  type StreamFunction,
} from "@mariozechner/pi-ai";
import type { ModelRegistry } from "@mariozechner/pi-coding-agent";

interface InlineToolCall {
  name: string;
  arguments: Record<string, unknown>;
  id: string;
}

/**
 * Parse <tool_call>JSON</tool_call> blocks from a text string.
 * Returns the extracted tool calls and the cleaned text (with the XML removed).
 */
function parseInlineToolCalls(text: string): {
  toolCalls: InlineToolCall[];
  cleanText: string;
} {
  const toolCalls: InlineToolCall[] = [];
  const regex = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;

  let match;
  while ((match = regex.exec(text)) !== null) {
    try {
      const parsed = JSON.parse(match[1]) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        const obj = parsed as Record<string, unknown>;
        const name = typeof obj.name === "string" ? obj.name.trim() : "";
        const args =
          obj.arguments &&
          typeof obj.arguments === "object" &&
          !Array.isArray(obj.arguments)
            ? (obj.arguments as Record<string, unknown>)
            : {};
        if (name) {
          toolCalls.push({ name, arguments: args, id: crypto.randomUUID() });
        }
      }
    } catch {
      // Ignore malformed JSON inside <tool_call> tags
    }
  }

  const cleanText =
    toolCalls.length > 0 ? text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim() : text;

  return { toolCalls, cleanText };
}

/**
 * Wrap a streamSimple function so that after the stream completes, if the
 * model placed tool calls inside text content as <tool_call> XML, those are
 * extracted and turned into proper toolCall blocks.
 *
 * The generic parameter preserves the specific API type so TypeScript is happy.
 */
function wrapWithToolCallFallback<A extends Api>(
  original: StreamFunction<A, SimpleStreamOptions>
): StreamFunction<A, SimpleStreamOptions> {
  return (
    model: Model<A>,
    context: Context,
    options?: SimpleStreamOptions
  ): AssistantMessageEventStream => {
    const outer = createAssistantMessageEventStream();
    const inner = original(model, context, options);

    (async () => {
      for await (const event of inner) {
        if (event.type === "done") {
          const message = event.message;

          // Only apply fallback if the model produced no native tool calls
          const hasNativeToolCalls = message.content.some(
            (b) => b.type === "toolCall"
          );

          if (!hasNativeToolCalls) {
            type ContentBlock = (typeof message.content)[number];
            const newContent: ContentBlock[] = [];
            let foundInlineToolCalls = false;

            for (const block of message.content) {
              if (block.type === "text" && "text" in block && typeof block.text === "string") {
                const { toolCalls, cleanText } = parseInlineToolCalls(block.text);
                if (toolCalls.length > 0) {
                  foundInlineToolCalls = true;
                  if (cleanText) {
                    newContent.push({ ...block, text: cleanText } as ContentBlock);
                  }
                  for (const tc of toolCalls) {
                    newContent.push({
                      type: "toolCall",
                      id: tc.id,
                      name: tc.name,
                      arguments: tc.arguments,
                    } as ContentBlock);
                  }
                } else {
                  newContent.push(block);
                }
              } else {
                newContent.push(block);
              }
            }

            if (foundInlineToolCalls) {
              outer.push({
                type: "done",
                reason: "toolUse",
                message: {
                  ...message,
                  content: newContent,
                  stopReason: "toolUse",
                },
              });
              outer.end();
              return;
            }
          }

          // No inline tool calls – pass through unchanged
          outer.push(event);
        } else {
          outer.push(event);
        }
      }
      outer.end();
    })();

    return outer;
  };
}

/**
 * Register wrapped stream providers in the model registry so that
 * <tool_call> XML in text content is transparently converted to real tool
 * calls. This overrides the built-in openai-completions and openai-responses
 * API providers, and the override is persisted across model registry refreshes.
 */
export function registerToolCallFallback(modelRegistry: ModelRegistry): void {
  modelRegistry.registerProvider("openai-completions-tool-call-fallback", {
    api: "openai-completions",
    streamSimple: wrapWithToolCallFallback(
      streamSimpleOpenAICompletions
    ) as StreamFunction<Api, SimpleStreamOptions>,
  });

  modelRegistry.registerProvider("openai-responses-tool-call-fallback", {
    api: "openai-responses",
    streamSimple: wrapWithToolCallFallback(
      streamSimpleOpenAIResponses
    ) as StreamFunction<Api, SimpleStreamOptions>,
  });
}
