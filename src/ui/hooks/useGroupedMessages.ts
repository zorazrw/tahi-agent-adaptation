import { useMemo } from "react";
import type { IndexedMessage } from "./useMessageWindow";
import type { StreamMessage } from "../types";
import type { SDKAssistantMessage, SDKUserMessage } from "@anthropic-ai/claude-agent-sdk";

export type TopLevelItem =
  | { kind: "message"; originalIndex: number; message: StreamMessage }
  | {
      kind: "task_group";
      taskToolUseId: string;
      taskDescription: string;
      subagentType: string;
      parentMessageIndex: number;
      parentMessage: StreamMessage;
      children: IndexedMessage[];
      resultMessage?: StreamMessage;
      status: "running" | "completed" | "error";
    };

type TaskMeta = {
  toolUseId: string;
  description: string;
  subagentType: string;
  parentMessageIndex: number;
  parentMessage: StreamMessage;
};

function getParentToolUseId(msg: StreamMessage): string | null {
  const sdk = msg as any;
  return sdk?.parent_tool_use_id ?? null;
}

export function useGroupedMessages(visibleMessages: IndexedMessage[]): TopLevelItem[] {
  return useMemo(() => {
    // Pass 1: Find all Task tool_use blocks and collect their IDs + metadata
    const taskMetas = new Map<string, TaskMeta>();

    for (const item of visibleMessages) {
      const msg = item.message;
      if (msg.type !== "assistant") continue;
      const assistant = msg as SDKAssistantMessage;
      for (const block of assistant.message.content) {
        if (block.type === "tool_use" && block.name === "Task") {
          const input = block.input as Record<string, any>;
          taskMetas.set(block.id, {
            toolUseId: block.id,
            description: input?.description ?? input?.prompt ?? "Task",
            subagentType: input?.subagent_type ?? "general-purpose",
            parentMessageIndex: item.originalIndex,
            parentMessage: item.message,
          });
        }
      }
    }

    // Fast path: no Task tool calls, return flat list
    if (taskMetas.size === 0) {
      return visibleMessages.map((item) => ({
        kind: "message" as const,
        originalIndex: item.originalIndex,
        message: item.message,
      }));
    }

    // Pass 2: Bucket children by parent_tool_use_id; find tool_result messages for each Task
    const childrenMap = new Map<string, IndexedMessage[]>();
    const resultMap = new Map<string, StreamMessage>();
    const consumedIndices = new Set<number>();

    for (const item of visibleMessages) {
      const parentId = getParentToolUseId(item.message);
      if (parentId && taskMetas.has(parentId)) {
        if (!childrenMap.has(parentId)) childrenMap.set(parentId, []);
        childrenMap.get(parentId)!.push(item);
        consumedIndices.add(item.originalIndex);
        continue;
      }

      // Check for tool_result that matches a Task tool_use_id
      if (item.message.type === "user") {
        const userMsg = item.message as SDKUserMessage;
        for (const block of userMsg.message.content) {
          if (
            block.type === "tool_result" &&
            typeof block.tool_use_id === "string" &&
            taskMetas.has(block.tool_use_id)
          ) {
            resultMap.set(block.tool_use_id, item.message);
            consumedIndices.add(item.originalIndex);
          }
        }
      }
    }

    // Pass 3: Build output array
    const output: TopLevelItem[] = [];
    // Track which Task IDs we've already emitted a group for
    const emittedTasks = new Set<string>();

    for (const item of visibleMessages) {
      if (consumedIndices.has(item.originalIndex)) continue;

      // Check if this assistant message contains Task tool_use blocks
      if (item.message.type === "assistant") {
        const assistant = item.message as SDKAssistantMessage;
        const taskBlocks: string[] = [];
        let hasNonTaskContent = false;

        for (const block of assistant.message.content) {
          if (block.type === "tool_use" && block.name === "Task") {
            taskBlocks.push(block.id);
          } else {
            hasNonTaskContent = true;
          }
        }

        // If message has non-Task content, emit the message (TaskToolCard will skip Task blocks)
        if (hasNonTaskContent) {
          output.push({
            kind: "message",
            originalIndex: item.originalIndex,
            message: item.message,
          });
        }

        // Emit a task_group for each Task tool_use
        for (const taskId of taskBlocks) {
          if (emittedTasks.has(taskId)) continue;
          emittedTasks.add(taskId);
          const meta = taskMetas.get(taskId)!;
          const children = childrenMap.get(taskId) ?? [];
          const resultMessage = resultMap.get(taskId);

          let status: "running" | "completed" | "error" = "running";
          if (resultMessage) {
            const userMsg = resultMessage as SDKUserMessage;
            const resultBlock = userMsg.message.content.find(
              (b: any) => b.type === "tool_result" && b.tool_use_id === taskId
            ) as any;
            status = resultBlock?.is_error ? "error" : "completed";
          }

          output.push({
            kind: "task_group",
            taskToolUseId: taskId,
            taskDescription: meta.description,
            subagentType: meta.subagentType,
            parentMessageIndex: meta.parentMessageIndex,
            parentMessage: meta.parentMessage,
            children,
            resultMessage,
            status,
          });
        }

        // If message ONLY had Task blocks (no other content), skip the message render
        if (!hasNonTaskContent) continue;
      } else {
        output.push({
          kind: "message",
          originalIndex: item.originalIndex,
          message: item.message,
        });
      }
    }

    return output;
  }, [visibleMessages]);
}
