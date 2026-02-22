import { useEffect, useMemo } from "react";
import type { TopLevelItem } from "../hooks/useGroupedMessages";
import type { IndexedMessage } from "../hooks/useMessageWindow";
import type { PermissionRequest } from "../store/useAppStore";
import type { PermissionResult, SDKAssistantMessage } from "@anthropic-ai/claude-agent-sdk";
import { useAppStore } from "../store/useAppStore";
import { MessageCard } from "./EventCard";
import { ErrorBoundary } from "./ErrorBoundary";
import { Tool, ToolHeader, ToolContent } from "../../components/ai-elements/tool";
import { WrenchIcon } from "lucide-react";

type TaskGroup = Extract<TopLevelItem, { kind: "task_group" }>;

export function TaskToolCard({
  group,
  isRunning,
  permissionRequest,
  onPermissionResult,
}: {
  group: TaskGroup;
  isRunning: boolean;
  permissionRequest?: PermissionRequest;
  onPermissionResult?: (toolUseId: string, result: PermissionResult) => void;
}) {
  const storeSetToolStatus = useAppStore((s) => s.setToolStatus);
  const setToolMeta = useAppStore((s) => s.setToolMeta);

  // Register tool meta and status so ToolUseCard doesn't double-render
  useEffect(() => {
    setToolMeta(group.taskToolUseId, { name: "Task", info: group.taskDescription });
    const status = group.status === "error" ? "error" : group.status === "completed" ? "success" : "pending";
    storeSetToolStatus(group.taskToolUseId, status);
  }, [group.taskToolUseId, group.status, group.taskDescription, storeSetToolStatus, setToolMeta]);

  const toolCount = useMemo(() => {
    let count = 0;
    for (const child of group.children) {
      if (child.message.type !== "assistant") continue;
      const assistant = child.message as SDKAssistantMessage;
      for (const block of assistant.message.content) {
        if (block.type === "tool_use") count++;
      }
    }
    return count;
  }, [group.children]);

  const toolState = group.status === "running" ? "running" : group.status === "error" ? "error" : "completed";
  const defaultOpen = group.status === "running";

  return (
    <div className="mt-3">
      <Tool defaultOpen={defaultOpen} className="border-l-2 border-l-primary/30">
        <ToolHeader
          icon={<WrenchIcon className="size-4" />}
          title="Task"
          description={group.taskDescription}
          suffix={
            <span className="inline-flex items-center gap-1.5">
              {toolCount > 0 && (
                <span className="inline-flex items-center rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary tabular-nums">
                  {toolCount} tool{toolCount !== 1 ? "s" : ""}
                </span>
              )}
              <span className="inline-flex items-center rounded-full bg-surface-tertiary px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
                {group.subagentType}
              </span>
            </span>
          }
          state={toolState}
        />
        <ToolContent className="p-0">
          <div className="max-h-[600px] overflow-y-auto pl-2 pr-1 py-1">
            {group.children.length === 0 && group.status === "running" && (
              <div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
                <span className="inline-grid grid-cols-2 gap-0.5 opacity-40">
                  <span className="h-1 w-1 rounded-full bg-current animate-pulse" />
                  <span className="h-1 w-1 rounded-full bg-current animate-pulse" style={{ animationDelay: "150ms" }} />
                  <span className="h-1 w-1 rounded-full bg-current animate-pulse" style={{ animationDelay: "300ms" }} />
                  <span className="h-1 w-1 rounded-full bg-current animate-pulse" style={{ animationDelay: "450ms" }} />
                </span>
                <span>Agent working...</span>
              </div>
            )}
            {group.children.map((child: IndexedMessage) => (
              <ErrorBoundary key={`task-child-${child.originalIndex}`}>
                <MessageCard
                  message={child.message}
                  isLast={false}
                  isRunning={isRunning && group.status === "running"}
                  permissionRequest={permissionRequest}
                  onPermissionResult={onPermissionResult}
                />
              </ErrorBoundary>
            ))}
          </div>
        </ToolContent>
      </Tool>
    </div>
  );
}
