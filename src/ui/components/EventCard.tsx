import { useEffect, useState } from "react";
import JsonView from "@uiw/react-json-view";
import type {
  AppPermissionResult,
  AskUserQuestionInput,
  LegacyAssistantMessage,
  LegacyMessage,
  LegacyResultMessage,
  LegacySystemMessage,
  LegacyToolResultBlock,
  LegacyUserMessage,
  NodeCompletedMessage,
  PiAssistantMessage,
  PiLlmDebugMessage,
  PiRunResultMessage,
  PiSystemInitMessage,
  PiToolResultMessage,
  StreamMessage,
} from "../types";
import { useAppStore } from "../store/useAppStore";
import type { PermissionRequest } from "../store/useAppStore";
import { DecisionPanel } from "./DecisionPanel";
import { WorkflowCard } from "./WorkflowCard";
import { NodeOutputSnippet } from "./StepOutputSnippet";

// ai-elements
import { MessageResponse } from "../../components/ai-elements/message";
import { Reasoning, ReasoningTrigger, ReasoningContent } from "../../components/ai-elements/reasoning";
import { Tool, ToolHeader, ToolContent } from "../../components/ai-elements/tool";
import Ansi from "ansi-to-react";

// @pierre/diffs for code rendering
import { MultiFileDiff, File as DiffsFile } from "@pierre/diffs/react";

// lucide icons for tool cards
import {
  TerminalSquareIcon,
  FileTextIcon,
  FileEditIcon,
  SearchIcon,
  GlobeIcon,
  WrenchIcon,
  FileIcon,
  PencilIcon,
} from "lucide-react";

type AnyAssistantContentBlock =
  | LegacyAssistantMessage["message"]["content"][number]
  | PiAssistantMessage["blocks"][number];
type ToolStatus = "pending" | "success" | "error";

const getAskUserQuestionSignature = (input?: AskUserQuestionInput | null) => {
  if (!input?.questions?.length) return "";
  return input.questions.map((question) => {
    const options = (question.options ?? []).map((o) => `${o.label}|${o.description ?? ""}`).join(",");
    return `${question.question}|${question.header ?? ""}|${question.multiSelect ? "1" : "0"}|${options}`;
  }).join("||");
};

const useToolStatus = (toolUseId: string | undefined) => {
  return useAppStore((s) => toolUseId ? s.toolStatuses[toolUseId] : undefined);
};

/* ── Tool icon map ── */
const getToolIcon = (name: string) => {
  switch (name.toLowerCase()) {
    case "bash": return <TerminalSquareIcon className="size-4" />;
    case "read": return <FileTextIcon className="size-4" />;
    case "write": return <PencilIcon className="size-4" />;
    case "edit": return <FileEditIcon className="size-4" />;
    case "glob":
    case "grep":
    case "find":
    case "ls":
      return <SearchIcon className="size-4" />;
    case "webfetch":
    case "websearch":
      return <GlobeIcon className="size-4" />;
    case "task": return <WrenchIcon className="size-4" />;
    default: return <FileIcon className="size-4" />;
  }
};

/* ── Tool description helper ── */
const getToolInfo = (name: string, input: Record<string, any>): string | null => {
  switch (name.toLowerCase()) {
    case "bash": return input?.command || null;
    case "read": case "write": case "edit": return input?.file_path || input?.path || null;
    case "glob": case "grep": case "find": return input?.pattern || input?.path || null;
    case "ls": return input?.path || null;
    case "task": return input?.description || null;
    case "webfetch": return input?.url || null;
    default: return null;
  }
};

/* ── Diff stat helper for Edit ── */
const getDiffStats = (oldStr: string, newStr: string): string => {
  const oldLines = oldStr.split("\n").length;
  const newLines = newStr.split("\n").length;
  return `+${newLines} -${oldLines}`;
};

/* ── Extract file name from path ── */
const getFileName = (filePath: string): string => {
  const parts = filePath.split("/");
  return parts[parts.length - 1] || filePath;
};

/* ── Strip cat -n line numbers from Read output ── */
const stripCatLineNumbers = (text: string): string => {
  // Format: "     1→content" or "     1\tcontent"
  const lines = text.split("\n");
  const stripped = lines.map((line) => {
    const match = line.match(/^\s*\d+[→\t](.*)/);
    return match ? match[1] : line;
  });
  return stripped.join("\n");
};

/* ── Strip system-reminder tags from Read output ── */
const stripSystemReminders = (text: string): string => {
  return text.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, "").trimEnd();
};

/* ── Get file extension for language detection ── */
const getFileExtension = (filePath: string): string => {
  const name = getFileName(filePath);
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot + 1) : "";
};

/* ── Session Result ── */
const SessionResult = ({ message }: { message: LegacyResultMessage | PiRunResultMessage }) => {
  if (message.type === "run_result") {
    const usage = message.usage;
    const totalCost = usage?.cost?.total;
    return (
      <div className="flex flex-col gap-2 mt-4">
        <div className="text-xs text-ink-500 uppercase tracking-wide font-semibold">Run Result</div>
        <div className="flex flex-col rounded-xl px-4 py-3 border border-ink-900/10 bg-surface-secondary space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-[14px]">
            <span className="font-normal">Status</span>
            <span className="inline-flex items-center rounded-full bg-surface-tertiary px-2.5 py-0.5 text-ink-700 text-[13px]">
              {message.status}
            </span>
            {typeof totalCost === "number" && (
              <span className="inline-flex items-center rounded-full bg-primary/10 px-2.5 py-0.5 text-primary text-[13px]">
                Cost ${totalCost.toFixed(4)}
              </span>
            )}
          </div>
          {message.error && (
            <pre className="text-sm text-error whitespace-pre-wrap">{message.error}</pre>
          )}
        </div>
      </div>
    );
  }

  const formatMinutes = (ms: number | undefined) => typeof ms !== "number" ? "-" : `${(ms / 60000).toFixed(2)} min`;
  const formatUsd = (usd: number | undefined) => typeof usd !== "number" ? "-" : usd.toFixed(2);
  const formatMillions = (tokens: number | undefined) => typeof tokens !== "number" ? "-" : `${(tokens / 1_000_000).toFixed(4)} M`;

  return (
    <div className="flex flex-col gap-2 mt-4">
      <div className="text-xs text-ink-500 uppercase tracking-wide font-semibold">Session Result</div>
      <div className="flex flex-col rounded-xl px-4 py-3 border border-ink-900/10 bg-surface-secondary space-y-2">
        <div className="flex flex-wrap items-center gap-2 text-[14px]">
          <span className="font-normal">Duration</span>
          <span className="inline-flex items-center rounded-full bg-surface-tertiary px-2.5 py-0.5 text-ink-700 text-[13px]">{formatMinutes(message.duration_ms)}</span>
          <span className="font-normal">API</span>
          <span className="inline-flex items-center rounded-full bg-surface-tertiary px-2.5 py-0.5 text-ink-700 text-[13px]">{formatMinutes(message.duration_api_ms)}</span>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[14px]">
          <span className="font-normal">Usage</span>
          <span className="inline-flex items-center rounded-full bg-primary/10 px-2.5 py-0.5 text-primary text-[13px]">Cost ${formatUsd(message.total_cost_usd)}</span>
          <span className="inline-flex items-center rounded-full bg-surface-tertiary px-2.5 py-0.5 text-ink-700 text-[13px]">Input {formatMillions(message.usage?.input_tokens)}</span>
          <span className="inline-flex items-center rounded-full bg-surface-tertiary px-2.5 py-0.5 text-ink-700 text-[13px]">Output {formatMillions(message.usage?.output_tokens)}</span>
        </div>
      </div>
    </div>
  );
};

const LlmDebugValue = ({ value }: { value: unknown }) => {
  if (value === undefined) {
    return <div className="text-xs text-ink-500">No data</div>;
  }

  if (value === null || typeof value !== "object") {
    return (
      <pre className="text-xs whitespace-pre-wrap font-mono text-ink-800">
        {typeof value === "string" ? value : JSON.stringify(value, null, 2)}
      </pre>
    );
  }

  return (
    <div className="text-xs overflow-auto">
      <JsonView value={value} collapsed={2} displayDataTypes={false} />
    </div>
  );
};

const LlmDebugCard = ({ message }: { message: PiLlmDebugMessage }) => {
  const description = message.model ? `${message.provider ?? "model"}/${message.model}` : message.provider;

  return (
    <div className="mt-3">
      <Tool defaultOpen={false}>
        <ToolHeader
          icon={<WrenchIcon className="size-4" />}
          title={message.title || "LLM Debug"}
          description={description || undefined}
          state={message.error ? "error" : "completed"}
        />
        <ToolContent className="space-y-3">
          {message.error && (
            <div className="rounded-lg border border-error/20 bg-error-light/50 px-3 py-2">
              <div className="text-[11px] uppercase tracking-wide text-error font-semibold mb-1">Error</div>
              <pre className="text-xs whitespace-pre-wrap font-mono text-error">{message.error}</pre>
            </div>
          )}
          <div>
            <div className="text-[11px] uppercase tracking-wide text-ink-500 font-semibold mb-2">Request</div>
            <LlmDebugValue value={message.request} />
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-wide text-ink-500 font-semibold mb-2">Response</div>
            <LlmDebugValue value={message.response} />
          </div>
        </ToolContent>
      </Tool>
    </div>
  );
};

export function isMarkdown(text: string): boolean {
  if (!text || typeof text !== "string") return false;
  const patterns: RegExp[] = [/^#{1,6}\s+/m, /```[\s\S]*?```/];
  return patterns.some((pattern) => pattern.test(text));
}

function extractTagContent(input: string, tag: string): string | null {
  const match = input.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`));
  return match ? match[1] : null;
}

/* ── Bash Tool Result ── */
const BashToolResult = ({ command, outputText, isError }: { command: string | null; outputText: string; isError: boolean }) => {
  return (
    <Tool defaultOpen={false}>
      <ToolHeader
        icon={<TerminalSquareIcon className="size-4" />}
        title="Shell"
        description={command || undefined}
        state={isError ? "error" : "completed"}
      />
      {outputText.trim() && (
        <ToolContent className="p-0">
          <pre className={`font-mono text-xs whitespace-pre overflow-auto max-h-64 p-3 ${isError ? "text-destructive" : ""}`}>
            <Ansi>{outputText}</Ansi>
          </pre>
        </ToolContent>
      )}
    </Tool>
  );
};

/* ── Edit Tool Result with Diff ── */
const EditToolResult = ({ filePath, editData, outputText, isError }: {
  filePath: string | null;
  editData?: { file_path: string; old_string: string; new_string: string };
  outputText: string;
  isError: boolean;
}) => {
  const [open, setOpen] = useState(false);
  const fileName = filePath ? getFileName(filePath) : "file";
  const diffStats = editData ? getDiffStats(editData.old_string, editData.new_string) : null;

  return (
    <Tool open={open} onOpenChange={setOpen}>
      <ToolHeader
        icon={<FileEditIcon className="size-4" />}
        title="Edit"
        description={filePath || undefined}
        suffix={diffStats && (
          <span className="text-xs font-mono text-muted-foreground">{diffStats}</span>
        )}
        state={isError ? "error" : "completed"}
      />
      <ToolContent className="p-0">
        {editData && !isError ? (
          <div className="overflow-auto max-h-80 text-xs">
            <MultiFileDiff
              oldFile={{ name: fileName, contents: editData.old_string }}
              newFile={{ name: fileName, contents: editData.new_string }}
              options={{
                theme: "pierre-light",
                diffStyle: "unified",
                disableFileHeader: true,
                diffIndicators: "bars",
                overflow: "scroll",
                disableLineNumbers: false,
              }}
            />
          </div>
        ) : (
          <pre className={`font-mono text-xs whitespace-pre p-3 overflow-auto max-h-64 ${isError ? "text-destructive" : ""}`}>
            {outputText}
          </pre>
        )}
      </ToolContent>
    </Tool>
  );
};

/* ── Write Tool Result with Diff (all additions) ── */
const WriteToolResult = ({ filePath, writeData, outputText, isError }: {
  filePath: string | null;
  writeData?: { file_path: string; content: string };
  outputText: string;
  isError: boolean;
}) => {
  const fileName = filePath ? getFileName(filePath) : "file";
  const lineCount = writeData ? writeData.content.split("\n").length : null;

  return (
    <Tool defaultOpen={false}>
      <ToolHeader
        icon={<PencilIcon className="size-4" />}
        title="Write"
        description={filePath || undefined}
        suffix={lineCount != null ? (
          <span className="text-xs font-mono text-muted-foreground">+{lineCount}</span>
        ) : undefined}
        state={isError ? "error" : "completed"}
      />
      <ToolContent className="p-0">
        {writeData && !isError ? (
          <div className="overflow-auto max-h-80 text-xs">
            <MultiFileDiff
              oldFile={{ name: fileName, contents: "" }}
              newFile={{ name: fileName, contents: writeData.content }}
              options={{
                theme: "pierre-light",
                diffStyle: "unified",
                disableFileHeader: true,
                diffIndicators: "bars",
                overflow: "scroll",
                disableLineNumbers: false,
              }}
            />
          </div>
        ) : (
          <pre className={`font-mono text-xs whitespace-pre p-3 overflow-auto max-h-64 ${isError ? "text-destructive" : ""}`}>
            {outputText}
          </pre>
        )}
      </ToolContent>
    </Tool>
  );
};

/* ── Read Tool Result with syntax highlighting ── */
const ReadToolResult = ({ filePath, outputText, isError }: {
  filePath: string | null;
  outputText: string;
  isError: boolean;
}) => {
  const fileName = filePath ? getFileName(filePath) : "file";
  const ext = filePath ? getFileExtension(filePath) : "";
  const cleanedOutput = !isError ? stripSystemReminders(stripCatLineNumbers(outputText)) : outputText;

  return (
    <Tool defaultOpen={false}>
      <ToolHeader
        icon={<FileTextIcon className="size-4" />}
        title="Read"
        description={filePath || undefined}
        state={isError ? "error" : "completed"}
      />
      <ToolContent className="p-0">
        {!isError && cleanedOutput.trim() ? (
          <div className="overflow-auto max-h-80 text-xs">
            <DiffsFile
              file={{ name: fileName, contents: cleanedOutput, lang: ext as any }}
              options={{
                theme: "pierre-light",
                disableFileHeader: true,
                overflow: "scroll",
              }}
            />
          </div>
        ) : (
          <pre className={`font-mono text-xs whitespace-pre p-3 overflow-auto max-h-64 ${isError ? "text-destructive" : ""}`}>
            {outputText}
          </pre>
        )}
      </ToolContent>
    </Tool>
  );
};

/* ── Generic Tool Result (Glob, Grep, etc.) ── */
const GenericToolResult = ({ toolName, toolInfo, outputText, isError }: {
  toolName: string;
  toolInfo: string | null;
  outputText: string;
  isError: boolean;
}) => {
  const [open, setOpen] = useState(false);

  return (
    <Tool open={open} onOpenChange={setOpen}>
      <ToolHeader
        icon={getToolIcon(toolName)}
        title={toolName}
        description={toolInfo || undefined}
        state={isError ? "error" : "completed"}
      />
      {outputText.trim() && (
        <ToolContent className="p-0">
          <pre className={`font-mono text-xs whitespace-pre p-3 overflow-auto max-h-64 ${isError ? "text-destructive" : ""}`}>
            {outputText}
          </pre>
        </ToolContent>
      )}
    </Tool>
  );
};

/* ── Tool Result (user message containing tool_result) ── */
const ToolResult = ({
  messageContent,
  directToolName,
}: {
  messageContent: LegacyToolResultBlock | PiToolResultMessage;
  directToolName?: string;
}) => {
  const storeSetToolStatus = useAppStore((s) => s.setToolStatus);
  const toolUseId =
    "tool_use_id" in messageContent ? messageContent.tool_use_id : messageContent.toolUseId;
  const toolMeta = useAppStore((s) => toolUseId ? s.toolMeta[toolUseId] : undefined);
  const isError = "is_error" in messageContent ? !!messageContent.is_error : !!("isError" in messageContent && messageContent.isError);
  const status: ToolStatus = isError ? "error" : "success";

  useEffect(() => { if (toolUseId) storeSetToolStatus(toolUseId, status); }, [toolUseId, status, storeSetToolStatus]);

  const toolName = directToolName ?? toolMeta?.name ?? ("toolName" in messageContent ? messageContent.toolName : "Tool");

  // Suppress rendering for WorkflowPlan tool results (just "plan registered" text)
  if (toolName.toLowerCase().includes("workflow")) return null;
  const toolInfo = toolMeta?.info;
  const editData = toolMeta?.editData;
  const writeData = toolMeta?.writeData;

  // Parse output text
  let outputText = "";
  if ("content" in messageContent && typeof messageContent.content === "string") {
    outputText = messageContent.content;
  } else if ("is_error" in messageContent && messageContent.is_error) {
    outputText = extractTagContent(String(messageContent.content), "tool_use_error") || String(messageContent.content);
  } else {
    try {
      const legacyContent = (messageContent as LegacyToolResultBlock).content;
      if (Array.isArray(legacyContent)) {
        outputText = legacyContent.map((item: any) => item.text || "").join("\n");
      } else {
        outputText = String(legacyContent);
      }
    } catch { outputText = JSON.stringify(messageContent, null, 2); }
  }

  return (
    <div className="mt-3">
      {toolName.toLowerCase() === "bash" ? (
        <BashToolResult command={toolInfo ?? null} outputText={outputText} isError={isError} />
      ) : toolName.toLowerCase() === "edit" ? (
        <EditToolResult filePath={toolInfo ?? null} editData={editData} outputText={outputText} isError={isError} />
      ) : toolName.toLowerCase() === "write" ? (
        <WriteToolResult filePath={toolInfo ?? null} writeData={writeData} outputText={outputText} isError={isError} />
      ) : toolName.toLowerCase() === "read" ? (
        <ReadToolResult filePath={toolInfo ?? null} outputText={outputText} isError={isError} />
      ) : (
        <GenericToolResult toolName={toolName} toolInfo={toolInfo ?? null} outputText={outputText} isError={isError} />
      )}
    </div>
  );
};

/* ── Thinking Block (Reasoning) ── */
const ThinkingBlock = ({ text, isStreaming = false }: { text: string; isStreaming?: boolean }) => (
  <div className="mt-4">
    <Reasoning isStreaming={isStreaming}>
      <ReasoningTrigger />
      <ReasoningContent>{text}</ReasoningContent>
    </Reasoning>
  </div>
);

/* ── Assistant Text Block ── */
const AssistantTextBlock = ({ text, showIndicator = false }: { text: string; showIndicator?: boolean }) => (
  <div className="mt-4">
    <MessageResponse isAnimating={showIndicator} caret="block">{text}</MessageResponse>
  </div>
);

/* ── WorkflowPlan Tool Use Card ── */
const WorkflowPlanToolUseCard = ({ messageContent }: { messageContent: AnyAssistantContentBlock }) => {
  if (messageContent.type !== "tool_use") return null;

  const setToolMeta = useAppStore((s) => s.setToolMeta);
  const storeSetToolStatus = useAppStore((s) => s.setToolStatus);

  const input = messageContent.input as { tasks?: Array<{ description: string; outputFiles: string[]; verifiers: string[]; children?: unknown[] }> } | null;
  const tasks = input?.tasks ?? [];

  useEffect(() => {
    if (messageContent.id) {
      setToolMeta(messageContent.id, { name: "WorkflowPlan", info: null });
      storeSetToolStatus(messageContent.id, "success");
    }
  }, [messageContent.id]); // eslint-disable-line react-hooks/exhaustive-deps

  if (tasks.length === 0) return null;

  // Flatten to show top-level descriptions in the card
  return (
    <div className="mt-4">
      <WorkflowCard
        steps={tasks.map((s) => s.description)}
        outputFiles={tasks.map((s) => s.outputFiles)}
        verifiers={tasks.map((s) => s.verifiers)}
      />
    </div>
  );
};

/* ── Tool Use Card (assistant side — pending state) ── */
const ToolUseCard = ({ messageContent }: { messageContent: AnyAssistantContentBlock }) => {
  if (messageContent.type !== "tool_use") return null;

  const toolStatus = useToolStatus(messageContent.id);
  const setToolMeta = useAppStore((s) => s.setToolMeta);
  const storeSetToolStatus = useAppStore((s) => s.setToolStatus);
  const toolStatuses = useAppStore((s) => s.toolStatuses);

  const input = messageContent.input as Record<string, any>;
  const toolInfo = getToolInfo(messageContent.name, input);

  // Build edit data for Edit tool
  const editData = messageContent.name.toLowerCase() === "edit" && input?.file_path && input?.old_string && input?.new_string
    ? { file_path: input.file_path, old_string: input.old_string, new_string: input.new_string }
    : undefined;

  // Build write data for Write tool
  const writeData = messageContent.name.toLowerCase() === "write" && input?.file_path && typeof input?.content === "string"
    ? { file_path: input.file_path, content: input.content }
    : undefined;

  useEffect(() => {
    if (messageContent?.id && !toolStatuses[messageContent.id]) storeSetToolStatus(messageContent.id, "pending");
  }, [messageContent?.id, storeSetToolStatus, toolStatuses]);

  useEffect(() => {
    setToolMeta(messageContent.id, { name: messageContent.name, info: toolInfo, editData, writeData });
  }, [messageContent.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // When result exists, ToolResult renders the unified card
  if (toolStatus === "success" || toolStatus === "error") return null;

  // Pending/running state — compact collapsible header
  return (
    <div className="mt-3">
      <Tool>
        <ToolHeader
          icon={getToolIcon(messageContent.name)}
          title={messageContent.name.toLowerCase() === "bash" ? "Shell" : messageContent.name}
          description={toolInfo || undefined}
          state="running"
        />
      </Tool>
    </div>
  );
};

/* ── AskUserQuestion Card ── */
const AskUserQuestionCard = ({
  messageContent,
  permissionRequest,
  onPermissionResult
}: {
  messageContent: AnyAssistantContentBlock;
  permissionRequest?: PermissionRequest;
  onPermissionResult?: (toolUseId: string, result: AppPermissionResult) => void;
}) => {
  if (messageContent.type !== "tool_use") return null;

  const input = messageContent.input as AskUserQuestionInput | null;
  const questions = input?.questions ?? [];
  const currentSignature = getAskUserQuestionSignature(input);
  const requestSignature = getAskUserQuestionSignature(permissionRequest?.input as AskUserQuestionInput | undefined);
  const isActiveRequest = permissionRequest && currentSignature === requestSignature;

  if (isActiveRequest && onPermissionResult) {
    return (
      <div className="mt-4">
        <DecisionPanel
          request={permissionRequest}
          onSubmit={(result) => onPermissionResult(permissionRequest.toolUseId, result)}
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2 rounded-[1rem] bg-surface-tertiary px-3 py-2 mt-4">
      <div className="flex flex-row items-center gap-2">
        <span className="inline-flex items-center rounded-md text-primary py-0.5 text-sm font-medium">AskUserQuestion</span>
      </div>
      {questions.map((q, idx) => (
        <div key={idx} className="text-sm text-ink-700 ml-4">{q.question}</div>
      ))}
    </div>
  );
};

/* ── System Info Card ── */
const LegacySystemInfoCard = ({ message }: { message: LegacySystemMessage }) => {
  if (message.type !== "system" || message.subtype !== "init") return null;

  const InfoItem = ({ name, value }: { name: string; value: string }) => (
    <div className="text-[14px]">
      <span className="mr-4 font-normal">{name}</span>
      <span className="font-light">{value}</span>
    </div>
  );

  return (
    <div className="flex flex-col gap-2 mt-2">
      <span className="text-xs text-ink-500 uppercase tracking-wide font-semibold">System Init</span>
      <div className="flex flex-col rounded-xl px-4 py-2 border border-ink-900/10 bg-surface-secondary space-y-1">
        <InfoItem name="Session ID" value={message.session_id || "-"} />
        <InfoItem name="Model Name" value={message.model || "-"} />
        <InfoItem name="Permission Mode" value={message.permissionMode || "-"} />
        <InfoItem name="Working Directory" value={message.cwd || "-"} />
      </div>
    </div>
  );
};

const PiSystemInfoCard = ({ message }: { message: PiSystemInitMessage }) => {
  const InfoItem = ({ name, value }: { name: string; value: string }) => (
    <div className="text-[14px]">
      <span className="mr-4 font-normal">{name}</span>
      <span className="font-light">{value}</span>
    </div>
  );

  return (
    <div className="flex flex-col gap-2 mt-2">
      <span className="text-xs text-ink-500 uppercase tracking-wide font-semibold">System Init</span>
      <div className="flex flex-col rounded-xl px-4 py-2 border border-ink-900/10 bg-surface-secondary space-y-1">
        <InfoItem name="Session File" value={message.sessionFile || "-"} />
        <InfoItem name="Model" value={message.model ? `${message.provider}/${message.model}` : "-"} />
        <InfoItem name="Thinking" value={message.thinkingLevel || "-"} />
        <InfoItem name="Working Directory" value={message.cwd || "-"} />
      </div>
    </div>
  );
};

/* ── User Message Card ── */
const UserMessageCard = ({ message }: { message: { type: "user_prompt"; prompt: string } }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.prompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col items-end mt-4 group">
      <div className="relative" style={{ maxWidth: "min(85%, 64ch)" }}>
        <button
          onClick={handleCopy}
          className="absolute -left-10 top-2 p-1.5 rounded-lg text-ink-400 hover:text-ink-600 hover:bg-ink-900/5 opacity-0 group-hover:opacity-100 transition-opacity duration-150"
          aria-label="Copy message"
        >
          {copied ? (
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12l4 4L19 6" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></svg>
          )}
        </button>
        <div className="rounded-2xl rounded-tr-sm bg-surface-secondary border border-ink-900/8 px-4 py-3">
          <MessageResponse>{message.prompt}</MessageResponse>
        </div>
      </div>
    </div>
  );
};

/* ── Main MessageCard ── */
export function MessageCard({
  message,
  isLast = false,
  isRunning = false,
  permissionRequest,
  onPermissionResult,
  skipTaskToolUse = false,
}: {
  message: StreamMessage;
  isLast?: boolean;
  isRunning?: boolean;
  permissionRequest?: PermissionRequest;
  onPermissionResult?: (toolUseId: string, result: AppPermissionResult) => void;
  skipTaskToolUse?: boolean;
}) {
  const showIndicator = isLast && isRunning;

  if (message.type === "user_prompt") {
    return <UserMessageCard message={message} />;
  }

  if (message.type === "verifier_label") {
    return null;
  }

  if (
    message.type === "edit_workflow" ||
    message.type === "edit_verifier" ||
    message.type === "file_edit"
  ) {
    return null;
  }

  if (message.type === "system_init") {
    return <PiSystemInfoCard message={message} />;
  }

  if (message.type === "tool_result") {
    return <ToolResult messageContent={message} directToolName={message.toolName} />;
  }

  if (message.type === "run_result") {
    return <SessionResult message={message} />;
  }

  if (message.type === "llm_debug") {
    return <LlmDebugCard message={message} />;
  }

  if (message.type === "node_completed") {
    return <NodeOutputSnippet message={message as NodeCompletedMessage} />;
  }

  const legacyMessage = message as LegacyMessage;

  if (legacyMessage.type === "system") {
    return <LegacySystemInfoCard message={legacyMessage as LegacySystemMessage} />;
  }

  if (legacyMessage.type === "result") {
    if ((legacyMessage as LegacyResultMessage).subtype === "success") {
      return <SessionResult message={legacyMessage as LegacyResultMessage} />;
    }
    return (
      <div className="flex flex-col gap-2 mt-4">
        <div className="text-xs text-error uppercase tracking-wide font-semibold">Session Error</div>
        <div className="rounded-xl bg-error-light p-3">
          <pre className="text-sm text-error whitespace-pre-wrap">{JSON.stringify(legacyMessage, null, 2)}</pre>
        </div>
      </div>
    );
  }

  if (message.type === "assistant" && "engine" in message && message.engine === "pi") {
    const contents = message.blocks;
    return (
      <>
        {contents.map((content, idx) => {
          const isLastContent = idx === contents.length - 1;
          if (content.type === "thinking") {
            return <ThinkingBlock key={idx} text={content.thinking} isStreaming={isLastContent && showIndicator} />;
          }
          if (content.type === "text") {
            return <AssistantTextBlock key={idx} text={content.text} showIndicator={isLastContent && showIndicator} />;
          }
          if (content.type === "tool_use") {
            if (content.name === "ask_user_question") {
              return <AskUserQuestionCard key={idx} messageContent={content} permissionRequest={permissionRequest} onPermissionResult={onPermissionResult} />;
            }
            if (content.name === "workflow_plan") {
              return <WorkflowPlanToolUseCard key={idx} messageContent={content} />;
            }
            return <ToolUseCard key={idx} messageContent={content} />;
          }
          return null;
        })}
      </>
    );
  }

  if (legacyMessage.type === "assistant") {
    let contents = (legacyMessage as LegacyAssistantMessage).message.content;
    if (skipTaskToolUse) {
      contents = contents.filter(
        (block) => !(block.type === "tool_use" && block.name === "Task")
      );
      if (contents.length === 0) return null;
    }
    return (
      <>
        {contents.map((content, idx: number) => {
          const isLastContent = idx === contents.length - 1;
          if (content.type === "thinking") {
            return <ThinkingBlock key={idx} text={content.thinking} isStreaming={isLastContent && showIndicator} />;
          }
          if (content.type === "text") {
            return <AssistantTextBlock key={idx} text={content.text} showIndicator={isLastContent && showIndicator} />;
          }
          if (content.type === "tool_use") {
            if (content.name === "AskUserQuestion") {
              return <AskUserQuestionCard key={idx} messageContent={content} permissionRequest={permissionRequest} onPermissionResult={onPermissionResult} />;
            }
            if (content.name.includes("WorkflowPlan")) {
              return <WorkflowPlanToolUseCard key={idx} messageContent={content} />;
            }
            return <ToolUseCard key={idx} messageContent={content} />;
          }
          return null;
        })}
      </>
    );
  }

  if (legacyMessage.type === "user") {
    const contents = (legacyMessage as LegacyUserMessage).message.content;
    return (
      <>
        {contents.map((content, idx: number) => {
          if (content.type === "tool_result") {
            return <ToolResult key={idx} messageContent={content} />;
          }
          return null;
        })}
      </>
    );
  }

  return null;
}

export { MessageCard as EventCard };
