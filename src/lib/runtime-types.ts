export type NodeStatus = "pending" | "running" | "completed" | "error";

export type SessionEngine = "legacy-claude" | "pi";

export type VerifierMark = "check" | "cross" | undefined;

export type NodeVerifierPatch = {
  nodeId: string;
  verifiers: string[];
  verifierMarks: VerifierMark[];
};

export type WorkflowNodeResumePoint = { entryId: string } | { uuid: string; claudeSessionId: string };

export type WorkflowNode = {
  id: string;
  description: string;
  outputFiles: string[];
  verifiers: string[];
  verifierMarks: VerifierMark[];
  children: WorkflowNode[];
  status: NodeStatus;
  depth: number;
  resumePoint?: WorkflowNodeResumePoint;
  originalOutputs?: { path: string; content: string }[];
};

export type UserPromptMessage = {
  type: "user_prompt";
  prompt: string;
};

export type NodeCompletedMessage = {
  type: "node_completed";
  nodeId: string;
  nodeLabel: string;
};

export type AppPermissionResult = {
  behavior: "allow" | "deny";
  updatedInput?: unknown;
  message?: string;
};

export type AskUserQuestionInput = {
  questions?: Array<{
    question: string;
    header?: string;
    options?: Array<{ label: string; description?: string }>;
    multiSelect?: boolean;
  }>;
  answers?: Record<string, string>;
};

export type PiTextBlock = {
  type: "text";
  text: string;
};

export type PiThinkingBlock = {
  type: "thinking";
  thinking: string;
};

export type PiToolUseBlock = {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
};

export type PiAssistantBlock = PiTextBlock | PiThinkingBlock | PiToolUseBlock;

export type PiSystemInitMessage = {
  type: "system_init";
  engine: "pi";
  sessionFile?: string;
  provider?: string;
  model?: string;
  cwd?: string;
  thinkingLevel?: string;
};

export type PiAssistantMessage = {
  type: "assistant";
  engine: "pi";
  id: string;
  blocks: PiAssistantBlock[];
  provider?: string;
  model?: string;
  stopReason?: string;
  timestamp?: number;
};

export type PiToolResultMessage = {
  type: "tool_result";
  engine: "pi";
  toolUseId: string;
  toolName: string;
  content: string;
  isError: boolean;
  details?: unknown;
  timestamp?: number;
};

export type PiRunResultMessage = {
  type: "run_result";
  engine: "pi";
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
  timestamp?: number;
};

export type PiLlmDebugMessage = {
  type: "llm_debug";
  engine: "pi";
  provider?: string;
  model?: string;
  request?: unknown;
  response?: unknown;
  error?: string;
  title?: string;
  timestamp?: number;
};

export type LegacyTextBlock = {
  type: "text";
  text: string;
};

export type LegacyThinkingBlock = {
  type: "thinking";
  thinking: string;
  thinkingSignature?: string;
};

export type LegacyToolUseBlock = {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
};

export type LegacyToolResultBlock = {
  type: "tool_result";
  tool_use_id: string;
  content: string | Array<{ type: "text"; text: string }>;
  is_error?: boolean;
};

export type LegacyAssistantMessage = {
  type: "assistant";
  uuid?: string;
  parent_tool_use_id?: string | null;
  message: {
    role: "assistant";
    content: Array<LegacyTextBlock | LegacyThinkingBlock | LegacyToolUseBlock>;
  };
};

export type LegacyUserMessage = {
  type: "user";
  uuid?: string;
  parent_tool_use_id?: string | null;
  message: {
    role: "user";
    content: LegacyToolResultBlock[];
  };
};

export type LegacySystemMessage = {
  type: "system";
  subtype?: "init";
  session_id?: string;
  model?: string;
  permissionMode?: string;
  cwd?: string;
};

export type LegacyResultMessage = {
  type: "result";
  subtype: "success" | "error";
  duration_ms?: number;
  duration_api_ms?: number;
  total_cost_usd?: number;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
  result?: string;
  error?: string;
};

export type LegacyStreamEventMessage = {
  type: "stream_event";
  event: Record<string, unknown>;
  delta?: Record<string, unknown>;
};

export type LegacyMessage =
  | LegacyAssistantMessage
  | LegacyUserMessage
  | LegacySystemMessage
  | LegacyResultMessage
  | LegacyStreamEventMessage;

export type BrainEditMessage = { type: "brain_edit" };
export type VerifierLabelMessage = { type: "verifier_label"; nodeId: string };
export type EditWorkflowMessage = { type: "edit_workflow" };
export type EditVerifierMessage = { type: "edit_verifier" };
/** LLM auto-refinement of verifiers after user prompts / file edits (not manual sidebar edits). */
export type UpdateVerifiersMessage = { type: "update_verifiers"; nodeId: string };
export type FileEditMessage = { type: "file_edit"; path: string };
/** Quoted selection + comment on a txt/md preview file; does not trigger an agent run by itself. */
export type FileCommentMessage = { type: "file_comment"; path: string; prompt: string };

export type StreamMessage =
  | UserPromptMessage
  | NodeCompletedMessage
  | PiSystemInitMessage
  | PiAssistantMessage
  | PiToolResultMessage
  | PiRunResultMessage
  | PiLlmDebugMessage
  | BrainEditMessage
  | VerifierLabelMessage
  | EditWorkflowMessage
  | EditVerifierMessage
  | UpdateVerifiersMessage
  | FileEditMessage
  | FileCommentMessage
  | LegacyMessage;

export type SessionStatus = "idle" | "running" | "completed" | "error";

export type SessionInfo = {
  id: string;
  title: string;
  status: SessionStatus;
  engine: SessionEngine;
  claudeSessionId?: string;
  piSessionFile?: string;
  cwd?: string;
  workflowTree?: WorkflowNode[];
  verificationDepth?: number;
  createdAt: number;
  updatedAt: number;
};

export type AgentSettings = {
  defaultProvider?: string;
  defaultModel?: string;
  defaultThinkingLevel?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
};

export type OpenAICompatibleApiFormat = "openai-completions" | "openai-responses";

export type OpenAICompatibleProviderConfig = {
  provider: "openai-compatible";
  baseUrl: string;
  model: string;
  apiFormat: OpenAICompatibleApiFormat;
  hasApiKey: boolean;
};

export type OpenAICompatibleProviderInput = {
  baseUrl: string;
  model: string;
  apiFormat: OpenAICompatibleApiFormat;
  apiKey?: string;
};

export type TinkerModelConfig = {
  id: string;
  baseModel: string;
  modelPath?: string;
  rendererName?: string;
  reasoning: boolean;
  contextWindow: number;
  maxTokens: number;
};

export type TinkerProviderConfig = {
  provider: "tinker";
  baseUrl?: string;
  hasApiKey: boolean;
  model: TinkerModelConfig;
};

export type TinkerProviderInput = {
  baseUrl?: string;
  apiKey?: string;
  model: string;
  baseModel: string;
  modelPath?: string;
  rendererName?: string;
  reasoning?: boolean;
  contextWindow?: number;
  maxTokens?: number;
};

export type AvailableModel = {
  provider: string;
  id: string;
  label: string;
  reasoning: boolean;
};

export type ProviderAuthStatus = {
  provider: string;
  hasAuth: boolean;
  authType?: "api_key" | "oauth" | "env";
  supportsOAuth: boolean;
  oauthName?: string;
};

export type ServerEvent =
  | { type: "stream.message"; payload: { sessionId: string; message: StreamMessage } }
  | { type: "stream.user_prompt"; payload: { sessionId: string; prompt: string } }
  | { type: "session.status"; payload: { sessionId: string; status: SessionStatus; title?: string; cwd?: string; error?: string } }
  | { type: "session.list"; payload: { sessions: SessionInfo[] } }
  | { type: "session.history"; payload: { sessionId: string; status: SessionStatus; messages: StreamMessage[]; workflowTree?: WorkflowNode[]; verificationDepth?: number; title?: string; engine?: SessionEngine } }
  | { type: "session.workflowTree"; payload: { sessionId: string; workflowTree: WorkflowNode[] } }
  | { type: "session.nodeVerifiers"; payload: { sessionId: string; updates: NodeVerifierPatch[] } }
  | { type: "session.verificationDepth"; payload: { sessionId: string; verificationDepth: number } }
  | { type: "session.title"; payload: { sessionId: string; title: string } }
  | { type: "session.deleted"; payload: { sessionId: string } }
  | { type: "session.nodeCompleted"; payload: { sessionId: string; nodeId: string } }
  | { type: "session.contextInduction"; payload: { sessionId: string; phase: string; ok?: boolean; trainingTriggered?: boolean; historyLen?: number; minSessions?: number } }
  | { type: "session.weightTraining"; payload: { phase: "started" | "finished" } }
  | { type: "session.verifierCheck"; payload: { sessionId: string; nodeId: string; phase: string } }
  | { type: "permission.request"; payload: { sessionId: string; toolUseId: string; toolName: string; input: unknown } }
  | { type: "runner.error"; payload: { sessionId?: string; message: string } }
  | { type: "workflow.plan"; payload: { sessionId: string; workflowTree: WorkflowNode[] } }
  | { type: "session.messagesReset"; payload: { sessionId: string; messages: StreamMessage[] } }
  | { type: "session.effectivePrompt"; payload: { sessionId: string; prompt: string } }
  | { type: "memory.readResult"; payload: { requestId: string; dir: string; sections: unknown; skillsDir: string; skillSections: unknown } }
  | { type: "memory.writeResult"; payload: { requestId: string; success: boolean; error?: string } }
  | { type: "skills.writeResult"; payload: { requestId: string; success: boolean; error?: string } };

export type ClientEvent =
  | {
      type: "session.start";
      payload: {
        title: string;
        prompt: string;
        cwd?: string;
        allowedTools?: string;
        autoContextInduction?: boolean;
        /** Expertise picker category slug, e.g. data-viz-html → memories/skills/<slug>.md */
        expertiseTask?: string;
      };
    }
  | { type: "session.continue"; payload: { sessionId: string; prompt: string; verificationNodeId?: string } }
  | { type: "session.addFileComment"; payload: { sessionId: string; path: string; prompt: string } }
  | { type: "session.stop"; payload: { sessionId: string } }
  | { type: "session.delete"; payload: { sessionId: string } }
  | { type: "session.updateWorkflowTree"; payload: { sessionId: string; workflowTree: WorkflowNode[] } }
  | { type: "session.updateVerificationDepth"; payload: { sessionId: string; verificationDepth: number } }
  | { type: "session.updateTitle"; payload: { sessionId: string; title: string } }
  | { type: "session.list" }
  | { type: "session.history"; payload: { sessionId: string } }
  | { type: "session.solveNode"; payload: { sessionId: string; nodeId: string } }
  | { type: "session.labelVerifiers"; payload: { sessionId: string; nodeId: string } }
  | { type: "session.regenerateWorkflow"; payload: { sessionId: string } }
  | { type: "permission.response"; payload: { sessionId: string; toolUseId: string; result: AppPermissionResult } }
  | { type: "memory.read"; payload: { requestId: string } }
  | { type: "memory.write"; payload: { requestId: string; sections: Array<{ fileName: string; content: string }>; deletedFileNames?: string[] } }
  | { type: "skills.write"; payload: { requestId: string; sections: Array<{ fileName?: string; content?: string }>; deletedFileNames?: string[] } }
  | { type: "session.recordBrainEdit"; payload: { sessionId: string } }
  | { type: "session.runContextInduction"; payload: { sessionId: string } }
  | { type: "session.uploadForTinkerTraining"; payload: { sessionId: string } }
  | {
      type: "session.setAutoContextInduction";
      payload: { sessionId: string; autoContextInduction: boolean };
    }
  | {
      type: "session.setExpertiseTask";
      payload: { sessionId: string; expertiseTask: string | null };
    };
