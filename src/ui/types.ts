import type { SDKMessage, PermissionResult } from "@anthropic-ai/claude-agent-sdk";

export type NodeStatus = "pending" | "running" | "completed" | "error";

export type WorkflowNode = {
  id: string;
  description: string;
  outputFiles: string[];
  verifiers: string[];
  verifierMarks: VerifierMark[];
  children: WorkflowNode[];
  status: NodeStatus;
  depth: number;
  resumePoint?: { uuid: string; claudeSessionId: string };
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

export type StreamMessage = SDKMessage | UserPromptMessage | NodeCompletedMessage;

export type SessionStatus = "idle" | "running" | "completed" | "error";

export type VerifierMark = "check" | "cross" | undefined;

export type SessionInfo = {
  id: string;
  title: string;
  status: SessionStatus;
  claudeSessionId?: string;
  cwd?: string;
  workflowTree?: WorkflowNode[];
  verificationDepth?: number;
  createdAt: number;
  updatedAt: number;
};

// Server -> Client events
export type ServerEvent =
  | { type: "stream.message"; payload: { sessionId: string; message: StreamMessage } }
  | { type: "stream.user_prompt"; payload: { sessionId: string; prompt: string } }
  | { type: "session.status"; payload: { sessionId: string; status: SessionStatus; title?: string; cwd?: string; error?: string } }
  | { type: "session.list"; payload: { sessions: SessionInfo[] } }
  | { type: "session.history"; payload: { sessionId: string; status: SessionStatus; messages: StreamMessage[]; workflowTree?: WorkflowNode[]; verificationDepth?: number; title?: string } }
  | { type: "session.workflowTree"; payload: { sessionId: string; workflowTree: WorkflowNode[] } }
  | { type: "session.verificationDepth"; payload: { sessionId: string; verificationDepth: number } }
  | { type: "session.title"; payload: { sessionId: string; title: string } }
  | { type: "session.deleted"; payload: { sessionId: string } }
  | { type: "session.nodeCompleted"; payload: { sessionId: string; nodeId: string } }
  | {
      type: "session.contextInduction";
      payload: { phase: "started" | "finished"; sessionId: string; ok?: boolean };
    }
  | { type: "permission.request"; payload: { sessionId: string; toolUseId: string; toolName: string; input: unknown } }
  | { type: "runner.error"; payload: { sessionId?: string; message: string } }
  | { type: "session.messagesReset"; payload: { sessionId: string; messages: StreamMessage[] } }
  | { type: "session.effectivePrompt"; payload: { sessionId: string; prompt: string } }
  | {
      type: "memory.readResult";
      payload: {
        requestId: string;
        dir: string;
        sections: { fileName: string; title: string; content: string }[];
        skillsDir: string;
        skillSections: { fileName: string; title: string; content: string }[];
      };
    }
  | { type: "memory.writeResult"; payload: { requestId: string; success: boolean; error?: string } }
  | { type: "skills.writeResult"; payload: { requestId: string; success: boolean; error?: string } };

// Client -> Server events
export type ClientEvent =
  | { type: "session.start"; payload: { title: string; prompt: string; cwd?: string; allowedTools?: string } }
  | { type: "session.continue"; payload: { sessionId: string; prompt: string } }
  | { type: "session.stop"; payload: { sessionId: string } }
  | { type: "session.delete"; payload: { sessionId: string } }
  | { type: "session.updateWorkflowTree"; payload: { sessionId: string; workflowTree: WorkflowNode[] } }
  | { type: "session.updateVerificationDepth"; payload: { sessionId: string; verificationDepth: number } }
  | { type: "session.updateTitle"; payload: { sessionId: string; title: string } }
  | { type: "session.list" }
  | { type: "session.history"; payload: { sessionId: string } }
  | { type: "session.solveNode"; payload: { sessionId: string; nodeId: string } }
  | { type: "session.regenerateWorkflow"; payload: { sessionId: string } }
  | { type: "permission.response"; payload: { sessionId: string; toolUseId: string; result: PermissionResult } }
  | { type: "memory.read"; payload: { requestId: string } }
  | {
      type: "memory.write";
      payload: {
        requestId: string;
        sections: { fileName: string; content: string }[];
        deletedFileNames?: string[];
      };
    }
  | {
      type: "skills.write";
      payload: {
        requestId: string;
        sections: { fileName: string; content: string }[];
        deletedFileNames?: string[];
      };
    };
