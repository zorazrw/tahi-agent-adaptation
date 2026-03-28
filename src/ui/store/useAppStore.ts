import { create } from 'zustand';
import type { ServerEvent, SessionStatus, StreamMessage, NodeCompletedMessage, VerifierMark, WorkflowNode } from "../types";

export type PermissionRequest = {
  toolUseId: string;
  toolName: string;
  input: unknown;
};

export type SessionView = {
  id: string;
  title: string;
  status: SessionStatus;
  cwd?: string;
  workflowTree?: WorkflowNode[];
  verificationDepth?: number;
  messages: StreamMessage[];
  permissionRequests: PermissionRequest[];
  lastPrompt?: string;
  lastEffectivePrompt?: string;
  createdAt?: number;
  updatedAt?: number;
  hydrated: boolean;
};

interface AppState {
  sessions: Record<string, SessionView>;
  activeSessionId: string | null;
  selectedNodeId: string | null;
  runningNodeId: string | null;
  collapsedNodeIds: Set<string>;
  highlightDepth: number | null;
  prompt: string;
  cwd: string;
  pendingStart: boolean;
  globalError: string | null;
  sessionsLoaded: boolean;
  showStartModal: boolean;
  showSettingsModal: boolean;
  historyRequested: Set<string>;
  apiConfigChecked: boolean;
  attachedFiles: string[];
  tempCwd: string | null;
  previewPanelOpen: boolean;
  /** Active context export + extract_context runs (memory/skill induction). */
  contextInductionDepth: number;

  setPrompt: (prompt: string) => void;
  setCwd: (cwd: string) => void;
  setPendingStart: (pending: boolean) => void;
  setGlobalError: (error: string | null) => void;
  setShowStartModal: (show: boolean) => void;
  setShowSettingsModal: (show: boolean) => void;
  setAttachedFiles: (files: string[]) => void;
  setTempCwd: (dir: string | null) => void;
  setActiveSessionId: (id: string | null) => void;
  setSelectedNodeId: (id: string | null) => void;
  setRunningNodeId: (id: string | null) => void;
  toggleNodeCollapsed: (nodeId: string) => void;
  setCollapsedNodeIds: (ids: Set<string>) => void;
  setHighlightDepth: (depth: number | null) => void;
  setPreviewPanelOpen: (open: boolean) => void;
  setApiConfigChecked: (checked: boolean) => void;
  markHistoryRequested: (sessionId: string) => void;
  resolvePermissionRequest: (sessionId: string, toolUseId: string) => void;
  updateWorkflowTree: (sessionId: string, workflowTree: WorkflowNode[]) => void;
  updateVerificationDepth: (sessionId: string, verificationDepth: number) => void;
  updateSessionTitle: (sessionId: string, title: string) => void;
  handleServerEvent: (event: ServerEvent) => void;

  toolStatuses: Record<string, "pending" | "success" | "error">;
  setToolStatus: (toolUseId: string, status: "pending" | "success" | "error") => void;

  toolMeta: Record<string, { name: string; info: string | null; editData?: { file_path: string; old_string: string; new_string: string }; writeData?: { file_path: string; content: string } }>;
  setToolMeta: (toolUseId: string, meta: { name: string; info: string | null; editData?: { file_path: string; old_string: string; new_string: string }; writeData?: { file_path: string; content: string } }) => void;
}

function createSession(id: string): SessionView {
  return { id, title: "", status: "idle", messages: [], permissionRequests: [], hydrated: false };
}

/** Find a node by id in a workflow tree. */
function findNode(tree: WorkflowNode[], id: string): WorkflowNode | undefined {
  for (const node of tree) {
    if (node.id === id) return node;
    const found = findNode(node.children, id);
    if (found) return found;
  }
  return undefined;
}

export const useAppStore = create<AppState>((set, get) => ({
  sessions: {},
  activeSessionId: null,
  selectedNodeId: null,
  runningNodeId: null,
  collapsedNodeIds: new Set(),
  highlightDepth: null,
  prompt: "",
  cwd: "",
  pendingStart: false,
  globalError: null,
  sessionsLoaded: false,
  showStartModal: false,
  showSettingsModal: false,
  historyRequested: new Set(),
  apiConfigChecked: false,
  attachedFiles: [],
  tempCwd: null,
  previewPanelOpen: false,
  contextInductionDepth: 0,

  setPrompt: (prompt) => set({ prompt }),
  setCwd: (cwd) => set({ cwd }),
  setAttachedFiles: (attachedFiles) => set({ attachedFiles }),
  setTempCwd: (tempCwd) => set({ tempCwd }),
  setPendingStart: (pendingStart) => set({ pendingStart }),
  setGlobalError: (globalError) => set({ globalError }),
  setShowStartModal: (showStartModal) => set({ showStartModal }),
  setShowSettingsModal: (showSettingsModal) => set({ showSettingsModal }),
  setActiveSessionId: (id) => set({ activeSessionId: id, selectedNodeId: null, previewPanelOpen: false }),
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),
  setRunningNodeId: (runningNodeId) => set({ runningNodeId }),
  toggleNodeCollapsed: (nodeId) => set((state) => {
    const next = new Set(state.collapsedNodeIds);
    if (next.has(nodeId)) next.delete(nodeId); else next.add(nodeId);
    return { collapsedNodeIds: next };
  }),
  setCollapsedNodeIds: (collapsedNodeIds) => set({ collapsedNodeIds }),
  setHighlightDepth: (highlightDepth) => set({ highlightDepth }),
  setPreviewPanelOpen: (previewPanelOpen) => set({ previewPanelOpen }),
  setApiConfigChecked: (apiConfigChecked) => set({ apiConfigChecked }),

  toolStatuses: {},
  setToolStatus: (toolUseId, status) => set((state) => ({
    toolStatuses: { ...state.toolStatuses, [toolUseId]: status }
  })),

  toolMeta: {},
  setToolMeta: (toolUseId, meta) => set((state) => ({
    toolMeta: { ...state.toolMeta, [toolUseId]: meta }
  })),

  updateWorkflowTree: (sessionId, workflowTree) => {
    set((state) => {
      const existing = state.sessions[sessionId];
      if (!existing) return {};
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: { ...existing, workflowTree }
        }
      };
    });
  },

  updateVerificationDepth: (sessionId, verificationDepth) => {
    set((state) => {
      const existing = state.sessions[sessionId];
      if (!existing) return {};
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: { ...existing, verificationDepth }
        }
      };
    });
  },

  updateSessionTitle: (sessionId, title) => {
    set((state) => {
      const existing = state.sessions[sessionId];
      if (!existing) return {};
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: { ...existing, title }
        }
      };
    });
  },

  markHistoryRequested: (sessionId) => {
    set((state) => {
      const next = new Set(state.historyRequested);
      next.add(sessionId);
      return { historyRequested: next };
    });
  },

  resolvePermissionRequest: (sessionId, toolUseId) => {
    set((state) => {
      const existing = state.sessions[sessionId];
      if (!existing) return {};
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: {
            ...existing,
            permissionRequests: existing.permissionRequests.filter(req => req.toolUseId !== toolUseId)
          }
        }
      };
    });
  },

  handleServerEvent: (event) => {
    const state = get();

    switch (event.type) {
      case "session.list": {
        const nextSessions: Record<string, SessionView> = {};
        for (const session of event.payload.sessions) {
          const existing = state.sessions[session.id] ?? createSession(session.id);
          nextSessions[session.id] = {
            ...existing,
            status: session.status,
            title: session.title ?? existing.title,
            cwd: session.cwd,
            workflowTree: session.workflowTree ?? existing.workflowTree,
            verificationDepth: session.verificationDepth ?? existing.verificationDepth,
            createdAt: session.createdAt,
            updatedAt: session.updatedAt
          };
        }

        set({ sessions: nextSessions, sessionsLoaded: true });

        const hasSessions = event.payload.sessions.length > 0;
        set({ showStartModal: !hasSessions });

        if (!hasSessions) {
          get().setActiveSessionId(null);
        }

        if (!state.activeSessionId && event.payload.sessions.length > 0) {
          const sorted = [...event.payload.sessions].sort((a, b) => {
            const aTime = a.updatedAt ?? a.createdAt ?? 0;
            const bTime = b.updatedAt ?? b.createdAt ?? 0;
            return aTime - bTime;
          });
          const latestSession = sorted[sorted.length - 1];
          if (latestSession) {
            get().setActiveSessionId(latestSession.id);
          }
        } else if (state.activeSessionId) {
          const stillExists = event.payload.sessions.some(
            (session) => session.id === state.activeSessionId
          );
          if (!stillExists) {
            get().setActiveSessionId(null);
          }
        }
        break;
      }

      case "session.effectivePrompt": {
        const { sessionId, prompt } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, lastEffectivePrompt: prompt }
            }
          };
        });
        break;
      }

      case "session.history": {
        const { sessionId, messages, status, workflowTree, verificationDepth, title } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...existing,
                status,
                messages,
                ...(workflowTree !== undefined && { workflowTree }),
                ...(verificationDepth !== undefined && { verificationDepth }),
                ...(title !== undefined && { title }),
                hydrated: true
              }
            }
          };
        });
        break;
      }

      case "session.workflowTree": {
        const { sessionId, workflowTree } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, workflowTree }
            }
          };
        });
        break;
      }

      case "session.verificationDepth": {
        const { sessionId, verificationDepth } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, verificationDepth }
            }
          };
        });
        break;
      }

      case "session.title": {
        const { sessionId, title } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, title }
            }
          };
        });
        break;
      }

      case "session.messagesReset": {
        const { sessionId, messages } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          const isActive = sessionId === state.activeSessionId;
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, messages }
            },
            ...(isActive ? { previewPanelOpen: false } : {})
          };
        });
        break;
      }

      case "session.contextInduction": {
        const { phase } = event.payload;
        set((state) => ({
          contextInductionDepth:
            phase === "started"
              ? state.contextInductionDepth + 1
              : Math.max(0, state.contextInductionDepth - 1),
        }));
        break;
      }

      case "session.nodeCompleted": {
        const { sessionId, nodeId } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          const isActive = sessionId === state.activeSessionId;
          const tree = existing.workflowTree ?? [];
          const node = findNode(tree, nodeId);
          const nodeLabel = node?.description ?? nodeId;

          // Inject synthetic node_completed message into chat
          const syntheticMsg: NodeCompletedMessage = { type: "node_completed", nodeId, nodeLabel };
          const nextMessages = [...existing.messages, syntheticMsg];

          return {
            runningNodeId: isActive && state.runningNodeId === nodeId ? null : state.runningNodeId,
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, messages: nextMessages }
            }
          };
        });
        break;
      }

      case "session.status": {
        const { sessionId, status, title, cwd } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          const isActive = sessionId === state.activeSessionId;
          return {
            ...(isActive && status !== "running" ? { runningNodeId: null } : {}),
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...existing,
                status,
                title: title ?? existing.title,
                cwd: cwd ?? existing.cwd,
                updatedAt: Date.now()
              }
            }
          };
        });

        if (state.pendingStart) {
          get().setActiveSessionId(sessionId);
          set({ pendingStart: false, showStartModal: false });
        }
        break;
      }

      case "session.deleted": {
        const { sessionId } = event.payload;
        const state = get();

        const nextSessions = { ...state.sessions };
        delete nextSessions[sessionId];

        const nextHistoryRequested = new Set(state.historyRequested);
        nextHistoryRequested.delete(sessionId);

        const hasRemaining = Object.keys(nextSessions).length > 0;

        set({
          sessions: nextSessions,
          historyRequested: nextHistoryRequested,
          showStartModal: !hasRemaining
        });

        if (state.activeSessionId === sessionId) {
          const remaining = Object.values(nextSessions).sort(
            (a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0)
          );
          get().setActiveSessionId(remaining[0]?.id ?? null);
        }
        break;
      }

      case "stream.message": {
        const { sessionId, message } = event.payload;
        if ((message as any).type === "stream_event") break;

        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, messages: [...existing.messages, message] }
            }
          };
        });
        break;
      }

      case "stream.user_prompt": {
        const { sessionId, prompt } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...existing,
                messages: [...existing.messages, { type: "user_prompt", prompt }]
              }
            }
          };
        });
        break;
      }

      case "permission.request": {
        const { sessionId, toolUseId, toolName, input } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...existing,
                permissionRequests: [...existing.permissionRequests, { toolUseId, toolName, input }]
              }
            }
          };
        });
        break;
      }

      case "runner.error": {
        set({ globalError: event.payload.message });
        break;
      }

      case "memory.readResult":
      case "memory.writeResult":
      case "skills.writeResult":
        break;
    }
  }
}));
