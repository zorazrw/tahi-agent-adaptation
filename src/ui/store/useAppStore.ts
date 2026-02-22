import { create } from 'zustand';
import type { ServerEvent, SessionStatus, StreamMessage, StepCompletedMessage, VerifierMark } from "../types";

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
  steps?: string[];
  completedStepIndices?: number[];
  outputFiles?: string[][];
  verificationCriteria?: string[][];
  verifierMarks?: VerifierMark[][];
  messages: StreamMessage[];
  permissionRequests: PermissionRequest[];
  lastPrompt?: string;
  createdAt?: number;
  updatedAt?: number;
  hydrated: boolean;
};

interface AppState {
  sessions: Record<string, SessionView>;
  activeSessionId: string | null;
  selectedStepIndex: number;
  previewStepIndex: number;
  previewPanelOpen: boolean;
  prompt: string;
  cwd: string;
  pendingStart: boolean;
  globalError: string | null;
  sessionsLoaded: boolean;
  showStartModal: boolean;
  showSettingsModal: boolean;
  historyRequested: Set<string>;
  apiConfigChecked: boolean;

  setPrompt: (prompt: string) => void;
  setCwd: (cwd: string) => void;
  setPendingStart: (pending: boolean) => void;
  setGlobalError: (error: string | null) => void;
  setShowStartModal: (show: boolean) => void;
  setShowSettingsModal: (show: boolean) => void;
  setActiveSessionId: (id: string | null) => void;
  setSelectedStepIndex: (index: number) => void;
  setPreviewStepIndex: (index: number) => void;
  setPreviewPanelOpen: (open: boolean) => void;
  setApiConfigChecked: (checked: boolean) => void;
  markHistoryRequested: (sessionId: string) => void;
  resolvePermissionRequest: (sessionId: string, toolUseId: string) => void;
  updateSessionSteps: (sessionId: string, steps: string[]) => void;
  updateSessionVerificationCriteria: (sessionId: string, verificationCriteria: string[][]) => void;
  updateSessionVerifierMarks: (sessionId: string, verifierMarks: VerifierMark[][]) => void;
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

export const useAppStore = create<AppState>((set, get) => ({
  sessions: {},
  activeSessionId: null,
  selectedStepIndex: 0,
  previewStepIndex: 0,
  previewPanelOpen: false,
  prompt: "",
  cwd: "",
  pendingStart: false,
  globalError: null,
  sessionsLoaded: false,
  showStartModal: false,
  showSettingsModal: false,
  historyRequested: new Set(),
  apiConfigChecked: false,

  setPrompt: (prompt) => set({ prompt }),
  setCwd: (cwd) => set({ cwd }),
  setPendingStart: (pendingStart) => set({ pendingStart }),
  setGlobalError: (globalError) => set({ globalError }),
  setShowStartModal: (showStartModal) => set({ showStartModal }),
  setShowSettingsModal: (showSettingsModal) => set({ showSettingsModal }),
  setActiveSessionId: (id) => set({ activeSessionId: id, selectedStepIndex: 0, previewStepIndex: 0, previewPanelOpen: false }),
  setSelectedStepIndex: (index) => set({ selectedStepIndex: index, previewStepIndex: index }),
  setPreviewStepIndex: (index) => set({ previewStepIndex: index }),
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

  updateSessionSteps: (sessionId, steps) => {
    set((state) => {
      const existing = state.sessions[sessionId];
      if (!existing) return {};
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: { ...existing, steps }
        }
      };
    });
  },

  updateSessionVerificationCriteria: (sessionId, verificationCriteria) => {
    set((state) => {
      const existing = state.sessions[sessionId];
      if (!existing) return {};
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: { ...existing, verificationCriteria }
        }
      };
    });
  },

  updateSessionVerifierMarks: (sessionId, verifierMarks) => {
    set((state) => {
      const existing = state.sessions[sessionId];
      if (!existing) return {};
      return {
        sessions: {
          ...state.sessions,
          [sessionId]: { ...existing, verifierMarks }
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
            steps: session.steps ?? existing.steps,
            completedStepIndices: session.completedStepIndices ?? existing.completedStepIndices,
            outputFiles: session.outputFiles ?? existing.outputFiles,
            verificationCriteria: session.verificationCriteria ?? existing.verificationCriteria,
            verifierMarks: session.verifierMarks ?? existing.verifierMarks,
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

      case "session.history": {
        const { sessionId, messages, status, steps, completedStepIndices, outputFiles, verificationCriteria, verifierMarks, title } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...existing,
                status,
                messages,
                ...(steps !== undefined && { steps }),
                ...(completedStepIndices !== undefined && { completedStepIndices }),
                ...(outputFiles !== undefined && { outputFiles }),
                ...(verificationCriteria !== undefined && { verificationCriteria }),
                ...(verifierMarks !== undefined && { verifierMarks }),
                ...(title !== undefined && { title }),
                hydrated: true
              }
            }
          };
        });
        break;
      }

      case "session.steps": {
        const { sessionId, steps } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, steps }
            }
          };
        });
        break;
      }

      case "session.outputFiles": {
        const { sessionId, outputFiles } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, outputFiles }
            }
          };
        });
        break;
      }

      case "session.verificationCriteria": {
        const { sessionId, verificationCriteria } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, verificationCriteria }
            }
          };
        });
        break;
      }

      case "session.verifierMarks": {
        const { sessionId, verifierMarks } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, verifierMarks }
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

      case "session.stepCompleted": {
        const { sessionId, stepIndex } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          const completed = existing.completedStepIndices ?? [];
          if (completed.includes(stepIndex)) return {};
          const nextCompleted = [...completed, stepIndex].sort((a, b) => a - b);
          const isActive = sessionId === state.activeSessionId;
          const nextStepIndex = Math.min(stepIndex + 1, (existing.steps?.length ?? 1) - 1);
          const stepLabel = existing.steps?.[stepIndex] ?? `Step ${stepIndex + 1}`;
          const hasOutputFiles = (existing.outputFiles?.[stepIndex]?.length ?? 0) > 0;

          // Inject synthetic step_completed message into chat
          const syntheticMsg: StepCompletedMessage = { type: "step_completed", stepIndex, stepLabel };
          const nextMessages = [...existing.messages, syntheticMsg];

          return {
            selectedStepIndex: isActive ? nextStepIndex : state.selectedStepIndex,
            previewStepIndex: isActive ? stepIndex : state.previewStepIndex,
            // Auto-open preview panel when step has output files
            previewPanelOpen: isActive && hasOutputFiles ? true : state.previewPanelOpen,
            sessions: {
              ...state.sessions,
              [sessionId]: { ...existing, completedStepIndices: nextCompleted, messages: nextMessages }
            }
          };
        });
        break;
      }

      case "session.status": {
        const { sessionId, status, title, cwd } = event.payload;
        set((state) => {
          const existing = state.sessions[sessionId] ?? createSession(sessionId);
          return {
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
        // Skip intermediate stream events (content_block_delta, etc.)
        // These are handled by partial message state in App.tsx
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
    }
  }
}));
