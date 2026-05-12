import { useCallback, useEffect, useRef } from "react";
import type { ClientEvent, PredictedUserActionSuggestion, WorkflowNode } from "../types";
import { useAppStore } from "../store/useAppStore";
import {
  executeAction,
  type ExecutableAction,
} from "../../lib/executable-actions";

const DEFAULT_ALLOWED_TOOLS = "Read,Edit,Bash";
const MAX_ROWS = 12;
const LINE_HEIGHT = 21;
const MAX_HEIGHT = MAX_ROWS * LINE_HEIGHT;
const AUTO_INDUCTION_KEY = "agent-cowork-auto-context-induction";

function isFakeUserPrediction(s: PredictedUserActionSuggestion): boolean {
  const raw = s.rawResponse?.trim();
  if (!raw) return false;
  try {
    const j = JSON.parse(raw) as { fakeUserPredict?: boolean };
    return j?.fakeUserPredict === true;
  } catch {
    return false;
  }
}

/** Fake `edit_workflow` accept: append a root “Visualization” step so Progress updates (UI test hook). */
function appendFakeVisualizationStep(tree: WorkflowNode[] | undefined): WorkflowNode[] {
  const clone = JSON.parse(JSON.stringify(tree ?? [])) as WorkflowNode[];
  const step: WorkflowNode = {
    id: crypto.randomUUID(),
    description: "Visualization",
    outputFiles: [],
    verifiers: [],
    verifierMarks: [],
    children: [],
    status: "pending",
    depth: 0,
  };
  return [...clone, step];
}

function readStoredAutoInduction(): boolean {
  try {
    const v = localStorage.getItem(AUTO_INDUCTION_KEY);
    if (v === "false") return false;
    if (v === "true") return true;
  } catch {
    /* ignore */
  }
  return true;
}

interface PromptInputProps {
  sendEvent: (event: ClientEvent) => void;
  onSendMessage?: () => void;
  disabled?: boolean;
  rightOffset?: string;
  predictedSuggestion?: PredictedUserActionSuggestion | null;
  isPredictingSuggestion?: boolean;
  onClearPredictedSuggestion?: () => void;
  onAcceptPredictedSuggestion?: () => void;
}

export function usePromptActions(sendEvent: (event: ClientEvent) => void) {
  const prompt = useAppStore((state) => state.prompt);
  const cwd = useAppStore((state) => state.cwd);
  const activeSessionId = useAppStore((state) => state.activeSessionId);
  const selectedNodeId = useAppStore((state) => state.selectedNodeId);
  const sessions = useAppStore((state) => state.sessions);
  const setPrompt = useAppStore((state) => state.setPrompt);
  const setPendingStart = useAppStore((state) => state.setPendingStart);
  const setGlobalError = useAppStore((state) => state.setGlobalError);

  const activeSession = activeSessionId ? sessions[activeSessionId] : undefined;
  const isRunning = activeSession?.status === "running";

  const handleSend = useCallback(async () => {
    if (!prompt.trim()) return;

    window.dispatchEvent(new CustomEvent("preview-flush-save"));
    await new Promise((resolve) => setTimeout(resolve, 200));

    if (!activeSessionId) {
      let title = "";
      try {
        setPendingStart(true);
        title = await window.electron.generateSessionTitle(prompt);
      } catch (error) {
        console.error(error);
        setPendingStart(false);
        setGlobalError("Failed to get session title.");
        return;
      }
      sendEvent({
        type: "session.start",
        payload: {
          title,
          prompt,
          cwd: cwd.trim() || undefined,
          allowedTools: DEFAULT_ALLOWED_TOOLS,
          autoContextInduction: readStoredAutoInduction(),
        }
      });
    } else {
      if (activeSession?.status === "running") {
        setGlobalError("Session is still running. Please wait for it to finish.");
        return;
      }
      sendEvent({
        type: "session.continue",
        payload: {
          sessionId: activeSessionId,
          prompt,
          ...(selectedNodeId ? { verificationNodeId: selectedNodeId } : {}),
        },
      });
    }
    setPrompt("");
  }, [activeSession, activeSessionId, cwd, prompt, selectedNodeId, sendEvent, setGlobalError, setPendingStart, setPrompt]);

  const handleStop = useCallback(() => {
    if (!activeSessionId) return;
    sendEvent({ type: "session.stop", payload: { sessionId: activeSessionId } });
  }, [activeSessionId, sendEvent]);

  const handleStartFromModal = useCallback(() => {
    if (!cwd.trim()) {
      setGlobalError("Working Directory is required to start a session.");
      return;
    }
    handleSend();
  }, [cwd, handleSend, setGlobalError]);

  return { prompt, setPrompt, isRunning, handleSend, handleStop, handleStartFromModal };
}

export function PromptInput({
  sendEvent,
  onSendMessage,
  disabled = false,
  rightOffset,
  predictedSuggestion,
  isPredictingSuggestion = false,
  onClearPredictedSuggestion,
  onAcceptPredictedSuggestion,
}: PromptInputProps) {
  const { prompt, setPrompt, isRunning, handleSend, handleStop } = usePromptActions(sendEvent);
  const promptRef = useRef<HTMLTextAreaElement | null>(null);

  const activeSessionId = useAppStore((state) => state.activeSessionId);
  const sessions = useAppStore((state) => state.sessions);
  const selectedNodeId = useAppStore((state) => state.selectedNodeId);
  const setRunningNodeId = useAppStore((state) => state.setRunningNodeId);
  const activeSession = activeSessionId ? sessions[activeSessionId] : undefined;

  // Find the selected node in the workflow tree
  const findNode = (tree: import("../types").WorkflowNode[], id: string): import("../types").WorkflowNode | undefined => {
    for (const node of tree) {
      if (node.id === id) return node;
      const found = findNode(node.children, id);
      if (found) return found;
    }
    return undefined;
  };

  const selectedNode = selectedNodeId && activeSession?.workflowTree
    ? findNode(activeSession.workflowTree, selectedNodeId)
    : undefined;

  // Determine if the selected node is pending (can be started)
  const hasPendingNode = !!(
    activeSessionId &&
    activeSession &&
    activeSession.status !== "running" &&
    selectedNode &&
    selectedNode.status !== "completed" &&
    selectedNode.status !== "running"
  );

  const canAcceptPrediction = Boolean(
    predictedSuggestion &&
    activeSessionId &&
    !isRunning &&
    !disabled &&
    !prompt.trim() &&
    (predictedSuggestion.actionType === "message"
      ? Boolean(predictedSuggestion.draftText.trim())
      : predictedSuggestion.actionType === "edit_workflow" ||
        predictedSuggestion.actionType === "edit_verifier" ||
        predictedSuggestion.actionType === "file_edit" ||
        predictedSuggestion.actionType === "brain_edit" ||
        predictedSuggestion.actionType === "stop" ||
        predictedSuggestion.actionType === "unknown")
  );

  const setGlobalError = useAppStore((s) => s.setGlobalError);

  const acceptPredictedSuggestion = useCallback(async () => {
    if (!predictedSuggestion || !activeSessionId) return;
    const sessionCwd = activeSession?.cwd;

    // Fake `edit_workflow` UI test hook: append a "Visualization" step locally
    // and sync, bypassing the LLM's executable payload.
    if (
      predictedSuggestion.actionType === "edit_workflow" &&
      isFakeUserPrediction(predictedSuggestion)
    ) {
      onAcceptPredictedSuggestion?.();
      const next = appendFakeVisualizationStep(activeSession?.workflowTree);
      const newStep = next[next.length - 1];
      useAppStore.getState().updateWorkflowTree(activeSessionId, next);
      sendEvent({
        type: "session.updateWorkflowTree",
        payload: { sessionId: activeSessionId, workflowTree: next },
      });
      if (newStep) useAppStore.getState().setSelectedNodeId(newStep.id);
      onSendMessage?.();
      return;
    }

    // Validated payload from the LLM: dispatch via the shared executeAction.
    if (predictedSuggestion.executable) {
      onAcceptPredictedSuggestion?.();
      // The LLM doesn't see the current node selection, so override
      // verificationNodeId on message actions from local UI state.
      const action: ExecutableAction =
        predictedSuggestion.executable.type === "message" && selectedNodeId
          ? { ...predictedSuggestion.executable, verificationNodeId: selectedNodeId }
          : predictedSuggestion.executable;

      try {
        await executeAction(action, {
          sessionId: activeSessionId,
          sendEvent,
          currentWorkflowTree: activeSession?.workflowTree,
          writeFile: async (filePath, contents) => {
            const result = await window.electron.writeFile(
              filePath,
              sessionCwd ?? null,
              contents,
              activeSessionId
            );
            if (!result?.success) {
              throw new Error(result?.error ?? "writeFile failed");
            }
          },
        });
        if (action.type === "message") setPrompt("");
        onSendMessage?.();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        console.error("Failed to execute predicted action:", err);
        setGlobalError(`Failed to execute ${action.type}: ${message}`);
      }
      return;
    }

    // Legacy fallback: model returned no validated executable. Preserve the
    // previous behaviour (message → continue, brain_edit → record, others → dismiss).
    const t = predictedSuggestion.actionType;
    if (t === "message") {
      const draftText = predictedSuggestion.draftText.trim();
      if (!draftText) return;
      onAcceptPredictedSuggestion?.();
      onSendMessage?.();
      sendEvent({
        type: "session.continue",
        payload: {
          sessionId: activeSessionId,
          prompt: draftText,
          ...(selectedNodeId ? { verificationNodeId: selectedNodeId } : {}),
        },
      });
      setPrompt("");
      return;
    }
    if (t === "brain_edit") {
      onAcceptPredictedSuggestion?.();
      sendEvent({ type: "session.recordBrainEdit", payload: { sessionId: activeSessionId } });
      onSendMessage?.();
      return;
    }
    onAcceptPredictedSuggestion?.();
    onSendMessage?.();
  }, [
    activeSession,
    activeSessionId,
    onAcceptPredictedSuggestion,
    onSendMessage,
    predictedSuggestion,
    selectedNodeId,
    sendEvent,
    setGlobalError,
    setPrompt,
  ]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape" && predictedSuggestion) {
      e.preventDefault();
      onClearPredictedSuggestion?.();
      return;
    }
    // Tab to start next pending step (prediction accept uses the suggestion “Accept” button, not Tab)
    if (e.key === "Tab" && hasPendingNode && selectedNodeId && !prompt.trim() && !canAcceptPrediction) {
      e.preventDefault();
      setRunningNodeId(selectedNodeId);
      sendEvent({ type: "session.solveNode", payload: { sessionId: activeSessionId!, nodeId: selectedNodeId } });
      return;
    }
    if (disabled && !isRunning) return;
    if (e.key !== "Enter" || e.shiftKey) return;
    e.preventDefault();
    if (isRunning) { handleStop(); return; }
    onSendMessage?.();
    handleSend();
  };

  const handleButtonClick = () => {
    if (disabled && !isRunning) return;
    if (isRunning) {
      handleStop();
    } else {
      onSendMessage?.();
      handleSend();
    }
  };

  const handleInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const target = e.currentTarget;
    target.style.height = "auto";
    const scrollHeight = target.scrollHeight;
    if (scrollHeight > MAX_HEIGHT) {
      target.style.height = `${MAX_HEIGHT}px`;
      target.style.overflowY = "auto";
    } else {
      target.style.height = `${scrollHeight}px`;
      target.style.overflowY = "hidden";
    }
  };

  useEffect(() => {
    if (!promptRef.current) return;
    promptRef.current.style.height = "auto";
    const scrollHeight = promptRef.current.scrollHeight;
    if (scrollHeight > MAX_HEIGHT) {
      promptRef.current.style.height = `${MAX_HEIGHT}px`;
      promptRef.current.style.overflowY = "auto";
    } else {
      promptRef.current.style.height = `${scrollHeight}px`;
      promptRef.current.style.overflowY = "hidden";
    }
  }, [prompt]);

  return (
    <section className={`fixed bottom-0 left-0 bg-gradient-to-t from-surface via-surface to-transparent pb-6 lg:pb-8 pt-8 lg:ml-[var(--sidebar-width)] ${rightOffset ? "px-4" : "px-2"}`} style={{ right: rightOffset ?? 0 }}>
      <div className={`mx-auto flex w-full max-w-full flex-col gap-2 ${rightOffset ? "" : "lg:max-w-3xl"}`}>
        {(predictedSuggestion || isPredictingSuggestion) && (
          <div className="rounded-2xl border border-ink-900/10 bg-white/95 px-4 py-3 shadow-card">
            {isPredictingSuggestion && !predictedSuggestion ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 animate-spin" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                  <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" />
                </svg>
                <span>Generating next action suggestion…</span>
              </div>
            ) : predictedSuggestion ? (
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="rounded-full bg-primary/10 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-primary">
                      {predictedSuggestion.actionType}
                    </span>
                    <span className="text-[11px] text-muted-foreground">
                      {(predictedSuggestion.confidence * 100).toFixed(0)}% self-reported confidence
                    </span>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    {canAcceptPrediction && (
                      <button
                        type="button"
                        onClick={() => acceptPredictedSuggestion()}
                        className="rounded-lg bg-primary px-2.5 py-1.5 text-xs font-medium text-white shadow-soft hover:bg-primary-hover transition-colors"
                      >
                        {predictedSuggestion.actionType === "message" ? "Accept & send" : "Accept"}
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => onClearPredictedSuggestion?.()}
                      className="rounded-lg px-2.5 py-1.5 text-xs font-medium text-muted-foreground hover:bg-ink-900/5 hover:text-ink-800 transition-colors"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
                {predictedSuggestion.actionType === "stop" && !predictedSuggestion.draftText.trim() && (
                  <p className="text-xs text-muted-foreground">
                    Predicted end of turn — no further user message or structural edit expected.
                  </p>
                )}
                {predictedSuggestion.draftText && (
                  <div className="rounded-xl border border-ink-900/8 bg-surface-secondary px-3 py-2 text-sm text-ink-800">
                    {predictedSuggestion.draftText}
                  </div>
                )}
                {predictedSuggestion.rationale && (
                  <div className="text-xs text-muted-foreground line-clamp-3">
                    Rationale: {predictedSuggestion.rationale}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        )}
        <div className="flex w-full items-end gap-3 rounded-2xl border border-ink-900/10 bg-white px-4 py-3 shadow-card transition-[border-color,box-shadow] duration-150 ease-out focus-within:border-ink-900/25 focus-within:shadow-[0_4px_16px_rgba(0,0,0,0.08),0_0_0_3px_rgba(217,119,87,0.08)]">
        <div className="flex-1 relative">
          <textarea
            rows={1}
            className="w-full resize-none bg-transparent py-1.5 text-sm text-ink-900 placeholder:text-ink-500 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60"
            placeholder={disabled ? "Create/select a task to start..." : "Describe what you want agent to handle..."}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            ref={promptRef}
            disabled={disabled && !isRunning}
          />
          {hasPendingNode && !prompt.trim() && !canAcceptPrediction && (
            <div className="absolute right-0 top-1/2 -translate-y-1/2 flex items-center gap-1.5 pointer-events-none">
              <kbd className="inline-flex items-center rounded border border-ink-900/15 bg-ink-900/5 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground leading-none">TAB</kbd>
              <span className="text-xs text-muted-foreground whitespace-nowrap">to start next step</span>
            </div>
          )}
        </div>
        <button
          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${isRunning ? "bg-error text-white hover:bg-error/90" : "bg-primary text-white hover:bg-primary-hover"}`}
          onClick={handleButtonClick}
          aria-label={isRunning ? "Stop session" : "Send prompt"}
          disabled={disabled && !isRunning}
        >
          {isRunning ? (
            <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" /></svg>
          ) : (
            <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true"><path d="M3.4 20.6 21 12 3.4 3.4l2.8 7.2L16 12l-9.8 1.4-2.8 7.2Z" fill="currentColor" /></svg>
          )}
        </button>
        </div>
      </div>
    </section>
  );
}
