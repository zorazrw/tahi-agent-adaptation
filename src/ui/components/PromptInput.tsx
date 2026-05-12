import { useCallback, useEffect, useRef } from "react";
import type { ClientEvent, PredictedUserActionSuggestion, WorkflowNode } from "../types";
import { useAppStore } from "../store/useAppStore";
import {
  executeAction,
  type ExecutableAction,
} from "../../lib/executable-actions";
import {
  BrainIcon,
  CircleStopIcon,
  FileTextIcon,
  GitBranchIcon,
  HelpCircleIcon,
  ListChecksIcon,
  MessageSquareTextIcon,
} from "lucide-react";

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

function getExecutableAction(suggestion: PredictedUserActionSuggestion): ExecutableAction | null {
  return suggestion.executable ?? null;
}

function countWorkflowNodes(nodes: WorkflowNode[]): number {
  return nodes.reduce((count, node) => count + 1 + countWorkflowNodes(node.children), 0);
}

function firstWorkflowDescriptions(nodes: WorkflowNode[], limit = 3): string[] {
  const descriptions: string[] = [];
  const visit = (items: WorkflowNode[]) => {
    for (const item of items) {
      if (descriptions.length >= limit) return;
      descriptions.push(item.description || "Untitled step");
      visit(item.children);
    }
  };
  visit(nodes);
  return descriptions;
}

function lineCount(text: string): number {
  if (!text) return 0;
  return text.split("\n").length;
}

function fileName(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

function PredictionActionPreview({
  suggestion,
}: {
  suggestion: PredictedUserActionSuggestion;
}) {
  const action = getExecutableAction(suggestion);
  const type = suggestion.actionType;

  const tone = (() => {
    switch (type) {
      case "message":
        return {
          title: "Message",
          accent: "text-primary",
          chip: "bg-primary-subtle text-primary",
          border: "border-primary/20",
          icon: <MessageSquareTextIcon className="size-4" aria-hidden />,
        };
      case "edit_workflow":
        return {
          title: "Workflow edit",
          accent: "text-info",
          chip: "bg-info-light text-info",
          border: "border-info/20",
          icon: <GitBranchIcon className="size-4" aria-hidden />,
        };
      case "edit_verifier":
        return {
          title: "Verifier edit",
          accent: "text-success",
          chip: "bg-success-light text-success",
          border: "border-success/20",
          icon: <ListChecksIcon className="size-4" aria-hidden />,
        };
      case "file_edit":
        return {
          title: "File edit",
          accent: "text-ink-700",
          chip: "bg-ink-900/5 text-ink-700",
          border: "border-ink-900/12",
          icon: <FileTextIcon className="size-4" aria-hidden />,
        };
      case "brain_edit":
        return {
          title: "Brain edit",
          accent: "text-[#8B5E2A]",
          chip: "bg-[#F1E3D1] text-[#7A4E20]",
          border: "border-[#D8B98F]",
          icon: <BrainIcon className="size-4" aria-hidden />,
        };
      case "stop":
        return {
          title: "Stop",
          accent: "text-error",
          chip: "bg-error-light text-error",
          border: "border-error/20",
          icon: <CircleStopIcon className="size-4" aria-hidden />,
        };
      case "unknown":
        return {
          title: "Unknown",
          accent: "text-muted-foreground",
          chip: "bg-surface-tertiary text-muted-foreground",
          border: "border-ink-900/10",
          icon: <HelpCircleIcon className="size-4" aria-hidden />,
        };
    }
  })();

  const body = (() => {
    if (action?.type === "message") {
      return (
        <>
          <div className="text-[11px] font-medium text-muted-foreground">Draft prompt</div>
          <div className="mt-1 max-h-28 overflow-y-auto whitespace-pre-wrap rounded-lg border border-ink-900/8 bg-white px-3 py-2 text-sm text-ink-900">
            {action.text}
          </div>
        </>
      );
    }

    if (action?.type === "edit_workflow") {
      const descriptions = firstWorkflowDescriptions(action.workflowTree);
      return (
        <>
          <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
            <span>{countWorkflowNodes(action.workflowTree)} steps</span>
            <span>{action.workflowTree.length} root items</span>
          </div>
          {descriptions.length > 0 && (
            <div className="mt-2 grid gap-1.5">
              {descriptions.map((description, index) => (
                <div key={`${description}-${index}`} className="flex items-center gap-2 rounded-lg bg-white px-2.5 py-1.5 text-xs text-ink-800">
                  <span className={`h-1.5 w-1.5 rounded-full ${index === 0 ? "bg-info" : "bg-ink-900/20"}`} />
                  <span className="truncate">{description}</span>
                </div>
              ))}
            </div>
          )}
        </>
      );
    }

    if (action?.type === "edit_verifier") {
      return (
        <>
          <div className="text-[11px] text-muted-foreground">
            Node <span className="font-mono text-ink-700">{action.nodeId}</span>
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {action.verifiers.length > 0 ? (
              action.verifiers.slice(0, 4).map((verifier, index) => (
                <span key={`${verifier}-${index}`} className="rounded-md border border-success/20 bg-white px-2 py-1 text-xs text-ink-800">
                  {verifier}
                </span>
              ))
            ) : (
              <span className="text-xs text-muted-foreground">Clears verifier list</span>
            )}
            {action.verifiers.length > 4 && (
              <span className="rounded-md bg-white px-2 py-1 text-xs text-muted-foreground">
                +{action.verifiers.length - 4} more
              </span>
            )}
          </div>
        </>
      );
    }

    if (action?.type === "file_edit") {
      return (
        <>
          <div className="flex min-w-0 items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-ink-900">{fileName(action.path)}</div>
              <div className="truncate font-mono text-[11px] text-muted-foreground">{action.path}</div>
            </div>
            <span className="shrink-0 rounded-md bg-white px-2 py-1 text-[11px] text-muted-foreground">
              {lineCount(action.contents)} lines
            </span>
          </div>
          {action.contents && (
            <pre className="mt-2 max-h-20 overflow-hidden rounded-lg border border-ink-900/8 bg-white px-3 py-2 text-xs text-ink-700">
              {action.contents.split("\n").slice(0, 3).join("\n")}
            </pre>
          )}
        </>
      );
    }

    if (action?.type === "brain_edit") {
      return (
        <>
          <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
            <span>{action.kind === "memory" ? "Memory" : "Skill"} update</span>
            <span>{action.sections.length} sections</span>
            {action.deletedFileNames?.length ? <span>{action.deletedFileNames.length} deletes</span> : null}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {action.sections.slice(0, 4).map((section) => (
              <span key={section.fileName} className="rounded-md border border-[#D8B98F] bg-white px-2 py-1 text-xs text-ink-800">
                {section.fileName}
              </span>
            ))}
            {action.sections.length > 4 && (
              <span className="rounded-md bg-white px-2 py-1 text-xs text-muted-foreground">
                +{action.sections.length - 4} more
              </span>
            )}
          </div>
        </>
      );
    }

    if (action?.type === "stop" || type === "stop") {
      return (
        <div className="text-sm text-ink-800">
          Predicted that the user is done for now, with no follow-up prompt or structural edit expected.
        </div>
      );
    }

    if (suggestion.draftText.trim()) {
      return (
        <div className="whitespace-pre-wrap rounded-lg border border-ink-900/8 bg-white px-3 py-2 text-sm text-ink-800">
          {suggestion.draftText}
        </div>
      );
    }

    return (
      <div className="text-sm text-muted-foreground">
        The predictor could not produce a validated action payload.
      </div>
    );
  })();

  return (
    <div className={`rounded-xl border ${tone.border} bg-surface-secondary/80 px-3 py-2.5`}>
      <div className="mb-2 flex items-center gap-2">
        <div className={`${tone.accent} flex h-7 w-7 items-center justify-center rounded-lg bg-white shadow-soft`}>
          {tone.icon}
        </div>
        <span className={`rounded-md px-2 py-1 text-[11px] font-semibold uppercase ${tone.chip}`}>
          {tone.title}
        </span>
      </div>
      {body}
    </div>
  );
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
                <PredictionActionPreview suggestion={predictedSuggestion} />
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
