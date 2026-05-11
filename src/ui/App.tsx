import { useCallback, useEffect, useRef, useState } from "react";
import { useIPC } from "./hooks/useIPC";
import { useMessageWindow } from "./hooks/useMessageWindow";
import { useAppStore } from "./store/useAppStore";
import type { AppPermissionResult, ClientEvent, PredictedUserActionSuggestion, ServerEvent } from "./types";
import { findNextRunnableWorkflowNodeId } from "./lib/workflow-run";
import { Sidebar } from "./components/Sidebar";
import { HomePromptInput } from "./components/HomePromptInput";
import { SettingsModal } from "./components/SettingsModal";
import { MemoryModal } from "./components/MemoryModal";
import { PromptInput } from "./components/PromptInput";
import { MessageCard } from "./components/EventCard";
import { TaskToolCard } from "./components/TaskToolCard";
import { useGroupedMessages } from "./hooks/useGroupedMessages";
import { FilePreview, getPreviewFileForNode } from "./components/FilePreview";
import { MessageResponse } from "../components/ai-elements/message";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { MessagesSquare, MessageSquareX } from "lucide-react";

const SCROLL_THRESHOLD = 50;

/**
 * Idea bulb like reference: thick upper semicircle (open below), sides curve inward to a short neck,
 * clear gap, then a full-width base bar — artwork fills the viewBox vertically (no empty band below).
 */
function IdeaBulbWithRays({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={`block shrink-0 ${className ?? ""}`}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
      preserveAspectRatio="xMidYMid meet"
    >
      <g stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        {/* Five rays — shorter dashes; inner ends sit back from the glass for a clear band of space */}
        <line x1="12" y1="2.35" x2="12" y2="3.45" />
        <line x1="4.05" y1="6.05" x2="5.35" y2="6.95" />
        <line x1="19.95" y1="6.05" x2="18.65" y2="6.95" />
        <line x1="1.95" y1="10.75" x2="5.35" y2="10.75" />
        <line x1="22.05" y1="10.75" x2="18.65" y2="10.75" />
        {/* Two 90° arcs = upper semicircle (center 12,10.75, r=5); Q curves inward to neck */}
        <path d="M7 10.75A5 5 0 0112 5.75A5 5 0 0117 10.75Q15.45 13.35 14.35 16.35H9.65Q8.55 13.35 7 10.75z" />
        {/* Base — clear gap under neck, bar low so the glyph fills the viewBox */}
        <line x1="8.25" y1="21.35" x2="15.75" y2="21.35" />
      </g>
    </svg>
  );
}

function AnimatedDots() {
  const [count, setCount] = useState(1);
  useEffect(() => {
    const id = setInterval(() => setCount((c) => (c % 3) + 1), 500);
    return () => clearInterval(id);
  }, []);
  return <>{".".repeat(count)}</>;
}

function App() {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const topSentinelRef = useRef<HTMLDivElement>(null);
  const partialMessageRef = useRef("");
  const [partialMessage, setPartialMessage] = useState("");
  const [showPartialMessage, setShowPartialMessage] = useState(false);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
  const [hasNewMessages, setHasNewMessages] = useState(false);
  const [previewWidthPct, setPreviewWidthPct] = useState(45);
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);
  const prevMessagesLengthRef = useRef(0);
  const animatedIndicesRef = useRef(new Set<number>());
  const scrollHeightBeforeLoadRef = useRef(0);
  const shouldRestoreScrollRef = useRef(false);
  const [showPromptInspector, setShowPromptInspector] = useState(false);
  const [showMemoryModal, setShowMemoryModal] = useState(false);
  const [predictedSuggestion, setPredictedSuggestion] = useState<PredictedUserActionSuggestion | null>(null);
  const [isPredictingSuggestion, setIsPredictingSuggestion] = useState(false);
  const [lastAutofillKey, setLastAutofillKey] = useState<string | null>(null);
  const explicitlyStoppedSessionIdsRef = useRef(new Set<string>());

  const sessions = useAppStore((s) => s.sessions);
  const activeSessionId = useAppStore((s) => s.activeSessionId);

  const showSettingsModal = useAppStore((s) => s.showSettingsModal);
  const setShowSettingsModal = useAppStore((s) => s.setShowSettingsModal);
  const globalError = useAppStore((s) => s.globalError);
  const setGlobalError = useAppStore((s) => s.setGlobalError);
  const historyRequested = useAppStore((s) => s.historyRequested);
  const markHistoryRequested = useAppStore((s) => s.markHistoryRequested);
  const resolvePermissionRequest = useAppStore((s) => s.resolvePermissionRequest);
  const handleServerEvent = useAppStore((s) => s.handleServerEvent);
  const apiConfigChecked = useAppStore((s) => s.apiConfigChecked);
  const setApiConfigChecked = useAppStore((s) => s.setApiConfigChecked);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const previewPanelOpen = useAppStore((s) => s.previewPanelOpen);
  const setPreviewPanelOpen = useAppStore((s) => s.setPreviewPanelOpen);
  const contextInductionDepth = useAppStore((s) => s.contextInductionDepth);
  const predictionAssistMode = useAppStore((s) => s.predictionAssistMode);

  // Helper function to extract partial message content
  const getPartialMessageContent = (eventMessage: any) => {
    try {
      const realType = eventMessage.delta.type.split("_")[0];
      return eventMessage.delta[realType];
    } catch (error) {
      console.error(error);
      return "";
    }
  };

  // Handle partial messages from stream events
  const handlePartialMessages = useCallback((partialEvent: ServerEvent) => {
    if (partialEvent.type !== "stream.message" || partialEvent.payload.message.type !== "stream_event") return;

    const message = partialEvent.payload.message as any;
    if (message.event.type === "content_block_start") {
      partialMessageRef.current = "";
      setPartialMessage(partialMessageRef.current);
      setShowPartialMessage(true);
    }

    if (message.event.type === "content_block_delta") {
      partialMessageRef.current += getPartialMessageContent(message.event) || "";
      setPartialMessage(partialMessageRef.current);
    }

    if (message.event.type === "content_block_stop") {
      setShowPartialMessage(false);
      partialMessageRef.current = "";
      setPartialMessage("");
    }
  }, []);

  // Combined event handler
  const onEvent = useCallback((event: ServerEvent) => {
    handleServerEvent(event);
    handlePartialMessages(event);
    // When session stops (idle/error/completed), clear "Thinking..." so it doesn't stick after Stop
    if (
      event.type === "session.status" &&
      event.payload.status !== "running" &&
      event.payload.sessionId === activeSessionId
    ) {
      setShowPartialMessage(false);
      setPartialMessage("");
      partialMessageRef.current = "";
    }
  }, [handleServerEvent, handlePartialMessages, activeSessionId]);

  const { connected, sendEvent } = useIPC(onEvent);
  const sendClientEvent = useCallback((event: ClientEvent) => {
    if (event.type === "session.stop") {
      explicitlyStoppedSessionIdsRef.current.add(event.payload.sessionId);
    } else if (
      event.type === "session.continue" ||
      event.type === "session.solveNode" ||
      event.type === "session.regenerateWorkflow"
    ) {
      explicitlyStoppedSessionIdsRef.current.delete(event.payload.sessionId);
    }
    sendEvent(event);
  }, [sendEvent]);

  const activeSession = activeSessionId ? sessions[activeSessionId] : undefined;
  const messages = activeSession?.messages ?? [];
  const permissionRequests = activeSession?.permissionRequests ?? [];
  const isRunning = activeSession?.status === "running";
  const hasNextRunnableWorkflowNode = Boolean(
    activeSession?.workflowTree?.length &&
    findNextRunnableWorkflowNodeId(activeSession.workflowTree, activeSession.verificationDepth ?? 0)
  );

  const showChatPanel = previewPanelOpen;

  const {
    visibleMessages,
    hasMoreHistory,
    isLoadingHistory,
    loadMoreMessages,
    resetToLatest,
    totalMessages,
  } = useMessageWindow(messages, permissionRequests, activeSessionId);

  const groupedItems = useGroupedMessages(visibleMessages);

  // Check API configuration on startup
  useEffect(() => {
    if (!apiConfigChecked) {
      window.electron.listAvailableModels()
        .then(async (models) => {
          const providers = [...new Set(models.map((model) => model.provider))];
          const statuses = await Promise.all(
            providers.map((provider) => window.electron.getProviderAuthStatus(provider))
          );
          setApiConfigChecked(true);
          if (!statuses.some((status) => status.hasAuth)) {
            setShowSettingsModal(true);
          }
        })
        .catch((err) => {
          console.error("Failed to load agent auth state:", err);
          setApiConfigChecked(true);
        });
    }
  }, [apiConfigChecked, setApiConfigChecked, setShowSettingsModal]);

  useEffect(() => {
    if (connected) sendEvent({ type: "session.list" });
  }, [connected, sendEvent]);

  useEffect(() => {
    if (!activeSessionId || !connected) return;
    const session = sessions[activeSessionId];
    if (session && !session.hydrated && !historyRequested.has(activeSessionId)) {
      markHistoryRequested(activeSessionId);
      sendEvent({ type: "session.history", payload: { sessionId: activeSessionId } });
    }
  }, [activeSessionId, connected, sessions, historyRequested, markHistoryRequested, sendEvent]);

  const handleScroll = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const isAtBottom = scrollTop + clientHeight >= scrollHeight - SCROLL_THRESHOLD;

    if (isAtBottom !== shouldAutoScroll) {
      setShouldAutoScroll(isAtBottom);
      if (isAtBottom) {
        setHasNewMessages(false);
      }
    }
  }, [shouldAutoScroll]);

  // Set up IntersectionObserver for top sentinel
  useEffect(() => {
    const sentinel = topSentinelRef.current;
    const container = scrollContainerRef.current;
    if (!sentinel || !container) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry.isIntersecting && hasMoreHistory && !isLoadingHistory) {
          scrollHeightBeforeLoadRef.current = container.scrollHeight;
          shouldRestoreScrollRef.current = true;
          loadMoreMessages();
        }
      },
      {
        root: container,
        rootMargin: "100px 0px 0px 0px",
        threshold: 0,
      }
    );

    observer.observe(sentinel);

    return () => {
      observer.disconnect();
    };
  }, [hasMoreHistory, isLoadingHistory, loadMoreMessages]);

  // Restore scroll position after loading history
  useEffect(() => {
    if (shouldRestoreScrollRef.current && !isLoadingHistory) {
      const container = scrollContainerRef.current;
      if (container) {
        const newScrollHeight = container.scrollHeight;
        const scrollDiff = newScrollHeight - scrollHeightBeforeLoadRef.current;
        container.scrollTop += scrollDiff;
      }
      shouldRestoreScrollRef.current = false;
    }
  }, [visibleMessages, isLoadingHistory]);

  // Reset scroll state on session change
  useEffect(() => {
    setShouldAutoScroll(true);
    setHasNewMessages(false);
    prevMessagesLengthRef.current = 0;
    animatedIndicesRef.current = new Set();
    setPredictedSuggestion(null);
    setIsPredictingSuggestion(false);
    setLastAutofillKey(null);
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "auto" });
    }, 100);
  }, [activeSessionId]);

  useEffect(() => {
    if (predictionAssistMode === "off") {
      setIsPredictingSuggestion(false);
      setPredictedSuggestion(null);
      return;
    }
    const isNaturalStop =
      activeSessionId &&
      activeSession &&
      activeSession.status === "completed" &&
      !hasNextRunnableWorkflowNode &&
      !explicitlyStoppedSessionIdsRef.current.has(activeSessionId);

    if (!isNaturalStop || messages.length === 0) {
      setIsPredictingSuggestion(false);
      setPredictedSuggestion(null);
      return;
    }
    const sessionId = activeSessionId;

    const lastMessage = messages[messages.length - 1];
    if (
      !lastMessage ||
      lastMessage.type === "user_prompt" ||
      (lastMessage.type === "run_result" && lastMessage.status !== "success")
    ) {
      setIsPredictingSuggestion(false);
      setPredictedSuggestion(null);
      return;
    }

    let cancelled = false;
    setIsPredictingSuggestion(true);

    window.electron.predictNextUserAction(sessionId)
      .then((suggestion) => {
        if (cancelled) return;
        setPredictedSuggestion(suggestion);
      })
      .catch((error) => {
        if (cancelled) return;
        console.error("Failed to predict next user action:", error);
        setPredictedSuggestion(null);
      })
      .finally(() => {
        if (!cancelled) {
          setIsPredictingSuggestion(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeSession, activeSessionId, predictionAssistMode, hasNextRunnableWorkflowNode, messages]);

  useEffect(() => {
    if (predictionAssistMode !== "autofill") return;
    if (!predictedSuggestion || predictedSuggestion.actionType !== "message") return;
    if (!predictedSuggestion.draftText.trim()) return;
    if (!activeSessionId || !activeSession || activeSession.status !== "completed") return;
    if (hasNextRunnableWorkflowNode || explicitlyStoppedSessionIdsRef.current.has(activeSessionId)) return;
    const sourceKey = `${activeSessionId}:${messages.length}:${predictedSuggestion.actionType}:${predictedSuggestion.draftText}`;
    if (lastAutofillKey === sourceKey) return;

    setLastAutofillKey(sourceKey);
    setPredictedSuggestion(null);
    setShouldAutoScroll(true);
    setHasNewMessages(false);
    resetToLatest();
    sendClientEvent({
      type: "session.continue",
      payload: {
        sessionId: activeSessionId,
        prompt: predictedSuggestion.draftText.trim(),
        ...(selectedNodeId ? { verificationNodeId: selectedNodeId } : {}),
      },
    });
  }, [
    activeSession,
    activeSessionId,
    lastAutofillKey,
    messages.length,
    predictedSuggestion,
    predictionAssistMode,
    hasNextRunnableWorkflowNode,
    resetToLatest,
    selectedNodeId,
    sendClientEvent,
  ]);

  // Track new finalized messages for badge / auto-scroll
  useEffect(() => {
    if (shouldAutoScroll) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    } else if (messages.length > prevMessagesLengthRef.current && prevMessagesLengthRef.current > 0) {
      setHasNewMessages(true);
    }
    prevMessagesLengthRef.current = messages.length;
  }, [messages, shouldAutoScroll]);

  // Auto-scroll during streaming partial updates (lightweight, no state changes)
  useEffect(() => {
    if (shouldAutoScroll && partialMessage) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [partialMessage, shouldAutoScroll]);

  const scrollToBottom = useCallback(() => {
    setShouldAutoScroll(true);
    setHasNewMessages(false);
    resetToLatest();
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [resetToLatest]);

  const handleNewSession = useCallback(() => {
    const store = useAppStore.getState();
    store.setActiveSessionId(null);
    store.setAttachedFiles([]);
    store.setTempCwd(null);
    store.setCwd("");
    store.setPrompt("");
  }, []);

  const handleDeleteSession = useCallback((sessionId: string) => {
    sendEvent({ type: "session.delete", payload: { sessionId } });
  }, [sendEvent]);

  const handlePermissionResult = useCallback((toolUseId: string, result: AppPermissionResult) => {
    if (!activeSessionId) return;
    sendEvent({ type: "permission.response", payload: { sessionId: activeSessionId, toolUseId, result } });
    resolvePermissionRequest(activeSessionId, toolUseId);
  }, [activeSessionId, sendEvent, resolvePermissionRequest]);

  const handleSendMessage = useCallback(() => {
    setShouldAutoScroll(true);
    setHasNewMessages(false);
    setPredictedSuggestion(null);
    resetToLatest();
  }, [resetToLatest]);

  // Horizontal drag handler for resizing chat / preview columns
  const handleSplitMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDraggingRef.current = true;
    const container = splitContainerRef.current;
    if (!container) return;

    const onMouseMove = (ev: MouseEvent) => {
      if (!isDraggingRef.current) return;
      const rect = container.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      // Preview is on the left (center), chat on the right; x = preview width
      const previewPct = Math.min(85, Math.max(30, (x / rect.width) * 100));
      setPreviewWidthPct(100 - previewPct);
    };
    const onMouseUp = () => {
      isDraggingRef.current = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  }, []);

  return (
    <div className="flex h-screen bg-surface">
      <Sidebar
        connected={connected}
        sendEvent={sendClientEvent}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
      />

      <main className="flex flex-1 flex-col ml-[var(--sidebar-width)] min-h-0 overflow-hidden bg-surface-cream">
        {activeSession && (
          <div
            className="relative flex shrink-0 items-center justify-end px-4 pt-3 pb-2 border-b border-ink-900/10 bg-surface-cream select-none"
            style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
          >
            <div className="absolute inset-x-0 flex justify-center pointer-events-none px-28 min-w-0 -translate-x-10">
              <span className="min-w-0 max-w-full text-center text-base font-semibold text-ink-900 tracking-tight truncate">
                {activeSession.title}
              </span>
            </div>
            <div className="relative z-10 flex items-center gap-0.5" style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}>
              <button
                type="button"
                onClick={() => setShowMemoryModal(true)}
                className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-ink-800 px-2.5 py-1 rounded-md hover:bg-ink-900/5 transition-colors"
                title={
                  contextInductionDepth > 0
                    ? "Updating memories and skills from the last completed step…"
                    : "Brain — edit memory and skill .md files injected into the agent"
                }
                aria-label="Brain: memories and skills"
                aria-busy={contextInductionDepth > 0}
              >
                <span
                  className={`flex h-5 w-5 shrink-0 items-center justify-center text-inherit ${contextInductionDepth > 0 ? "brain-inducing" : ""}`}
                  aria-hidden
                >
                  <IdeaBulbWithRays className="h-full w-full" />
                </span>
                Brain
              </button>
              <button
                type="button"
                onClick={() => setPreviewPanelOpen(!previewPanelOpen)}
                className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-ink-800 px-2.5 py-1 rounded-md hover:bg-ink-900/5 transition-colors"
                title={previewPanelOpen ? "Close chat" : "Open chat"}
              >
                <span className="flex h-4 w-4 shrink-0 items-center justify-center" aria-hidden>
                  {previewPanelOpen ? (
                    <MessageSquareX className="size-full stroke-[1.5]" />
                  ) : (
                    <MessagesSquare className="size-full stroke-[1.5]" />
                  )}
                </span>
                Chat
              </button>
            </div>
          </div>
        )}

        {!activeSession ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-6 px-8 text-center min-h-0">
            <p className="text-3xl sm:text-4xl font-semibold text-ink-900 tracking-tight max-w-2xl leading-snug">
              What&apos;s on your mind?
            </p>
            <HomePromptInput sendEvent={sendClientEvent} />
          </div>
        ) : (
        <>
        <div className="flex flex-1 flex-col min-h-0" style={{ paddingBottom: "var(--prompt-bar-height)" }}>
          <div ref={splitContainerRef} className="flex flex-1 flex-row min-h-0 overflow-hidden">
          {/* Left (center) column: preview — always visible */}
          <div className="min-w-0 overflow-hidden flex flex-col bg-surface-cream" style={{ flex: showChatPanel ? `${100 - previewWidthPct} 1 0px` : "1 1 0px" }}>
            <div className="flex-1 min-h-0 overflow-hidden p-4">
              <div className="flex flex-col h-full">
                <ErrorBoundary>
                  <FilePreview
                    sessionId={activeSessionId}
                    filePath={(() => {
                      if (!selectedNodeId || !activeSession?.workflowTree) return null;
                      const findNode = (tree: import("./types").WorkflowNode[], id: string): import("./types").WorkflowNode | undefined => {
                        for (const n of tree) {
                          if (n.id === id) return n;
                          const f = findNode(n.children, id);
                          if (f) return f;
                        }
                        return undefined;
                      };
                      const node = findNode(activeSession.workflowTree, selectedNodeId);
                      return getPreviewFileForNode(node?.outputFiles);
                    })()}
                    cwd={activeSession?.cwd}
                    stepCompleted={(() => {
                      if (!selectedNodeId || !activeSession?.workflowTree) return false;
                      const findNode = (tree: import("./types").WorkflowNode[], id: string): import("./types").WorkflowNode | undefined => {
                        for (const n of tree) {
                          if (n.id === id) return n;
                          const f = findNode(n.children, id);
                          if (f) return f;
                        }
                        return undefined;
                      };
                      const node = findNode(activeSession.workflowTree, selectedNodeId);
                      return node?.status === "completed";
                    })()}
                  />
                </ErrorBoundary>
              </div>
            </div>
          </div>

          {/* Vertical drag handle (only when chat panel is open) */}
          {showChatPanel && (
            <div
              onMouseDown={handleSplitMouseDown}
              className="shrink-0 w-3 cursor-col-resize relative group flex items-center justify-center border-l border-r border-ink-900/8 hover:border-ink-900/15 hover:bg-primary/5 transition-all duration-150"
            >
              <div className="w-[3px] h-12 rounded-full bg-ink-900/20 group-hover:bg-primary/40 group-active:bg-primary/50 transition-colors duration-150" />
            </div>
          )}

          {/* Right column: chat / model log (only when Chat is toggled) */}
          {showChatPanel && (
            <div className="min-w-0 overflow-hidden flex flex-col bg-surface-cream" style={{ flex: `${previewWidthPct} 1 0px` }}>
              <div className="flex items-center justify-between px-4 pt-3 pb-1 border-b border-ink-900/10">
                <span className="text-xs font-semibold uppercase tracking-wide text-ink-500">Conversation</span>
                <button
                  type="button"
                  onClick={() => setShowPromptInspector((v) => !v)}
                  className="text-[11px] px-2 py-1 rounded-md border border-ink-900/10 bg-white/70 text-ink-500 hover:text-ink-800 hover:border-ink-900/30 hover:bg-white transition-colors"
                >
                  {showPromptInspector ? "Hide LM input" : "Show LM input"}
                </button>
              </div>
              {showPromptInspector && (
                <div className="px-4 pt-2 pb-1 border-b border-ink-900/10 bg-surface">
                  <div className="text-[11px] font-medium text-ink-600 mb-1">Last LM input</div>
                  <pre className="max-h-56 overflow-auto whitespace-pre-wrap text-[11px] text-ink-700 bg-white/80 rounded-md border border-ink-900/10 px-3 py-2">
                    {activeSession?.lastEffectivePrompt ?? "No LM input captured yet. Run a task or send a message to see the constructed prompt."}
                  </pre>
                </div>
              )}
              <div
                ref={scrollContainerRef}
                onScroll={handleScroll}
                className="flex-1 min-h-0 overflow-y-auto pb-4 pt-4 px-8"
              >
                <div className="mx-auto max-w-3xl">
                  <div ref={topSentinelRef} className="h-1" />

                  {!hasMoreHistory && totalMessages > 0 && (
                    <div className="flex items-center justify-center py-2 mb-2">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <div className="h-px w-12 bg-ink-900/10" />
                        <span>Beginning of conversation</span>
                        <div className="h-px w-12 bg-ink-900/10" />
                      </div>
                    </div>
                  )}

                  {isLoadingHistory && (
                    <div className="flex items-center justify-center py-2 mb-2">
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        <span>Loading...</span>
                      </div>
                    </div>
                  )}

                  {visibleMessages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                      <div className="text-lg font-medium text-ink-700">No messages yet</div>
                      <p className="mt-2 text-sm text-muted-foreground">Start a conversation with agent cowork</p>
                    </div>
                  ) : (
                    groupedItems.map((item, idx) => {
                      if (item.kind === "task_group") {
                        const shouldAnimate = isRunning && !animatedIndicesRef.current.has(item.parentMessageIndex) && item.parentMessageIndex >= prevMessagesLengthRef.current - 1;
                        if (shouldAnimate) animatedIndicesRef.current.add(item.parentMessageIndex);
                        return (
                          <div key={`${activeSessionId}-task-${item.taskToolUseId}`} className={`message-card ${shouldAnimate ? "animate-fade-up" : ""}`}>
                            <ErrorBoundary>
                              <TaskToolCard
                                group={item}
                                isRunning={isRunning}
                                permissionRequest={permissionRequests[0]}
                                onPermissionResult={handlePermissionResult}
                              />
                            </ErrorBoundary>
                          </div>
                        );
                      }
                      const shouldAnimate = isRunning && !animatedIndicesRef.current.has(item.originalIndex) && item.originalIndex >= prevMessagesLengthRef.current - 1;
                      if (shouldAnimate) animatedIndicesRef.current.add(item.originalIndex);
                      return (
                        <div key={`${activeSessionId}-msg-${item.originalIndex}`} className={`message-card ${shouldAnimate ? "animate-fade-up" : ""}`}>
                          <ErrorBoundary>
                            <MessageCard
                              message={item.message}
                              isLast={idx === groupedItems.length - 1}
                              isRunning={isRunning}
                              permissionRequest={permissionRequests[0]}
                              onPermissionResult={handlePermissionResult}
                              skipTaskToolUse
                            />
                          </ErrorBoundary>
                        </div>
                      );
                    })
                  )}

                  <div className="partial-message">
                    {showPartialMessage && !partialMessage.trim() ? (
                      <div className="flex items-center gap-2 py-3 text-sm text-muted-foreground">
                        <span className="inline-grid grid-cols-2 gap-0.5 opacity-40">
                          <span className="h-1 w-1 rounded-full bg-current animate-pulse" />
                          <span className="h-1 w-1 rounded-full bg-current animate-pulse" style={{ animationDelay: "150ms" }} />
                          <span className="h-1 w-1 rounded-full bg-current animate-pulse" style={{ animationDelay: "300ms" }} />
                          <span className="h-1 w-1 rounded-full bg-current animate-pulse" style={{ animationDelay: "450ms" }} />
                        </span>
                        <span>Thinking<AnimatedDots /></span>
                      </div>
                    ) : (
                      <MessageResponse isAnimating={showPartialMessage} caret="block">{partialMessage}</MessageResponse>
                    )}
                  </div>

                  <div ref={messagesEndRef} />
                </div>
              </div>
            </div>
          )}
          </div>
        </div>

        <PromptInput
          sendEvent={sendClientEvent}
          onSendMessage={handleSendMessage}
          disabled={visibleMessages.length === 0}
          rightOffset={undefined}
          predictedSuggestion={predictionAssistMode === "off" ? null : predictedSuggestion}
          isPredictingSuggestion={isPredictingSuggestion}
          onClearPredictedSuggestion={() => setPredictedSuggestion(null)}
        />
        </>
        )}

        {hasNewMessages && !shouldAutoScroll && (
          <button
            onClick={scrollToBottom}
            className="fixed bottom-28 left-1/2 ml-[calc(var(--sidebar-width)/2)] z-40 -translate-x-1/2 flex items-center gap-2 rounded-full bg-primary px-4 py-2 text-sm font-medium text-white shadow-lg transition-all hover:bg-primary-hover hover:scale-105 animate-bounce-subtle"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12l7 7 7-7" />
            </svg>
            <span>New messages</span>
          </button>
        )}
      </main>

      {showSettingsModal && (
        <SettingsModal onClose={() => setShowSettingsModal(false)} />
      )}

      {showMemoryModal && (
        <MemoryModal onClose={() => setShowMemoryModal(false)} taskSessionId={activeSessionId} />
      )}

      {globalError && (
        <div className="fixed bottom-24 left-1/2 z-50 -translate-x-1/2 rounded-xl border border-error/20 bg-error-light px-4 py-3 shadow-lg">
          <div className="flex items-center gap-3">
            <span className="text-sm text-error">{globalError}</span>
            <button className="text-error hover:text-error/80" onClick={() => setGlobalError(null)}>
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12" /></svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
