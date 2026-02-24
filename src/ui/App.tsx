import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PermissionResult } from "@anthropic-ai/claude-agent-sdk";
import { useIPC } from "./hooks/useIPC";
import { useMessageWindow } from "./hooks/useMessageWindow";
import { useAppStore } from "./store/useAppStore";
import type { ServerEvent, StreamMessage } from "./types";
import { Sidebar } from "./components/Sidebar";
import { StartSessionModal } from "./components/StartSessionModal";
import { SettingsModal } from "./components/SettingsModal";
import { PromptInput, usePromptActions } from "./components/PromptInput";
import { MessageCard } from "./components/EventCard";
import { TaskToolCard } from "./components/TaskToolCard";
import { useGroupedMessages } from "./hooks/useGroupedMessages";
import { FilePreview, getPreviewFileForStep } from "./components/FilePreview";
import { PreviewPanelHeader } from "./components/PreviewPanelHeader";
import { MessageResponse } from "../components/ai-elements/message";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { PanelRightOpenIcon, PanelRightCloseIcon } from "lucide-react";

const SCROLL_THRESHOLD = 50;

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

  const sessions = useAppStore((s) => s.sessions);
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const showStartModal = useAppStore((s) => s.showStartModal);
  const setShowStartModal = useAppStore((s) => s.setShowStartModal);
  const showSettingsModal = useAppStore((s) => s.showSettingsModal);
  const setShowSettingsModal = useAppStore((s) => s.setShowSettingsModal);
  const globalError = useAppStore((s) => s.globalError);
  const setGlobalError = useAppStore((s) => s.setGlobalError);
  const historyRequested = useAppStore((s) => s.historyRequested);
  const markHistoryRequested = useAppStore((s) => s.markHistoryRequested);
  const resolvePermissionRequest = useAppStore((s) => s.resolvePermissionRequest);
  const handleServerEvent = useAppStore((s) => s.handleServerEvent);
  const prompt = useAppStore((s) => s.prompt);
  const setPrompt = useAppStore((s) => s.setPrompt);
  const cwd = useAppStore((s) => s.cwd);
  const setCwd = useAppStore((s) => s.setCwd);
  const pendingStart = useAppStore((s) => s.pendingStart);
  const apiConfigChecked = useAppStore((s) => s.apiConfigChecked);
  const setApiConfigChecked = useAppStore((s) => s.setApiConfigChecked);
  const previewStepIndex = useAppStore((s) => s.previewStepIndex);
  const previewPanelOpen = useAppStore((s) => s.previewPanelOpen);
  const setPreviewPanelOpen = useAppStore((s) => s.setPreviewPanelOpen);

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
  }, [handleServerEvent, handlePartialMessages]);

  const { connected, sendEvent } = useIPC(onEvent);
  const { handleStartFromModal } = usePromptActions(sendEvent);

  const activeSession = activeSessionId ? sessions[activeSessionId] : undefined;
  const messages = activeSession?.messages ?? [];
  const permissionRequests = activeSession?.permissionRequests ?? [];
  const isRunning = activeSession?.status === "running";

  // Determine if any step has output files (for toggle button visibility)
  const hasAnyPreviewFiles = useMemo(() => {
    const outputFiles = activeSession?.outputFiles;
    if (!outputFiles) return false;
    return outputFiles.some((files) => files?.length > 0);
  }, [activeSession?.outputFiles]);

  const showPreviewPanel = previewPanelOpen && hasAnyPreviewFiles;

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
      window.electron.checkApiConfig().then((result) => {
        setApiConfigChecked(true);
        if (!result.hasConfig) {
          setShowSettingsModal(true);
        }
      }).catch((err) => {
        console.error("Failed to check API config:", err);
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
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "auto" });
    }, 100);
  }, [activeSessionId]);

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
    useAppStore.getState().setActiveSessionId(null);
    setShowStartModal(true);
  }, [setShowStartModal]);

  const handleDeleteSession = useCallback((sessionId: string) => {
    sendEvent({ type: "session.delete", payload: { sessionId } });
  }, [sendEvent]);

  const handlePermissionResult = useCallback((toolUseId: string, result: PermissionResult) => {
    if (!activeSessionId) return;
    sendEvent({ type: "permission.response", payload: { sessionId: activeSessionId, toolUseId, result } });
    resolvePermissionRequest(activeSessionId, toolUseId);
  }, [activeSessionId, sendEvent, resolvePermissionRequest]);

  const handleSendMessage = useCallback(() => {
    setShouldAutoScroll(true);
    setHasNewMessages(false);
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
      // Preview is on the right, so previewWidthPct = 100 - chatPct
      const chatPct = Math.min(85, Math.max(30, (x / rect.width) * 100));
      setPreviewWidthPct(100 - chatPct);
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

  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);

  // Filter to only high-level messages (skip raw stream events)
  const transcriptMessages = useMemo(() => {
    const validTypes = new Set(["user_prompt", "assistant", "user", "system", "result", "step_completed"]);
    return messages.filter((m) => validTypes.has(m.type));
  }, [messages]);

  const messagesToMarkdown = useCallback((msgs: StreamMessage[]) => {
    const lines: string[] = [];
    for (const msg of msgs) {
      if (msg.type === "step_completed") {
        lines.push(`---\n**Step ${msg.stepIndex + 1} completed:** ${msg.stepLabel}\n`);
      } else if (msg.type === "user_prompt") {
        lines.push(`## User\n\n${msg.prompt}\n`);
      } else if (msg.type === "assistant") {
        for (const block of (msg as any).message.content) {
          if (block.type === "text") lines.push(`## Assistant\n\n${block.text}\n`);
          else if (block.type === "tool_use") lines.push(`### Tool Use: ${block.name}\n\n\`\`\`json\n${JSON.stringify(block.input, null, 2)}\n\`\`\`\n`);
        }
      } else if (msg.type === "user") {
        for (const block of (msg as any).message.content) {
          if (block.type === "tool_result") {
            const text = Array.isArray(block.content) ? block.content.map((c: any) => c.text || "").join("\n") : String(block.content ?? "");
            if (text.trim()) lines.push(`### Tool Result${block.is_error ? " (Error)" : ""}\n\n\`\`\`\n${text}\n\`\`\`\n`);
          }
        }
      } else if (msg.type === "result") {
        const r = msg as any;
        lines.push(`## Result (${r.subtype})\n\nCost: $${r.total_cost_usd?.toFixed(2) ?? "?"} | Duration: ${r.duration_ms ? (r.duration_ms / 60000).toFixed(1) + "min" : "?"}\n`);
      }
    }
    return lines.join("\n");
  }, []);

  const handleCopyTranscript = useCallback((format: "json" | "markdown") => {
    const data = format === "json" ? JSON.stringify(transcriptMessages, null, 2) : messagesToMarkdown(transcriptMessages);
    navigator.clipboard.writeText(data).then(() => {
      setCopyFeedback(format === "json" ? "JSON" : "Markdown");
      setTimeout(() => setCopyFeedback(null), 1500);
    });
  }, [transcriptMessages, messagesToMarkdown]);

  return (
    <div className="flex h-screen bg-surface">
      <Sidebar
        connected={connected}
        sendEvent={sendEvent}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
      />

      <main className="flex flex-1 flex-col ml-[var(--sidebar-width)] min-h-0 overflow-hidden bg-surface-cream">
        <div
          className="flex shrink-0 items-center justify-between h-12 px-4 border-b border-ink-900/10 bg-surface-cream select-none"
          style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
        >
          <span className="text-sm font-medium text-ink-700">{activeSession?.title || "Agent Cowork"}</span>
          {activeSession && (
            <div className="flex items-center gap-1" style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}>
              {copyFeedback ? (
                <span className="text-xs text-success px-2 py-0.5">Copied {copyFeedback}!</span>
              ) : (
                <>
                  <button onClick={() => handleCopyTranscript("markdown")} className="text-xs text-muted-foreground hover:text-ink-700 px-2 py-0.5 rounded hover:bg-ink-900/5 transition-colors" title="Copy as Markdown">
                    Copy MD
                  </button>
                  <button onClick={() => handleCopyTranscript("json")} className="text-xs text-muted-foreground hover:text-ink-700 px-2 py-0.5 rounded hover:bg-ink-900/5 transition-colors" title="Copy as JSON">
                    Copy JSON
                  </button>
                </>
              )}
              {hasAnyPreviewFiles && (
                <button
                  onClick={() => setPreviewPanelOpen(!previewPanelOpen)}
                  className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-ink-700 px-2 py-0.5 rounded hover:bg-ink-900/5 transition-colors"
                  title={previewPanelOpen ? "Close preview" : "Open preview"}
                >
                  {previewPanelOpen ? (
                    <PanelRightCloseIcon className="size-3.5" />
                  ) : (
                    <PanelRightOpenIcon className="size-3.5" />
                  )}
                  Preview
                </button>
              )}
            </div>
          )}
        </div>

        {!activeSession ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4 px-8 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
              <svg viewBox="0 0 24 24" className="h-8 w-8 text-primary" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 6.042A8.967 8.967 0 0 0 6 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 0 1 6 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 0 1 6-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0 0 18 18a8.967 8.967 0 0 0-6 2.292m0-14.25v14.25" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-semibold text-ink-800">Agent Cowork</h2>
              <p className="mt-1 text-sm text-muted-foreground max-w-md">Create a new task to start working with an AI agent. Define steps, set verification criteria, and preview outputs.</p>
            </div>
            <button
              className="rounded-full bg-primary px-6 py-2.5 text-sm font-medium text-white shadow-soft hover:bg-primary-hover transition-colors"
              onClick={handleNewSession}
            >
              + New Task
            </button>
          </div>
        ) : (
        <div ref={splitContainerRef} className="flex flex-1 flex-row min-h-0 overflow-hidden">
          {/* Left column: chat */}
          <div className="min-w-0 overflow-hidden flex flex-col bg-surface-cream" style={{ flex: `${100 - previewWidthPct} 1 0px` }}>
            {/* Messages */}
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
                    // kind: "message"
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

                {/* Partial message display */}
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

            {/* Spacer for fixed PromptInput */}
            <div className="h-24 shrink-0 lg:h-28" aria-hidden />
            <PromptInput
              sendEvent={sendEvent}
              onSendMessage={handleSendMessage}
              disabled={visibleMessages.length === 0}
              rightOffset={showPreviewPanel ? `calc(${previewWidthPct}% - var(--sidebar-width) * ${previewWidthPct} / 100 + 12px)` : undefined}
            />
          </div>

          {/* Vertical drag handle (only when preview panel is open) */}
          {showPreviewPanel && (
            <div
              onMouseDown={handleSplitMouseDown}
              className="shrink-0 w-3 cursor-col-resize relative group flex items-center justify-center border-l border-r border-ink-900/8 hover:border-ink-900/15 hover:bg-primary/5 transition-all duration-150"
            >
              <div className="w-[3px] h-12 rounded-full bg-ink-900/20 group-hover:bg-primary/40 group-active:bg-primary/50 transition-colors duration-150" />
            </div>
          )}

          {/* Right column: preview panel (only when open) */}
          {showPreviewPanel && (
            <div className="min-w-0 overflow-hidden flex flex-col bg-surface-cream" style={{ flex: `${previewWidthPct} 1 0px` }}>
              <PreviewPanelHeader />
              <div className="flex-1 min-h-0 overflow-hidden p-4">
                <div className="flex flex-col h-full">
                <ErrorBoundary>
                  <FilePreview
                    filePath={getPreviewFileForStep(activeSession?.outputFiles, previewStepIndex)}
                    cwd={activeSession?.cwd}
                    stepCompleted={activeSession?.completedStepIndices?.includes(previewStepIndex) ?? false}
                  />
                </ErrorBoundary>
                </div>
              </div>
            </div>
          )}
        </div>
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

      {showStartModal && (
        <StartSessionModal
          cwd={cwd}
          prompt={prompt}
          pendingStart={pendingStart}
          onCwdChange={setCwd}
          onPromptChange={setPrompt}
          onStart={handleStartFromModal}
          onClose={() => setShowStartModal(false)}
        />
      )}

      {showSettingsModal && (
        <SettingsModal onClose={() => setShowSettingsModal(false)} />
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
