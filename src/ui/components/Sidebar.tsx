import { useEffect, useMemo, useRef, useState } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import * as Dialog from "@radix-ui/react-dialog";
import type { ClientEvent } from "../types";
import { useAppStore } from "../store/useAppStore";

interface SidebarProps {
  connected: boolean;
  sendEvent: (event: ClientEvent) => void;
  onNewSession: () => void;
  onDeleteSession: (sessionId: string) => void;
}

export function Sidebar({
  sendEvent,
  onNewSession,
  onDeleteSession
}: SidebarProps) {
  const sessions = useAppStore((state) => state.sessions);
  const activeSessionId = useAppStore((state) => state.activeSessionId);
  const setActiveSessionId = useAppStore((state) => state.setActiveSessionId);
  const selectedStepIndex = useAppStore((state) => state.selectedStepIndex);
  const setSelectedStepIndex = useAppStore((state) => state.setSelectedStepIndex);
  const updateSessionSteps = useAppStore((state) => state.updateSessionSteps);
  const updateSessionVerificationCriteria = useAppStore((state) => state.updateSessionVerificationCriteria);
  const updateSessionVerifierMarks = useAppStore((state) => state.updateSessionVerifierMarks);
  const updateSessionTitle = useAppStore((state) => state.updateSessionTitle);
  const runningStepIndex = useAppStore((state) => state.runningStepIndex);
  const setRunningStepIndex = useAppStore((state) => state.setRunningStepIndex);
  const [resumeSessionId, setResumeSessionId] = useState<string | null>(null);
  const [deleteConfirmSessionId, setDeleteConfirmSessionId] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const closeTimerRef = useRef<number | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const titleInputRef = useRef<HTMLInputElement | null>(null);

  const activeSession = activeSessionId ? sessions[activeSessionId] : undefined;
  const progressSteps = activeSession?.steps ?? [];

  const [verificationCriteriaByStep, setVerificationCriteriaByStep] = useState<string[][]>([]);
  const [editingStepIndex, setEditingStepIndex] = useState<number | null>(null);
  const [editingStepDraft, setEditingStepDraft] = useState("");
  const editingStepInputRef = useRef<HTMLInputElement | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [draftText, setDraftText] = useState("");
  const editInputRef = useRef<HTMLTextAreaElement | null>(null);

  // Sync verification criteria from store when session or steps change (and not editing a criterion).
  useEffect(() => {
    if (editingIndex !== null) return;
    const fromStore = activeSession?.verificationCriteria ?? [];
    const n = progressSteps.length;
    const synced =
      fromStore.length >= n
        ? fromStore.slice(0, n)
        : [...fromStore, ...Array.from({ length: n - fromStore.length }, () => [])];
    setVerificationCriteriaByStep(synced);
  }, [activeSessionId, activeSession?.verificationCriteria, progressSteps.length, editingIndex]);

  useEffect(() => {
    if (selectedStepIndex >= progressSteps.length) {
      setSelectedStepIndex(Math.max(0, progressSteps.length - 1));
    }
  }, [progressSteps.length, selectedStepIndex]);

  useEffect(() => {
    if (editingStepIndex !== null) {
      editingStepInputRef.current?.focus();
    }
  }, [editingStepIndex]);

  useEffect(() => {
    if (editingTitle) {
      titleInputRef.current?.focus();
    }
  }, [editingTitle]);

  useEffect(() => {
    setEditingTitle(false);
    setTitleDraft("");
  }, [activeSessionId]);

  const startEditTitle = () => {
    if (!activeSessionId || !sessions[activeSessionId]) return;
    setTitleDraft(sessions[activeSessionId].title ?? "");
    setEditingTitle(true);
  };

  const saveTitle = () => {
    if (!activeSessionId || !editingTitle) return;
    const trimmed = titleDraft.trim();
    if (trimmed) {
      updateSessionTitle(activeSessionId, trimmed);
      sendEvent({ type: "session.updateTitle", payload: { sessionId: activeSessionId, title: trimmed } });
    }
    setEditingTitle(false);
    setTitleDraft("");
  };

  const startEditStepLabel = (index: number) => {
    setEditingStepIndex(index);
    setEditingStepDraft(progressSteps[index] ?? "");
  };

  const saveStepLabelEdit = () => {
    if (editingStepIndex === null || !activeSessionId) return;
    const trimmed = editingStepDraft.trim();
    const newSteps = [...progressSteps];
    if (trimmed) {
      newSteps[editingStepIndex] = trimmed;
    } else {
      newSteps.splice(editingStepIndex, 1);
    }
    const stepsToSave = newSteps.length > 0 ? newSteps : [];
    if (newSteps.length > 0) {
      updateSessionSteps(activeSessionId, newSteps);
    } else {
      updateSessionSteps(activeSessionId, []);
    }
    sendEvent({ type: "session.updateSteps", payload: { sessionId: activeSessionId, steps: stepsToSave } });
    const n = newSteps.length;
    const criteria = verificationCriteriaByStep.slice(0, n);
    while (criteria.length < n) criteria.push([]);
    setVerificationCriteriaByStep(criteria);
    updateSessionVerificationCriteria(activeSessionId, criteria);
    sendEvent({ type: "session.updateVerificationCriteria", payload: { sessionId: activeSessionId, verificationCriteria: criteria } });
    setEditingStepIndex(null);
    setEditingStepDraft("");
  };

  const deleteStep = (index: number) => {
    if (!activeSessionId) return;
    const newSteps = progressSteps.filter((_, i) => i !== index);
    const stepsToSave = newSteps.length > 0 ? newSteps : [];
    updateSessionSteps(activeSessionId, stepsToSave);
    sendEvent({ type: "session.updateSteps", payload: { sessionId: activeSessionId, steps: stepsToSave } });
    const criteria = verificationCriteriaByStep.filter((_, i) => i !== index);
    setVerificationCriteriaByStep(criteria);
    updateSessionVerificationCriteria(activeSessionId, criteria);
    sendEvent({ type: "session.updateVerificationCriteria", payload: { sessionId: activeSessionId, verificationCriteria: criteria } });
    setEditingStepIndex(null);
    if (selectedStepIndex >= newSteps.length) {
      setSelectedStepIndex(Math.max(0, newSteps.length - 1));
    } else if (selectedStepIndex > index) {
      setSelectedStepIndex(selectedStepIndex - 1);
    }
  };

  const addStep = () => {
    if (!activeSessionId) return;
    const newSteps = [...progressSteps, ""];
    updateSessionSteps(activeSessionId, newSteps);
    sendEvent({ type: "session.updateSteps", payload: { sessionId: activeSessionId, steps: newSteps } });
    const criteria = [...verificationCriteriaByStep, []];
    setVerificationCriteriaByStep(criteria);
    updateSessionVerificationCriteria(activeSessionId, criteria);
    sendEvent({ type: "session.updateVerificationCriteria", payload: { sessionId: activeSessionId, verificationCriteria: criteria } });
    setEditingStepIndex(newSteps.length - 1);
    setEditingStepDraft("");
  };

  const verificationCriteria = verificationCriteriaByStep[selectedStepIndex] ?? [];

  const formatCwd = (cwd?: string) => {
    if (!cwd) return "Working dir unavailable";
    const parts = cwd.split(/[\\/]+/).filter(Boolean);
    const tail = parts.slice(-2).join("/");
    return `/${tail || cwd}`;
  };

  const sessionList = useMemo(() => {
    const list = Object.values(sessions);
    list.sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0));
    return list;
  }, [sessions]);

  useEffect(() => {
    setCopied(false);
    if (closeTimerRef.current) {
      window.clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  }, [resumeSessionId]);

  useEffect(() => {
    return () => {
      if (closeTimerRef.current) {
        window.clearTimeout(closeTimerRef.current);
        closeTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (editingIndex !== null) {
      editInputRef.current?.focus();
    }
  }, [editingIndex]);

  const startAddCriterion = () => {
    if (!activeSessionId) return;
    const list = verificationCriteriaByStep[selectedStepIndex] ?? [];
    const newIndex = list.length;
    const next = verificationCriteriaByStep.slice();
    const stepList = next[selectedStepIndex] ?? [];
    next[selectedStepIndex] = [...stepList, ""];
    setVerificationCriteriaByStep(next);
    setEditingIndex(newIndex);
    setDraftText("");
    updateSessionVerificationCriteria(activeSessionId, next);
    sendEvent({ type: "session.updateVerificationCriteria", payload: { sessionId: activeSessionId, verificationCriteria: next } });
  };

  const startEditCriterion = (index: number) => {
    setEditingIndex(index);
    setDraftText(verificationCriteria[index] ?? "");
  };

  const saveEdit = () => {
    if (editingIndex === null || !activeSessionId) return;
    const trimmed = draftText.trim();
    const next = verificationCriteriaByStep.slice();
    const stepList = [...(next[selectedStepIndex] ?? [])];
    if (trimmed) {
      stepList[editingIndex] = trimmed;
    } else {
      stepList.splice(editingIndex, 1);
    }
    next[selectedStepIndex] = stepList;
    setVerificationCriteriaByStep(next);
    setEditingIndex(null);
    setDraftText("");
    updateSessionVerificationCriteria(activeSessionId, next);
    sendEvent({ type: "session.updateVerificationCriteria", payload: { sessionId: activeSessionId, verificationCriteria: next } });
  };

  const currentVerifierMarks = activeSession?.verifierMarks?.[selectedStepIndex] ?? [];

  const removeCriterion = (index: number) => {
    if (!activeSessionId) return;
    const nextCriteria = verificationCriteriaByStep.slice();
    const stepList = [...(nextCriteria[selectedStepIndex] ?? [])];
    stepList.splice(index, 1);
    nextCriteria[selectedStepIndex] = stepList;
    setVerificationCriteriaByStep(nextCriteria);
    updateSessionVerificationCriteria(activeSessionId, nextCriteria);
    sendEvent({ type: "session.updateVerificationCriteria", payload: { sessionId: activeSessionId, verificationCriteria: nextCriteria } });
    const allMarks = activeSession?.verifierMarks ?? [];
    const stepMarks = [...(allMarks[selectedStepIndex] ?? [])];
    stepMarks.splice(index, 1);
    const nextFull = allMarks.slice(0, progressSteps.length);
    while (nextFull.length <= selectedStepIndex) nextFull.push([]);
    nextFull[selectedStepIndex] = stepMarks;
    updateSessionVerifierMarks(activeSessionId, nextFull);
    sendEvent({ type: "session.updateVerifierMarks", payload: { sessionId: activeSessionId, verifierMarks: nextFull } });
    if (editingIndex === index) {
      setEditingIndex(null);
      setDraftText("");
    } else if (editingIndex != null && editingIndex > index) {
      setEditingIndex(editingIndex - 1);
    }
  };

  const toggleVerifierMark = (index: number) => {
    if (!activeSessionId) return;
    const allMarks = activeSession?.verifierMarks ?? [];
    const stepMarks = [...(allMarks[selectedStepIndex] ?? [])];
    while (stepMarks.length <= index) stepMarks.push(undefined);
    const cur = stepMarks[index];
    stepMarks[index] = cur === undefined ? "check" : cur === "check" ? "cross" : undefined;
    const nextFull = allMarks.slice(0, progressSteps.length);
    while (nextFull.length <= selectedStepIndex) nextFull.push([]);
    nextFull[selectedStepIndex] = stepMarks;
    updateSessionVerifierMarks(activeSessionId, nextFull);
    sendEvent({ type: "session.updateVerifierMarks", payload: { sessionId: activeSessionId, verifierMarks: nextFull } });
  };

  const handleCopyCommand = async () => {
    if (!resumeSessionId) return;
    const command = `claude --resume ${resumeSessionId}`;
    try {
      await navigator.clipboard.writeText(command);
    } catch {
      return;
    }
    setCopied(true);
    if (closeTimerRef.current) {
      window.clearTimeout(closeTimerRef.current);
    }
    closeTimerRef.current = window.setTimeout(() => {
      setResumeSessionId(null);
    }, 3000);
  };

  return (
    <aside className="fixed inset-y-0 left-0 flex h-full w-[var(--sidebar-width)] flex-col border-r border-ink-900/5 bg-[#FAF9F6] px-4 pb-4 pt-12">
      <div 
        className="absolute top-0 left-0 right-0 h-12"
        style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
      />
      <div className="flex shrink-0 gap-2 mt-4">
        <button
          className="flex-1 rounded-xl border border-ink-900/10 bg-surface px-4 py-2.5 text-sm font-medium text-ink-700 hover:bg-surface-tertiary hover:border-ink-900/20 transition-colors"
          onClick={onNewSession}
        >
          + New Task
        </button>
        <button
          className="rounded-xl border border-ink-900/10 bg-surface px-4 py-3 text-sm text-ink-700 hover:bg-surface-tertiary hover:border-ink-900/20 transition-colors"
          onClick={() => useAppStore.getState().setShowSettingsModal(true)}
          aria-label="Settings"
        >
          <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.8">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.08a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.08a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        </button>
      </div>
      {/* Top half: current task box + switch dropdown */}
      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <div className="py-2">
          {sessionList.length === 0 ? (
            <div className="rounded-xl border border-ink-900/5 bg-surface px-4 py-5 text-center text-xs text-muted-foreground">
              No sessions yet. Click "+ New Task" to start.
            </div>
          ) : (
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <div
                  role="button"
                  tabIndex={0}
                  className="flex w-full cursor-pointer items-center gap-2 rounded-xl border border-ink-900/10 bg-surface px-3 py-3 text-left transition hover:bg-surface-tertiary hover:border-ink-900/20 focus:outline-none focus:ring-2 focus:ring-primary/30 data-[state=open]:border-primary/30 data-[state=open]:bg-primary-subtle"
                >
                  <div
                    className="flex min-w-0 flex-1 flex-col overflow-hidden"
                    onClick={(e) => e.stopPropagation()}
                    onPointerDown={(e) => e.stopPropagation()}
                    onKeyDown={(e) => e.stopPropagation()}
                  >
                    {activeSessionId && sessions[activeSessionId] ? (
                      editingTitle ? (
                        <input
                          ref={titleInputRef}
                          type="text"
                          className="w-full rounded border border-ink-900/20 bg-white px-1.5 py-0.5 text-[12px] font-medium text-ink-800 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20"
                          value={titleDraft}
                          onChange={(e) => setTitleDraft(e.target.value)}
                          onBlur={saveTitle}
                          onKeyDown={(e) => {
                            e.stopPropagation();
                            if (e.key === "Enter") saveTitle();
                            if (e.key === "Escape") { setEditingTitle(false); setTitleDraft(""); }
                          }}
                          onClick={(e) => e.stopPropagation()}
                          onPointerDown={(e) => e.stopPropagation()}
                          aria-label="Edit task title"
                        />
                      ) : (
                        <>
                          <div
                            className={`text-[12px] font-medium ${sessions[activeSessionId].status === "running" ? "text-info" : sessions[activeSessionId].status === "completed" ? "text-success" : sessions[activeSessionId].status === "error" ? "text-error" : "text-ink-800"} truncate cursor-text hover:underline`}
                            onClick={(e) => { e.stopPropagation(); e.preventDefault(); startEditTitle(); }}
                            onPointerDown={(e) => { e.stopPropagation(); e.preventDefault(); }}
                            role="button"
                            tabIndex={0}
                            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); startEditTitle(); } }}
                            aria-label="Edit task title"
                          >
                            {sessions[activeSessionId].title}
                          </div>
                          <div className="mt-0.5 text-xs text-muted-foreground">
                            <span className="truncate">{formatCwd(sessions[activeSessionId].cwd)}</span>
                          </div>
                        </>
                      )
                    ) : (
                      <span className="text-xs text-muted-foreground">Select a task</span>
                    )}
                  </div>
                  <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0 text-ink-500" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </div>
              </DropdownMenu.Trigger>
              <DropdownMenu.Portal>
                <DropdownMenu.Content className="z-50 max-h-[min(60vh,320px)] min-w-[240px] overflow-y-auto rounded-xl border border-ink-900/10 bg-white p-1 shadow-lg" align="start" sideOffset={6}>
                  <div className="px-2 py-1.5 text-xs font-medium text-ink-500">Switch task</div>
                  {sessionList.map((session) => (
                    <DropdownMenu.Item
                      key={session.id}
                      className="flex cursor-pointer flex-col items-start gap-0.5 rounded-lg px-3 py-2 text-left text-sm text-ink-700 outline-none hover:bg-ink-900/5 data-[highlighted]:bg-ink-900/5"
                      onSelect={() => setActiveSessionId(session.id)}
                    >
                      <span className={`font-medium ${session.status === "running" ? "text-info" : session.status === "completed" ? "text-success" : session.status === "error" ? "text-error" : "text-ink-800"}`}>
                        {session.title}
                      </span>
                      <span className="text-xs text-muted-foreground">{formatCwd(session.cwd)}</span>
                    </DropdownMenu.Item>
                  ))}
                  {activeSessionId && (
                    <>
                      <DropdownMenu.Separator className="my-1 h-px bg-ink-900/10" />
                      <DropdownMenu.Item className="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm text-ink-700 outline-none hover:bg-ink-900/5" onSelect={() => activeSessionId && setDeleteConfirmSessionId(activeSessionId)}>
                        <svg viewBox="0 0 24 24" className="h-4 w-4 text-error/80" fill="none" stroke="currentColor" strokeWidth="1.8">
                          <path d="M4 7h16" /><path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" /><path d="M7 7l1 12a1 1 0 0 0 1 .9h6a1 1 0 0 0 1-.9l1-12" />
                        </svg>
                        Delete this session
                      </DropdownMenu.Item>
                      <DropdownMenu.Item className="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm text-ink-700 outline-none hover:bg-ink-900/5" onSelect={() => activeSessionId && setResumeSessionId(activeSessionId)}>
                        <svg viewBox="0 0 24 24" className="h-4 w-4 text-ink-500" fill="none" stroke="currentColor" strokeWidth="1.8">
                          <path d="M4 5h16v14H4z" /><path d="M7 9h10M7 12h6" /><path d="M13 15l3 2-3 2" />
                        </svg>
                        Resume in Claude Code
                      </DropdownMenu.Item>
                    </>
                  )}
                </DropdownMenu.Content>
              </DropdownMenu.Portal>
            </DropdownMenu.Root>
          )}
        </div>
        {/* Progress: vertical timeline */}
        {sessionList.length > 0 && (
          <div className="shrink-0 border-t border-ink-900/10 pt-3">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-3 w-0.5 rounded-full bg-primary" />
              <span className="text-xs font-semibold uppercase tracking-wide text-ink-500">Progress</span>
            </div>
            {progressSteps.length === 0 ? (
              <p className="text-xs text-muted-foreground py-1 mb-1">No steps defined yet. Add steps to track progress.</p>
            ) : (
              <div className="flex flex-col gap-0">
                {progressSteps.map((label, i) => {
                  const isCompleted = activeSession?.completedStepIndices?.includes(i);
                  const isSelected = selectedStepIndex === i;
                  const isRunning = runningStepIndex === i && activeSession?.status === "running";
                  return (
                    <div key={i} className="flex gap-2.5 min-w-0">
                      {/* Timeline column: dot + connector */}
                      <div className="flex flex-col items-center shrink-0 w-4 pt-[3px]">
                        <button
                          type="button"
                          className="shrink-0 focus:outline-none"
                          onClick={() => { setSelectedStepIndex(i); setEditingIndex(null); }}
                          aria-label={`Select step ${i + 1}`}
                        >
                          {isRunning ? (
                            <div className="h-3 w-3 rounded-full border-2 border-primary border-t-transparent animate-spin" />
                          ) : (
                            <div className={`h-3 w-3 rounded-full border-2 transition-colors ${
                              isCompleted
                                ? "step-circle-completed"
                                : isSelected
                                  ? "border-primary bg-primary/20"
                                  : "border-ink-900/25 bg-surface"
                            }`} />
                          )}
                        </button>
                        {i < progressSteps.length - 1 && (
                          <div className="w-px flex-1 min-h-[12px] bg-ink-900/15 mt-0.5" />
                        )}
                      </div>
                      {/* Label column */}
                      <div className="flex-1 min-w-0 pb-2.5">
                        {editingStepIndex === i ? (
                          <input
                            ref={editingStepIndex === i ? (el) => { editingStepInputRef.current = el; } : undefined}
                            type="text"
                            className="w-full rounded border border-ink-900/20 bg-surface px-2 py-1 text-xs text-ink-800 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20"
                            value={editingStepDraft}
                            onChange={(e) => setEditingStepDraft(e.target.value)}
                            onBlur={saveStepLabelEdit}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") { e.preventDefault(); saveStepLabelEdit(); }
                              if (e.key === "Escape") { setEditingStepIndex(null); setEditingStepDraft(""); }
                            }}
                          />
                        ) : (
                          <div className="group flex items-start gap-1 min-w-0">
                            <button
                              type="button"
                              className={`flex-1 text-left text-xs leading-snug rounded px-1 -mx-1 py-0.5 transition-colors hover:bg-ink-900/5 line-clamp-2 break-words ${isSelected ? "font-medium text-ink-800" : "text-ink-700"}`}
                              onClick={() => { setSelectedStepIndex(i); setEditingIndex(null); }}
                            >
                              {label || <span className="italic text-muted-foreground">Untitled step</span>}
                            </button>
                            <div className="shrink-0 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                              <button
                                type="button"
                                className="rounded p-0.5 text-ink-400 hover:text-ink-600 hover:bg-ink-900/10 focus:outline-none focus:ring-2 focus:ring-primary/30"
                                onClick={(e) => { e.stopPropagation(); startEditStepLabel(i); }}
                                aria-label="Edit step label"
                              >
                                <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2">
                                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                                </svg>
                              </button>
                              <button
                                type="button"
                                className="rounded p-0.5 text-ink-400 hover:text-error hover:bg-ink-900/10 focus:outline-none focus:ring-2 focus:ring-primary/30"
                                onClick={(e) => { e.stopPropagation(); deleteStep(i); }}
                                aria-label="Delete step"
                              >
                                <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2">
                                  <path d="M18 6L6 18M6 6l12 12" />
                                </svg>
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            {/* Action buttons: Add step + Run selected step */}
            <div className="flex gap-2 mt-1">
              <button
                type="button"
                className="flex-1 flex items-center justify-center gap-1.5 rounded-lg border border-dashed border-ink-900/20 px-2 py-1.5 text-xs text-muted-foreground hover:border-ink-900/30 hover:text-ink-600 hover:bg-ink-900/5 transition-colors"
                onClick={addStep}
                aria-label="Add step"
              >
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 5v14M5 12h14" />
                </svg>
                <span>Add step</span>
              </button>
              {progressSteps.length > 0 && activeSessionId && (() => {
                const isStepCompleted = activeSession?.completedStepIndices?.includes(selectedStepIndex);
                const isSelectedStepRunning = runningStepIndex === selectedStepIndex && activeSession?.status === "running";
                if (isSelectedStepRunning) {
                  return (
                    <div className="flex items-center justify-center gap-1.5 rounded-lg bg-ink-900/8 px-3 py-1.5 text-xs font-medium text-muted-foreground">
                      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0 animate-spin" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <circle cx="12" cy="12" r="9" strokeOpacity="0.25" />
                        <path d="M12 3a9 9 0 0 1 9 9" strokeLinecap="round" />
                      </svg>
                      <span>Running step {selectedStepIndex + 1}...</span>
                    </div>
                  );
                }
                return (
                  <button
                    type="button"
                    className="flex items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-white hover:bg-primary-hover transition-colors shadow-soft"
                    onClick={() => {
                      setRunningStepIndex(selectedStepIndex);
                      sendEvent({ type: "session.solveStep", payload: { sessionId: activeSessionId, stepIndex: selectedStepIndex } });
                    }}
                    aria-label={`${isStepCompleted ? "Rerun" : "Run"} step ${selectedStepIndex + 1}`}
                  >
                    {isStepCompleted ? (
                      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="1 4 1 10 7 10" />
                        <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                      </svg>
                    ) : (
                      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0" fill="currentColor">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    )}
                    <span>{isStepCompleted ? "Rerun" : "Run"} step {selectedStepIndex + 1}</span>
                  </button>
                );
              })()}
            </div>
          </div>
        )}
      </div>
      {/* Files: expected output file name(s) for the current step */}
      <div className="shrink-0 flex flex-col border-t border-ink-900/10 pt-2 pb-2">
        <div className="shrink-0 flex items-center gap-2 mb-1.5">
          <div className="h-3 w-0.5 rounded-full bg-ink-400" />
          <span className="text-xs font-semibold uppercase tracking-wide text-ink-500">
            Files {progressSteps.length > 0 ? `(Step ${selectedStepIndex + 1})` : ""}
          </span>
        </div>
        <div className="min-h-0 overflow-y-auto max-h-[88px] flex flex-col gap-1">
          {(activeSession?.outputFiles?.[selectedStepIndex] ?? []).length === 0 ? (
            <p className="text-xs text-muted-foreground py-0.5">No output file name for this step.</p>
          ) : (
            (activeSession?.outputFiles?.[selectedStepIndex] ?? []).map((fileName, i) => (
              <div key={i} className="font-mono text-xs text-ink-700 truncate rounded bg-ink-900/5 px-2 py-1" title={fileName}>
                {fileName}
              </div>
            ))
          )}
        </div>
      </div>
      {/* Verifier area (per-step criteria from workflow or user) */}
      <div className="flex-1 min-h-0 max-h-[260px] flex flex-col overflow-hidden border-t border-ink-900/10 pt-2">
        <div className="shrink-0 flex items-center gap-2 mb-2">
          <div className="h-3 w-0.5 rounded-full bg-ink-400" />
          <span className="text-xs font-semibold uppercase tracking-wide text-ink-500">
            Verifier {progressSteps.length > 0 ? `(Step ${selectedStepIndex + 1})` : ""}
          </span>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-2">
          {verificationCriteria.length === 0 && !editingIndex ? (
            <p className="text-xs text-muted-foreground py-1">No verifiers for this step. Add criteria to check output files and quality.</p>
          ) : null}
          {verificationCriteria.map((text, index) => {
            const mark = currentVerifierMarks[index];
            return (
              <div key={index} className="shrink-0 flex items-start gap-2">
                <div className="flex-1 min-w-0">
                  {editingIndex === index ? (
                    <div className="rounded-xl border border-primary/40 bg-surface p-1.5">
                      <textarea
                        ref={editingIndex === index ? (el) => { editInputRef.current = el; } : undefined}
                        className="w-full min-h-[52px] resize-none rounded-lg border border-ink-900/10 bg-white px-2.5 py-1.5 text-xs text-ink-800 placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30"
                        placeholder="Enter verification criterion..."
                        value={draftText}
                        onChange={(e) => setDraftText(e.target.value)}
                        onBlur={saveEdit}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            saveEdit();
                          }
                        }}
                      />
                    </div>
                  ) : (
                    <div
                      className="cursor-pointer rounded-xl border border-ink-900/10 bg-surface px-3 py-2 text-left text-xs text-ink-700 hover:bg-surface-tertiary hover:border-ink-900/20 transition-colors min-h-[38px] flex items-center"
                      onClick={() => startEditCriterion(index)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); startEditCriterion(index); } }}
                    >
                      <span className="line-clamp-2 break-words">{text || "Click to edit"}</span>
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleVerifierMark(index);
                  }}
                  className="shrink-0 flex items-center justify-center w-8 min-h-[38px] rounded-lg border border-ink-900/15 bg-surface text-ink-500 hover:bg-ink-900/10 hover:text-ink-700 focus:outline-none focus:ring-2 focus:ring-primary/30 transition-colors mt-0.5"
                  aria-label={mark === "check" ? "Mark as failed (cross)" : "Mark as passed (check)"}
                  title={mark === "check" ? "Mark as failed" : "Mark as passed"}
                >
                  {mark === "check" ? (
                    <svg viewBox="0 0 24 24" className="h-4 w-4 text-emerald-600" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M20 6L9 17l-5-5" />
                    </svg>
                  ) : mark === "cross" ? (
                    <svg viewBox="0 0 24 24" className="h-4 w-4 text-red-500" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M18 6L6 18M6 6l12 12" />
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10" />
                    </svg>
                  )}
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeCriterion(index);
                  }}
                  className="shrink-0 flex items-center justify-center w-8 min-h-[38px] rounded-lg border border-ink-900/15 bg-surface text-ink-500 hover:bg-red-50 hover:text-red-600 hover:border-red-200 focus:outline-none focus:ring-2 focus:ring-primary/30 transition-colors mt-0.5"
                  aria-label="Remove verifier"
                  title="Remove verifier"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    <line x1="10" y1="11" x2="10" y2="17" />
                    <line x1="14" y1="11" x2="14" y2="17" />
                  </svg>
                </button>
              </div>
            );
          })}
          <button
            type="button"
            className="flex shrink-0 items-center justify-center rounded-xl border border-dashed border-ink-900/20 bg-surface/50 py-2.5 text-muted-foreground hover:bg-surface hover:border-ink-900/30 hover:text-ink-600 transition-colors min-h-[38px] w-full"
            onClick={startAddCriterion}
            aria-label="Add verification criterion"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
        </div>
      </div>
      <Dialog.Root open={!!resumeSessionId} onOpenChange={(open) => !open && setResumeSessionId(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-ink-900/40 backdrop-blur-sm animate-fade-in" />
          <Dialog.Content className="fixed left-1/2 top-1/2 w-full max-w-xl -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-6 shadow-xl animate-scale-in">
            <div className="flex items-start justify-between gap-4">
              <Dialog.Title className="text-lg font-semibold text-ink-800">Resume</Dialog.Title>
              <Dialog.Close asChild>
                <button className="rounded-full p-1 text-ink-500 hover:bg-ink-900/10" aria-label="Close dialog">
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M6 6l12 12M18 6l-12 12" />
                  </svg>
                </button>
              </Dialog.Close>
            </div>
            <div className="mt-4 flex items-center gap-2 rounded-xl border border-ink-900/10 bg-surface px-3 py-2 font-mono text-xs text-ink-700">
              <span className="flex-1 break-all">{resumeSessionId ? `claude --resume ${resumeSessionId}` : ""}</span>
              <button className="rounded-lg p-1.5 text-ink-600 hover:bg-ink-900/10" onClick={handleCopyCommand} aria-label="Copy resume command">
                {copied ? (
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12l4 4L19 6" /></svg>
                ) : (
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V5a2 2 0 0 1 2-2h10" /></svg>
                )}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      <Dialog.Root open={!!deleteConfirmSessionId} onOpenChange={(open) => !open && setDeleteConfirmSessionId(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-ink-900/40 backdrop-blur-sm animate-fade-in" />
          <Dialog.Content className="fixed left-1/2 top-1/2 w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-6 shadow-xl animate-scale-in">
            <Dialog.Title className="text-lg font-semibold text-ink-800">Delete session?</Dialog.Title>
            <p className="mt-2 text-sm text-muted-foreground">This action cannot be undone. The session and its history will be permanently removed.</p>
            <div className="mt-5 flex gap-3">
              <Dialog.Close asChild>
                <button className="flex-1 rounded-xl border border-ink-900/10 bg-surface px-4 py-2.5 text-sm font-medium text-ink-700 hover:bg-surface-tertiary transition-colors">
                  Cancel
                </button>
              </Dialog.Close>
              <button
                className="flex-1 rounded-xl bg-error px-4 py-2.5 text-sm font-medium text-white hover:bg-error/90 transition-colors"
                onClick={() => {
                  if (deleteConfirmSessionId) {
                    onDeleteSession(deleteConfirmSessionId);
                    setDeleteConfirmSessionId(null);
                  }
                }}
              >
                Delete
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </aside>
  );
}
