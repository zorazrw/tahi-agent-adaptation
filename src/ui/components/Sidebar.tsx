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
  const updateSessionSteps = useAppStore((state) => state.updateSessionSteps);
  const updateSessionVerificationCriteria = useAppStore((state) => state.updateSessionVerificationCriteria);
  const updateSessionTitle = useAppStore((state) => state.updateSessionTitle);
  const [resumeSessionId, setResumeSessionId] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const closeTimerRef = useRef<number | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const titleInputRef = useRef<HTMLInputElement | null>(null);

  const activeSession = activeSessionId ? sessions[activeSessionId] : undefined;
  const DEFAULT_STEPS = ["Step 1", "Step 2", "Step 3", "Step 4"];
  const progressSteps = activeSession?.steps?.length ? activeSession.steps : DEFAULT_STEPS;

  const [verificationCriteriaByStep, setVerificationCriteriaByStep] = useState<string[][]>(() =>
    Array.from({ length: DEFAULT_STEPS.length }, () => [])
  );
  const [selectedStepIndex, setSelectedStepIndex] = useState(0);
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
    <aside className="fixed inset-y-0 left-0 flex h-full w-[280px] flex-col border-r border-ink-900/5 bg-[#FAF9F6] px-4 pb-4 pt-12">
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
            <div className="rounded-xl border border-ink-900/5 bg-surface px-4 py-5 text-center text-xs text-muted">
              No sessions yet. Click "+ New Task" to start.
            </div>
          ) : (
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <div
                  role="button"
                  tabIndex={0}
                  className="flex w-full cursor-pointer items-center gap-2 rounded-xl border border-ink-900/10 bg-surface px-3 py-3 text-left transition hover:bg-surface-tertiary hover:border-ink-900/20 focus:outline-none focus:ring-2 focus:ring-accent/30 data-[state=open]:border-accent/30 data-[state=open]:bg-accent-subtle"
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
                          className="w-full rounded border border-ink-900/20 bg-white px-1.5 py-0.5 text-[12px] font-medium text-ink-800 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/20"
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
                          <div className="mt-0.5 text-xs text-muted">
                            <span className="truncate">{formatCwd(sessions[activeSessionId].cwd)}</span>
                          </div>
                        </>
                      )
                    ) : (
                      <span className="text-xs text-muted">Select a task</span>
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
                      <span className="text-xs text-muted">{formatCwd(session.cwd)}</span>
                    </DropdownMenu.Item>
                  ))}
                  {activeSessionId && (
                    <>
                      <DropdownMenu.Separator className="my-1 h-px bg-ink-900/10" />
                      <DropdownMenu.Item className="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm text-ink-700 outline-none hover:bg-ink-900/5" onSelect={() => activeSessionId && onDeleteSession(activeSessionId)}>
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
        {/* Progress: connected dots multi-step plan (step labels editable) */}
        {sessionList.length > 0 && (
          <div className="shrink-0 border-t border-ink-900/10 pt-3">
            <div className="mb-2 text-sm font-semibold text-ink-600">Progress</div>
            <div className="flex flex-col">
              {progressSteps.map((label, i) => (
                <div key={i} className="flex items-center gap-3 w-full min-w-0">
                  <button
                    type="button"
                    className="flex flex-col items-center shrink-0 rounded-full p-1.5 -m-1.5 focus:outline-none focus:ring-2 focus:ring-accent/30 focus:ring-inset min-w-[32px] min-h-[32px] justify-center"
                    onClick={() => {
                      setSelectedStepIndex(i);
                      setEditingIndex(null);
                      if (activeSessionId) {
                        sendEvent({ type: "session.solveStep", payload: { sessionId: activeSessionId, stepIndex: i } });
                      }
                    }}
                    aria-label={`Run step ${i + 1}`}
                  >
                    <div
                      className={`h-4 w-4 rounded-full border-2 transition-colors flex-shrink-0 ${
                        activeSession?.completedStepIndices?.includes(i)
                          ? "step-circle-completed"
                          : selectedStepIndex === i
                            ? "border-accent bg-accent/20"
                            : "border-ink-900/30 bg-surface"
                      }`}
                    />
                    {i < progressSteps.length - 1 && (
                      <div className="w-px h-4 bg-ink-900/20 shrink-0 mt-0.5" />
                    )}
                  </button>
                  <div className="flex-1 min-w-0 py-0.5">
                    {editingStepIndex === i ? (
                      <input
                        ref={editingStepIndex === i ? (el) => { editingStepInputRef.current = el; } : undefined}
                        type="text"
                        className="w-full rounded border border-ink-900/20 bg-surface px-2 py-1 text-xs text-ink-800 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/20"
                        value={editingStepDraft}
                        onChange={(e) => setEditingStepDraft(e.target.value)}
                        onBlur={saveStepLabelEdit}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            saveStepLabelEdit();
                          }
                          if (e.key === "Escape") {
                            setEditingStepIndex(null);
                            setEditingStepDraft("");
                          }
                        }}
                      />
                    ) : (
                      <div className="flex items-center gap-0.5 min-w-0 group">
                        <button
                          type="button"
                          className={`flex-1 text-left text-xs truncate rounded px-1 -mx-1 py-0.5 transition-colors hover:bg-ink-900/5 ${selectedStepIndex === i ? "font-medium text-ink-800" : "text-ink-700"}`}
                          onClick={() => {
                            setSelectedStepIndex(i);
                            setEditingIndex(null);
                          }}
                        >
                          {label}
                        </button>
                        <button
                          type="button"
                          className="shrink-0 rounded p-0.5 text-ink-400 hover:text-ink-600 hover:bg-ink-900/10 opacity-0 group-hover:opacity-100 focus:opacity-100 focus:outline-none focus:ring-2 focus:ring-accent/30"
                          onClick={(e) => {
                            e.stopPropagation();
                            startEditStepLabel(i);
                          }}
                          aria-label="Edit step label"
                        >
                          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                          </svg>
                        </button>
                        <button
                          type="button"
                          className="shrink-0 rounded p-0.5 text-ink-400 hover:text-error hover:bg-ink-900/10 opacity-0 group-hover:opacity-100 focus:opacity-100 focus:outline-none focus:ring-2 focus:ring-accent/30"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteStep(i);
                          }}
                          aria-label="Delete step"
                        >
                          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M5 12h14" />
                          </svg>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div className="flex items-center gap-3 w-full min-w-0">
                <div className="shrink-0 w-8 h-4" aria-hidden />
                <button
                  type="button"
                  className="flex-1 flex items-center gap-1.5 rounded-lg border border-dashed border-ink-900/20 px-2 py-1.5 mt-0.5 text-xs text-muted hover:border-ink-900/30 hover:text-ink-600 hover:bg-ink-900/5 transition-colors"
                  onClick={addStep}
                  aria-label="Add step"
                >
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 shrink-0" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 5v14M5 12h14" />
                  </svg>
                  <span>Add step</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
      {/* Files: expected output file name(s) for the current step */}
      <div className="shrink-0 flex flex-col border-t border-ink-900/10 pt-2 pb-2">
        <div className="shrink-0 text-sm font-semibold text-ink-600 mb-1.5">
          Files {progressSteps.length > 0 ? `(Step ${selectedStepIndex + 1})` : ""}
        </div>
        <div className="min-h-0 overflow-y-auto max-h-[88px] flex flex-col gap-1">
          {(activeSession?.outputFiles?.[selectedStepIndex] ?? []).length === 0 ? (
            <p className="text-xs text-muted py-0.5">No output file name for this step.</p>
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
        <div className="shrink-0 text-sm font-semibold text-ink-600 mb-2">
          Verifier {progressSteps.length > 0 ? `(Step ${selectedStepIndex + 1})` : ""}
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-2">
          {verificationCriteria.length === 0 && !editingIndex ? (
            <p className="text-xs text-muted py-1">No verifiers for this step. Add criteria to check output files and quality.</p>
          ) : null}
          {verificationCriteria.map((text, index) => (
            <div key={index} className="shrink-0">
              {editingIndex === index ? (
                <div className="rounded-xl border border-accent/40 bg-surface p-1.5">
                  <textarea
                    ref={editingIndex === index ? (el) => { editInputRef.current = el; } : undefined}
                    className="w-full min-h-[52px] resize-none rounded-lg border border-ink-900/10 bg-white px-2.5 py-1.5 text-xs text-ink-800 placeholder:text-muted focus:outline-none focus:ring-2 focus:ring-accent/30"
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
          ))}
          <button
            type="button"
            className="flex shrink-0 items-center justify-center rounded-xl border border-dashed border-ink-900/20 bg-surface/50 py-2.5 text-muted hover:bg-surface hover:border-ink-900/30 hover:text-ink-600 transition-colors min-h-[38px] w-full"
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
          <Dialog.Overlay className="fixed inset-0 bg-ink-900/40 backdrop-blur-sm" />
          <Dialog.Content className="fixed left-1/2 top-1/2 w-full max-w-xl -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-6 shadow-xl">
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
    </aside>
  );
}
