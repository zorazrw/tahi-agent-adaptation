import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { Combobox, ComboboxContent, ComboboxEmpty, ComboboxInput, ComboboxItem, ComboboxList } from "@/ui/components/ui/combobox";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/ui/components/ui/tooltip";
import type { ClientEvent, WorkflowNode, VerifierMark } from "../types";
import { useAppStore } from "../store/useAppStore";

interface SidebarProps {
  connected: boolean;
  sendEvent: (event: ClientEvent) => void;
  onNewSession: () => void;
  onDeleteSession: (sessionId: string) => void;
}

function getMaxDepth(tree: WorkflowNode[]): number {
  let max = 0;
  for (const node of tree) {
    max = Math.max(max, node.depth);
    if (node.children.length > 0) max = Math.max(max, getMaxDepth(node.children));
  }
  return max;
}

function findNode(tree: WorkflowNode[], id: string): WorkflowNode | undefined {
  for (const node of tree) {
    if (node.id === id) return node;
    const found = findNode(node.children, id);
    if (found) return found;
  }
  return undefined;
}

function depthLabel(depth: number): string {
  return `Level ${depth}`;
}

// ─── Compact Tree Node ────────────────────────────────────────────────

function TreeNode({
  node,
  selectedNodeId,
  runningNodeId,
  collapsedNodeIds,
  effectiveDepth,
  editingNodeId,
  editingNodeDraft,
  editingNodeInputRef,
  onSelectNode,
  onToggleCollapse,
  onEditNode,
  onDeleteNode,
  onEditDraftChange,
  onEditSave,
  onEditCancel,
}: {
  node: WorkflowNode;
  selectedNodeId: string | null;
  runningNodeId: string | null;
  collapsedNodeIds: Set<string>;
  effectiveDepth: number;
  editingNodeId: string | null;
  editingNodeDraft: string;
  editingNodeInputRef: React.RefObject<HTMLInputElement | null>;
  onSelectNode: (id: string) => void;
  onToggleCollapse: (id: string) => void;
  onEditNode: (id: string) => void;
  onDeleteNode: (id: string) => void;
  onEditDraftChange: (val: string) => void;
  onEditSave: () => void;
  onEditCancel: () => void;
}) {
  const isSelected = selectedNodeId === node.id;
  const isRunning = runningNodeId === node.id && node.status === "running";
  const isCompleted = node.status === "completed";
  const isCollapsed = collapsedNodeIds.has(node.id);
  const hasChildren = node.children.length > 0;
  const isEditing = editingNodeId === node.id;
  const isHighlighted = node.depth === effectiveDepth;

  return (
    <div style={{ paddingLeft: node.depth > 0 ? 20 : 0 }}>
      {/* Node row */}
      <div
        className={`group flex items-start gap-1 py-[3px] px-1 rounded-md cursor-pointer select-none transition-colors ${isSelected ? "bg-primary/8" : "hover:bg-ink-900/4"}`}
        style={isHighlighted ? { borderLeft: '2px solid rgba(217, 119, 87, 0.6)', paddingLeft: 2, backgroundColor: isSelected ? undefined : 'rgba(217, 119, 87, 0.04)' } : undefined}
        onClick={() => onSelectNode(node.id)}
      >
        {/* Chevron for collapsible */}
        {hasChildren ? (
          <button
            type="button"
            className="shrink-0 mt-[2px] p-0 text-ink-400 hover:text-ink-600"
            onClick={(e) => { e.stopPropagation(); onToggleCollapse(node.id); }}
          >
            <svg viewBox="0 0 16 16" className={`h-3.5 w-3.5 transition-transform ${isCollapsed ? "" : "rotate-90"}`} fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M6 3l5 5-5 5" />
            </svg>
          </button>
        ) : (
          <span className="shrink-0 w-3.5 mt-[2px]" />
        )}

        {/* Status circle */}
        <span className="shrink-0 mt-[5px]">
          {isRunning ? (
            <span className="block h-2.5 w-2.5 rounded-full border-[1.5px] border-primary border-t-transparent animate-spin" />
          ) : (
            <span className={`block h-2.5 w-2.5 rounded-full border-[1.5px] ${
              isCompleted
                ? "border-emerald-500 bg-emerald-500"
                : node.status === "error"
                  ? "border-error bg-error/30"
                  : isSelected
                    ? "border-primary bg-primary/25"
                    : "border-ink-900/25 bg-transparent"
            }`} />
          )}
        </span>

        {/* Label */}
        {isEditing ? (
          <input
            ref={editingNodeInputRef}
            type="text"
            className="flex-1 min-w-0 rounded border border-ink-900/20 bg-white px-1.5 py-0.5 text-[12px] text-ink-800 focus:border-primary focus:outline-none"
            value={editingNodeDraft}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => onEditDraftChange(e.target.value)}
            onBlur={onEditSave}
            onKeyDown={(e) => {
              if (e.key === "Enter") { e.preventDefault(); onEditSave(); }
              if (e.key === "Escape") onEditCancel();
            }}
          />
        ) : (
          <span className={`flex-1 min-w-0 text-[12px] leading-[18px] break-words ${isSelected ? "font-medium text-ink-800" : "text-ink-600"}`}>
            {node.description || <span className="italic text-ink-400">Untitled</span>}
          </span>
        )}

        {/* Hover actions */}
        {!isEditing && (
          <span className="shrink-0 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              type="button"
              className="p-0.5 rounded text-ink-300 hover:text-ink-600"
              onClick={(e) => { e.stopPropagation(); onEditNode(node.id); }}
            >
              <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M11.5 1.5l3 3L5 14l-3.5.5L2 11l9.5-9.5z" /></svg>
            </button>
            <button
              type="button"
              className="p-0.5 rounded text-ink-300 hover:text-error"
              onClick={(e) => { e.stopPropagation(); onDeleteNode(node.id); }}
            >
              <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 4L4 12M4 4l8 8" /></svg>
            </button>
          </span>
        )}
      </div>

      {/* Children */}
      {hasChildren && !isCollapsed && node.children.map((child) => (
        <TreeNode
          key={child.id}
          node={child}
          selectedNodeId={selectedNodeId}
          runningNodeId={runningNodeId}
          collapsedNodeIds={collapsedNodeIds}
          effectiveDepth={effectiveDepth}
          editingNodeId={editingNodeId}
          editingNodeDraft={editingNodeDraft}
          editingNodeInputRef={editingNodeInputRef}
          onSelectNode={onSelectNode}
          onToggleCollapse={onToggleCollapse}
          onEditNode={onEditNode}
          onDeleteNode={onDeleteNode}
          onEditDraftChange={onEditDraftChange}
          onEditSave={onEditSave}
          onEditCancel={onEditCancel}
        />
      ))}
    </div>
  );
}

// ─── Granularity Slider ───────────────────────────────────────────────

function GranularitySlider({
  maxDepth, verificationDepth, highlightDepth,
  onHighlightChange, onDepthCommit,
}: {
  maxDepth: number; verificationDepth: number; highlightDepth: number | null;
  onHighlightChange: (depth: number | null) => void;
  onDepthCommit: (depth: number) => void;
}) {
  if (maxDepth <= 0) return null;
  return (
    <div className="flex flex-col gap-0.5 px-0.5">
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>Verify at:</span>
        <span className="font-medium text-ink-600">{depthLabel(highlightDepth ?? verificationDepth)}</span>
      </div>
      <input
        type="range" min={0} max={maxDepth}
        value={highlightDepth ?? verificationDepth}
        className="w-full h-1 accent-primary cursor-pointer"
        onInput={(e) => onHighlightChange(Number((e.target as HTMLInputElement).value))}
        onChange={(e) => { onHighlightChange(null); onDepthCommit(Number((e.target as HTMLInputElement).value)); }}
        onMouseLeave={() => onHighlightChange(null)}
      />
      <div className="flex justify-between text-[9px] text-muted-foreground">
        {Array.from({ length: maxDepth + 1 }, (_, i) => <span key={i}>{depthLabel(i)}</span>)}
      </div>
    </div>
  );
}

// ─── Main Sidebar ─────────────────────────────────────────────────────

export function Sidebar({ sendEvent, onNewSession, onDeleteSession }: SidebarProps) {
  const sessions = useAppStore((s) => s.sessions);
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const setActiveSessionId = useAppStore((s) => s.setActiveSessionId);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useAppStore((s) => s.setSelectedNodeId);
  const runningNodeId = useAppStore((s) => s.runningNodeId);
  const setRunningNodeId = useAppStore((s) => s.setRunningNodeId);
  const collapsedNodeIds = useAppStore((s) => s.collapsedNodeIds);
  const toggleNodeCollapsed = useAppStore((s) => s.toggleNodeCollapsed);
  const setCollapsedNodeIds = useAppStore((s) => s.setCollapsedNodeIds);
  const highlightDepth = useAppStore((s) => s.highlightDepth);
  const setHighlightDepth = useAppStore((s) => s.setHighlightDepth);
  const updateWorkflowTree = useAppStore((s) => s.updateWorkflowTree);
  const updateVerificationDepth = useAppStore((s) => s.updateVerificationDepth);
  const updateSessionTitle = useAppStore((s) => s.updateSessionTitle);

  const [deleteConfirmSessionId, setDeleteConfirmSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const titleInputRef = useRef<HTMLInputElement | null>(null);
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
  const [editingNodeDraft, setEditingNodeDraft] = useState("");
  const editingNodeInputRef = useRef<HTMLInputElement | null>(null);

  const activeSession = activeSessionId ? sessions[activeSessionId] : undefined;
  const workflowTree = activeSession?.workflowTree ?? [];
  const verificationDepth = activeSession?.verificationDepth ?? 0;
  const maxDepth = useMemo(() => getMaxDepth(workflowTree), [workflowTree]);
  const selectedNode = useMemo(() => selectedNodeId ? findNode(workflowTree, selectedNodeId) : undefined, [workflowTree, selectedNodeId]);
  const effectiveDepth = highlightDepth ?? verificationDepth;

  // Auto-collapse nodes below verification depth when depth changes
  const prevDepthRef = useRef(verificationDepth);
  useEffect(() => {
    if (workflowTree.length === 0) return;
    // Only reset collapsed state when verificationDepth actually changes
    if (prevDepthRef.current === verificationDepth) return;
    prevDepthRef.current = verificationDepth;
    const toCollapse = new Set<string>();
    function walk(nodes: WorkflowNode[]) {
      for (const node of nodes) {
        if (node.children.length > 0 && node.depth >= verificationDepth) {
          toCollapse.add(node.id);
        }
        walk(node.children);
      }
    }
    walk(workflowTree);
    setCollapsedNodeIds(toCollapse);
  }, [verificationDepth, workflowTree]);

  // Auto-collapse on initial tree load
  const initialCollapseRef = useRef(false);
  useEffect(() => {
    if (workflowTree.length === 0 || initialCollapseRef.current) return;
    initialCollapseRef.current = true;
    const toCollapse = new Set<string>();
    function walk(nodes: WorkflowNode[]) {
      for (const node of nodes) {
        if (node.children.length > 0 && node.depth >= verificationDepth) {
          toCollapse.add(node.id);
        }
        walk(node.children);
      }
    }
    walk(workflowTree);
    setCollapsedNodeIds(toCollapse);
  }, [workflowTree]);

  // Auto-select first node at effective depth if nothing selected
  useEffect(() => {
    if (selectedNodeId || workflowTree.length === 0) return;
    function findFirstAtDepth(nodes: WorkflowNode[], target: number): WorkflowNode | undefined {
      for (const node of nodes) {
        if (node.depth === target) return node;
        const found = findFirstAtDepth(node.children, target);
        if (found) return found;
      }
      return undefined;
    }
    const first = findFirstAtDepth(workflowTree, verificationDepth);
    if (first) setSelectedNodeId(first.id);
  }, [workflowTree, selectedNodeId, verificationDepth]);

  useEffect(() => { if (editingNodeId) editingNodeInputRef.current?.focus(); }, [editingNodeId]);
  useEffect(() => { if (editingTitle) titleInputRef.current?.focus(); }, [editingTitle]);
  useEffect(() => { setEditingTitle(false); setTitleDraft(""); }, [activeSessionId]);

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
    setEditingTitle(false); setTitleDraft("");
  };

  const handleEditNode = useCallback((nodeId: string) => {
    const node = findNode(workflowTree, nodeId);
    if (!node) return;
    setEditingNodeId(nodeId); setEditingNodeDraft(node.description);
  }, [workflowTree]);

  const saveNodeEdit = () => {
    if (!editingNodeId || !activeSessionId) return;
    const trimmed = editingNodeDraft.trim();
    if (!trimmed) { setEditingNodeId(null); return; }
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const node = findNode(newTree, editingNodeId);
    if (node) node.description = trimmed;
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
    setEditingNodeId(null); setEditingNodeDraft("");
  };

  const handleDeleteNode = useCallback((nodeId: string) => {
    if (!activeSessionId) return;
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    function removeNode(nodes: WorkflowNode[]): WorkflowNode[] {
      return nodes.filter(n => { if (n.id === nodeId) return false; n.children = removeNode(n.children); return true; });
    }
    const pruned = removeNode(newTree);
    updateWorkflowTree(activeSessionId, pruned);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: pruned } });
    if (selectedNodeId === nodeId) setSelectedNodeId(null);
  }, [activeSessionId, workflowTree, selectedNodeId]);

  const handleDepthCommit = useCallback((depth: number) => {
    if (!activeSessionId) return;
    updateVerificationDepth(activeSessionId, depth);
    sendEvent({ type: "session.updateVerificationDepth", payload: { sessionId: activeSessionId, verificationDepth: depth } });
    // Auto-select first node at the new depth
    function findFirstAtDepth(nodes: WorkflowNode[], target: number): WorkflowNode | undefined {
      for (const node of nodes) {
        if (node.depth === target) return node;
        const found = findFirstAtDepth(node.children, target);
        if (found) return found;
      }
      return undefined;
    }
    const first = findFirstAtDepth(workflowTree, depth);
    if (first) setSelectedNodeId(first.id);
  }, [activeSessionId, workflowTree]);

  const currentVerifiers = selectedNode?.verifiers ?? [];
  const currentVerifierMarks = selectedNode?.verifierMarks ?? [];

  const toggleVerifierMark = (index: number) => {
    if (!activeSessionId || !selectedNodeId) return;
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const node = findNode(newTree, selectedNodeId);
    if (!node) return;
    while (node.verifierMarks.length <= index) node.verifierMarks.push(undefined);
    const cur = node.verifierMarks[index];
    node.verifierMarks[index] = cur === undefined ? "check" : cur === "check" ? "cross" : undefined;
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
  };

  const formatCwd = (cwd?: string) => {
    if (!cwd) return "Working dir unavailable";
    const parts = cwd.split(/[\\/]+/).filter(Boolean);
    return `/${parts.slice(-2).join("/") || cwd}`;
  };

  const sessionList = useMemo(() => {
    const list = Object.values(sessions);
    list.sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0));
    return list;
  }, [sessions]);

  return (
    <aside className="fixed inset-y-0 left-0 flex h-full w-[var(--sidebar-width)] flex-col border-r border-ink-900/5 bg-[#FAF9F6] px-3 pb-3 pt-12">
      <div className="absolute top-0 left-0 right-0 h-12" style={{ WebkitAppRegion: 'drag' } as React.CSSProperties} />

      {/* New Task + Settings */}
      <div className="flex shrink-0 gap-2 mt-4">
        <button className="flex-1 rounded-xl border border-ink-900/10 bg-surface px-4 py-2.5 text-sm font-medium text-ink-700 hover:bg-surface-tertiary hover:border-ink-900/20 transition-colors" onClick={onNewSession}>+ New Task</button>
        <button className="rounded-xl border border-ink-900/10 bg-surface px-4 py-3 text-sm text-ink-700 hover:bg-surface-tertiary hover:border-ink-900/20 transition-colors" onClick={() => useAppStore.getState().setShowSettingsModal(true)} aria-label="Settings">
          <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.08a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.08a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
        </button>
      </div>

      {/* Session picker + meta */}
      <div className="shrink-0 py-2">
        {sessionList.length === 0 ? (
          <div className="rounded-xl border border-ink-900/5 bg-surface px-4 py-5 text-center text-xs text-muted-foreground">No sessions yet. Click "+ New Task" to start.</div>
        ) : (
          <div className="space-y-1">
            <Combobox items={sessionList} value={activeSessionId ? sessionList.find((s) => s.id === activeSessionId) ?? null : null} onValueChange={(session) => { if (session) setActiveSessionId(session.id); }} itemToStringLabel={(s) => s.title} itemToStringValue={(s) => s.title}>
              <ComboboxInput placeholder="Search tasks..." className="w-full" />
              <ComboboxContent>
                <ComboboxEmpty>No tasks found.</ComboboxEmpty>
                <ComboboxList>
                  {(session) => (
                    <ComboboxItem key={session.id} value={session}>
                      <span className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${session.status === "running" ? "bg-info" : session.status === "completed" ? "bg-success" : session.status === "error" ? "bg-error" : "bg-ink-300"}`} />
                      <span className="truncate">{session.title}</span>
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
            {activeSessionId && sessions[activeSessionId] && (
              <div className="flex items-center gap-1 px-0.5">
                <span className="flex-1 min-w-0 truncate text-[11px] text-muted-foreground">{formatCwd(sessions[activeSessionId].cwd)}</span>
                <TooltipProvider delayDuration={300}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button className="shrink-0 rounded p-1 text-ink-400 hover:text-ink-700 hover:bg-ink-900/5 transition-colors" onClick={startEditTitle} aria-label="Edit title">
                        <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" /><path d="m15 5 4 4" /></svg>
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="right">Edit</TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button className="shrink-0 rounded p-1 text-ink-400 hover:text-error hover:bg-ink-900/5 transition-colors" onClick={() => setDeleteConfirmSessionId(activeSessionId)} aria-label="Delete session">
                        <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 7h16" /><path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" /><path d="M7 7l1 12a1 1 0 0 0 1 .9h6a1 1 0 0 0 1-.9l1-12" /></svg>
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="right">Delete</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            )}
            {editingTitle && activeSessionId && (
              <div className="px-0.5">
                <input ref={titleInputRef} type="text" className="w-full rounded-md border border-ink-900/20 bg-white px-2 py-1 text-sm text-ink-800 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20" value={titleDraft} onChange={(e) => setTitleDraft(e.target.value)} onBlur={saveTitle} onKeyDown={(e) => { if (e.key === "Enter") saveTitle(); if (e.key === "Escape") { setEditingTitle(false); setTitleDraft(""); } }} />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Progress: tree + slider + run button — takes all remaining space */}
      {sessionList.length > 0 && (
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden border-t border-ink-900/10 pt-2">
          <div className="flex items-center gap-2 mb-1.5 shrink-0">
            <div className="h-3 w-0.5 rounded-full bg-primary" />
            <span className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">Progress</span>
          </div>

          {workflowTree.length > 0 && maxDepth > 0 && (
            <div className="shrink-0 mb-1.5">
              <GranularitySlider maxDepth={maxDepth} verificationDepth={verificationDepth} highlightDepth={highlightDepth} onHighlightChange={setHighlightDepth} onDepthCommit={handleDepthCommit} />
            </div>
          )}

          {workflowTree.length === 0 ? (
            <p className="text-xs text-muted-foreground py-1">No workflow yet. Send a message to generate the plan.</p>
          ) : (
            <div className="flex-1 min-h-0 overflow-y-auto">
              {workflowTree.map((root) => (
                <TreeNode
                  key={root.id}
                  node={root}
                  selectedNodeId={selectedNodeId}
                  runningNodeId={runningNodeId}
                  collapsedNodeIds={collapsedNodeIds}
                  effectiveDepth={effectiveDepth}
                  editingNodeId={editingNodeId}
                  editingNodeDraft={editingNodeDraft}
                  editingNodeInputRef={editingNodeInputRef}
                  onSelectNode={setSelectedNodeId}
                  onToggleCollapse={toggleNodeCollapsed}
                  onEditNode={handleEditNode}
                  onDeleteNode={handleDeleteNode}
                  onEditDraftChange={setEditingNodeDraft}
                  onEditSave={saveNodeEdit}
                  onEditCancel={() => { setEditingNodeId(null); setEditingNodeDraft(""); }}
                />
              ))}
            </div>
          )}

          {/* Run button */}
          {workflowTree.length > 0 && activeSessionId && selectedNode && (() => {
            const isNodeRunning = runningNodeId === selectedNodeId && activeSession?.status === "running";
            const isNodeCompleted = selectedNode.status === "completed";
            if (isNodeRunning) {
              return (
                <div className="shrink-0 mt-1.5 flex items-center justify-center gap-1.5 rounded-lg bg-ink-900/8 px-3 py-2 text-xs font-medium text-muted-foreground">
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 animate-spin" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="9" strokeOpacity="0.25" /><path d="M12 3a9 9 0 0 1 9 9" strokeLinecap="round" /></svg>
                  Running...
                </div>
              );
            }
            return (
              <button
                type="button"
                className="shrink-0 mt-1.5 flex items-center justify-center gap-1.5 rounded-lg bg-primary px-3 py-2 text-xs font-medium text-white hover:bg-primary-hover transition-colors shadow-soft w-full"
                onClick={() => { setRunningNodeId(selectedNodeId); sendEvent({ type: "session.solveNode", payload: { sessionId: activeSessionId, nodeId: selectedNodeId! } }); }}
              >
                {isNodeCompleted ? (
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" /></svg>
                ) : (
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
                )}
                {isNodeCompleted ? "Rerun" : "Run"}
              </button>
            );
          })()}
        </div>
      )}

      {/* Files (compact) */}
      {selectedNode && (selectedNode.outputFiles.length > 0) && (
        <div className="shrink-0 border-t border-ink-900/10 pt-1.5 pb-1">
          <div className="flex items-center gap-1.5 mb-1">
            <div className="h-2.5 w-0.5 rounded-full bg-ink-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wide text-ink-500">Files</span>
          </div>
          <div className="flex flex-col gap-0.5 max-h-[60px] overflow-y-auto">
            {selectedNode.outputFiles.map((f, i) => (
              <span key={i} className="font-mono text-[11px] text-ink-600 rounded bg-ink-900/5 px-1.5 py-0.5 break-all">{f}</span>
            ))}
          </div>
        </div>
      )}

      {/* Verifiers (compact) */}
      {selectedNode && currentVerifiers.length > 0 && (
        <div className="shrink-0 max-h-[180px] flex flex-col overflow-hidden border-t border-ink-900/10 pt-1.5">
          <div className="flex items-center gap-1.5 mb-1 shrink-0">
            <div className="h-2.5 w-0.5 rounded-full bg-ink-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wide text-ink-500">Verifiers</span>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-1">
            {currentVerifiers.map((text, index) => {
              const mark = currentVerifierMarks[index];
              return (
                <div key={index} className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => toggleVerifierMark(index)}
                    className="shrink-0 flex items-center justify-center w-5 h-5 rounded border border-ink-900/15 bg-surface text-ink-400 hover:bg-ink-900/8 transition-colors"
                  >
                    {mark === "check" ? (
                      <svg viewBox="0 0 16 16" className="h-3 w-3 text-emerald-600" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 4L6 11 3 8" /></svg>
                    ) : mark === "cross" ? (
                      <svg viewBox="0 0 16 16" className="h-3 w-3 text-red-500" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 4L4 12M4 4l8 8" /></svg>
                    ) : (
                      <span className="block h-1.5 w-1.5 rounded-full bg-ink-900/15" />
                    )}
                  </button>
                  <span className="text-[11px] text-ink-600 leading-tight break-words min-w-0">{text}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Delete confirmation dialog */}
      <Dialog.Root open={!!deleteConfirmSessionId} onOpenChange={(open) => !open && setDeleteConfirmSessionId(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-ink-900/40 backdrop-blur-sm animate-fade-in" />
          <Dialog.Content className="fixed left-1/2 top-1/2 w-full max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-6 shadow-xl animate-scale-in">
            <Dialog.Title className="text-lg font-semibold text-ink-800">Delete session?</Dialog.Title>
            <p className="mt-2 text-sm text-muted-foreground">This action cannot be undone. The session and its history will be permanently removed.</p>
            <div className="mt-5 flex gap-3">
              <Dialog.Close asChild>
                <button className="flex-1 rounded-xl border border-ink-900/10 bg-surface px-4 py-2.5 text-sm font-medium text-ink-700 hover:bg-surface-tertiary transition-colors">Cancel</button>
              </Dialog.Close>
              <button className="flex-1 rounded-xl bg-error px-4 py-2.5 text-sm font-medium text-white hover:bg-error/90 transition-colors" onClick={() => { if (deleteConfirmSessionId) { onDeleteSession(deleteConfirmSessionId); setDeleteConfirmSessionId(null); } }}>Delete</button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </aside>
  );
}
