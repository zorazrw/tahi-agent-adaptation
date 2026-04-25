import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { Combobox, ComboboxContent, ComboboxEmpty, ComboboxInput, ComboboxItem, ComboboxList } from "@/ui/components/ui/combobox";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/ui/components/ui/tooltip";
import type { ClientEvent, WorkflowNode } from "../types";
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

function findFirstAtDepth(tree: WorkflowNode[], target: number): WorkflowNode | undefined {
  for (const node of tree) {
    if (node.depth === target) return node;
    const found = findFirstAtDepth(node.children, target);
    if (found) return found;
  }
  return undefined;
}

/** Prefer a node at `target` depth; if the tree has no such nodes (e.g. detail mode on a flat plan), use the first root. */
function findFirstAtDepthOrFallback(tree: WorkflowNode[], target: number): WorkflowNode | undefined {
  return findFirstAtDepth(tree, target) ?? findFirstAtDepth(tree, 0);
}

function findNode(tree: WorkflowNode[], id: string): WorkflowNode | undefined {
  for (const node of tree) {
    if (node.id === id) return node;
    const found = findNode(node.children, id);
    if (found) return found;
  }
  return undefined;
}

function createEmptyNode(depth: number): WorkflowNode {
  return {
    id: crypto.randomUUID(),
    description: "",
    outputFiles: [],
    verifiers: [],
    verifierMarks: [],
    children: [],
    status: "pending",
    depth,
  };
}

/** Basename only, aligned with plan output file rules (no directories in stored names). */
function normalizeOutputFileName(raw: string): string | null {
  const s = String(raw ?? "").trim();
  if (!s) return null;
  const norm = s.replace(/\\/g, "/").split("/").pop()?.trim() ?? "";
  if (!norm) return null;
  return norm;
}

/** Whether nodeId is the same as or a descendant of ancestorId. */
function isDescendant(tree: WorkflowNode[], nodeId: string, ancestorId: string): boolean {
  if (nodeId === ancestorId) return true;
  for (const node of tree) {
    if (node.id === ancestorId) {
      return findNode(node.children, nodeId) != null;
    }
    if (isDescendant(node.children, nodeId, ancestorId)) return true;
  }
  return false;
}

/** Remove node by id from tree (mutates). Returns the removed node or null. */
function removeNodeFromTree(nodes: WorkflowNode[], nodeId: string): WorkflowNode | null {
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i].id === nodeId) {
      const [removed] = nodes.splice(i, 1);
      return removed ?? null;
    }
    const found = removeNodeFromTree(nodes[i].children, nodeId);
    if (found) return found;
  }
  return null;
}

/** Set depth on node and all descendants. */
function setDepthRecursive(node: WorkflowNode, depth: number): void {
  node.depth = depth;
  for (const child of node.children) setDepthRecursive(child, depth + 1);
}

/** Insert node into tree: under parentId as last child, or at root before index. Mutates tree. */
function insertNodeIntoTree(
  roots: WorkflowNode[],
  node: WorkflowNode,
  parentId: string | null,
  siblingIndex: number
): void {
  if (parentId === null) {
    setDepthRecursive(node, 0);
    roots.splice(siblingIndex, 0, node);
    return;
  }
  const parent = findNode(roots, parentId);
  if (!parent) return;
  setDepthRecursive(node, parent.depth + 1);
  parent.children.splice(siblingIndex, 0, node);
}

type ParentAndIndex = { parentId: string | null; siblings: WorkflowNode[]; index: number } | null;

/** Find parent of node and its index among siblings. Roots have parentId null. */
function findParentAndIndex(roots: WorkflowNode[], nodeId: string): ParentAndIndex {
  for (let i = 0; i < roots.length; i++) {
    if (roots[i].id === nodeId) return { parentId: null, siblings: roots, index: i };
  }
  function walk(nodes: WorkflowNode[], parentId: string): ParentAndIndex {
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].id === nodeId) return { parentId, siblings: nodes, index: i };
      const found = walk(nodes[i].children, nodes[i].id);
      if (found) return found;
    }
    return null;
  }
  for (const root of roots) {
    const result = walk(root.children, root.id);
    if (result) return result;
  }
  return null;
}

// ─── Compact Tree Node ────────────────────────────────────────────────

function TreeNode({
  node,
  selectedNodeId,
  runningNodeId,
  collapsedNodeIds,
  effectiveDepth,
  verificationDepth,
  editingNodeId,
  editingNodeDraft,
  editingNodeInputRef,
  draggedNodeId,
  dropTargetNodeId,
  onSelectNode,
  onToggleCollapse,
  onEditNode,
  onDeleteNode,
  onAddChild,
  onDragStart,
  onDragOver,
  onDragLeave,
  onDrop,
  onDragEnd,
  onEditDraftChange,
  onEditSave,
  onEditCancel,
}: {
  node: WorkflowNode;
  selectedNodeId: string | null;
  runningNodeId: string | null;
  collapsedNodeIds: Set<string>;
  effectiveDepth: number;
  /** Session verification depth; roots always use Auto-style row chrome. */
  verificationDepth: number;
  editingNodeId: string | null;
  editingNodeDraft: string;
  editingNodeInputRef: React.RefObject<HTMLInputElement | null>;
  draggedNodeId: string | null;
  dropTargetNodeId: string | null;
  onSelectNode: (id: string) => void;
  onToggleCollapse: (id: string) => void;
  onEditNode: (id: string) => void;
  onDeleteNode: (id: string) => void;
  onAddChild?: (parentId: string) => void;
  onDragStart?: (nodeId: string) => void;
  onDragOver?: (nodeId: string) => void;
  onDragLeave?: () => void;
  onDrop?: (targetNodeId: string) => void;
  onDragEnd?: () => void;
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
  const isDragging = draggedNodeId === node.id;
  const isDropTarget = dropTargetNodeId === node.id;
  /** Root steps match Auto chrome; nested steps in Detail use expand chevron + inset padding. */
  const autoLikeRow = verificationDepth === 0 || node.depth === 0;
  const showChevronColumn = verificationDepth > 0 && node.depth > 0;
  const depthIndent = node.depth > 0 ? (verificationDepth > 0 ? 20 : 12) : 0;
  const autoRowPad =
    autoLikeRow && isHighlighted && !isSelected
      ? "gap-2 pl-2 pr-0.5"
      : autoLikeRow
        ? "gap-2 pl-1.5 pr-0.5"
        : "";

  return (
    <div style={{ paddingLeft: depthIndent }}>
      {/* Node row */}
      <div
        className={`group flex items-center py-1 rounded-md cursor-pointer select-none transition-colors ${autoLikeRow ? autoRowPad : "gap-2 px-1"} ${isSelected ? "bg-ink-900/6" : "hover:bg-ink-900/4"} ${isDragging ? "opacity-50" : ""} ${isDropTarget ? "ring-1 ring-ink-900/15 ring-inset" : ""} ${isHighlighted && !isSelected ? (autoLikeRow ? "border-l-2 border-ink-900/20" : "border-l-2 border-ink-900/20 -ml-px pl-[calc(0.25rem-1px)]") : ""}`}
        onClick={() => onSelectNode(node.id)}
        draggable={!isEditing}
        onDragStart={(e) => {
          if (isEditing) return;
          e.dataTransfer.setData("application/x-workflow-node-id", node.id);
          e.dataTransfer.effectAllowed = "move";
          onDragStart?.(node.id);
        }}
        onDragOver={(e) => {
          if (!draggedNodeId || draggedNodeId === node.id) return;
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          onDragOver?.(node.id);
        }}
        onDragLeave={() => onDragLeave?.()}
        onDrop={(e) => {
          e.preventDefault();
          const id = e.dataTransfer.getData("application/x-workflow-node-id");
          if (id && id !== node.id) onDrop?.(node.id);
        }}
        onDragEnd={() => onDragEnd?.()}
      >
        {/* Chevron only for nested steps in Detail; roots stay Auto-style */}
        {showChevronColumn && hasChildren ? (
          <button
            type="button"
            className="shrink-0 self-center p-0 text-ink-400 hover:text-ink-600"
            onClick={(e) => { e.stopPropagation(); onToggleCollapse(node.id); }}
          >
            <svg viewBox="0 0 16 16" className={`h-4 w-4 transition-transform ${isCollapsed ? "" : "rotate-90"}`} fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M6 3l5 5-5 5" />
            </svg>
          </button>
        ) : showChevronColumn ? (
          <span className="shrink-0 w-4 self-center" />
        ) : null}

        {/* Status circle */}
        <span className="shrink-0 self-center">
          {isRunning ? (
            <span className="block h-3 w-3 rounded-full border-[1.5px] border-primary border-t-transparent animate-spin" />
          ) : (
            <span className={`block h-3 w-3 rounded-full border-[1.5px] ${
              isCompleted
                ? "border-[#adc178] bg-[#adc178]"
                : node.status === "error"
                  ? "border-error bg-error/25"
                  : isSelected
                    ? "border-ink-500 bg-ink-900/10"
                    : "border-ink-900/25 bg-transparent"
            }`} />
          )}
        </span>

        {/* Label + hover actions overlayed to give label more width */}
        <div className="flex-1 min-w-0 relative pl-1.5">
          {isEditing ? (
            <input
              ref={editingNodeInputRef}
              type="text"
              className="w-full rounded border border-ink-900/20 bg-white px-1.5 py-0.5 text-sm text-ink-800 focus:border-primary focus:outline-none"
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
            <span className={`block pr-6 text-sm leading-snug break-words ${isSelected ? "font-medium text-ink-800" : "text-ink-600"}`}>
              {node.description || <span className="italic text-ink-400">Untitled</span>}
            </span>
          )}

          {!isEditing && (
            <span className="absolute right-0 bottom-0 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                type="button"
                className="p-0.5 rounded text-ink-300 hover:text-ink-600"
                onClick={(e) => { e.stopPropagation(); onEditNode(node.id); }}
                aria-label="Edit"
              >
                <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M11.5 1.5l3 3L5 14l-3.5.5L2 11l9.5-9.5z" /></svg>
              </button>
              {onAddChild && (
                <TooltipProvider delayDuration={300}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        className="p-0.5 rounded text-ink-300 hover:text-primary"
                        onClick={(e) => { e.stopPropagation(); onAddChild(node.id); }}
                        aria-label="Add sub-step"
                      >
                        <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 3v10M3 8h10" /></svg>
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="right">Add sub-step</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
              <button
                type="button"
                className="p-0.5 rounded text-ink-300 hover:text-error"
                onClick={(e) => { e.stopPropagation(); onDeleteNode(node.id); }}
                aria-label="Remove"
              >
                <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 4L4 12M4 4l8 8" /></svg>
              </button>
            </span>
          )}
        </div>
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
          verificationDepth={verificationDepth}
          editingNodeId={editingNodeId}
          editingNodeDraft={editingNodeDraft}
          editingNodeInputRef={editingNodeInputRef}
          draggedNodeId={draggedNodeId}
          dropTargetNodeId={dropTargetNodeId}
          onSelectNode={onSelectNode}
          onToggleCollapse={onToggleCollapse}
          onEditNode={onEditNode}
          onDeleteNode={onDeleteNode}
          onAddChild={onAddChild}
          onDragStart={onDragStart}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onDragEnd={onDragEnd}
          onEditDraftChange={onEditDraftChange}
          onEditSave={onEditSave}
          onEditCancel={onEditCancel}
        />
      ))}
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
  const updateWorkflowTree = useAppStore((s) => s.updateWorkflowTree);
  const updateVerificationDepth = useAppStore((s) => s.updateVerificationDepth);
  const updateSessionTitle = useAppStore((s) => s.updateSessionTitle);
  const workflowRunMode = useAppStore((s) => s.workflowRunMode);
  const setWorkflowRunMode = useAppStore((s) => s.setWorkflowRunMode);
  const verifierCheckSessionId = useAppStore((s) => s.verifierCheckSessionId);
  const verifierCheckNodeId = useAppStore((s) => s.verifierCheckNodeId);

  const [deleteConfirmSessionId, setDeleteConfirmSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const titleInputRef = useRef<HTMLInputElement | null>(null);
  const [editingNodeId, setEditingNodeId] = useState<string | null>(null);
  const [editingNodeDraft, setEditingNodeDraft] = useState("");
  const editingNodeInputRef = useRef<HTMLInputElement | null>(null);
  const [draggedNodeId, setDraggedNodeId] = useState<string | null>(null);
  const [dropTargetNodeId, setDropTargetNodeId] = useState<string | null>(null);
  const [editingOutputFileIndex, setEditingOutputFileIndex] = useState<number | null>(null);
  const [editingOutputFileDraft, setEditingOutputFileDraft] = useState("");
  const editingOutputFileInputRef = useRef<HTMLInputElement | null>(null);
  const newOutputFileInputRef = useRef<HTMLInputElement | null>(null);
  const [newOutputFileDraft, setNewOutputFileDraft] = useState("");
  const [addingOutputFile, setAddingOutputFile] = useState(false);
  const skipNextOutputFileBlurSave = useRef(false);

  const activeSession = activeSessionId ? sessions[activeSessionId] : undefined;
  const workflowTree = activeSession?.workflowTree ?? [];
  const verificationDepth = activeSession?.verificationDepth ?? 0;
  const maxDepth = useMemo(() => getMaxDepth(workflowTree), [workflowTree]);
  const selectedNode = useMemo(() => selectedNodeId ? findNode(workflowTree, selectedNodeId) : undefined, [workflowTree, selectedNodeId]);
  /** When the plan is flat, treat highlight depth as coarse so the step list stays consistent in both modes. */
  const effectiveDepth = maxDepth === 0 ? 0 : verificationDepth;
  const detailTargetDepth = maxDepth > 0 ? maxDepth : 1;
  const isVerifierChecking = Boolean(
    activeSessionId &&
      verifierCheckSessionId === activeSessionId &&
      verifierCheckNodeId &&
      (verifierCheckNodeId === selectedNodeId || verifierCheckNodeId === runningNodeId)
  );

  /** Stale when session or plan shape changes; same string when only in-place edits keep the same root ids. */
  const workflowTreeIdentity = useMemo(() => {
    if (!activeSessionId || workflowTree.length === 0) return "";
    return `${activeSessionId}:${workflowTree.map((r) => r.id).join("/")}`;
  }, [activeSessionId, workflowTree]);

  const lastAppliedCollapseRef = useRef<{ identity: string; depth: number } | null>(null);

  // Collapse branches that should be hidden at the current verification depth (coarse = collapse all nested parents).
  useEffect(() => {
    if (!workflowTreeIdentity || workflowTree.length === 0) return;

    const prev = lastAppliedCollapseRef.current;
    const identityChanged = !prev || prev.identity !== workflowTreeIdentity;
    const depthChanged = !prev || prev.depth !== verificationDepth;

    if (!identityChanged && !depthChanged) return;

    lastAppliedCollapseRef.current = { identity: workflowTreeIdentity, depth: verificationDepth };

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
  }, [workflowTreeIdentity, verificationDepth, workflowTree]);

  // Auto-select first node at effective depth if nothing selected
  useEffect(() => {
    if (selectedNodeId || workflowTree.length === 0) return;
    const first = findFirstAtDepthOrFallback(workflowTree, verificationDepth);
    if (first) setSelectedNodeId(first.id);
  }, [workflowTree, selectedNodeId, verificationDepth]);

  useEffect(() => { if (editingNodeId) editingNodeInputRef.current?.focus(); }, [editingNodeId]);
  useEffect(() => { if (editingTitle) titleInputRef.current?.focus(); }, [editingTitle]);
  useEffect(() => {
    if (editingOutputFileIndex !== null) {
      const el = editingOutputFileInputRef.current;
      el?.focus({ preventScroll: true });
      el?.select();
    }
  }, [editingOutputFileIndex]);
  useEffect(() => { setEditingTitle(false); setTitleDraft(""); }, [activeSessionId]);
  useEffect(() => {
    setEditingOutputFileIndex(null);
    setEditingOutputFileDraft("");
    setAddingOutputFile(false);
    setNewOutputFileDraft("");
  }, [selectedNodeId, activeSessionId]);

  useEffect(() => {
    if (addingOutputFile) {
      newOutputFileInputRef.current?.focus({ preventScroll: true });
    }
  }, [addingOutputFile]);

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

  const handleAddStepAtRoot = useCallback(() => {
    if (!activeSessionId) return;
    const newNode = createEmptyNode(0);
    const newTree = [...workflowTree, newNode];
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
    setSelectedNodeId(newNode.id);
    setEditingNodeId(newNode.id);
    setEditingNodeDraft("");
  }, [activeSessionId, workflowTree]);

  const handleAddChildNode = useCallback((parentId: string) => {
    if (!activeSessionId) return;
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const parent = findNode(newTree, parentId);
    if (!parent) return;
    const childDepth = parent.depth + 1;
    const newNode = createEmptyNode(childDepth);
    parent.children = [...parent.children, newNode];
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
    setCollapsedNodeIds(new Set([...collapsedNodeIds].filter((id) => id !== parentId)));
    setSelectedNodeId(newNode.id);
    setEditingNodeId(newNode.id);
    setEditingNodeDraft("");
  }, [activeSessionId, workflowTree, collapsedNodeIds]);

  const handleMoveNode = useCallback((draggedId: string, targetId: string) => {
    if (!activeSessionId || draggedId === targetId) return;
    if (isDescendant(workflowTree, targetId, draggedId)) return;
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const dragged = removeNodeFromTree(newTree, draggedId);
    if (!dragged) return;
    const from = findParentAndIndex(workflowTree, draggedId);
    const to = findParentAndIndex(newTree, targetId);
    if (!to) return;
    const sameParent = from?.parentId === to.parentId && from?.parentId != null;
    const sameParentRoot = from?.parentId === null && to.parentId === null;
    if (sameParent || sameParentRoot) {
      insertNodeIntoTree(newTree, dragged, to.parentId, to.index);
    } else {
      const targetNode = findNode(newTree, targetId);
      if (!targetNode) return;
      insertNodeIntoTree(newTree, dragged, targetId, targetNode.children.length);
      if (collapsedNodeIds.has(targetId)) setCollapsedNodeIds(new Set([...collapsedNodeIds].filter((id) => id !== targetId)));
    }
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
    setDraggedNodeId(null);
    setDropTargetNodeId(null);
  }, [activeSessionId, workflowTree, collapsedNodeIds]);

  const handleDepthCommit = useCallback((depth: number) => {
    if (!activeSessionId) return;
    updateVerificationDepth(activeSessionId, depth);
    sendEvent({ type: "session.updateVerificationDepth", payload: { sessionId: activeSessionId, verificationDepth: depth } });
    const first = findFirstAtDepthOrFallback(workflowTree, depth);
    if (first) setSelectedNodeId(first.id);
  }, [activeSessionId, workflowTree]);

  const currentVerifiers = selectedNode?.verifiers ?? [];
  const currentVerifierMarks = selectedNode?.verifierMarks ?? [];
  const [newVerifierDraft, setNewVerifierDraft] = useState("");
  const [addingVerifier, setAddingVerifier] = useState(false);
  const [trainingUploadState, setTrainingUploadState] = useState<
    | { kind: "idle" }
    | { kind: "uploading" }
    | { kind: "success" }
    | { kind: "error"; message: string }
  >({ kind: "idle" });

  const handleTriggerTraining = useCallback(async () => {
    if (!activeSessionId) return;
    setTrainingUploadState({ kind: "uploading" });
    try {
      const res = await window.electron.postSessionToTrainer(activeSessionId);
      if (res.success) {
        setTrainingUploadState({ kind: "success" });
        setTimeout(() => {
          setTrainingUploadState((prev) => (prev.kind === "success" ? { kind: "idle" } : prev));
        }, 2500);
      } else {
        setTrainingUploadState({ kind: "error", message: res.error ?? "Upload failed" });
      }
    } catch (e) {
      setTrainingUploadState({
        kind: "error",
        message: e instanceof Error ? e.message : String(e),
      });
    }
  }, [activeSessionId]);

  const toggleVerifierMark = (index: number) => {
    if (!activeSessionId || !selectedNodeId) return;
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const node = findNode(newTree, selectedNodeId);
    if (!node) return;
    while (node.verifierMarks.length <= index) node.verifierMarks.push(undefined);
    const cur = node.verifierMarks[index];
    node.verifierMarks[index] = cur === "check" ? "cross" : "check";
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
  };

  const handleDeleteVerifier = (index: number) => {
    if (!activeSessionId || !selectedNodeId) return;
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const node = findNode(newTree, selectedNodeId);
    if (!node) return;
    if (index < 0 || index >= node.verifiers.length) return;
    node.verifiers.splice(index, 1);
    if (Array.isArray(node.verifierMarks)) {
      node.verifierMarks.splice(index, 1);
    }
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
  };

  const handleAddVerifier = () => {
    if (!activeSessionId || !selectedNodeId) return;
    const text = newVerifierDraft.trim();
    if (!text) {
      setAddingVerifier(false);
      setNewVerifierDraft("");
      return;
    }
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const node = findNode(newTree, selectedNodeId);
    if (!node) return;
    node.verifiers.push(text);
    if (Array.isArray(node.verifierMarks)) {
      node.verifierMarks.push(undefined);
    }
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
    setAddingVerifier(false);
    setNewVerifierDraft("");
  };

  const startEditOutputFile = (index: number) => {
    if (!selectedNode) return;
    const path = selectedNode.outputFiles[index];
    if (path === undefined) return;
    skipNextOutputFileBlurSave.current = false;
    setEditingOutputFileIndex(index);
    setEditingOutputFileDraft(path);
  };

  const saveOutputFileEdit = () => {
    if (skipNextOutputFileBlurSave.current) {
      skipNextOutputFileBlurSave.current = false;
      return;
    }
    if (editingOutputFileIndex === null || !activeSessionId || !selectedNodeId) return;
    const idx = editingOutputFileIndex;
    const trimmed = editingOutputFileDraft.trim();
    setEditingOutputFileIndex(null);
    setEditingOutputFileDraft("");
    if (!trimmed) return;
    const nextName = normalizeOutputFileName(trimmed) ?? trimmed;
    if (!nextName) return;
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const node = findNode(newTree, selectedNodeId);
    if (!node || idx < 0 || idx >= node.outputFiles.length) return;
    node.outputFiles[idx] = nextName;
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
  };

  const handleAddOutputFile = () => {
    if (!activeSessionId || !selectedNodeId) return;
    const trimmed = newOutputFileDraft.trim();
    if (!trimmed) {
      setAddingOutputFile(false);
      setNewOutputFileDraft("");
      return;
    }
    const name = normalizeOutputFileName(trimmed);
    if (!name) {
      setAddingOutputFile(false);
      setNewOutputFileDraft("");
      return;
    }
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const node = findNode(newTree, selectedNodeId);
    if (!node) return;
    if (node.outputFiles.includes(name)) {
      setAddingOutputFile(false);
      setNewOutputFileDraft("");
      return;
    }
    node.outputFiles.push(name);
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
    setAddingOutputFile(false);
    setNewOutputFileDraft("");
  };

  const handleDeleteOutputFile = (index: number) => {
    if (!activeSessionId || !selectedNodeId) return;
    const newTree = JSON.parse(JSON.stringify(workflowTree)) as WorkflowNode[];
    const node = findNode(newTree, selectedNodeId);
    if (!node || index < 0 || index >= node.outputFiles.length) return;
    node.outputFiles.splice(index, 1);
    updateWorkflowTree(activeSessionId, newTree);
    sendEvent({ type: "session.updateWorkflowTree", payload: { sessionId: activeSessionId, workflowTree: newTree } });
  };

  const cancelOutputFileEdit = () => {
    skipNextOutputFileBlurSave.current = true;
    setEditingOutputFileIndex(null);
    setEditingOutputFileDraft("");
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
        <button className="flex-1 rounded-xl border border-ink-900/10 bg-surface px-3 py-2 text-[15px] font-medium leading-tight text-ink-700 hover:bg-surface-tertiary hover:border-ink-900/20 transition-colors" onClick={onNewSession}>+ New Task</button>
        <button type="button" className="inline-flex shrink-0 items-center justify-center rounded-xl border border-ink-900/10 bg-surface px-3 py-2 text-ink-700 hover:bg-surface-tertiary hover:border-ink-900/20 transition-colors" onClick={() => useAppStore.getState().setShowSettingsModal(true)} aria-label="Settings">
          <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1.08 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.08a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.08a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
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

      {/* Progress: tree + slider + run button — only when a task is selected (hidden on new-task home) */}
      {activeSessionId && (
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden border-t border-ink-900/10 pt-2.5">
          <div className="flex items-center gap-2 mb-2.5 shrink-0 min-w-0">
            <span className="text-base font-semibold text-ink-900 truncate tracking-tight shrink-0">Progress</span>
            {workflowTree.length > 0 && (
              <div className="flex items-center gap-1.5 shrink-0 text-xs">
                <TooltipProvider delayDuration={300}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        disabled={!activeSessionId}
                        onClick={() => handleDepthCommit(verificationDepth === 0 ? detailTargetDepth : 0)}
                        className="inline-flex h-5 w-14 shrink-0 items-center justify-center rounded-md border border-[#f3d5b5] bg-transparent px-0 py-0 text-ink-700 text-[11px] font-medium leading-none transition-colors hover:border-primary/30 hover:bg-primary/5 disabled:opacity-40 disabled:pointer-events-none"
                        aria-label={
                          verificationDepth === 0
                            ? "Verification: coarse (high level). Click to switch to detail."
                            : "Verification: detail. Click to switch to coarse."
                        }
                      >
                        {verificationDepth === 0 ? "Coarse" : "Detail"}
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      {maxDepth === 0
                        ? verificationDepth === 0
                          ? "Coarse level (no sub-steps in this plan yet). Click to show Detail mode; step list stays the same until you add sub-steps."
                          : "Detail mode; add sub-steps under a step to verify at finer granularity. Click for coarse."
                        : verificationDepth === 0
                          ? "Verifying at coarse (high) level. Click for detailed sub-steps."
                          : "Verifying with detailed sub-steps. Click for coarse level only."}
                    </TooltipContent>
                  </Tooltip>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => setWorkflowRunMode(workflowRunMode === "manual" ? "auto" : "manual")}
                        className="inline-flex h-5 w-14 shrink-0 items-center justify-center rounded-md border border-[#f3d5b5] bg-transparent px-0 py-0 text-ink-700 text-[11px] font-medium leading-none transition-colors hover:border-primary/30 hover:bg-primary/5"
                        aria-label={
                          workflowRunMode === "manual"
                            ? "Steps: wait (pause after each). Click to run all steps automatically."
                            : "Steps: auto. Click to wait after each step."
                        }
                      >
                        {workflowRunMode === "manual" ? "Wait" : "Auto"}
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom">
                      {workflowRunMode === "manual"
                        ? "Wait after each step. Click to run remaining steps automatically."
                        : "Run steps in sequence until finished. Click to wait after each step."}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            )}
            <div className="flex-1 min-w-0" aria-hidden />
            <TooltipProvider delayDuration={300}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    disabled={!activeSessionId || activeSession?.status === "running"}
                    onClick={() => {
                      if (activeSessionId) {
                        sendEvent({ type: "session.regenerateWorkflow", payload: { sessionId: activeSessionId } });
                      }
                    }}
                    className="shrink-0 rounded p-1 text-ink-400 hover:text-primary hover:bg-ink-900/5 transition-colors disabled:opacity-40 disabled:pointer-events-none"
                    aria-label="Re-generate workflow steps"
                  >
                    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" /><path d="M16 21h5v-5" /></svg>
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom">Re-generate workflow steps</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>

          {workflowTree.length > 0 ? (
            <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-0.5">
              {workflowTree.map((root) => (
                <TreeNode
                  key={root.id}
                  node={root}
                  selectedNodeId={selectedNodeId}
                  runningNodeId={runningNodeId}
                  collapsedNodeIds={collapsedNodeIds}
                  effectiveDepth={effectiveDepth}
                  verificationDepth={verificationDepth}
                  editingNodeId={editingNodeId}
                  editingNodeDraft={editingNodeDraft}
                  editingNodeInputRef={editingNodeInputRef}
                  draggedNodeId={draggedNodeId}
                  dropTargetNodeId={dropTargetNodeId}
                  onSelectNode={setSelectedNodeId}
                  onToggleCollapse={toggleNodeCollapsed}
                  onEditNode={handleEditNode}
                  onDeleteNode={handleDeleteNode}
                  onAddChild={handleAddChildNode}
                  onDragStart={(nodeId) => setDraggedNodeId(nodeId)}
                  onDragOver={(nodeId) => setDropTargetNodeId(nodeId)}
                  onDragLeave={() => setDropTargetNodeId(null)}
                  onDrop={(targetId) => { if (draggedNodeId) handleMoveNode(draggedNodeId, targetId); }}
                  onDragEnd={() => { setDraggedNodeId(null); setDropTargetNodeId(null); }}
                  onEditDraftChange={setEditingNodeDraft}
                  onEditSave={saveNodeEdit}
                  onEditCancel={() => { setEditingNodeId(null); setEditingNodeDraft(""); }}
                />
              ))}
              <button
                type="button"
                className="shrink-0 flex items-center gap-1 py-1 px-0.5 text-sm text-muted-foreground hover:text-ink-600 transition-colors"
                onClick={handleAddStepAtRoot}
              >
                <svg viewBox="0 0 16 16" className="h-4 w-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 3v10M3 8h10" /></svg>
                <span>Add step</span>
              </button>
            </div>
          ) : null}

          {/* Run button */}
          {workflowTree.length > 0 && activeSessionId && selectedNode && (() => {
            const isNodeRunning = runningNodeId === selectedNodeId && activeSession?.status === "running";
            const isNodeCompleted = selectedNode.status === "completed";
            if (isNodeRunning) {
              return (
                <div className="shrink-0 mt-1 flex items-center justify-center gap-1.5 py-1 text-sm text-muted-foreground">
                  <svg viewBox="0 0 24 24" className="h-4 w-4 animate-spin shrink-0" fill="none" stroke="currentColor" strokeWidth="2.5"><circle cx="12" cy="12" r="9" strokeOpacity="0.25" /><path d="M12 3a9 9 0 0 1 9 9" strokeLinecap="round" /></svg>
                  Running…
                </div>
              );
            }
            return (
              <button
                type="button"
                className="shrink-0 mt-1 flex w-full items-center justify-center gap-1.5 rounded-md border border-ink-900/15 bg-ink-900/[0.04] px-2.5 py-2 text-sm font-medium text-ink-800 hover:bg-ink-900/[0.07] transition-colors"
                onClick={() => { setRunningNodeId(selectedNodeId); sendEvent({ type: "session.solveNode", payload: { sessionId: activeSessionId, nodeId: selectedNodeId! } }); }}
              >
                {isNodeCompleted ? (
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" /></svg>
                ) : (
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
                )}
                {isNodeCompleted ? "Rerun" : "Run"}
              </button>
            );
          })()}
        </div>
      )}

      {/* Files (compact) */}
      {selectedNode && (
        <div className="shrink-0 border-t border-ink-900/10 pt-2.5 pb-3">
          <div className="flex items-center justify-between gap-1.5 mb-2.5 shrink-0">
            <span className="text-base font-semibold text-ink-900 truncate tracking-tight">Files</span>
            <button
              type="button"
              disabled={!activeSessionId}
              onClick={() => {
                setAddingOutputFile(true);
                setNewOutputFileDraft("");
                setEditingOutputFileIndex(null);
                setEditingOutputFileDraft("");
              }}
              className="shrink-0 p-0.5 rounded text-ink-300 hover:text-primary hover:bg-ink-900/5 transition-colors disabled:opacity-40"
              aria-label="Add output file"
            >
              <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M8 3v10M3 8h10" />
              </svg>
            </button>
          </div>
          <div
            className={
              editingOutputFileIndex !== null
                ? "flex flex-col gap-0.5 min-w-0 overflow-visible"
                : "flex flex-col gap-0.5 min-w-0 max-h-24 overflow-y-auto"
            }
          >
            {addingOutputFile && (
              <div className="flex items-center gap-1.5 min-w-0">
                <input
                  ref={newOutputFileInputRef}
                  type="text"
                  value={newOutputFileDraft}
                  onChange={(e) => setNewOutputFileDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleAddOutputFile();
                    }
                    if (e.key === "Escape") {
                      e.preventDefault();
                      setAddingOutputFile(false);
                      setNewOutputFileDraft("");
                    }
                  }}
                  onBlur={handleAddOutputFile}
                  placeholder="e.g. report.md"
                  className="flex-1 min-w-0 rounded border border-ink-900/20 bg-white px-1.5 py-0.5 text-sm text-ink-800 placeholder:text-ink-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
                />
              </div>
            )}
            {selectedNode.outputFiles.length === 0 && !addingOutputFile ? (
              <p className="text-sm text-muted-foreground leading-snug">No output files listed for this step.</p>
            ) : (
              selectedNode.outputFiles.map((f, i) =>
                editingOutputFileIndex === i ? (
                  <input
                    key={`${selectedNode.id}-outfile-${i}`}
                    ref={editingOutputFileInputRef}
                    type="text"
                    className="w-full min-w-0 rounded border border-ink-900/20 bg-white px-1.5 py-0.5 text-sm text-ink-800 overflow-x-hidden focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
                    value={editingOutputFileDraft}
                    onChange={(e) => setEditingOutputFileDraft(e.target.value)}
                    onBlur={saveOutputFileEdit}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        saveOutputFileEdit();
                      }
                      if (e.key === "Escape") {
                        e.preventDefault();
                        cancelOutputFileEdit();
                      }
                    }}
                  />
                ) : (
                  <div key={`${selectedNode.id}-outfile-${i}`} className="flex items-start gap-1.5 min-w-0">
                    <span
                      className="min-w-0 flex-1 text-sm leading-snug text-ink-600 break-words cursor-text rounded px-0.5 -mx-0.5 hover:text-ink-800"
                      title="Double-click to edit"
                      onDoubleClick={(e) => {
                        e.preventDefault();
                        startEditOutputFile(i);
                      }}
                    >
                      {f}
                    </span>
                    <button
                      type="button"
                      onClick={() => handleDeleteOutputFile(i)}
                      className="shrink-0 p-0.5 rounded text-ink-300 hover:text-error hover:bg-ink-900/5 transition-colors mt-0.5"
                      aria-label={`Remove ${f}`}
                    >
                      <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M12 4L4 12M4 4l8 8" />
                      </svg>
                    </button>
                  </div>
                )
              )
            )}
          </div>
        </div>
      )}

      {/* Verifiers (compact) */}
      {selectedNode && (
        <div className="shrink-0 max-h-[200px] flex flex-col overflow-hidden border-t border-ink-900/10 pt-2.5 pb-3">
          <div className="flex items-center justify-between gap-1.5 mb-2.5 shrink-0">
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-base font-semibold text-ink-900 truncate tracking-tight">Verifiers</span>
              {isVerifierChecking && (
                <span className="flex items-center gap-1 text-xs text-muted-foreground shrink-0">
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 animate-spin" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <circle cx="12" cy="12" r="9" strokeOpacity="0.25" />
                    <path d="M12 3a9 9 0 0 1 9 9" strokeLinecap="round" />
                  </svg>
                  Checking…
                </span>
              )}
            </div>
            <button
              type="button"
              disabled={isVerifierChecking}
              onClick={() => { setAddingVerifier(true); setNewVerifierDraft(""); }}
              className="shrink-0 p-0.5 rounded text-ink-300 hover:text-primary hover:bg-ink-900/5 transition-colors disabled:opacity-40"
              aria-label="Add verifier"
            >
              <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M8 3v10M3 8h10" />
              </svg>
            </button>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-2">
            {addingVerifier && (
              <div className="flex items-center gap-1.5">
                <span className="block h-1.5 w-1.5 rounded-full bg-ink-900/15" />
                <input
                  type="text"
                  value={newVerifierDraft}
                  onChange={(e) => setNewVerifierDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleAddVerifier();
                    if (e.key === "Escape") { setAddingVerifier(false); setNewVerifierDraft(""); }
                  }}
                  onBlur={handleAddVerifier}
                  placeholder="Describe what to verify…"
                  className="flex-1 min-w-0 rounded border border-ink-900/20 bg-white px-1.5 py-0.5 text-sm text-ink-800 placeholder:text-ink-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30"
                />
              </div>
            )}
            {currentVerifiers.map((text, index) => {
              const mark = currentVerifierMarks[index];
              const pass = mark === "check";
              const fail = mark === "cross";
              return (
                <div key={index} className="flex items-center gap-1.5">
                  <button
                    type="button"
                    disabled={isVerifierChecking}
                    onClick={() => { if (!isVerifierChecking) toggleVerifierMark(index); }}
                    className="shrink-0 flex items-center justify-center w-4 h-4 rounded-full border border-ink-900/25 bg-white text-ink-400 hover:bg-ink-900/5 transition-colors disabled:opacity-50 disabled:pointer-events-none"
                  >
                    {pass ? (
                      <svg viewBox="0 0 16 16" className="h-3 w-3 text-emerald-600" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 4L6 11 3 8" /></svg>
                    ) : fail ? (
                      <svg viewBox="0 0 16 16" className="h-3 w-3 text-red-500" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 4L4 12M4 4l8 8" /></svg>
                    ) : (
                      null
                    )}
                  </button>
                  <span className="min-w-0 flex-1 text-sm leading-snug text-ink-600 break-words">
                    {text}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleDeleteVerifier(index)}
                    className="shrink-0 p-0.5 rounded text-ink-300 hover:text-error hover:bg-ink-900/5 transition-colors"
                    aria-label="Delete verifier"
                  >
                    <svg viewBox="0 0 16 16" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M12 4L4 12M4 4l8 8" />
                    </svg>
                  </button>
                </div>
              );
            })}
          </div>
          <div className="shrink-0 mt-2 pt-2 border-t border-ink-900/5">
            <button
              type="button"
              onClick={handleTriggerTraining}
              disabled={!activeSessionId || trainingUploadState.kind === "uploading"}
              className="w-full flex items-center justify-center gap-1.5 rounded-md border border-primary/30 bg-primary/5 px-2.5 py-1.5 text-xs font-medium text-primary hover:bg-primary/10 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Upload this session to the training proxy"
            >
              {trainingUploadState.kind === "uploading" ? (
                <>
                  <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 animate-spin" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <circle cx="12" cy="12" r="9" strokeOpacity="0.25" />
                    <path d="M12 3a9 9 0 0 1 9 9" strokeLinecap="round" />
                  </svg>
                  Training…
                </>
              ) : trainingUploadState.kind === "success" ? (
                <>
                  <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M13 4L6 11 3 8" />
                  </svg>
                  Sent for training
                </>
              ) : (
                <>
                  <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.6">
                    <path d="M8 2v8M4.5 6.5L8 10l3.5-3.5M3 13h10" />
                  </svg>
                  Train on this session
                </>
              )}
            </button>
            {trainingUploadState.kind === "error" && (
              <p className="mt-1 text-[11px] text-error leading-snug break-words">
                {trainingUploadState.message}
              </p>
            )}
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
