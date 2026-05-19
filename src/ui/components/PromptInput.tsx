import { useCallback, useEffect, useRef, useState } from "react";
import { MultiFileDiff } from "@pierre/diffs/react";
import type { ClientEvent, PredictedUserActionSuggestion, WorkflowNode } from "../types";
import { useAppStore } from "../store/useAppStore";
import {
  applyWorkflowPatch,
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

function getExecutableAction(suggestion: PredictedUserActionSuggestion): ExecutableAction | null {
  return suggestion.executable ?? null;
}

function findWorkflowNode(nodes: WorkflowNode[] | undefined, nodeId: string): WorkflowNode | undefined {
  for (const node of nodes ?? []) {
    if (node.id === nodeId) return node;
    const found = findWorkflowNode(node.children, nodeId);
    if (found) return found;
  }
  return undefined;
}

function shortNodeId(nodeId: string): string {
  return nodeId.length > 8 ? nodeId.slice(0, 8) : nodeId;
}

function findWorkflowParent(
  nodes: WorkflowNode[] | undefined,
  nodeId: string,
  parent?: WorkflowNode
): WorkflowNode | undefined {
  for (const node of nodes ?? []) {
    if (node.id === nodeId) return parent;
    const found = findWorkflowParent(node.children, nodeId, node);
    if (found) return found;
  }
  return undefined;
}

type VerifierDiffRow = {
  key: string;
  text: string;
  kind: "added" | "removed" | "unchanged";
};

type WorkflowNodePreview = Omit<WorkflowNode, "children" | "depth"> & {
  children?: WorkflowNodePreview[];
  depth?: number;
};

type WorkflowNodeChange = "added" | "deleted" | "edited" | "unchanged";

type WorkflowPreviewNode = Omit<WorkflowNodePreview, "children"> & {
  change: WorkflowNodeChange;
  children: WorkflowPreviewNode[];
  previousDescription?: string;
  previousStatus?: WorkflowNode["status"];
  movedFrom?: string;
};

function verifierDiffRows(before: string[], after: string[]): VerifierDiffRow[] {
  const remainingAfter = new Map<string, number>();
  for (const verifier of after) {
    remainingAfter.set(verifier, (remainingAfter.get(verifier) ?? 0) + 1);
  }

  const rows: VerifierDiffRow[] = [];
  for (const verifier of before) {
    const count = remainingAfter.get(verifier) ?? 0;
    if (count > 0) {
      remainingAfter.set(verifier, count - 1);
      rows.push({ key: `same-${rows.length}-${verifier}`, text: verifier, kind: "unchanged" });
    } else {
      rows.push({ key: `removed-${rows.length}-${verifier}`, text: verifier, kind: "removed" });
    }
  }

  for (const verifier of after) {
    const beforeCount = before.filter((item) => item === verifier).length;
    const unchangedCount = rows.filter((row) => row.kind === "unchanged" && row.text === verifier).length;
    const alreadyAdded = rows.filter((row) => row.kind === "added" && row.text === verifier).length;
    if (unchangedCount + alreadyAdded < after.filter((item) => item === verifier).length && unchangedCount >= beforeCount) {
      rows.push({ key: `added-${rows.length}-${verifier}`, text: verifier, kind: "added" });
    } else if (beforeCount === 0) {
      rows.push({ key: `added-${rows.length}-${verifier}`, text: verifier, kind: "added" });
    }
  }

  return rows;
}

function DiffRows({ rows, emptyLabel }: { rows: VerifierDiffRow[]; emptyLabel: string }) {
  return (
    <div className="mt-2 grid gap-1.5">
      {rows.length > 0 ? (
        rows.map((row) => (
          <div
            key={row.key}
            className={`flex min-w-0 items-start gap-2 rounded-lg border px-2.5 py-1.5 text-xs ${
              row.kind === "added"
                ? "border-success/20 bg-success-light text-ink-900"
                : row.kind === "removed"
                  ? "border-error/20 bg-error-light text-ink-800"
                  : "border-ink-900/8 bg-white text-muted-foreground"
            }`}
          >
            <span className={`mt-0.5 shrink-0 font-mono text-[11px] font-semibold ${
              row.kind === "added"
                ? "text-success"
                : row.kind === "removed"
                  ? "text-error"
                  : "text-muted-foreground"
            }`}>
              {row.kind === "added" ? "+" : row.kind === "removed" ? "-" : " "}
            </span>
            <span className={row.kind === "removed" ? "min-w-0 flex-1 whitespace-pre-wrap break-words line-through" : "min-w-0 flex-1 whitespace-pre-wrap break-words"}>
              {row.text}
            </span>
          </div>
        ))
      ) : (
        <span className="rounded-lg border border-ink-900/8 bg-white px-2.5 py-1.5 text-xs text-muted-foreground">
          {emptyLabel}
        </span>
      )}
    </div>
  );
}

function collectNodeIds(node: WorkflowNodePreview, ids = new Set<string>()): Set<string> {
  ids.add(node.id);
  for (const child of node.children ?? []) collectNodeIds(child, ids);
  return ids;
}

function cloneWorkflowPreviewNode(
  node: WorkflowNodePreview,
  change: WorkflowNodeChange,
  details?: Partial<WorkflowPreviewNode>
): WorkflowPreviewNode {
  return {
    ...node,
    outputFiles: [...(node.outputFiles ?? [])],
    verifiers: [...(node.verifiers ?? [])],
    verifierMarks: [...(node.verifierMarks ?? [])],
    children: (node.children ?? []).map((child) => cloneWorkflowPreviewNode(child, change)),
    change,
    ...details,
  };
}

function findPreviewNode(nodes: WorkflowPreviewNode[], nodeId: string): WorkflowPreviewNode | undefined {
  for (const node of nodes) {
    if (node.id === nodeId) return node;
    const found = findPreviewNode(node.children, nodeId);
    if (found) return found;
  }
  return undefined;
}

function nodeIndex(nodes: WorkflowNode[] | undefined, nodeId: string): number {
  return (nodes ?? []).findIndex((node) => node.id === nodeId);
}

function buildWorkflowPreview(
  action: Extract<ExecutableAction, { type: "edit_workflow" }>,
  currentWorkflowTree: WorkflowNode[] | undefined
): { roots: WorkflowPreviewNode[]; fallbackMessage?: string } {
  const beforeTree = currentWorkflowTree ?? [];
  const addedIds = new Set<string>();
  const editedIds = new Set<string>();
  const deletedIds = new Set<string>();
  const movedIds = new Set<string>();
  const previousDescriptions = new Map<string, string>();
  const previousStatuses = new Map<string, WorkflowNode["status"]>();
  const movedFrom = new Map<string, string>();
  const deletedRoots: Array<{ node: WorkflowPreviewNode; parentId: string | null; index: number }> = [];

  for (const operation of action.patch) {
    if (operation.op === "add_node") {
      collectNodeIds(operation.node).forEach((id) => addedIds.add(id));
      continue;
    }

    if (operation.op === "update_node") {
      editedIds.add(operation.nodeId);
      const target = findWorkflowNode(beforeTree, operation.nodeId);
      if (target) {
        if (operation.description !== undefined && operation.description !== target.description) {
          previousDescriptions.set(operation.nodeId, target.description);
        }
        if (operation.status !== undefined && operation.status !== target.status) {
          previousStatuses.set(operation.nodeId, target.status);
        }
      }
      continue;
    }

    if (operation.op === "delete_node") {
      const target = findWorkflowNode(beforeTree, operation.nodeId);
      if (!target) continue;
      collectNodeIds(target).forEach((id) => deletedIds.add(id));
      const parent = findWorkflowParent(beforeTree, operation.nodeId);
      deletedRoots.push({
        node: cloneWorkflowPreviewNode(target, "deleted"),
        parentId: parent?.id ?? null,
        index: nodeIndex(parent ? parent.children : beforeTree, operation.nodeId),
      });
      continue;
    }

    if (operation.op === "move_node") {
      movedIds.add(operation.nodeId);
      const oldParent = findWorkflowParent(beforeTree, operation.nodeId);
      movedFrom.set(operation.nodeId, oldParent?.description ?? "top level");
    }
  }

  let afterTree: WorkflowNode[];
  try {
    afterTree = applyWorkflowPatch(beforeTree, action.patch);
  } catch (error) {
    const addRoots = action.patch.flatMap((operation) =>
      operation.op === "add_node" ? [cloneWorkflowPreviewNode(operation.node, "added")] : []
    );
    return {
      roots: [...addRoots, ...deletedRoots.map((item) => item.node)],
      fallbackMessage: error instanceof Error ? error.message : String(error),
    };
  }

  const roots = afterTree.map((node) => {
    const change: WorkflowNodeChange = addedIds.has(node.id)
      ? "added"
      : editedIds.has(node.id) || movedIds.has(node.id)
        ? "edited"
        : "unchanged";
    return cloneWorkflowPreviewNode(node, change, {
      previousDescription: previousDescriptions.get(node.id),
      previousStatus: previousStatuses.get(node.id),
      movedFrom: movedFrom.get(node.id),
    });
  });

  const decorate = (nodes: WorkflowPreviewNode[]) => {
    for (const node of nodes) {
      if (addedIds.has(node.id)) node.change = "added";
      else if (editedIds.has(node.id) || movedIds.has(node.id)) node.change = "edited";
      node.previousDescription = previousDescriptions.get(node.id);
      node.previousStatus = previousStatuses.get(node.id);
      node.movedFrom = movedFrom.get(node.id);
      decorate(node.children);
    }
  };
  decorate(roots);

  for (const deleted of deletedRoots) {
    const parent = deleted.parentId ? findPreviewNode(roots, deleted.parentId) : undefined;
    const siblings = parent ? parent.children : roots;
    const insertAt = deleted.index >= 0 ? Math.min(deleted.index, siblings.length) : siblings.length;
    if (!deleted.parentId || parent) siblings.splice(insertAt, 0, deleted.node);
  }

  if (!roots.length && deletedRoots.length) return { roots: deletedRoots.map((item) => item.node) };
  return { roots };
}

function workflowPreviewNodeClasses(change: WorkflowNodeChange): string {
  if (change === "added") return "border-success/30 bg-success-light text-ink-900 shadow-[inset_3px_0_0_rgba(22,163,74,0.85)]";
  if (change === "deleted") return "border-error/30 bg-error-light text-ink-800 shadow-[inset_3px_0_0_rgba(220,38,38,0.85)]";
  if (change === "edited") return "border-[#D99A20]/35 bg-[#FEF3C7] text-ink-900 shadow-[inset_3px_0_0_rgba(217,154,32,0.9)]";
  return "border-ink-900/8 bg-white text-ink-800";
}

function workflowPreviewDotClasses(change: WorkflowNodeChange): string {
  if (change === "added") return "border-success bg-success";
  if (change === "deleted") return "border-error bg-error";
  if (change === "edited") return "border-[#D99A20] bg-[#FBBF24]";
  return "border-ink-900/25 bg-white";
}

function WorkflowPreviewNodeView({ node, depth = 0 }: { node: WorkflowPreviewNode; depth?: number }) {
  const hasChildren = node.children.length > 0;
  const showDetails = node.outputFiles.length > 0 || node.verifiers.length > 0 || node.previousDescription || node.previousStatus || node.movedFrom;
  return (
    <div className="relative" style={{ paddingLeft: depth > 0 ? 18 : 0 }}>
      {depth > 0 && (
        <div className="absolute left-[7px] top-0 h-full border-l border-ink-900/10" />
      )}
      <div className={`relative rounded-lg border px-3 py-2 ${workflowPreviewNodeClasses(node.change)}`}>
        <div className="flex min-w-0 items-start gap-2">
          <span className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full border ${workflowPreviewDotClasses(node.change)}`} />
          <div className="min-w-0 flex-1">
            {node.previousDescription && (
              <div className="mb-0.5 break-words text-xs text-ink-600 line-through">
                {node.previousDescription}
              </div>
            )}
            <div className={`break-words text-sm font-medium leading-snug ${node.change === "deleted" ? "line-through" : ""}`}>
              {node.description || "Untitled"}
            </div>
            {showDetails && (
              <div className="mt-1.5 grid gap-1 text-xs">
                {node.previousStatus && (
                  <div className="text-ink-700">
                    <span className="line-through">{node.previousStatus}</span>
                    <span className="mx-1 text-ink-400">to</span>
                    <span>{node.status}</span>
                  </div>
                )}
                {node.movedFrom && (
                  <div className="text-ink-700">
                    moved from {node.movedFrom}
                  </div>
                )}
                {node.outputFiles.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {node.outputFiles.map((outputFile) => (
                      <span key={outputFile} className="rounded border border-ink-900/10 bg-white/65 px-1.5 py-0.5 font-mono text-[11px]">
                        {outputFile}
                      </span>
                    ))}
                  </div>
                )}
                {node.verifiers.length > 0 && (
                  <div className="grid gap-1">
                    {node.verifiers.map((verifier, index) => (
                      <div key={`${verifier}-${index}`} className="flex items-start gap-1.5 text-ink-700">
                        <span className="mt-[0.45rem] h-1 w-1 shrink-0 rounded-full bg-current opacity-50" />
                        <span className={node.change === "deleted" ? "line-through" : ""}>{verifier}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      {hasChildren && (
        <div className="mt-1.5 grid gap-1.5">
          {node.children.map((child) => (
            <WorkflowPreviewNodeView key={child.id} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
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
  currentWorkflowTree,
}: {
  suggestion: PredictedUserActionSuggestion;
  currentWorkflowTree?: WorkflowNode[];
}) {
  const action = getExecutableAction(suggestion);
  const type = suggestion.actionType;
  const [showBrainDiff, setShowBrainDiff] = useState(false);

  const body = (() => {
    if (action?.type === "message") {
      return (
        <>
          <div className="text-[11px] font-medium text-muted-foreground">Draft prompt</div>
          <div className="mt-1 max-h-28 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-ink-900/8 bg-white px-3 py-2 text-sm text-ink-900">
            {action.text}
          </div>
        </>
      );
    }

    if (action?.type === "edit_workflow") {
      const preview = buildWorkflowPreview(action, currentWorkflowTree);

      return (
        <>
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
            <span>{action.patch.length} workflow change{action.patch.length === 1 ? "" : "s"}</span>
            <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-success" /> added</span>
            <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-[#FBBF24]" /> edited</span>
            <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-error" /> deleted</span>
          </div>
          <div className="mt-2 grid max-h-72 min-w-0 gap-1.5 overflow-auto rounded-lg border border-ink-900/8 bg-white/60 p-2">
            {preview.roots.map((node) => (
              <WorkflowPreviewNodeView key={node.id} node={node} />
            ))}
          </div>
          {preview.fallbackMessage && (
            <div className="mt-2 rounded-lg border border-[#D99A20]/25 bg-[#FEF3C7] px-2.5 py-1.5 text-xs text-ink-800">
              Preview uses partial patch because the full patch could not be applied.
            </div>
          )}
        </>
      );
    }

    if (action?.type === "edit_verifier") {
      const targetNode = findWorkflowNode(currentWorkflowTree, action.nodeId);
      const beforeVerifiers = targetNode?.verifiers ?? [];
      const rows = verifierDiffRows(beforeVerifiers, action.verifiers);
      return (
        <>
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-ink-900">
              edit node: {targetNode?.description || `Step ${shortNodeId(action.nodeId)}`}
            </div>
            <div className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
              {targetNode ? `node ${shortNodeId(action.nodeId)}` : `node not found: ${action.nodeId}`}
            </div>
          </div>
          <DiffRows rows={rows} emptyLabel="Clears verifier list" />
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
            <pre className="mt-2 max-h-20 overflow-hidden whitespace-pre-wrap break-words rounded-lg border border-ink-900/8 bg-white px-3 py-2 text-xs text-ink-700">
              {action.contents.split("\n").slice(0, 3).join("\n")}
            </pre>
          )}
        </>
      );
    }

    if (action?.type === "brain_edit") {
      return (
        <>
          <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
            <div className="flex min-w-0 flex-wrap gap-2 text-[11px] text-muted-foreground">
              <span>{action.kind === "memory" ? "Memory" : "Skill"} update</span>
              <span>{action.sections.length} sections</span>
              {action.deletedFileNames?.length ? <span>{action.deletedFileNames.length} deletes</span> : null}
            </div>
            {action.sections.length > 0 && (
              <button
                type="button"
                onClick={() => setShowBrainDiff((open) => !open)}
                className="shrink-0 rounded-md border border-[#D8B98F] bg-white px-2 py-1 text-[11px] font-medium text-[#7A4E20] transition-colors hover:bg-[#F1E3D1]"
              >
                {showBrainDiff ? "Hide diff" : "Show diff"}
              </button>
            )}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {action.sections.slice(0, 4).map((section) => (
              <span key={section.fileName} className="max-w-full break-words rounded-md border border-[#D8B98F] bg-white px-2 py-1 text-xs text-ink-800">
                {section.fileName}
              </span>
            ))}
            {action.sections.length > 4 && (
              <span className="rounded-md bg-white px-2 py-1 text-xs text-muted-foreground">
                +{action.sections.length - 4} more
              </span>
            )}
          </div>
          {showBrainDiff && (
            <div className="mt-3 grid max-h-72 gap-3 overflow-auto rounded-lg border border-ink-900/10 bg-white p-2">
              {action.sections.map((section) => (
                <div key={section.fileName} className="min-w-0 overflow-hidden rounded-md border border-ink-900/8 bg-surface">
                  <div className="truncate border-b border-ink-900/8 bg-white px-2.5 py-1.5 font-mono text-[11px] text-ink-700">
                    {section.fileName}
                  </div>
                  <div className="max-h-56 overflow-auto text-xs">
                    <MultiFileDiff
                      oldFile={{ name: section.fileName, contents: "" }}
                      newFile={{ name: section.fileName, contents: section.content }}
                      options={{
                        theme: "pierre-light",
                        diffStyle: "unified",
                        disableFileHeader: true,
                        diffIndicators: "bars",
                        overflow: "scroll",
                        disableLineNumbers: false,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
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

  return <div className="min-w-0">{body}</div>;
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
      <div className={`mx-auto flex w-full max-w-full min-w-0 flex-col gap-2 ${rightOffset ? "" : "lg:max-w-3xl"}`}>
        {(predictedSuggestion || isPredictingSuggestion) && (
          <div className="max-h-[min(48vh,28rem)] overflow-y-auto overflow-x-hidden rounded-2xl border border-ink-900/10 bg-white/95 px-4 py-3 shadow-card">
            {isPredictingSuggestion && !predictedSuggestion ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5 animate-spin" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                  <path d="M12 2a10 10 0 0 1 10 10" strokeLinecap="round" />
                </svg>
                <span>Generating next action suggestion…</span>
              </div>
            ) : predictedSuggestion ? (
              <div className="flex min-w-0 flex-col gap-2">
                <div className="flex min-w-0 items-center justify-between gap-3">
                  <span className="min-w-0 truncate rounded-full bg-primary/10 px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-primary">
                    {predictedSuggestion.actionType}
                  </span>
                  <div className="ml-auto flex shrink-0 items-center gap-2">
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
                <PredictionActionPreview
                  suggestion={predictedSuggestion}
                  currentWorkflowTree={activeSession?.workflowTree}
                />
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
