import { BrowserWindow } from "electron";
import type { ClientEvent, ServerEvent, WorkflowNode } from "./types.js";
import { runClaude, buildPromptForNode, buildRegenerateWorkflowPrompt, type RunnerHandle } from "./libs/runner.js";
import { SessionStore, type Session } from "./libs/session-store.js";
import {
  findNodeById,
  findParentNode,
  getNextIncompleteChild,
  isNodeFullyComplete,
  getNodePath,
  updateNodeStatus,
  resetNode,
  completeNodeAndDescendants,
} from "./libs/workflow-tree-utils.js";
import { app } from "electron";
import { join, relative, resolve, isAbsolute as pathIsAbsolute } from "path";
import { readFileSync, rmSync } from "fs";
import { ensureMemoriesDir, readAllMemorySections, writeMemorySections, getMemoriesDir } from "./libs/memory-store.js";
import {
  readAllFlatSkillSections,
  writeFlatSkillSections,
  getAppSkillsDir,
  syncAppSkills,
  isValidFlatSkillMdFileName,
} from "./libs/skill-store.js";
import { runFullSessionExportAndExtract, setContextInductionNotifier } from "./libs/context-export.js";
import { labelVerifiersForNode } from "./libs/verifier-labeler.js";
import {
  buildExportEnvironmentSnapshot,
  buildExportEnvironmentSnapshotWithPreviewWrittenFile,
  shouldWriteSnapshotForSdkMessage,
} from "./libs/message-state-snapshot.js";
import { classifyUserWorkflowTreeEdit } from "./libs/workflow-edit-classify.js";
import { createPiSessionManager, getPiSessionsDir } from "./libs/pi-config.js";
import { generateUpdatedVerifiersForNode } from "./libs/verifier-generator.js";

/** Build a compact line-based diff between original and current text, with only changed hunks and small context. */
function buildTextDiff(original: string, current: string, maxHunks = 8, contextLines = 1): string {
  const origLines = original.split(/\r?\n/);
  const currLines = current.split(/\r?\n/);
  const n = origLines.length;
  const m = currLines.length;

  // LCS DP table
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      if (origLines[i] === currLines[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  type Op = { type: "equal" | "del" | "add"; line: string; i: number; j: number };
  const ops: Op[] = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (origLines[i] === currLines[j]) {
      ops.push({ type: "equal", line: origLines[i], i, j });
      i++; j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      ops.push({ type: "del", line: origLines[i], i, j });
      i++;
    } else {
      ops.push({ type: "add", line: currLines[j], i, j });
      j++;
    }
  }
  while (i < n) {
    ops.push({ type: "del", line: origLines[i], i, j });
    i++;
  }
  while (j < m) {
    ops.push({ type: "add", line: currLines[j], i, j });
    j++;
  }

  // Group into hunks with context
  const hunks: { start: number; end: number }[] = [];
  for (let k = 0; k < ops.length; k++) {
    if (ops[k].type === "equal") continue;
    const start = Math.max(0, k - contextLines);
    let end = Math.min(ops.length - 1, k + contextLines);
    // extend end forward while within context window and diff continues
    while (end + 1 < ops.length && ops[end + 1].type !== "equal") end++;
    // merge with previous hunk if overlapping
    if (hunks.length > 0 && start <= hunks[hunks.length - 1].end + 1) {
      hunks[hunks.length - 1].end = Math.max(hunks[hunks.length - 1].end, end);
    } else {
      hunks.push({ start, end });
    }
  }

  if (hunks.length === 0) return "";

  const lines: string[] = [];
  const limitedHunks = hunks.slice(0, maxHunks);
  for (let h = 0; h < limitedHunks.length; h++) {
    const { start: s, end: e } = limitedHunks[h];
    if (h > 0) lines.push("...");
    for (let k = s; k <= e; k++) {
      const op = ops[k];
      if (op.type === "equal") {
        lines.push(`  ${op.line}`);
      } else if (op.type === "del") {
        lines.push(`- ${op.line}`);
      } else {
        lines.push(`+ ${op.line}`);
      }
    }
  }
  return lines.join("\n");
}

let sessions: SessionStore;
const runnerHandles = new Map<string, RunnerHandle>();

/** While a node-solving run is in progress, maps sessionId -> nodeId so we can emit nodeCompleted on result. */
const sessionCurrentNodeId = new Map<string, string>();

/**
 * After session.continue, maps sessionId -> workflow node to re-run verifier labeling on success
 * (follow-up user messages that are not session.solveNode).
 */
const sessionContinueVerificationNodeId = new Map<string, string>();

/** Last workflow node the session was driving (solve or explicit selection); used to verify on continue when UI omits verificationNodeId. */
const sessionLastVerificationNodeId = new Map<string, string>();
type VerifierExampleState = { removed: string[]; added: string[] };
const sessionVerifierExamplesByNodeId = new Map<string, Map<string, VerifierExampleState>>();

function normalizeVerifierText(v: string): string {
  return v.replace(/\s+/g, " ").trim().toLowerCase();
}

function uniqueByNormalized(values: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const raw of values) {
    const cleaned = String(raw ?? "").trim();
    if (!cleaned) continue;
    const key = normalizeVerifierText(cleaned);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(cleaned);
  }
  return out;
}

function indexNodesById(tree: WorkflowNode[]): Map<string, WorkflowNode> {
  const map = new Map<string, WorkflowNode>();
  const stack = [...tree];
  while (stack.length > 0) {
    const node = stack.pop()!;
    map.set(node.id, node);
    for (const child of node.children ?? []) stack.push(child);
  }
  return map;
}

function collectVerifierExampleUpdates(
  oldTree: WorkflowNode[],
  newTree: WorkflowNode[]
): Map<string, VerifierExampleState> {
  const out = new Map<string, VerifierExampleState>();
  const oldIdx = indexNodesById(oldTree);
  const newIdx = indexNodesById(newTree);
  for (const [nodeId, beforeNode] of oldIdx.entries()) {
    const afterNode = newIdx.get(nodeId);
    if (!afterNode) continue;
    const before = uniqueByNormalized([...(beforeNode.verifiers ?? [])]);
    const after = uniqueByNormalized([...(afterNode.verifiers ?? [])]);
    const beforeKeys = new Set(before.map(normalizeVerifierText));
    const afterKeys = new Set(after.map(normalizeVerifierText));
    const removed = before.filter((v) => !afterKeys.has(normalizeVerifierText(v)));
    const added = after.filter((v) => !beforeKeys.has(normalizeVerifierText(v)));
    if (removed.length > 0 || added.length > 0) {
      out.set(nodeId, { removed, added });
    }
  }
  return out;
}

function initializeSessions() {
  if (!sessions) {
    const DB_PATH = join(app.getPath("userData"), "sessions.db");
    sessions = new SessionStore(DB_PATH);
  }
  return sessions;
}

function broadcast(event: ServerEvent) {
  const payload = JSON.stringify(event);
  const windows = BrowserWindow.getAllWindows();
  for (const win of windows) {
    win.webContents.send("server-event", payload);
  }
}

setContextInductionNotifier((ev) => {
  if (ev.kind === "start") {
    broadcast({
      type: "session.contextInduction",
      payload: { phase: "started", sessionId: ev.sessionId },
    });
  } else {
    broadcast({
      type: "session.contextInduction",
      payload: { phase: "finished", sessionId: ev.sessionId, ok: ev.ok },
    });
  }
});

function hasLiveSession(sessionId: string): boolean {
  return Boolean(initializeSessions().getSession(sessionId));
}

/** After Brain dialog saves memory + skills: persist a user action + env snapshot on the active task session. */
function recordBrainEditAction(sessionId: string): void {
  const store = initializeSessions();
  const sess = store.getSession(sessionId);
  if (!sess) return;
  const rowId = store.recordMessage(sessionId, { type: "brain_edit" });
  store.writeMessageSnapshot(rowId, buildExportEnvironmentSnapshot(sess));
  broadcast({
    type: "stream.message",
    payload: { sessionId, message: { type: "brain_edit" } },
  });
}

/** SDK result message: treat as success for post-step finalization (verifiers, nodeCompleted, status). */
function isSuccessfulAgentResult(message: unknown): boolean {
  if (!message || typeof message !== "object") return false;
  const m = message as { type?: string; subtype?: string; is_error?: boolean };
  if (m.type !== "result") return false;
  if (m.subtype === "success") return true;
  if (m.is_error === false) return true;
  return false;
}

function isAgentResultMessage(message: unknown): boolean {
  return Boolean(message && typeof message === "object" && (message as { type?: string }).type === "result");
}

/** Re-read output files and refresh verifier marks for one node (Messages API). */
async function runVerifierLabelingForNode(sessionId: string, nodeId: string): Promise<void> {
  const store = initializeSessions();

  broadcast({
    type: "session.verifierCheck",
    payload: { sessionId, nodeId, phase: "started" },
  });

  try {
    let session = store.getSession(sessionId);
    if (session?.workflowTree) {
      const node = findNodeById(session.workflowTree, nodeId);
      if (node && node.verifiers.length > 0) {
        try {
          const marks = await labelVerifiersForNode(session, session.workflowTree, node);
          node.verifierMarks = node.verifiers.map((_, i) => marks[i] ?? undefined);
          store.updateSession(sessionId, { workflowTree: session.workflowTree });
          const treePayload = JSON.parse(JSON.stringify(session.workflowTree)) as WorkflowNode[];
          broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree: treePayload } });
        } catch (e) {
          console.error("[ipc] verifier labeling failed:", e);
        }
        const verifyPayload = { type: "verifier_label" as const, nodeId };
        const verifyRowId = store.recordMessage(sessionId, verifyPayload);
        const sessAfter = store.getSession(sessionId);
        if (sessAfter) {
          store.writeMessageSnapshot(verifyRowId, buildExportEnvironmentSnapshot(sessAfter));
        }
        broadcast({
          type: "stream.message",
          payload: { sessionId, message: verifyPayload },
        });
      }
    }
  } finally {
    broadcast({
      type: "session.verifierCheck",
      payload: { sessionId, nodeId, phase: "finished" },
    });
  }
}

function runPostSolverExport(sessionId: string, _nodeId: string): void {
  const store = initializeSessions();
  const session = store.getSession(sessionId);
  if (!session) return;

  const treeAfter = session.workflowTree;
  const planFullyDone = Boolean(treeAfter?.length && treeAfter.every(isNodeFullyComplete));
  if (planFullyDone) {
    runFullSessionExportAndExtract(sessionId);
  }
}

async function finalizeNodeSolveAfterVerifierPass(sessionId: string, nodeId: string) {
  await runVerifierLabelingForNode(sessionId, nodeId);

  const store = initializeSessions();
  const session = store.getSession(sessionId);
  if (!session) return;

  broadcast({ type: "session.nodeCompleted", payload: { sessionId, nodeId } });

  runPostSolverExport(sessionId, nodeId);

  emit({
    type: "session.status",
    payload: {
      sessionId,
      status: "completed",
      title: session.title,
      cwd: session.cwd,
    },
  });
}

/** After a free-form session.continue finishes: re-check verifiers + export; no nodeCompleted (runner already set status). */
async function finalizeContinueWithVerification(sessionId: string, nodeId: string) {
  await runVerifierLabelingForNode(sessionId, nodeId);
  runPostSolverExport(sessionId, nodeId);
  const session = initializeSessions().getSession(sessionId);
  if (!session) return;
  emit({
    type: "session.status",
    payload: {
      sessionId,
      status: "completed",
      title: session.title,
      cwd: session.cwd,
    },
  });
}

function emit(event: ServerEvent) {
  initializeSessions();

  if (
    (event.type === "session.status" ||
      event.type === "stream.message" ||
      event.type === "stream.user_prompt" ||
      event.type === "permission.request") &&
    !hasLiveSession(event.payload.sessionId)
  ) {
    return;
  }

  if (
    event.type === "session.status" &&
    event.payload.status === "completed" &&
    (sessionCurrentNodeId.has(event.payload.sessionId) || sessionContinueVerificationNodeId.has(event.payload.sessionId))
  ) {
    // Force verifier pass to be the final action for execution runs before surfacing "completed".
    return;
  }

  if (event.type === "session.status") {
    sessions!.updateSession(event.payload.sessionId, { status: event.payload.status });
  }
  if (event.type === "workflow.plan") {
    const { sessionId, workflowTree } = event.payload;
    const session = sessions!.getSession(sessionId);
    if (session) {
      const defaultDepth = 0;
      sessions!.updateSession(sessionId, {
        workflowTree,
        verificationDepth: defaultDepth,
        status: "idle"
      });
      broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree } });
      broadcast({ type: "session.verificationDepth", payload: { sessionId, verificationDepth: defaultDepth } });
      broadcast({
        type: "session.status",
        payload: { sessionId, status: "idle", title: session.title, cwd: session.cwd }
      });
      // Force verifier refinement from latest user prompts right after initial plan registration.
      void autoRefineVerifiersFromUserMessages(sessionId, flattenWorkflowNodeIds(workflowTree));
    }
    return;
  }
  if (event.type === "stream.message") {
    const { sessionId, message } = event.payload;
    const messageRowId = sessions!.recordMessage(sessionId, message);

    // When a node-solving run completes, mark that node as completed and handle auto-cascade.
    const m = message as { type?: string; subtype?: string; status?: string };
    if (m.type === "result" || m.type === "run_result") {
      const nodeId = sessionCurrentNodeId.get(sessionId);
      if (nodeId !== undefined) {
        sessionCurrentNodeId.delete(sessionId);
        const didSucceed = m.type === "result" ? m.subtype === "success" : m.status === "success";
        if (didSucceed) {
          const session = sessions!.getSession(sessionId);
          if (session && session.workflowTree) {
            const completedNode = findNodeById(session.workflowTree, nodeId);
            if (completedNode) {
              // If we don't yet have an originalOutputs snapshot, capture the initial model-written content now
              if (!completedNode.originalOutputs && completedNode.outputFiles.length > 0) {
                const cwd = session.cwd ?? process.cwd();
                const originals: { path: string; content: string }[] = [];
                for (const relPath of completedNode.outputFiles) {
                  try {
                    const absPath = join(cwd, relPath);
                    const content = readFileSync(absPath, "utf8");
                    originals.push({ path: relPath, content });
                  } catch {
                    // ignore missing/unreadable files
                  }
                }
                if (originals.length > 0) {
                  completedNode.originalOutputs = originals;
                }
              }
            }

            // Mark this node and all its descendants as completed
            if (completedNode) {
              completeNodeAndDescendants(completedNode);
            }

            // Bubble up: mark parents complete if all children are done
            let parentNode = findParentNode(session.workflowTree!, nodeId);
            while (parentNode && isNodeFullyComplete(parentNode)) {
              parentNode.status = "completed";
              parentNode = findParentNode(session.workflowTree!, parentNode.id);
            }

            sessions!.updateSession(sessionId, { workflowTree: session.workflowTree });
            const treePayload = JSON.parse(JSON.stringify(session.workflowTree)) as WorkflowNode[];
            broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree: treePayload } });
          }
        }
        void finalizeNodeSolveAfterVerifierPass(sessionId, nodeId).catch((e) => {
          console.error("[ipc] finalizeNodeSolveAfterVerifierPass:", e);
          emit({
            type: "session.status",
            payload: {
              sessionId,
              status: "error",
              title: sessions!.getSession(sessionId)?.title,
              cwd: sessions!.getSession(sessionId)?.cwd,
              error: String(e),
            },
          });
        });
      } else {
        const continueNodeId = sessionContinueVerificationNodeId.get(sessionId);
        if (continueNodeId !== undefined) {
          sessionContinueVerificationNodeId.delete(sessionId);
          void finalizeContinueWithVerification(sessionId, continueNodeId).catch((e) => {
            console.error("[ipc] finalizeContinueWithVerification:", e);
          });
        }
      }
    } else if (isAgentResultMessage(message)) {
      sessionCurrentNodeId.delete(sessionId);
      sessionContinueVerificationNodeId.delete(sessionId);
    }
    if (shouldWriteSnapshotForSdkMessage(message)) {
      const sess = sessions!.getSession(sessionId);
      if (sess) {
        sessions!.writeMessageSnapshot(messageRowId, buildExportEnvironmentSnapshot(sess));
      }
    }
  }
  if (event.type === "stream.user_prompt") {
    const { sessionId, prompt } = event.payload;
    const promptRowId = sessions!.recordMessage(sessionId, {
      type: "user_prompt",
      prompt
    });
    const sess = sessions!.getSession(sessionId);
    if (sess) {
      sessions!.writeMessageSnapshot(promptRowId, buildExportEnvironmentSnapshot(sess));
    }
  }
  if (event.type === "session.messagesReset") {
    sessions.replaceMessages(event.payload.sessionId, event.payload.messages);
  }
  broadcast(event);
}

function isLegacySession(sessionId: string): boolean {
  const session = sessions.getSession(sessionId);
  return session?.engine === "legacy-claude";
}

function emitLegacyReadonlyError(sessionId: string, action: string) {
  broadcast({
    type: "runner.error",
    payload: {
      sessionId,
      message: `Cannot ${action}: this is a legacy Claude-backed session. Legacy sessions remain viewable but are read-only after the Pi migration.`
    }
  });
}

/** Remove persisted Pi session artifacts on disk for a deleted session. */
function cleanupPiSessionArtifacts(sessionId: string, session: Session | undefined): void {
  // Remove explicit session file if tracked.
  const piSessionFile = session?.piSessionFile?.trim();
  if (piSessionFile) {
    try {
      rmSync(piSessionFile, { force: true });
    } catch (error) {
      console.error(`[ipc] failed deleting pi session file for ${sessionId}:`, error);
    }
  }

  // Remove the whole per-session directory (~.../pi-agent/sessions/<sessionId>).
  try {
    const sessionDir = getPiSessionsDir(sessionId);
    rmSync(sessionDir, { recursive: true, force: true });
  } catch (error) {
    console.error(`[ipc] failed deleting pi session dir for ${sessionId}:`, error);
  }
}

function flattenWorkflowNodeIds(tree: WorkflowNode[]): string[] {
  const ids: string[] = [];
  const stack = [...tree];
  while (stack.length > 0) {
    const node = stack.pop()!;
    ids.push(node.id);
    for (const child of node.children ?? []) stack.push(child);
  }
  return ids;
}

function gatherUserPromptHistory(sessionId: string): string[] {
  return initializeSessions()
    .getMessages(sessionId)
    .filter((m): m is { type: "user_prompt"; prompt: string } => m.type === "user_prompt")
    .map((m) => String(m.prompt ?? "").trim())
    .filter(Boolean);
}

function verifiersEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

async function autoRefineVerifiersFromUserMessages(sessionId: string, targetNodeIds: string[]): Promise<void> {
  const store = initializeSessions();
  const session = store.getSession(sessionId);
  if (!session?.workflowTree?.length) return;
  if (targetNodeIds.length === 0) return;

  const promptHistory = gatherUserPromptHistory(sessionId);
  if (promptHistory.length === 0) return;

  const byNodeExamples = sessionVerifierExamplesByNodeId.get(sessionId) ?? new Map<string, VerifierExampleState>();
  let didChangeAny = false;

  for (const nodeId of targetNodeIds) {
    const node = findNodeById(session.workflowTree, nodeId);
    if (!node) continue;
    const examples = byNodeExamples.get(nodeId);
    try {
      const updated = await generateUpdatedVerifiersForNode(
        session,
        session.workflowTree,
        node,
        promptHistory,
        examples?.removed ?? [],
        examples?.added ?? []
      );
      if (!updated) continue;
      const nextVerifiers = uniqueByNormalized(updated);
      if (nextVerifiers.length === 0 || verifiersEqual(node.verifiers, nextVerifiers)) continue;
      node.verifiers = nextVerifiers;
      node.verifierMarks = nextVerifiers.map(() => undefined);
      didChangeAny = true;
    } catch (error) {
      console.error(`[ipc] auto verifier refinement failed for node ${nodeId}:`, error);
    }
  }

  if (!didChangeAny) return;

  store.updateSession(sessionId, { workflowTree: session.workflowTree });
  const treePayload = JSON.parse(JSON.stringify(session.workflowTree)) as WorkflowNode[];
  broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree: treePayload } });

  const rowId = store.recordMessage(sessionId, { type: "edit_verifier" });
  const sessAfter = store.getSession(sessionId);
  if (sessAfter) {
    store.writeMessageSnapshot(rowId, buildExportEnvironmentSnapshot(sessAfter));
  }
  broadcast({
    type: "stream.message",
    payload: { sessionId, message: { type: "edit_verifier" } },
  });
}

/** Starts a task-solving LLM call for the given workflow node. */
function triggerNodeSolve(sessionId: string, nodeId: string) {
  const store = initializeSessions();
  const session = store.getSession(sessionId);
  if (!session) return;
  if (session.status === "running") {
    broadcast({
      type: "runner.error",
      payload: { sessionId, message: "Session is already running. Please wait or stop it first." }
    });
    return;
  }
  if (!session.workflowTree?.length) {
    broadcast({
      type: "runner.error",
      payload: { sessionId, message: "No workflow tree yet. Send a message first to generate the workflow." }
    });
    return;
  }
  if (session.engine === "legacy-claude") {
    broadcast({
      type: "runner.error",
      payload: {
        sessionId,
        message: "Cannot solve node: this is a legacy Claude-backed session. Legacy sessions are read-only after the Pi migration."
      }
    });
    return;
  }

  const verificationDepth = session.verificationDepth ?? 0;
  const node = findNodeById(session.workflowTree, nodeId);
  if (!node) {
    broadcast({
      type: "runner.error",
      payload: { sessionId, message: `Node ${nodeId} not found in workflow tree.` }
    });
    return;
  }

  // If this node is above the verification depth and has children,
  // delegate to the first incomplete child at verification depth
  if (node.children.length > 0 && node.depth < verificationDepth) {
    const firstIncomplete = getNextIncompleteChild(node);
    if (!firstIncomplete) {
      broadcast({
        type: "runner.error",
        payload: { sessionId, message: "All children of this node are already completed." }
      });
      return;
    }
    triggerNodeSolve(sessionId, firstIncomplete.id);
    return;
  }

  // Execute this node directly (at or below verification depth, or leaf)
  const isRerun = node.status === "completed";
  let branchEntryId: string | undefined;
  let humanEdits: string | undefined;

  if (isRerun) {
    const resumeData = node.resumePoint;
    if (!resumeData || !("entryId" in resumeData) || !resumeData.entryId) {
      broadcast({
        type: "runner.error",
        payload: { sessionId, message: `Cannot rerun node: no resume point recorded.` }
      });
      return;
    }

    branchEntryId = resumeData.entryId;

    // Reset this node and all subsequent siblings + their children
    resetNode(node);

    const parent = findParentNode(session.workflowTree, nodeId);
    if (parent) {
      let foundSelf = false;
      for (const sibling of parent.children) {
        if (sibling.id === nodeId) { foundSelf = true; continue; }
        if (foundSelf) resetNode(sibling);
      }
      // Re-check parent status
      parent.status = "pending";
    }
    store.updateSession(sessionId, { workflowTree: session.workflowTree });

    // Compute human edits summary between originalOutputs snapshot and current file contents
    if (node.originalOutputs && node.originalOutputs.length > 0) {
      const cwd = session.cwd ?? process.cwd();
      const parts: string[] = [];
      for (const snap of node.originalOutputs) {
        try {
          const absPath = join(cwd, snap.path);
          const current = readFileSync(absPath, "utf8");
          if (current !== snap.content) {
            parts.push(
              [
                `File: ${snap.path}`,
                "",
                "(1) Original model output:",
                snap.content,
                "",
                "(2) Current version after human edits (USE THIS as the base for any further updates):",
                current,
              ].join("\n")
            );
          }
        } catch {
          // ignore if file missing or unreadable
        }
      }
      if (parts.length > 0) {
        humanEdits = parts.join("\n\n");
      }
    }

    const messages = store.getMessages(sessionId);
    broadcast({
      type: "session.messagesReset",
      payload: { sessionId, messages }
    });
  } else {
    try {
      const sessionManager = createPiSessionManager(session.id, session.cwd ?? process.cwd(), session.piSessionFile);
      const leafId = sessionManager.getLeafId();
      if (leafId) {
        node.resumePoint = { entryId: leafId };
        store.updateSession(sessionId, { workflowTree: session.workflowTree });
      }
    } catch {
      // Ignore missing session linkage; runner will still attempt the node solve.
    }
  }

  // Mark node as running
  node.status = "running";
  store.updateSession(sessionId, { workflowTree: session.workflowTree });
  broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree: session.workflowTree } });

  sessionCurrentNodeId.set(sessionId, nodeId);
  sessionLastVerificationNodeId.set(sessionId, nodeId);
  sessionContinueVerificationNodeId.delete(sessionId);
  const pathContext = getNodePath(session.workflowTree, nodeId);
  const nodePrompt = buildPromptForNode(node.description, pathContext, node.outputFiles, humanEdits, session.cwd);
  store.updateSession(sessionId, { status: "running", lastPrompt: nodePrompt });
  broadcast({
    type: "session.status",
    payload: { sessionId, status: "running", title: session.title, cwd: session.cwd }
  });
  // Persist the node-solving prompt in session message history (not just UI broadcast),
  // so downstream exporters can segment trajectories per workflow node.
  emit({
    type: "stream.user_prompt",
    payload: { sessionId, prompt: nodePrompt }
  });

  runClaude({
    prompt: nodePrompt,
    session,
    branchEntryId,
    onEvent: emit,
    onSessionUpdate: (updates) => {
      store.updateSession(session.id, updates);
    }
  })
    .then((handle) => {
      runnerHandles.set(session.id, handle);
    })
    .catch((error) => {
      store.updateSession(session.id, { status: "error" });
      broadcast({
        type: "session.status",
        payload: {
          sessionId: session.id,
          status: "error",
          title: session.title,
          cwd: session.cwd,
          error: String(error)
        }
      });
    });
}

export function handleClientEvent(event: ClientEvent) {
  const sessions = initializeSessions();

  if (event.type === "memory.read") {
    ensureMemoriesDir();
    const { requestId } = event.payload;
    const dir = getMemoriesDir();
    const sections = readAllMemorySections();
    const skillsDir = getAppSkillsDir();
    const skillSections = readAllFlatSkillSections();
    broadcast({
      type: "memory.readResult",
      payload: { requestId, dir, sections, skillsDir, skillSections },
    });
    return;
  }

  if (event.type === "memory.write") {
    const { requestId, sections, deletedFileNames } = event.payload;
    try {
      writeMemorySections(sections, deletedFileNames);
      broadcast({ type: "memory.writeResult", payload: { requestId, success: true } });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      broadcast({ type: "memory.writeResult", payload: { requestId, success: false, error: message } });
    }
    return;
  }

  if (event.type === "skills.write") {
    const { requestId, sections, deletedFileNames } = event.payload;
    try {
      const normalizedSections = sections
        .map((s) => ({
          fileName: String(s.fileName ?? "").trim(),
          content: s.content == null ? "" : String(s.content),
        }))
        .filter((s) => isValidFlatSkillMdFileName(s.fileName));
      const filteredDeletes = deletedFileNames?.filter((n) => isValidFlatSkillMdFileName(n));
      writeFlatSkillSections(normalizedSections, filteredDeletes);
      syncAppSkills();
      broadcast({ type: "skills.writeResult", payload: { requestId, success: true } });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      broadcast({ type: "skills.writeResult", payload: { requestId, success: false, error: message } });
    }
    return;
  }

  if (event.type === "session.recordBrainEdit") {
    const sid = String(event.payload.sessionId ?? "").trim();
    if (sid) {
      recordBrainEditAction(sid);
    }
    return;
  }

  if (event.type === "session.list") {
    emit({
      type: "session.list",
      payload: { sessions: sessions.listSessions() }
    });
    return;
  }

  if (event.type === "session.history") {
    const history = sessions.getSessionHistory(event.payload.sessionId);
    if (!history) {
      emit({ type: "session.deleted", payload: { sessionId: event.payload.sessionId } });
      return;
    }
    emit({
      type: "session.history",
      payload: {
        sessionId: history.session.id,
        status: history.session.status,
        messages: history.messages,
        workflowTree: history.session.workflowTree,
        verificationDepth: history.session.verificationDepth,
        title: history.session.title,
        engine: history.session.engine
      }
    });
    return;
  }

  if (event.type === "session.start") {
    const session = sessions.createSession({
      cwd: event.payload.cwd,
      title: event.payload.title,
      allowedTools: event.payload.allowedTools,
      prompt: event.payload.prompt,
      engine: "pi"
    });

    sessions.updateSession(session.id, {
      status: "running",
      lastPrompt: event.payload.prompt
    });
    emit({
      type: "session.status",
      payload: { sessionId: session.id, status: "running", title: session.title, cwd: session.cwd }
    });

    emit({
      type: "stream.user_prompt",
      payload: { sessionId: session.id, prompt: event.payload.prompt }
    });

    runClaude({
      prompt: event.payload.prompt,
      session,
      onEvent: emit,
      onSessionUpdate: (updates) => {
        sessions.updateSession(session.id, updates);
      }
    })
      .then((handle) => {
        runnerHandles.set(session.id, handle);
        sessions.setAbortController(session.id, undefined);
      })
      .catch((error) => {
        sessions.updateSession(session.id, { status: "error" });
        emit({
          type: "session.status",
          payload: {
            sessionId: session.id,
            status: "error",
            title: session.title,
            cwd: session.cwd,
            error: String(error)
          }
        });
      });

    return;
  }

  if (event.type === "session.continue") {
    const session = sessions.getSession(event.payload.sessionId);
    if (!session) {
      emit({ type: "session.deleted", payload: { sessionId: event.payload.sessionId } });
      emit({
        type: "runner.error",
        payload: { sessionId: event.payload.sessionId, message: "Session no longer exists." }
      });
      return;
    }

    if (session.engine === "legacy-claude") {
      emitLegacyReadonlyError(session.id, "continue this session");
      return;
    }

    sessions.updateSession(session.id, { status: "running", lastPrompt: event.payload.prompt });
    emit({
      type: "session.status",
      payload: { sessionId: session.id, status: "running", title: session.title, cwd: session.cwd }
    });

    emit({
      type: "stream.user_prompt",
      payload: { sessionId: session.id, prompt: event.payload.prompt }
    });

    const explicit = event.payload.verificationNodeId?.trim();
    const fallback = sessionLastVerificationNodeId.get(session.id);
    const vNode = explicit || fallback;
    if (vNode) {
      sessionLastVerificationNodeId.set(session.id, vNode);
      sessionContinueVerificationNodeId.set(session.id, vNode);
      // Force verifier refinement on each incoming user message.
      void autoRefineVerifiersFromUserMessages(session.id, [vNode]);
    } else {
      sessionContinueVerificationNodeId.delete(session.id);
      const allNodeIds = session.workflowTree?.length ? flattenWorkflowNodeIds(session.workflowTree) : [];
      if (allNodeIds.length > 0) {
        void autoRefineVerifiersFromUserMessages(session.id, allNodeIds);
      }
    }

    runClaude({
      prompt: event.payload.prompt,
      session,
      onEvent: emit,
      onSessionUpdate: (updates) => {
        sessions.updateSession(session.id, updates);
      }
    })
      .then((handle) => {
        runnerHandles.set(session.id, handle);
        sessions.setAbortController(session.id, undefined);
      })
      .catch((error) => {
        sessions.updateSession(session.id, { status: "error" });
        emit({
          type: "session.status",
          payload: {
            sessionId: session.id,
            status: "error",
            title: session.title,
            cwd: session.cwd,
            error: String(error)
          }
        });
      });

    return;
  }

  if (event.type === "session.solveNode") {
    const { sessionId, nodeId } = event.payload;
    if (isLegacySession(sessionId)) {
      emitLegacyReadonlyError(sessionId, "solve a node");
      return;
    }
    triggerNodeSolve(sessionId, nodeId);
    return;
  }

  if (event.type === "session.stop") {
    const session = sessions.getSession(event.payload.sessionId);
    if (!session) return;

    sessionContinueVerificationNodeId.delete(session.id);

    const handle = runnerHandles.get(session.id);
    if (handle) {
      handle.abort();
      runnerHandles.delete(session.id);
    }

    sessions.updateSession(session.id, { status: "idle" });
    emit({
      type: "session.status",
      payload: { sessionId: session.id, status: "idle", title: session.title, cwd: session.cwd }
    });
    return;
  }

  if (event.type === "session.updateWorkflowTree") {
    const { sessionId, workflowTree } = event.payload;
    const sessBefore = sessions.getSession(sessionId);
    const persistedTree = sessions.getPersistedWorkflowTree(sessionId);
    const oldTree =
      persistedTree !== undefined
        ? (JSON.parse(JSON.stringify(persistedTree)) as WorkflowNode[])
        : sessBefore?.workflowTree
          ? (JSON.parse(JSON.stringify(sessBefore.workflowTree)) as WorkflowNode[])
          : [];
    sessions.persistWorkflowTree(sessionId, workflowTree);
    if (hasLiveSession(sessionId)) {
      sessions.updateSession(sessionId, { workflowTree });
      broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree } });
    }
    const { workflow: wfEdit, verifier: verEdit } = classifyUserWorkflowTreeEdit(oldTree, workflowTree);
    const nodeExampleUpdates = collectVerifierExampleUpdates(oldTree, workflowTree);
    if (nodeExampleUpdates.size > 0) {
      const existing = sessionVerifierExamplesByNodeId.get(sessionId) ?? new Map<string, VerifierExampleState>();
      for (const [nodeId, delta] of nodeExampleUpdates.entries()) {
        const prev = existing.get(nodeId) ?? { removed: [], added: [] };
        existing.set(nodeId, {
          removed: uniqueByNormalized([...prev.removed, ...delta.removed]),
          added: uniqueByNormalized([...prev.added, ...delta.added]),
        });
      }
      sessionVerifierExamplesByNodeId.set(sessionId, existing);
    }
    const sessAfter = sessions.getSession(sessionId);
    if (sessAfter && wfEdit) {
      const rowId = sessions.recordMessage(sessionId, { type: "edit_workflow" });
      sessions.writeMessageSnapshot(rowId, buildExportEnvironmentSnapshot(sessAfter));
      broadcast({
        type: "stream.message",
        payload: { sessionId, message: { type: "edit_workflow" } },
      });
    } else if (sessAfter && verEdit) {
      const rowId = sessions.recordMessage(sessionId, { type: "edit_verifier" });
      sessions.writeMessageSnapshot(rowId, buildExportEnvironmentSnapshot(sessAfter));
      broadcast({
        type: "stream.message",
        payload: { sessionId, message: { type: "edit_verifier" } },
      });
    }
    return;
  }

  if (event.type === "session.updateVerificationDepth") {
    const { sessionId, verificationDepth } = event.payload;
    sessions.persistVerificationDepth(sessionId, verificationDepth);
    if (hasLiveSession(sessionId)) {
      sessions.updateSession(sessionId, { verificationDepth });
      broadcast({ type: "session.verificationDepth", payload: { sessionId, verificationDepth } });
    }
    return;
  }

  if (event.type === "session.updateTitle") {
    const { sessionId, title } = event.payload;
    sessions.persistTitle(sessionId, title);
    if (hasLiveSession(sessionId)) {
      sessions.updateSession(sessionId, { title });
      broadcast({ type: "session.title", payload: { sessionId, title } });
    }
    return;
  }

  if (event.type === "session.regenerateWorkflow") {
    const sessionId = event.payload.sessionId;
    const session = sessions.getSession(sessionId);
    if (!session) {
      emit({ type: "session.deleted", payload: { sessionId } });
      return;
    }
    if (session.engine === "legacy-claude") {
      emitLegacyReadonlyError(sessionId, "regenerate the workflow");
      return;
    }
    const messages = sessions.getMessages(sessionId);
    const userPrompts = messages
      .filter((m): m is { type: "user_prompt"; prompt: string } => m.type === "user_prompt")
      .map((m) => m.prompt.trim())
      .filter(Boolean);
    const taskSummary =
      userPrompts.length > 0
        ? userPrompts.map((p, i) => `Message ${i + 1}:\n${p}`).join("\n\n")
        : session.lastPrompt?.trim() || session.title || "Current task";
    sessions.updateSession(sessionId, { status: "running" });
    emit({
      type: "session.status",
      payload: { sessionId, status: "running", title: session.title, cwd: session.cwd }
    });
    runClaude({
      prompt: buildRegenerateWorkflowPrompt(taskSummary),
      session,
      regenerateWorkflow: true,
      onEvent: emit,
      onSessionUpdate: (updates) => {
        sessions.updateSession(sessionId, updates);
      }
    }).catch((error) => {
      sessions.updateSession(sessionId, { status: "idle" });
      emit({
        type: "session.status",
        payload: { sessionId, status: "idle", title: session.title, cwd: session.cwd }
      });
      emit({
        type: "runner.error",
        payload: { sessionId, message: String(error) }
      });
    });
    return;
  }

  if (event.type === "session.delete") {
    const sessionId = event.payload.sessionId;
    const session = sessions.getSession(sessionId);
    const handle = runnerHandles.get(sessionId);
    if (handle) {
      handle.abort();
      runnerHandles.delete(sessionId);
    }

    sessionCurrentNodeId.delete(sessionId);
    sessionContinueVerificationNodeId.delete(sessionId);
    sessionLastVerificationNodeId.delete(sessionId);
    sessionVerifierExamplesByNodeId.delete(sessionId);

    cleanupPiSessionArtifacts(sessionId, session);
    sessions.deleteSession(sessionId);
    emit({
      type: "session.deleted",
      payload: { sessionId }
    });
    return;
  }

  if (event.type === "permission.response") {
    const session = sessions.getSession(event.payload.sessionId);
    if (!session) return;

    const pending = session.pendingPermissions.get(event.payload.toolUseId);
    if (pending) {
      pending.resolve(event.payload.result);
    }
    return;
  }
}

/** If ``absNorm`` lies under session cwd, return posix relative path for storage/export; else absolute. */
function fileEditPathForMessage(sess: Session, absNorm: string): string {
  const cwd = sess.cwd?.trim();
  if (!cwd) return absNorm;
  try {
    const abs = resolve(absNorm);
    const root = resolve(cwd);
    const relPath = relative(root, abs);
    if (relPath && !relPath.startsWith("..") && !pathIsAbsolute(relPath)) {
      return relPath.replace(/\\/g, "/");
    }
  } catch {
    /* keep absolute */
  }
  return absNorm;
}

/** Called from main after a successful preview-panel ``write-file`` (``file_edit`` row + env snapshot including written HTML/text). */
export function recordFileEditAfterPreviewSave(
  sessionId: string,
  editedAbsPath: string,
  editedContent?: string
): void {
  const store = initializeSessions();
  const sess = store.getSession(sessionId);
  if (!sess) return;
  const pathNormAbs = editedAbsPath.replace(/\\/g, "/");
  const pathForMessage = fileEditPathForMessage(sess, pathNormAbs);
  const rowId = store.recordMessage(sessionId, { type: "file_edit", path: pathForMessage });
  const snapshot =
    typeof editedContent === "string"
      ? buildExportEnvironmentSnapshotWithPreviewWrittenFile(sess, pathNormAbs, editedContent)
      : buildExportEnvironmentSnapshot(sess);
  store.writeMessageSnapshot(rowId, snapshot);
  broadcast({
    type: "stream.message",
    payload: { sessionId, message: { type: "file_edit", path: pathForMessage } },
  });
}

export function cleanupAllSessions(): void {
  for (const [, handle] of runnerHandles) {
    handle.abort();
  }
  runnerHandles.clear();
  if (sessions) {
    sessions.close();
  }
}

export { sessions };
