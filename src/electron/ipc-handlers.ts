import { BrowserWindow } from "electron";
import type { ClientEvent, ServerEvent, WorkflowNode } from "./types.js";
import { runClaude, buildPromptForNode, buildRegenerateWorkflowPrompt, type RunnerHandle } from "./libs/runner.js";
import { SessionStore } from "./libs/session-store.js";
import {
  findNodeById,
  findParentNode,
  getNextIncompleteChild,
  isNodeFullyComplete,
  getMaxDepth,
  getNodePath,
  updateNodeStatus,
  resetNode,
  completeNodeAndDescendants,
} from "./libs/workflow-tree-utils.js";
import { app } from "electron";
import { join } from "path";
import { readFileSync } from "fs";
import { ensureMemoriesDir, readAllMemorySections, writeMemorySections, getMemoriesDir } from "./libs/memory-store.js";
import {
  readAllFlatSkillSections,
  writeFlatSkillSections,
  getAppSkillsDir,
  syncAppSkills,
  isValidFlatSkillMdFileName,
} from "./libs/skill-store.js";
import { runExportAndExtractContext, setContextInductionNotifier } from "./libs/context-export.js";

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
  if (!sessions) return false;
  return Boolean(sessions.getSession(sessionId));
}


function emit(event: ServerEvent) {
  if (
    (event.type === "session.status" ||
      event.type === "stream.message" ||
      event.type === "stream.user_prompt" ||
      event.type === "permission.request") &&
    !hasLiveSession(event.payload.sessionId)
  ) {
    return;
  }

  if (event.type === "session.status") {
    sessions.updateSession(event.payload.sessionId, { status: event.payload.status });
  }
  if (event.type === "workflow.plan") {
    const { sessionId, workflowTree } = event.payload;
    const session = sessions.getSession(sessionId);
    if (session) {
      const maxD = getMaxDepth(workflowTree);
      const defaultDepth = Math.max(0, Math.floor(maxD / 2));
      sessions.updateSession(sessionId, {
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
    }
    return;
  }
  if (event.type === "stream.message") {
    const { sessionId, message } = event.payload;
    sessions.recordMessage(sessionId, message);

    // When a node-solving run completes, mark that node as completed and handle auto-cascade.
    const m = message as { type?: string; subtype?: string };
    if (m.type === "result") {
      const nodeId = sessionCurrentNodeId.get(sessionId);
      if (nodeId !== undefined) {
        sessionCurrentNodeId.delete(sessionId);
        if (m.subtype === "success") {
          const session = sessions.getSession(sessionId);
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

              // Mark this node and all its descendants as completed
              completeNodeAndDescendants(completedNode);
            }

            // Bubble up: mark parents complete if all children are done
            let parentNode = findParentNode(session.workflowTree, nodeId);
            while (parentNode && isNodeFullyComplete(parentNode)) {
              parentNode.status = "completed";
              parentNode = findParentNode(session.workflowTree, parentNode.id);
            }

            sessions.updateSession(sessionId, { workflowTree: session.workflowTree });
            broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree: session.workflowTree } });
          }
          broadcast({ type: "session.nodeCompleted", payload: { sessionId, nodeId } });
          runExportAndExtractContext(sessionId, nodeId);
        }
      }
    }
  }
  if (event.type === "stream.user_prompt") {
    sessions.recordMessage(event.payload.sessionId, {
      type: "user_prompt",
      prompt: event.payload.prompt
    });
  }
  broadcast(event);
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
  if (!session.claudeSessionId) {
    broadcast({
      type: "runner.error",
      payload: { sessionId, message: "Cannot solve node: session has no resume id yet." }
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
  let resumeSessionAt: string | undefined;
  let claudeSessionIdForResume = session.claudeSessionId;
  let humanEdits: string | undefined;

  if (isRerun) {
    const resumeData = node.resumePoint;
    if (!resumeData?.uuid) {
      broadcast({
        type: "runner.error",
        payload: { sessionId, message: `Cannot rerun node: no resume point recorded.` }
      });
      return;
    }

    resumeSessionAt = resumeData.uuid;
    claudeSessionIdForResume = resumeData.claudeSessionId ?? session.claudeSessionId;

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

    store.deleteMessagesAfter(sessionId, resumeData.uuid);
    store.updateSession(sessionId, {
      workflowTree: session.workflowTree,
      claudeSessionId: claudeSessionIdForResume
    });

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
    // First run: record resume point
    const lastUuid = store.getLastAssistantMessageUuid(sessionId);
    if (lastUuid) {
      node.resumePoint = { uuid: lastUuid, claudeSessionId: session.claudeSessionId! };
      store.updateSession(sessionId, { workflowTree: session.workflowTree });
    }
  }

  // Mark node as running
  node.status = "running";
  store.updateSession(sessionId, { workflowTree: session.workflowTree });
  broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree: session.workflowTree } });

  sessionCurrentNodeId.set(sessionId, nodeId);
  const pathContext = getNodePath(session.workflowTree, nodeId);
  const nodePrompt = buildPromptForNode(node.description, pathContext, node.outputFiles, humanEdits);
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
    resumeSessionId: claudeSessionIdForResume,
    resumeSessionAt,
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
        title: history.session.title
      }
    });
    return;
  }

  if (event.type === "session.start") {
    const session = sessions.createSession({
      cwd: event.payload.cwd,
      title: event.payload.title,
      allowedTools: event.payload.allowedTools,
      prompt: event.payload.prompt
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
      resumeSessionId: session.claudeSessionId,
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

    if (!session.claudeSessionId) {
      emit({
        type: "runner.error",
        payload: { sessionId: session.id, message: "Session has no resume id yet." }
      });
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

    runClaude({
      prompt: event.payload.prompt,
      session,
      resumeSessionId: session.claudeSessionId,
      onEvent: emit,
      onSessionUpdate: (updates) => {
        sessions.updateSession(session.id, updates);
      }
    })
      .then((handle) => {
        runnerHandles.set(session.id, handle);
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
    triggerNodeSolve(sessionId, nodeId);
    return;
  }

  if (event.type === "session.stop") {
    const session = sessions.getSession(event.payload.sessionId);
    if (!session) return;

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
    sessions.persistWorkflowTree(sessionId, workflowTree);
    if (hasLiveSession(sessionId)) {
      sessions.updateSession(sessionId, { workflowTree });
      broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree } });
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
    const handle = runnerHandles.get(sessionId);
    if (handle) {
      handle.abort();
      runnerHandles.delete(sessionId);
    }

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
