import { BrowserWindow } from "electron";
import type { ClientEvent, ServerEvent, WorkflowNode } from "./types.js";
import { runClaude, buildPromptForNode, type RunnerHandle } from "./libs/runner.js";
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
    if (session && (!session.workflowTree || session.workflowTree.length === 0)) {
      // Set default verification depth to middle of tree
      const maxD = getMaxDepth(workflowTree);
      const defaultDepth = Math.max(0, Math.floor(maxD / 2));
      sessions.updateSession(sessionId, { workflowTree, verificationDepth: defaultDepth });
      broadcast({ type: "session.workflowTree", payload: { sessionId, workflowTree } });
      broadcast({ type: "session.verificationDepth", payload: { sessionId, verificationDepth: defaultDepth } });

      sessions.updateSession(sessionId, { status: "idle" });
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
  const nodePrompt = buildPromptForNode(node.description, pathContext);
  store.updateSession(sessionId, { status: "running", lastPrompt: nodePrompt });
  broadcast({
    type: "session.status",
    payload: { sessionId, status: "running", title: session.title, cwd: session.cwd }
  });
  broadcast({
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
