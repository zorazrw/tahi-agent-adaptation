import { BrowserWindow } from "electron";
import type { ClientEvent, ServerEvent } from "./types.js";
import { runClaude, buildPromptForStep, type RunnerHandle } from "./libs/runner.js";
import { SessionStore } from "./libs/session-store.js";
import { app } from "electron";
import { join } from "path";

let sessions: SessionStore;
const runnerHandles = new Map<string, RunnerHandle>();

/** While a step-solving run is in progress, maps sessionId -> stepIndex so we can emit stepCompleted on result. */
const sessionCurrentStepIndex = new Map<string, number>();

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
  // If a session was deleted, drop late events that would resurrect it in the UI.
  // (Session history lookups are DB-backed, so these late events commonly lead to "Unknown session".)
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
    const { sessionId, steps, outputFiles, verificationCriteria } = event.payload;
    const session = sessions.getSession(sessionId);
    if (session && !session.steps?.length) {
      sessions.updateSession(sessionId, { steps, outputFiles, verificationCriteria });
      broadcast({ type: "session.steps", payload: { sessionId, steps } });
      broadcast({ type: "session.outputFiles", payload: { sessionId, outputFiles } });
      broadcast({ type: "session.verificationCriteria", payload: { sessionId, verificationCriteria } });

      // Transition to idle so the human can drive step-by-step execution.
      // The runner will abort itself after the tool result is committed to the session.
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
    // When a step-solving run completes, mark that step as completed and persist.
    const m = message as { type?: string; subtype?: string };
    if (m.type === "result") {
      const stepIndex = sessionCurrentStepIndex.get(sessionId);
      if (stepIndex !== undefined) {
        sessionCurrentStepIndex.delete(sessionId);
        if (m.subtype === "success") {
          const session = sessions.getSession(sessionId);
          if (session) {
            const completed = [...(session.completedStepIndices ?? []), stepIndex].sort((a, b) => a - b);
            sessions.updateSession(sessionId, { completedStepIndices: completed });
          }
          broadcast({ type: "session.stepCompleted", payload: { sessionId, stepIndex } });
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

/** Extract uuid (and optional claudeSessionId) from a resume point, handling legacy string format. */
function parseResumePoint(
  point: string | { uuid: string; claudeSessionId: string } | undefined
): { uuid: string; claudeSessionId?: string } | undefined {
  if (!point) return undefined;
  if (typeof point === "string") return { uuid: point };
  return point;
}

/** Starts a task-solving LLM call for the given workflow step index (0-based). */
function triggerStepSolve(sessionId: string, stepIndex: number) {
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
  if (!session.steps?.length || stepIndex < 0 || stepIndex >= session.steps.length) {
    broadcast({
      type: "runner.error",
      payload: { sessionId, message: "No workflow steps yet. Send a message first to generate the workflow." }
    });
    return;
  }
  if (!session.claudeSessionId) {
    broadcast({
      type: "runner.error",
      payload: { sessionId, message: "Cannot solve step: session has no resume id yet." }
    });
    return;
  }

  const isRerun = session.completedStepIndices?.includes(stepIndex) ?? false;
  let resumeSessionAt: string | undefined;
  // For reruns, use the claudeSessionId saved at the resume point (not the current one,
  // which may have been overwritten by a previous failed rerun attempt).
  let claudeSessionIdForResume = session.claudeSessionId;

  if (isRerun) {
    // --- Rerun: rewind conversation to before this step ---
    const resumeData = parseResumePoint(session.stepResumePoints?.[stepIndex]);
    if (!resumeData?.uuid) {
      broadcast({
        type: "runner.error",
        payload: { sessionId, message: `Cannot rerun step ${stepIndex + 1}: no resume point recorded.` }
      });
      return;
    }

    resumeSessionAt = resumeData.uuid;
    claudeSessionIdForResume = resumeData.claudeSessionId ?? session.claudeSessionId;

    // Clear completed indices for this step and all subsequent steps
    const newCompleted = (session.completedStepIndices ?? []).filter(i => i < stepIndex);

    // Clear resume points for steps >= stepIndex (keep the current one for potential re-rerun)
    const newResumePoints: Record<number, string | { uuid: string; claudeSessionId: string }> = {};
    for (const [k, v] of Object.entries(session.stepResumePoints ?? {})) {
      if (Number(k) <= stepIndex) newResumePoints[Number(k)] = v;
    }

    // Delete messages from DB after the resume point
    store.deleteMessagesAfter(sessionId, resumeData.uuid);

    // Restore the pre-step claudeSessionId so future attempts use the correct session
    store.updateSession(sessionId, {
      completedStepIndices: newCompleted,
      stepResumePoints: newResumePoints,
      claudeSessionId: claudeSessionIdForResume
    });

    // Reload truncated messages and broadcast reset to UI
    const messages = store.getMessages(sessionId);
    broadcast({
      type: "session.messagesReset",
      payload: { sessionId, messages, completedStepIndices: newCompleted }
    });
  } else {
    // --- First run: record resume point (assistant UUID + claudeSessionId) for this step ---
    const lastUuid = store.getLastAssistantMessageUuid(sessionId);
    if (lastUuid) {
      const newResumePoints = {
        ...(session.stepResumePoints ?? {}),
        [stepIndex]: { uuid: lastUuid, claudeSessionId: session.claudeSessionId }
      };
      store.updateSession(sessionId, { stepResumePoints: newResumePoints });
    }
  }

  sessionCurrentStepIndex.set(sessionId, stepIndex);
  const stepPrompt = buildPromptForStep(session.steps[stepIndex], stepIndex, session.steps.length);
  store.updateSession(sessionId, { status: "running", lastPrompt: stepPrompt });
  broadcast({
    type: "session.status",
    payload: { sessionId, status: "running", title: session.title, cwd: session.cwd }
  });
  broadcast({
    type: "stream.user_prompt",
    payload: { sessionId, prompt: stepPrompt }
  });

  runClaude({
    prompt: stepPrompt,
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
  // Initialize sessions on first event
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
      // Session may have been deleted (or deleted concurrently). Treat as a sync event rather than an error toast.
      emit({ type: "session.deleted", payload: { sessionId: event.payload.sessionId } });
      return;
    }
    emit({
      type: "session.history",
      payload: {
        sessionId: history.session.id,
        status: history.session.status,
        messages: history.messages,
        steps: history.session.steps,
        completedStepIndices: history.session.completedStepIndices,
        outputFiles: history.session.outputFiles,
        verificationCriteria: history.session.verificationCriteria,
        verifierMarks: history.session.verifierMarks,
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

  if (event.type === "session.solveStep") {
    const { sessionId, stepIndex } = event.payload;
    triggerStepSolve(sessionId, stepIndex);
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

  if (event.type === "session.updateSteps") {
    const { sessionId, steps } = event.payload;
    // Always persist to DB so steps survive relaunch (even if session not in memory yet).
    sessions.persistSteps(sessionId, steps);
    if (hasLiveSession(sessionId)) {
      sessions.updateSession(sessionId, { steps });
      broadcast({ type: "session.steps", payload: { sessionId, steps } });
    }
    return;
  }

  if (event.type === "session.updateVerificationCriteria") {
    const { sessionId, verificationCriteria } = event.payload;
    sessions.persistVerificationCriteria(sessionId, verificationCriteria);
    if (hasLiveSession(sessionId)) {
      sessions.updateSession(sessionId, { verificationCriteria });
      broadcast({ type: "session.verificationCriteria", payload: { sessionId, verificationCriteria } });
    }
    return;
  }

  if (event.type === "session.updateVerifierMarks") {
    const { sessionId, verifierMarks } = event.payload;
    sessions.persistVerifierMarks(sessionId, verifierMarks);
    if (hasLiveSession(sessionId)) {
      sessions.updateSession(sessionId, { verifierMarks });
      broadcast({ type: "session.verifierMarks", payload: { sessionId, verifierMarks } });
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

    // Always try to delete and emit deleted event
    // Don't emit error if session doesn't exist - it may have already been deleted
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
