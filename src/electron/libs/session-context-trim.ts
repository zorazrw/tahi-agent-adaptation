import type { Agent } from "@mariozechner/pi-agent-core";
import {
  estimateTokens,
  getLatestCompactionEntry,
  type AgentSession,
  type SessionEntry,
  type SessionManager,
} from "@mariozechner/pi-coding-agent";

/** Max agent tool calls kept in LLM context during task execution (node solve / follow-ups). */
export const EXECUTION_CONTEXT_MAX_ACTIONS = 10;
/** Minimum prior agent tool calls retained during overflow trimming retries. */
export const MIN_EXECUTION_CONTEXT_ACTIONS = 2;

const CONTEXT_LENGTH_RE =
  /prompt length plus max_tokens|exceeds the (?:model's )?context window|exceeds the context window|prompt is too long|context window exceeds|too many tokens|token limit exceeded|maximum context length|input token count.*exceeds the maximum|too large for model with \d+ maximum context length/i;

export function isContextLengthExceededError(error: unknown): boolean {
  const msg =
    typeof error === "string"
      ? error
      : error instanceof Error
        ? error.message
        : error && typeof error === "object" && "errorMessage" in error
          ? String((error as { errorMessage?: unknown }).errorMessage ?? "")
          : String(error ?? "");
  return Boolean(msg && CONTEXT_LENGTH_RE.test(msg));
}

type AssistantTail = { stopReason?: string; errorMessage?: string };

function lastAssistant(agent: Agent): AssistantTail | undefined {
  const { messages } = agent.state;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role === "assistant") return m as AssistantTail;
  }
  return undefined;
}

function isContextOverflow(agent: Agent): boolean {
  const a = lastAssistant(agent);
  return a?.stopReason === "error" && Boolean(a.errorMessage && isContextLengthExceededError(a.errorMessage));
}

function assertTurnSucceeded(agent: Agent): void {
  const a = lastAssistant(agent);
  if (a?.stopReason === "error" && !isContextLengthExceededError(a.errorMessage)) {
    throw new Error(a.errorMessage ?? "Assistant turn failed");
  }
}

function findLastUserEntryId(path: SessionEntry[]): string | null {
  for (let i = path.length - 1; i >= 0; i--) {
    const e = path[i];
    if (e.type === "message" && e.message.role === "user") return e.id;
  }
  return null;
}

/** Branch leaf to latest user so a failed assistant turn is not kept in compacted context. */
function rewindToLastUser(sessionManager: SessionManager): string | null {
  const leafId = sessionManager.getLeafId();
  if (!leafId) return null;
  const userId = findLastUserEntryId(sessionManager.getBranch(leafId));
  if (userId) sessionManager.branch(userId);
  return userId;
}

/** ``agent.continue()`` needs last message to be ``user`` or ``toolResult``. */
function prepareAgentForContinue(agent: Agent): void {
  const messages = agent.state.messages.slice();
  while (messages.length > 0 && messages[messages.length - 1]?.role === "assistant") {
    messages.pop();
  }
  agent.replaceMessages(messages);
}

function appendCompactionIfNew(
  sessionManager: SessionManager,
  firstKeptEntryId: string,
  summary: string
): boolean {
  const leafId = sessionManager.getLeafId();
  if (!leafId) return false;
  const path = sessionManager.getBranch(leafId);
  if (getLatestCompactionEntry(path)?.firstKeptEntryId === firstKeptEntryId) return false;
  const messages = sessionManager.buildSessionContext().messages;
  const tokensBefore = messages.reduce((sum, msg) => sum + estimateTokens(msg), 0);
  sessionManager.appendCompaction(summary, firstKeptEntryId, tokensBefore);
  return true;
}

async function continueAfterTrim(piSession: AgentSession): Promise<void> {
  const { messages } = piSession.sessionManager.buildSessionContext();
  piSession.agent.replaceMessages(messages);
  prepareAgentForContinue(piSession.agent);
  await piSession.agent.continue();
  while (piSession.isRetrying) {
    await new Promise((r) => setTimeout(r, 50));
  }
}

async function trimAndContinue(
  piSession: AgentSession,
  maxActions: { value: number },
  overflowMessage?: string
): Promise<void> {
  const sm = piSession.sessionManager;
  const userEntryId = rewindToLastUser(sm);

  let trimmed = false;
  if (maxActions.value > MIN_EXECUTION_CONTEXT_ACTIONS) {
    const target = maxActions.value - 1;
    if (trimSessionToLastAgentActions(sm, target)) {
      maxActions.value = target;
      trimmed = true;
    }
  }
  if (!trimmed && userEntryId) {
    trimmed = appendCompactionIfNew(
      sm,
      userEntryId,
      "Earlier conversation is omitted from context to fit the model window. Continue from the current task."
    );
  }
  if (!trimmed) {
    throw new Error(overflowMessage ?? "Context length exceeded; could not trim session further.");
  }
  await continueAfterTrim(piSession);
}

/**
 * Task-execution prompt with retry: on context-window errors, compact older tool turns and
 * ``agent.continue()`` (never re-send the user message).
 */
export async function promptWithExecutionContextRetry(
  piSession: AgentSession,
  prompt: string,
  initialMaxActions: number
): Promise<void> {
  const maxActions = { value: initialMaxActions };

  try {
    await piSession.prompt(prompt);
  } catch (error) {
    if (!isContextLengthExceededError(error)) throw error;
    await trimAndContinue(piSession, maxActions, error instanceof Error ? error.message : String(error));
  }

  for (let i = 0; i < initialMaxActions + 3; i++) {
    assertTurnSucceeded(piSession.agent);
    if (!isContextOverflow(piSession.agent)) return;
    await trimAndContinue(piSession, maxActions, lastAssistant(piSession.agent)?.errorMessage);
  }

  assertTurnSucceeded(piSession.agent);
  if (isContextOverflow(piSession.agent)) {
    throw new Error("Context length exceeded after trimming execution context.");
  }
}

function countToolCallsInEntry(entry: SessionEntry): number {
  if (entry.type !== "message") return 0;
  const msg = entry.message;
  if (msg.role !== "assistant" || !Array.isArray(msg.content)) return 0;
  let n = 0;
  for (const block of msg.content) {
    if (!block || typeof block !== "object") continue;
    const type = (block as { type?: string }).type;
    if (type === "toolCall" || type === "tool_use") n += 1;
  }
  return n;
}

/** Entry id of the oldest assistant message to keep when retaining the last ``maxActions`` tool calls. */
export function findFirstKeptEntryIdForLastActions(path: SessionEntry[], maxActions: number): string | null {
  if (maxActions <= 0) return null;
  let seen = 0;
  for (let i = path.length - 1; i >= 0; i--) {
    const calls = countToolCallsInEntry(path[i]);
    if (calls === 0) continue;
    seen += calls;
    if (seen >= maxActions) return path[i].id;
  }
  return null;
}

/** Drop older tool history via compaction, keeping the last ``maxActions`` tool calls. */
export function trimSessionToLastAgentActions(
  sessionManager: SessionManager,
  maxActions = EXECUTION_CONTEXT_MAX_ACTIONS
): boolean {
  const leafId = sessionManager.getLeafId();
  if (!leafId) return false;
  const path = sessionManager.getBranch(leafId);
  const total = path.reduce((sum, e) => sum + countToolCallsInEntry(e), 0);
  if (total <= maxActions) return false;
  const firstKeptEntryId = findFirstKeptEntryIdForLastActions(path, maxActions);
  if (!firstKeptEntryId) return false;
  return appendCompactionIfNew(
    sessionManager,
    firstKeptEntryId,
    `Earlier agent actions (${total - maxActions} tool call(s)) are omitted from context. Continue from the last ${maxActions} actions.`
  );
}
