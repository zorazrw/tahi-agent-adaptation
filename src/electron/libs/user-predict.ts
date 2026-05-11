import { existsSync, readFileSync } from "fs";
import { join, resolve } from "path";
import type {
  PredictedUserActionSuggestion,
  StreamMessage,
  UserPredictionJudgeResult,
  WorkflowNode,
} from "../types.js";
import { runPiTextPrompt } from "./pi-prompt.js";

type ExportedTrajectoryStep = {
  actor?: string;
  action?: string;
  environment?: {
    workflow?: Array<{
      description?: string;
      status?: string;
      verifiers?: Array<{ criterion?: string; status?: string }>;
      children?: unknown[];
    }>;
  };
};

function extractJsonObject(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced?.[1]?.trim() || trimmed;
  const first = candidate.indexOf("{");
  const last = candidate.lastIndexOf("}");
  if (first === -1 || last === -1 || last <= first) return null;
  try {
    const parsed = JSON.parse(candidate.slice(first, last + 1));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

function clamp01(value: unknown, fallback = 0.5): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(1, n));
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function normalizeActionType(value: unknown): PredictedUserActionSuggestion["actionType"] {
  const raw = asString(value).trim().toLowerCase();
  if (
    raw === "message" ||
    raw === "edit_workflow" ||
    raw === "edit_verifier" ||
    raw === "file_edit" ||
    raw === "brain_edit" ||
    raw === "stop"
  ) {
    return raw;
  }
  return "unknown";
}

/** True when `AGENT_COWORK_FAKE_USER_PREDICT` is set (see `maybeFakeUserPredictAction`). */
export function isFakeUserPredictEnabled(): boolean {
  return Boolean(process.env.AGENT_COWORK_FAKE_USER_PREDICT?.trim());
}

/**
 * TEMPORARY UI testing: skip the LLM and return a fixed prediction.
 * Set env `AGENT_COWORK_FAKE_USER_PREDICT` to one of:
 * `message` | `edit_workflow` | `edit_verifier` | `file_edit` | `edit_file` (alias) | `brain_edit` | `stop` | `unknown`
 * Example: `AGENT_COWORK_FAKE_USER_PREDICT=file_edit npm run dev`
 * Remove this hook when finished testing.
 */
function maybeFakeUserPredictAction(): PredictedUserActionSuggestion | null {
  const raw = process.env.AGENT_COWORK_FAKE_USER_PREDICT?.trim();
  if (!raw) return null;

  const lowered = raw.toLowerCase();
  const aliased = lowered === "edit_file" ? "file_edit" : lowered;
  const actionType = normalizeActionType(aliased);

  const fixed: Record<
    PredictedUserActionSuggestion["actionType"],
    { draftText: string; rationale: string }
  > = {
    message: {
      draftText: "[Fake UI test] Tighten the legend spacing and bump axis label font one step.",
      rationale: "Placeholder: predicted next user chat for layout polish (testing message + Tab path).",
    },
    edit_workflow: {
      draftText:
        "[Fake UI test] Accept adds a new root step “Visualization” in Progress (workflow plan update).",
      rationale:
        "Placeholder: edit_workflow. With AGENT_COWORK_FAKE_USER_PREDICT=edit_workflow, Accept appends that step to the tree.",
    },
    edit_verifier: {
      draftText: "[Fake UI test] User would open the verifier panel and relax the “no duplicate dates” check.",
      rationale: "Placeholder: edit_verifier (testing non-message layout).",
    },
    file_edit: {
      draftText: "[Fake UI test] User would open `artifacts/chart.html` for a direct HTML tweak.",
      rationale: "Placeholder: file_edit (testing draft block with path-like text).",
    },
    brain_edit: {
      draftText: "[Fake UI test] User would edit project memory (e.g. AGENTS.md or brain panel).",
      rationale: "Placeholder: brain_edit.",
    },
    stop: {
      draftText: "",
      rationale: "Placeholder: user is done for this turn (testing stop + empty draft copy).",
    },
    unknown: {
      draftText: "[Fake UI test] Model could not classify the next surface action.",
      rationale: "Placeholder: unknown actionType (testing fallback styling).",
    },
  };

  const row = fixed[actionType];
  return {
    actionType,
    draftText: row.draftText,
    confidence: 0.77,
    rationale: row.rationale,
    rawResponse: JSON.stringify({ fakeUserPredict: true, requested: raw, actionType }),
  };
}

function normalizeJudgeVerdict(value: unknown): UserPredictionJudgeResult["verdict"] {
  const raw = asString(value).trim().toLowerCase();
  if (raw === "accurate" || raw === "partially_accurate" || raw === "inaccurate") {
    return raw;
  }
  return "inaccurate";
}

export function loadUserProfileMarkdown(cwd?: string): { profileMarkdown: string; profilePath?: string } {
  const candidates = [
    cwd ? resolve(cwd, "USER_PROFILE.md") : "",
    resolve(process.cwd(), "USER_PROFILE.md"),
  ].filter(Boolean);

  for (const path of candidates) {
    if (existsSync(path)) {
      return {
        profileMarkdown: readFileSync(path, "utf8"),
        profilePath: path,
      };
    }
  }

  return { profileMarkdown: "" };
}

function summarizeWorkflowNodes(
  nodes: WorkflowNode[] | Array<{ description?: string; status?: string; verifiers?: Array<{ criterion?: string; status?: string }>; children?: unknown[] }>,
  depth = 0
): string[] {
  const lines: string[] = [];
  for (const node of nodes ?? []) {
    const indent = "  ".repeat(depth);
    const desc = String(node.description ?? "").trim() || "(unnamed)";
    const status = String(node.status ?? "unknown");
    lines.push(`${indent}- ${desc} [${status}]`);
    const verifiers = Array.isArray(node.verifiers) ? node.verifiers : [];
    for (const verifier of verifiers) {
      const criterion =
        typeof verifier === "string"
          ? verifier
          : String((verifier as { criterion?: unknown }).criterion ?? "").trim();
      const vStatus =
        typeof verifier === "string"
          ? ""
          : String((verifier as { status?: unknown }).status ?? "").trim();
      if (criterion) {
        lines.push(`${indent}  verifier: ${criterion}${vStatus ? ` [${vStatus}]` : ""}`);
      }
    }
    const children = Array.isArray(node.children) ? (node.children as typeof nodes) : [];
    lines.push(...summarizeWorkflowNodes(children, depth + 1));
  }
  return lines;
}

function assistantBlocksToText(message: StreamMessage): string {
  if (message.type !== "assistant") return "";
  if ("blocks" in message && Array.isArray(message.blocks)) {
    return message.blocks
      .map((block) => {
        if (!block || typeof block !== "object") return "";
        if (block.type === "text") return block.text;
        if (block.type === "tool_use") return `[tool:${block.name}]`;
        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  const legacyMessage = "message" in message ? message.message : undefined;
  const content = legacyMessage?.content;
  return Array.isArray(content)
    ? content
        .map((block) => {
          if (!block || typeof block !== "object") return "";
          if (block.type === "text" && "text" in block) return String(block.text ?? "");
          if (block.type === "tool_use" && "name" in block) return `[tool:${String(block.name ?? "")}]`;
          return "";
        })
        .filter(Boolean)
        .join("\n")
        .trim()
    : "";
}

export function buildTranscriptFromStreamMessages(messages: StreamMessage[], limit = 14): string {
  const recent = messages.slice(-limit);
  const lines: string[] = [];
  for (const message of recent) {
    switch (message.type) {
      case "user_prompt":
        lines.push(`User: ${message.prompt}`);
        break;
      case "assistant": {
        const text = assistantBlocksToText(message);
        if (text) lines.push(`Agent: ${text}`);
        break;
      }
      case "run_result":
        lines.push(`System: run_result [${message.status}]`);
        break;
      case "node_completed":
        lines.push(`System: node_completed [${message.nodeLabel}]`);
        break;
      case "file_edit":
        lines.push(`UserAction: file_edit(${message.path})`);
        break;
      case "edit_workflow":
      case "edit_verifier":
      case "brain_edit":
        lines.push(`UserAction: ${message.type}`);
        break;
      case "tool_result":
        lines.push(`ToolResult: ${message.toolName}${message.isError ? " [error]" : ""}`);
        break;
      default:
        break;
    }
  }
  return lines.join("\n");
}

export function buildTranscriptFromExportedSteps(steps: ExportedTrajectoryStep[], limit = 14): string {
  const recent = steps.slice(-limit);
  return recent
    .map((step) => {
      const actor = step.actor === "agent" ? "Agent" : "User";
      return `${actor}: ${String(step.action ?? "").trim()}`;
    })
    .join("\n");
}

export function buildWorkflowSummaryFromExportedSteps(steps: ExportedTrajectoryStep[]): string {
  for (let i = steps.length - 1; i >= 0; i--) {
    const workflow = steps[i]?.environment?.workflow;
    if (Array.isArray(workflow) && workflow.length > 0) {
      return summarizeWorkflowNodes(workflow).join("\n");
    }
  }
  return "";
}

function buildPredictionPrompt(args: {
  userProfileMarkdown: string;
  transcript: string;
  workflowSummary?: string;
  sessionTitle?: string;
}): string {
  return [
    "You are predicting Zora's most likely immediate next action in Agent Cowork.",
    "Return ONLY JSON.",
    'Schema: {"actionType":"file_edit|edit_verifier|edit_workflow|brain_edit|message|stop|unknown","draftText":"string","confidence":0.0,"rationale":"string"}',
    "Use actionType \"stop\" liberally when appropriate: if the agent's latest output plausibly satisfies the user's original request, if remaining issues are minor/optional, or if another prompt would mostly be scope creep — prefer \"stop\" over speculative follow-ups.",
    "When uncertain between \"stop\" and a vague or low-value \"message\", choose \"stop\". Reserve \"message\" for clear, necessary next steps that stay on-task.",
    "Use actionType \"stop\" when the user is most likely done for now (satisfied, switching tasks, or not sending another prompt/structural edit). This is intentional completion — not a model error.",
    "Use actionType \"unknown\" only when the next move is unclear or does not fit message, workflow/verifier edits, file_edit, brain_edit, or stop.",
    "Anchor on the user's initial request (usually the first substantive \"User:\" turn in the transcript). Do not predict draftText or any next action that pivots to a new goal, new deliverable, unrelated benchmark, or \"improvement\" beyond what that initial request asked for.",
    "Never predict a message whose purpose is to drag the task toward unrelated exploration, extra features, or reframing the assignment. If the work already matches the initial ask, use \"stop\" instead of inventing follow-on work.",
    "Predicted edit_workflow, edit_verifier, file_edit, or brain_edit must likewise serve the same initial request — not introduce a new assignment or tangent.",
    "Do not default to actionType \"message\". Pick the single best type. Use \"message\" only when the next step is most likely a normal chat prompt.",
    "Use edit_workflow when the user would likely adjust the task tree or steps; edit_verifier when they would tweak criteria or checks; file_edit when they would open and edit a specific file path next; brain_edit when they would change saved memory/skills.",
    "If the workflow summary shows failing or stale verifiers the user would plausibly fix before chatting, prefer edit_verifier or edit_workflow over message.",
    "The transcript uses \"User:\" for chat prompts and \"UserAction: edit_workflow\" / \"UserAction: edit_verifier\" / \"UserAction: file_edit(path)\" / \"UserAction: brain_edit\" for non-chat actions. Prefer the action type that matches how the user has been acting.",
    "draftText should be the most likely next user message if actionType is message.",
    "If actionType is stop, draftText should be empty.",
    "If the likely action is not message, draftText may be empty or a short representative utterance.",
    "Optimize for the immediate next move, not the long-term intent.",
    "",
    args.sessionTitle ? `Session title: ${args.sessionTitle}` : "",
    "USER PROFILE (markdown):",
    args.userProfileMarkdown || "(none)",
    "",
    "RECENT CONVERSATION / ACTION CONTEXT:",
    args.transcript || "(none)",
    "",
    args.workflowSummary
      ? ["WORKFLOW / VERIFIER SUMMARY:", args.workflowSummary, ""].join("\n")
      : "",
    "Choose the actionType that best matches the next single user move among message, edit_workflow, edit_verifier, file_edit, brain_edit, stop, and unknown.",
  ]
    .filter(Boolean)
    .join("\n");
}

export async function predictNextUserAction(args: {
  cwd: string;
  userProfileMarkdown: string;
  transcript: string;
  workflowSummary?: string;
  sessionTitle?: string;
}): Promise<PredictedUserActionSuggestion> {
  const fake = maybeFakeUserPredictAction();
  if (fake) return fake;

  const rawResponse = await runPiTextPrompt({
    cwd: args.cwd,
    prompt: buildPredictionPrompt(args),
  });

  const parsed = extractJsonObject(rawResponse);
  if (!parsed) {
    return {
      actionType: "unknown",
      draftText: "",
      confidence: 0.2,
      rationale: rawResponse.trim() || "Model returned an unparsable prediction.",
      rawResponse,
    };
  }

  return {
    actionType: normalizeActionType(parsed.actionType),
    draftText: asString(parsed.draftText).trim(),
    confidence: clamp01(parsed.confidence, 0.5),
    rationale: asString(parsed.rationale).trim(),
    rawResponse,
  };
}

function buildJudgePrompt(args: {
  transcript: string;
  workflowSummary?: string;
  prediction: PredictedUserActionSuggestion;
  actualActionType: string;
  actualActionText: string;
}): string {
  return [
    "You are grading whether a predicted next user action was vaguely accurate.",
    "Return ONLY JSON.",
    'Schema: {"verdict":"accurate|partially_accurate|inaccurate","score":0.0,"rationale":"string"}',
    "Judge underlying intent more than exact UI action label.",
    "A prediction can be partially accurate if it identifies the same complaint or intervention target but misses the exact surface action.",
    "",
    "RECENT CONTEXT:",
    args.transcript || "(none)",
    "",
    args.workflowSummary
      ? ["WORKFLOW / VERIFIER SUMMARY:", args.workflowSummary, ""].join("\n")
      : "",
    "PREDICTION:",
    `actionType: ${args.prediction.actionType}`,
    `draftText: ${args.prediction.draftText || "(empty)"}`,
    `confidence: ${args.prediction.confidence}`,
    `rationale: ${args.prediction.rationale || "(empty)"}`,
    "",
    "ACTUAL NEXT USER STEP:",
    `actionType: ${args.actualActionType}`,
    `actionText: ${args.actualActionText || "(empty)"}`,
    "",
    "Would the prediction have been useful to anticipate what the user was about to do?",
  ]
    .filter(Boolean)
    .join("\n");
}

export async function judgePredictedAction(args: {
  cwd: string;
  transcript: string;
  workflowSummary?: string;
  prediction: PredictedUserActionSuggestion;
  actualActionType: string;
  actualActionText: string;
}): Promise<UserPredictionJudgeResult> {
  const rawResponse = await runPiTextPrompt({
    cwd: args.cwd,
    prompt: buildJudgePrompt(args),
  });

  const parsed = extractJsonObject(rawResponse);
  if (!parsed) {
    return {
      verdict: "inaccurate",
      score: 0,
      rationale: rawResponse.trim() || "Model returned an unparsable judgment.",
      rawResponse,
    };
  }

  return {
    verdict: normalizeJudgeVerdict(parsed.verdict),
    score: clamp01(parsed.score, 0),
    rationale: asString(parsed.rationale).trim(),
    rawResponse,
  };
}

export function buildWorkflowSummaryFromTree(workflowTree?: WorkflowNode[]): string {
  return workflowTree && workflowTree.length > 0 ? summarizeWorkflowNodes(workflowTree).join("\n") : "";
}

export function findUserProfilePath(cwd?: string): string | undefined {
  const { profilePath } = loadUserProfileMarkdown(cwd);
  return profilePath;
}

export function resolveAppUserProfileMarkdown(cwd?: string): { profileMarkdown: string; profilePath?: string } {
  const fromCwd = cwd ? loadUserProfileMarkdown(cwd) : { profileMarkdown: "" };
  if (fromCwd.profileMarkdown.trim()) return fromCwd;
  return loadUserProfileMarkdown(process.cwd());
}

export function getUserProfileRepoPath(): string {
  return join(process.cwd(), "USER_PROFILE.md");
}
