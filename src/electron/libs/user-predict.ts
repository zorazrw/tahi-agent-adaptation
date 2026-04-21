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
    raw === "brain_edit"
  ) {
    return raw;
  }
  return "unknown";
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
    'Schema: {"actionType":"message|edit_workflow|edit_verifier|file_edit|brain_edit|unknown","draftText":"string","confidence":0.0,"rationale":"string"}',
    "draftText should be the most likely next user message if actionType is message.",
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
    "Think about whether Zora is most likely to send a terse correction, edit the workflow, edit a verifier, or directly edit a file.",
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
