import type { Session } from "./session-store.js";
import type { WorkflowNode } from "../types.js";
import { resolveVerifierApiConfig } from "./pi-config.js";
import { getNodePath } from "./workflow-tree-utils.js";

function messagesApiUrl(baseURL: string): string {
  const base = baseURL.replace(/\/*$/, "");
  return base.endsWith("/v1") ? `${base}/messages` : `${base}/v1/messages`;
}

function parseJsonFromModelText(text: string): unknown {
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const raw = (fence ? fence[1] : text).trim();
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    throw new Error("No JSON object in model response");
  }
  return JSON.parse(raw.slice(start, end + 1)) as unknown;
}

function normalizeVerifierLines(input: unknown, fallback: string[]): string[] {
  if (!Array.isArray(input)) return fallback;
  const cleaned = input
    .map((v) => (typeof v === "string" ? v.trim() : ""))
    .filter(Boolean);
  return cleaned.length > 0 ? cleaned : fallback;
}

/**
 * Refresh verifier criteria before a follow-up task execution call.
 * Uses user messages, human file_edit diffs, verifier examples, and existing verifiers for the target node.
 */
export async function generateUpdatedVerifiersForNode(
  session: Session,
  workflowTree: WorkflowNode[],
  node: WorkflowNode,
  userMessages: string[],
  userRemovedExamples: string[] = [],
  userAddedExamples: string[] = [],
  fileEditDiffs: string[] = []
): Promise<string[] | null> {
  const config = await resolveVerifierApiConfig();
  if (!config) return null;

  const messageList = userMessages
    .map((m, i) => `Message ${i + 1}: ${m}`)
    .join("\n");
  const existing = node.verifiers.length > 0 ? node.verifiers : ["(none yet)"];
  const pathCtx = getNodePath(workflowTree, node.id);

  const prompt = [
    "Revise verifier criteria for one workflow step. Keep useful existing verifiers unless they conflict with newer requirements.",
    "Inputs: use the conversation, file edits, and existing verifiers to produce an updated list aligned with the latest requirements.",
    "Falsifiability: write short, concrete, falsifiable criteria checkable directly from output content.",
    "Distinctness: each verifier must cover one distinct requirement; merge, rewrite, or drop overlapping or duplicate criteria.",
    "Conciseness: use plain, concise language; avoid jargon and vague words (e.g., good, complete, quality, reasonable, appropriate).",
    "Explicitness: prefer explicit required elements, constraints, and forbidden claims.",
    "Non-restatement: do not add criteria that only restate values, labels, or specs already given in the step task; verify correctness, formatting, or relationships instead (e.g., prefer 'numeric values are accurately presented' over echoing each figure from the task).",
    "Visual layout: for visual tasks, include at least one criterion that items are clearly separated and not messily overlapping.",
    "User signals: honor user-removed examples by avoiding similar criteria; honor user-added examples by preserving or adding similar ones when relevant.",
    'Reply with ONLY JSON: {"verifiers":["...","..."]}',
    "",
    `Step path: ${pathCtx}`,
    `Step task: ${node.description}`,
    "",
    "Conversation messages (oldest -> newest):",
    messageList || "(none)",
    "",
    "Existing verifiers:",
    ...existing.map((v: string, i: number) => `${i + 1}. ${v}`),
    "",
    "User-removed verifier examples (negative; avoid similar criteria):",
    ...(userRemovedExamples.length > 0 ? userRemovedExamples.map((v: string, i: number) => `${i + 1}. ${v}`) : ["(none)"]),
    "",
    "User-added verifier examples (positive; prefer similar/preserved criteria when relevant):",
    ...(userAddedExamples.length > 0 ? userAddedExamples.map((v: string, i: number) => `${i + 1}. ${v}`) : ["(none)"]),
    "",
    "Human file edits (line diff; - removed, + added, space = context):",
    ...(fileEditDiffs.length > 0 ? fileEditDiffs : ["(none)"]),
  ].join("\n");

  const res = await fetch(messagesApiUrl(config.baseURL), {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": config.apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: config.model,
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Verifier generation API ${res.status}: ${errText.slice(0, 400)}`);
  }

  const data = (await res.json()) as {
    content?: Array<{ type?: string; text?: string }>;
  };
  const text =
    data.content?.find((b) => b.type === "text")?.text ??
    (typeof data.content?.[0]?.text === "string" ? data.content[0].text : "");
  if (!text) {
    throw new Error("Empty model content");
  }

  const parsed = parseJsonFromModelText(text) as { verifiers?: unknown };
  return normalizeVerifierLines(parsed.verifiers, node.verifiers);
}
