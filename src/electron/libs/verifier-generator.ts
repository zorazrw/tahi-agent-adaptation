import type { Session } from "./session-store.js";
import type { WorkflowNode } from "../types.js";
import { loadApiConfig, type ApiConfig } from "./config-store.js";
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
  const config = loadApiConfig();
  if (!config) return null;

  const messageList = userMessages
    .map((m, i) => `Message ${i + 1}: ${m}`)
    .join("\n");
  const existing = node.verifiers.length > 0 ? node.verifiers : ["(none yet)"];
  const pathCtx = getNodePath(workflowTree, node.id);

  const prompt = [
    "Revise verifier criteria for one workflow step.",
    "Use conversation + existing verifiers to produce an updated list aligned with latest user requirements.",
    "Write short, concrete, falsifiable criteria that can be checked directly from output content.",
    "Each verifier must test a distinct requirement; verifiers must not overlap in meaning or checks.",
    "If two criteria overlap, merge or rewrite them so each verifier covers one unique thing.",
    "Use the simplest, most concise language possible for normal users.",
    "Avoid jargon and unnecessary detail; prefer plain words and short sentences.",
    "Prefer explicit required elements, constraints, and forbidden claims.",
    "For visual tasks, always include at least one criterion that checks items are clearly separated and not messily overlapping.",
    "Avoid vague wording (e.g., good, complete, quality, reasonable, appropriate).",
    "No duplicates. Keep useful existing verifiers unless they conflict with newer requirements.",
    "Use user-removed examples as negative guidance; avoid similar criteria.",
    "Use user-added examples as positive guidance; preserve or add similar criteria when relevant.",
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
