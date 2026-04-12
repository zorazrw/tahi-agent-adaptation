import { readFileSync, existsSync } from "fs";
import { join } from "path";
import type { VerifierMark } from "../types.js";
import type { WorkflowNode } from "../types.js";
import type { Session } from "./session-store.js";
import { getNodePath } from "./workflow-tree-utils.js";
import { loadApiConfig, type ApiConfig } from "./config-store.js";

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

/**
 * One Messages API call: for each verifier criterion, decide pass/fail from step outputs on disk.
 */
export async function labelVerifiersForNode(
  session: Session,
  workflowTree: WorkflowNode[],
  node: WorkflowNode
): Promise<VerifierMark[]> {
  const n = node.verifiers.length;
  if (n === 0) return [];

  const config = loadApiConfig();
  if (!config) {
    return node.verifiers.map(() => undefined);
  }

  const cwd = session.cwd ?? process.cwd();
  const fileBlocks: string[] = [];
  for (const rel of node.outputFiles) {
    const abs = join(cwd, rel);
    if (existsSync(abs)) {
      try {
        let t = readFileSync(abs, "utf8");
        if (t.length > 14_000) {
          t = t.slice(0, 14_000) + "\n... [truncated]";
        }
        fileBlocks.push(`### ${rel}\n\n${t}`);
      } catch {
        fileBlocks.push(`### ${rel}\n(unreadable)`);
      }
    } else {
      fileBlocks.push(`### ${rel}\n(file missing)`);
    }
  }

  const pathCtx = getNodePath(workflowTree, node.id);
  const numbered = node.verifiers
    .map((c: string, i: number) => `${i}. ${c}`)
    .join("\n");

  const systemContext = [
    "You are an automated checker for a completed workflow step.",
    "Given verifier criteria and the current output files (below), decide whether each criterion is satisfied.",
    'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
    "The results array must have exactly one object per verifier line, in the same order (indices 0 .. n-1).",
    "pass: true means the criterion is satisfied; false means it is not.",
    "",
    `Step path: ${pathCtx}`,
    `Step task: ${node.description}`,
    "",
    "Verifier criteria (in order):",
    numbered,
    "",
    "Output files and contents:",
    fileBlocks.length > 0 ? fileBlocks.join("\n\n---\n\n") : "(no output files listed)",
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
      messages: [{ role: "user", content: systemContext }],
    }),
  });

  if (!res.ok) {
    const errText = await res.text().catch(() => "");
    throw new Error(`Verifier API ${res.status}: ${errText.slice(0, 400)}`);
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

  const parsed = parseJsonFromModelText(text) as { results?: unknown };
  const results = parsed.results;
  if (!Array.isArray(results)) {
    throw new Error("Missing results array");
  }

  const out: VerifierMark[] = node.verifiers.map(() => undefined);
  for (let i = 0; i < n && i < results.length; i++) {
    const r = results[i];
    if (r && typeof r === "object" && "pass" in r) {
      out[i] = (r as { pass: boolean }).pass === true ? "check" : "cross";
    }
  }
  return out;
}
