import { readFileSync, existsSync } from "fs";
import { extname, join } from "path";
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
  const scanCandidates = (source: string): string[] => {
    const candidates: string[] = [];
    let start: number | null = null;
    let depth = 0;
    let inString = false;
    let escape = false;
    for (let i = 0; i < source.length; i++) {
      const ch = source[i]!;
      if (start === null) {
        if (ch === "{") {
          start = i;
          depth = 1;
          inString = false;
          escape = false;
        }
        continue;
      }

      if (inString) {
        if (escape) {
          escape = false;
        } else if (ch === "\\") {
          escape = true;
        } else if (ch === '"') {
          inString = false;
        }
        continue;
      }

      if (ch === '"') {
        inString = true;
      } else if (ch === "{") {
        depth += 1;
      } else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          candidates.push(source.slice(start, i + 1));
          start = null;
        }
      }
    }
    return candidates;
  };

  const sources: string[] = [];
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fence?.[1]) sources.push(fence[1].trim());
  sources.push(text.trim());

  let lastError: unknown;
  const seen = new Set<string>();
  for (const source of sources) {
    for (const candidate of [...scanCandidates(source)].reverse()) {
      if (seen.has(candidate)) continue;
      seen.add(candidate);
      try {
        return JSON.parse(candidate) as unknown;
      } catch (err) {
        lastError = err;
      }
    }
  }

  if (lastError instanceof Error) {
    throw new Error(`Unable to parse JSON object in model response: ${lastError.message}`);
  }
  throw new Error("No JSON object in model response");
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
  const imageBlocks: Array<{ type: "image"; source: { type: "base64"; media_type: string; data: string } }> = [];
  const mimeByExt: Record<string, string> = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
  };
  for (const rel of node.outputFiles) {
    const abs = join(cwd, rel);
    if (existsSync(abs)) {
      try {
        const ext = extname(rel).toLowerCase();
        const imageMime = mimeByExt[ext];
        if (imageMime) {
          const bytes = readFileSync(abs);
          fileBlocks.push(`### ${rel}\n(image attached as base64 ${imageMime})`);
          imageBlocks.push({
            type: "image",
            source: {
              type: "base64",
              media_type: imageMime,
              data: bytes.toString("base64"),
            },
          });
          continue;
        }

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

  const userContent: Array<
    | { type: "text"; text: string }
    | { type: "image"; source: { type: "base64"; media_type: string; data: string } }
  > = [{ type: "text", text: systemContext }, ...imageBlocks];

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
      messages: [{ role: "user", content: userContent }],
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
