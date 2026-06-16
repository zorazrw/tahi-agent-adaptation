import { readFileSync, existsSync } from "fs";
import { extname, join } from "path";
import type { VerifierMark } from "../types.js";
import type { WorkflowNode } from "../types.js";
import type { Session } from "./session-store.js";
import { getNodePath } from "./workflow-tree-utils.js";
import { resolveVerifierApiConfig } from "./pi-config.js";

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

function resultsArrayInstructions(count: number): string {
  return (
    `There are ${count} criteria below. The results array must contain exactly ${count} objects ` +
    `(no more, no fewer).\n` +
    `results[0] is the verdict for criterion 1, results[1] for criterion 2, ` +
    `..., results[${count - 1}] for criterion ${count}.`
  );
}

function marksFromModelText(text: string, n: number): VerifierMark[] {
  const parsed = parseJsonFromModelText(text) as { results?: unknown };
  const results = parsed.results;
  if (!Array.isArray(results)) {
    throw new Error("Missing results array");
  }
  if (results.length !== n) {
    throw new Error(`Expected exactly ${n} entries in results, got ${results.length}`);
  }

  const out: VerifierMark[] = Array.from({ length: n }, () => undefined);
  for (let i = 0; i < n; i++) {
    const r = results[i];
    if (r && typeof r === "object" && "pass" in r) {
      out[i] = (r as { pass: boolean }).pass === true ? "check" : "cross";
    }
  }
  return out;
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

  const config = await resolveVerifierApiConfig();
  if (!config) {
    return node.verifiers.map(() => undefined);
  }

  const apiConfig = config;

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
  const numbered = node.verifiers.map((c: string, i: number) => `${i + 1}. ${c}`).join("\n");

  const systemContext = [
    "You are an automated checker for a completed workflow step.",
    "Given verifier criteria and the current output files (below), decide whether each criterion is satisfied.",
    'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
    resultsArrayInstructions(n),
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

  async function callModel(extraHint = ""): Promise<string> {
    const text = extraHint ? `${systemContext}\n\n${extraHint}` : systemContext;
    const content: typeof userContent = [{ type: "text", text }, ...imageBlocks];
    const res = await fetch(messagesApiUrl(apiConfig.baseURL), {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": apiConfig.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: apiConfig.model,
        max_tokens: 1024,
        messages: [{ role: "user", content }],
      }),
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      throw new Error(`Verifier API ${res.status}: ${errText.slice(0, 400)}`);
    }

    const data = (await res.json()) as {
      content?: Array<{ type?: string; text?: string }>;
    };
    const modelText =
      data.content?.find((b) => b.type === "text")?.text ??
      (typeof data.content?.[0]?.text === "string" ? data.content[0].text : "");
    if (!modelText) {
      throw new Error("Empty model content");
    }
    return modelText;
  }

  const retryHint =
    `Your previous reply had the wrong number of results entries. ` +
    `Return exactly ${n} objects in results — one per numbered line (1 through ${n}) above.`;

  try {
    return marksFromModelText(await callModel(), n);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (!msg.includes("Expected exactly")) {
      throw err;
    }
    return marksFromModelText(await callModel(retryHint), n);
  }
}
