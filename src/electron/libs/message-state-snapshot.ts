/**
 * Per-message environment snapshots for export: workflow (steps + verifier status) and output files.
 * Shape matches export_task_sessions.build_environment_state(include_files=True) (minus legacy-only edge cases).
 */
import { readFileSync, existsSync } from "fs";
import { join, resolve, relative, isAbsolute } from "path";
import type { Session } from "./session-store.js";
import type { VerifierMark, WorkflowNode } from "../types.js";

const MAX_OUTPUT_FILE_BYTES = 500_000;

export type ExportEnvironmentSnapshot = {
  workflow: ReturnType<typeof workflowNestedForExport>;
  file: ReturnType<typeof buildOutputFileEntries>;
};

function verifierStatusForExport(mark: VerifierMark | undefined): "success" | "failure" {
  if (mark === "check") return "success";
  return "failure";
}

function workflowNestedForExport(nodes: WorkflowNode[]): Array<{
  id: string;
  description: string;
  outputFiles: string[];
  verifiers: Array<{ criterion: string; status: "success" | "failure" }>;
  status: string;
  children: ReturnType<typeof workflowNestedForExport>;
}> {
  return nodes.map((n) => {
    const crits = Array.isArray(n.verifiers) ? n.verifiers : [];
    const marksRaw = Array.isArray(n.verifierMarks) ? n.verifierMarks : [];
    const verifiers = crits.map((c, j) => ({
      criterion: typeof c === "string" ? c : String(c),
      status: verifierStatusForExport(marksRaw[j] as VerifierMark | undefined),
    }));
    const ofs = Array.isArray(n.outputFiles) ? n.outputFiles : [];
    return {
      id: n.id,
      description: String(n.description ?? ""),
      outputFiles: ofs.map((x) => String(x)),
      verifiers,
      status: n.status,
      children: workflowNestedForExport(Array.isArray(n.children) ? n.children : []),
    };
  });
}

function orderedOutputRelPathsFromTree(tree: WorkflowNode[]): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  function walk(nodeList: WorkflowNode[]) {
    for (const n of nodeList) {
      for (const f of n.outputFiles ?? []) {
        const s = String(f).trim();
        if (s && !seen.has(s)) {
          seen.add(s);
          ordered.push(s);
        }
      }
      walk(n.children ?? []);
    }
  }
  walk(tree);
  return ordered;
}

function collectOriginalOutputsMap(tree: WorkflowNode[]): Record<string, string> {
  const out: Record<string, string> = {};
  function walk(nodeList: WorkflowNode[]) {
    for (const n of nodeList) {
      const oo = n.originalOutputs;
      if (Array.isArray(oo)) {
        for (const item of oo) {
          if (item?.path && typeof item.content === "string") {
            out[item.path] = item.content;
          }
        }
      }
      walk(n.children ?? []);
    }
  }
  walk(tree);
  return out;
}

function readTextLimited(absPath: string, maxBytes: number): { text: string | null; err: string | null } {
  try {
    if (!existsSync(absPath)) return { text: null, err: "not_a_file" };
    const buf = readFileSync(absPath);
    const truncated = buf.length > maxBytes;
    const chunk = buf.subarray(0, maxBytes);
    let text = chunk.toString("utf8");
    if (truncated) text += "\n[... export truncated: file larger than max bytes ...]";
    return { text, err: null };
  } catch (e) {
    return { text: null, err: e instanceof Error ? e.message : String(e) };
  }
}

function buildOutputFileEntries(
  cwd: string | undefined,
  relPaths: string[],
  originals: Record<string, string>
): Array<{ path: string; content: string | null; content_source: string | null; error: string | null }> {
  const entries: Array<{
    path: string;
    content: string | null;
    content_source: string | null;
    error: string | null;
  }> = [];
  let base: string | undefined;
  try {
    if (cwd?.trim()) base = resolve(cwd);
  } catch {
    base = undefined;
  }

  for (const rel of relPaths) {
    const item: { path: string; content: string | null; content_source: string | null; error: string | null } = {
      path: rel,
      content: null,
      content_source: null,
      error: null,
    };
    let readOk = false;
    if (base && rel && !rel.startsWith("/") && !rel.startsWith("\\")) {
      try {
        const absP = resolve(join(base, rel));
        if (base && isPathInsideDir(base, absP)) {
          const { text, err } = readTextLimited(absP, MAX_OUTPUT_FILE_BYTES);
          if (text != null) {
            item.content = text;
            item.content_source = "filesystem";
            readOk = true;
          } else if (err) item.error = err;
        } else item.error = "path_outside_cwd";
      } catch {
        item.error = "resolve_or_read_failed";
      }
    }
    if (!readOk && originals[rel]) {
      item.content = originals[rel];
      item.content_source = "originalOutputs";
      item.error = null;
    } else if (!readOk && item.content == null && item.error == null) {
      item.error = !base ? "no_cwd_or_missing_file" : "missing_or_unreadable";
    }
    entries.push(item);
  }
  return entries;
}

function isPathInsideDir(rootDir: string, candidatePath: string): boolean {
  const root = resolve(rootDir);
  const cand = resolve(candidatePath);
  const rel = relative(root, cand);
  return rel !== "" && !rel.startsWith("..") && !isAbsolute(rel);
}

/**
 * Build snapshot from current in-memory session (workflow tree + on-disk output files under cwd).
 */
export function buildExportEnvironmentSnapshot(session: Session): ExportEnvironmentSnapshot {
  const tree = session.workflowTree ?? [];
  const wf = workflowNestedForExport(tree);
  const relPaths = orderedOutputRelPathsFromTree(tree);
  const originals = collectOriginalOutputsMap(tree);
  const files = buildOutputFileEntries(session.cwd, relPaths, originals);
  return { workflow: wf, file: files };
}

/** Whether to persist a snapshot for this SDK message (per meaningful agent turn / tool outcome). */
export function shouldWriteSnapshotForSdkMessage(message: { type?: string }): boolean {
  const t = message.type;
  if (t === "stream_event" || t === "system") return false;
  if (t === "assistant") return false;
  if (t === "user" || t === "result") return true;
  return false;
}
