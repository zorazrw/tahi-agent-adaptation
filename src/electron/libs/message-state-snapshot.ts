/**
 * Per-message environment snapshots for export: workflow (steps + verifier status) and output files.
 * Shape matches export_task_sessions.build_environment_state(include_files=True) (minus legacy-only edge cases).
 */
import { readFileSync, existsSync } from "fs";
import { join, resolve, relative, isAbsolute } from "path";
import type { Session } from "./session-store.js";
import type { VerifierMark, WorkflowNode } from "../types.js";
import { readAllMemorySections } from "./memory-store.js";
import { readAllFlatSkillSections } from "./skill-store.js";

const MAX_OUTPUT_FILE_BYTES = 500_000;

type OutputContentEncoding = "utf8" | "base64";

type OutputFileEntry = {
  path: string;
  content: string | null;
  content_source: string | null;
  content_encoding?: OutputContentEncoding | null;
  error: string | null;
};

export type ExportEnvironmentSnapshot = {
  workflow: ReturnType<typeof workflowNestedForExport>;
  file: ReturnType<typeof buildOutputFileEntries>;
  /** Per memory .md file under userData/memories — file name → raw contents (truncated). */
  memory: Record<string, string>;
  /** Per top-level skill .md under userData/skills — file name → raw contents (truncated). */
  skill: Record<string, string>;
};

/** Align with in-app sidebar: no mark means not yet labeled, not a failed check. */
function verifierStatusForExport(mark: VerifierMark | undefined): "success" | "failure" | "unchecked" {
  if (mark === "check") return "success";
  if (mark === "cross") return "failure";
  return "unchecked";
}

function workflowNestedForExport(nodes: WorkflowNode[]): Array<{
  id: string;
  description: string;
  outputFiles: string[];
  verifiers: Array<{ criterion: string; status: "success" | "failure" | "unchecked" }>;
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

function isValidUtf8(buf: Buffer): boolean {
  const decoded = buf.toString("utf8");
  const roundTrip = Buffer.from(decoded, "utf8");
  return roundTrip.equals(buf);
}

function readFileContentLimited(
  absPath: string,
  maxBytes: number
): { content: string | null; contentEncoding: OutputContentEncoding | null; err: string | null } {
  try {
    if (!existsSync(absPath)) return { content: null, contentEncoding: null, err: "not_a_file" };
    const buf = readFileSync(absPath);
    const truncated = buf.length > maxBytes;
    const chunk = buf.subarray(0, maxBytes);
    if (isValidUtf8(chunk)) {
      let text = chunk.toString("utf8");
      if (truncated) text += "\n[... export truncated: file larger than max bytes ...]";
      return { content: text, contentEncoding: "utf8", err: null };
    }
    return { content: chunk.toString("base64"), contentEncoding: "base64", err: null };
  } catch (e) {
    return { content: null, contentEncoding: null, err: e instanceof Error ? e.message : String(e) };
  }
}

function buildOutputFileEntries(
  cwd: string | undefined,
  relPaths: string[],
  originals: Record<string, string>
): OutputFileEntry[] {
  const entries: OutputFileEntry[] = [];
  let base: string | undefined;
  try {
    if (cwd?.trim()) base = resolve(cwd);
  } catch {
    base = undefined;
  }

  for (const rel of relPaths) {
    const item: OutputFileEntry = {
      path: rel,
      content: null,
      content_source: null,
      content_encoding: null,
      error: null,
    };
    let readOk = false;
    if (rel && isAbsolute(rel)) {
      try {
        const absP = resolve(rel);
        if (existsSync(absP)) {
          const { content, contentEncoding, err } = readFileContentLimited(absP, MAX_OUTPUT_FILE_BYTES);
          if (content != null) {
            item.content = content;
            item.content_source = "filesystem";
            item.content_encoding = contentEncoding;
            readOk = true;
          } else if (err) item.error = err;
        } else {
          item.error = "not_a_file";
        }
      } catch {
        item.error = "resolve_or_read_failed";
      }
    } else if (base && rel) {
      try {
        const absP = resolve(join(base, rel));
        if (isPathInsideDir(base, absP)) {
          const { content, contentEncoding, err } = readFileContentLimited(absP, MAX_OUTPUT_FILE_BYTES);
          if (content != null) {
            item.content = content;
            item.content_source = "filesystem";
            item.content_encoding = contentEncoding;
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
      item.content_encoding = "utf8";
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

function truncateUtf8ForExport(text: string, maxBytes: number): string {
  const buf = Buffer.from(text, "utf8");
  if (buf.length <= maxBytes) return text;
  const chunk = buf.subarray(0, maxBytes);
  return chunk.toString("utf8") + "\n[... export truncated: file larger than max bytes ...]";
}

function memorySkillMapsForExport(): Pick<ExportEnvironmentSnapshot, "memory" | "skill"> {
  const memory: Record<string, string> = {};
  const skill: Record<string, string> = {};
  for (const { fileName, content } of readAllMemorySections()) {
    memory[fileName] = truncateUtf8ForExport(content ?? "", MAX_OUTPUT_FILE_BYTES);
  }
  for (const { fileName, content } of readAllFlatSkillSections()) {
    skill[fileName] = truncateUtf8ForExport(content ?? "", MAX_OUTPUT_FILE_BYTES);
  }
  return { memory, skill };
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
  const { memory, skill } = memorySkillMapsForExport();
  return { workflow: wf, file: files, memory, skill };
}

/** Canonical path key for matching workflow output paths (absolute vs cwd-relative mix). */
function resolvedFileKey(cwd: string | undefined, filePath: string): string {
  const p = String(filePath ?? "").trim();
  if (!p) return "";
  try {
    if (isAbsolute(p)) return resolve(p).replace(/\\/g, "/");
    if (cwd?.trim()) return resolve(cwd.trim(), p).replace(/\\/g, "/");
    return resolve(p).replace(/\\/g, "/");
  } catch {
    return p.replace(/\\/g, "/");
  }
}

/**
 * Same as ``buildExportEnvironmentSnapshot``, but guarantees ``editedRelPath`` appears in ``file``
 * with the exact post-write ``editedContent`` (preview Text / Move save). Paths only in the workflow
 * tree are otherwise included; this upserts or appends so HTML edits are visible in export DB rows.
 */
export function buildExportEnvironmentSnapshotWithPreviewWrittenFile(
  session: Session,
  editedRelPath: string,
  editedContent: string
): ExportEnvironmentSnapshot {
  const base = buildExportEnvironmentSnapshot(session);
  const cwd = session.cwd;
  const editedKey = resolvedFileKey(cwd, editedRelPath);
  const content = truncateUtf8ForExport(editedContent, MAX_OUTPUT_FILE_BYTES);
  const files = base.file.map((f) => ({ ...f }));
  const idx = files.findIndex((f) => resolvedFileKey(cwd, String(f.path)) === editedKey);
  const displayPath = idx >= 0 ? String(files[idx].path) : editedRelPath.replace(/\\/g, "/");
  const entry: (typeof base.file)[number] = {
    path: displayPath,
    content,
    content_source: "preview_write",
    content_encoding: "utf8",
    error: null,
  };
  if (idx >= 0) {
    files[idx] = { ...files[idx], ...entry };
  } else {
    files.push(entry);
  }
  return { workflow: base.workflow, file: files, memory: base.memory, skill: base.skill };
}

/** Whether to persist a snapshot for this SDK message (per meaningful agent turn / tool outcome). */
export function shouldWriteSnapshotForSdkMessage(message: { type?: string }): boolean {
  const t = message.type;
  if (t === "stream_event" || t === "system") return false;
  if (t === "assistant") return false;
  if (t === "user" || t === "result") return true;
  if (t === "tool_result" || t === "run_result") return true;
  return false;
}
