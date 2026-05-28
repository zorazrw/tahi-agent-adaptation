import { spawn, type ChildProcess } from "child_process";
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { app } from "electron";
import { startTinkerAutoUpdateWatcher } from "./tinker-auto-update.js";
import {
  getTrainingProxyBaseUrl,
  isTrainingProxyDisabled,
  TRAINING_PROXY_START_HINT,
} from "./training-proxy.js";

const LOG_BASENAME = "context-export.log";

export type ContextInductionNotifierEvent =
  | { kind: "start"; sessionId: string }
  | { kind: "end"; sessionId: string; ok: boolean; trainingUpload?: boolean };

let inductionNotifier: ((ev: ContextInductionNotifierEvent) => void) | null = null;

/** Optional UI hook (e.g. brain icon): export / induce / training upload lifecycle. */
export function setContextInductionNotifier(
  fn: ((ev: ContextInductionNotifierEvent) => void) | null
): void {
  inductionNotifier = fn;
}

function logLine(message: string): void {
  const line = `[${new Date().toISOString()}] ${message}\n`;
  try {
    appendFileSync(join(app.getPath("userData"), LOG_BASENAME), line);
  } catch {
    /* ignore disk errors */
  }
  console.error(`[context-export] ${message}`);
}

const EXPORT_SCRIPT_REL = join("tasks", "export_task_sessions.py");
const INDUCE_SCRIPT_REL = "induce.py";
const MD_FILE_RE = /^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.md$/;

function slugifySessionName(name: string, fallback = "session"): string {
  const raw = String(name ?? "").trim().toLowerCase();
  const slug = raw.replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return (slug || fallback).slice(0, 100);
}

function listTopLevelMdFiles(dir: string): string[] {
  if (!existsSync(dir)) return [];
  try {
    return readdirSync(dir).filter((n) => MD_FILE_RE.test(n));
  } catch {
    return [];
  }
}

type ExportedTaskUnit = { actor?: unknown; trajectory?: Array<{ action?: unknown }> };
type ExportedSessionBlob = {
  name?: unknown;
  task_units?: ExportedTaskUnit[];
  trajectory?: Array<{ action?: unknown; actor?: unknown }>;
};

function readSessionBlobForFallback(fullJsonPath: string): ExportedSessionBlob | null {
  try {
    const raw = JSON.parse(readFileSync(fullJsonPath, "utf8")) as unknown;
    if (Array.isArray(raw)) {
      const first = raw.find((row) => row && typeof row === "object");
      return first && typeof first === "object" ? (first as ExportedSessionBlob) : null;
    }
    if (raw && typeof raw === "object") {
      const obj = raw as { sessions?: unknown[] };
      if (Array.isArray(obj.sessions)) {
        const first = obj.sessions.find((row) => row && typeof row === "object");
        return first && typeof first === "object" ? (first as ExportedSessionBlob) : null;
      }
      return raw as ExportedSessionBlob;
    }
  } catch (e) {
    logLine(`Fallback parse failed: ${e instanceof Error ? e.message : String(e)}`);
  }
  return null;
}

function collectAgentActions(blob: ExportedSessionBlob | null): string[] {
  if (!blob) return [];
  const out: string[] = [];

  if (Array.isArray(blob.task_units)) {
    for (const unit of blob.task_units) {
      if (!unit || typeof unit !== "object") continue;
      if (String(unit.actor ?? "").toLowerCase() !== "agent") continue;
      const traj = unit.trajectory;
      if (!Array.isArray(traj)) continue;
      for (const step of traj) {
        const action = step?.action;
        if (typeof action === "string" && action.trim()) out.push(action.trim());
      }
    }
  }

  if (out.length === 0 && Array.isArray(blob.trajectory)) {
    for (const step of blob.trajectory) {
      if (!step || typeof step !== "object") continue;
      if (String(step.actor ?? "").toLowerCase() !== "agent") continue;
      const action = step.action;
      if (typeof action === "string" && action.trim()) out.push(action.trim());
    }
  }

  return out;
}

function writeFallbackInductionOutputs(userData: string, fullJsonPath: string): void {
  const memDir = join(userData, "memories");
  const skillsDir = join(userData, "skills");
  mkdirSync(memDir, { recursive: true });
  mkdirSync(skillsDir, { recursive: true });

  const beforeMem = listTopLevelMdFiles(memDir);
  const beforeSkills = listTopLevelMdFiles(skillsDir);
  const blob = readSessionBlobForFallback(fullJsonPath);
  const actions = collectAgentActions(blob);
  if (actions.length === 0) return;

  const stem = slugifySessionName(typeof blob?.name === "string" ? blob.name : "");
  const targetMem = join(memDir, `${stem}.md`);
  const targetSkill = join(skillsDir, `${stem}.md`);

  const memoryBody = [
    "## Auto memory (fallback)",
    "",
    "Generated because induce.py completed without creating memory/skill files.",
    "",
    `- Session: ${typeof blob?.name === "string" && blob.name.trim() ? blob.name.trim() : stem}`,
    `- Agent actions observed: ${actions.length}`,
    "",
    "Recent actions:",
    ...actions.slice(0, 8).map((a, i) => `${i + 1}. ${a}`),
    "",
  ].join("\n");
  const skillBody = [
    "Auto-induction fallback skill",
    "1. Review the latest exported session trajectory.",
    "2. Extract stable preferences/facts from agent actions.",
    "3. Convert recurring approach into reusable skill steps.",
    "",
  ].join("\n");

  writeFileSync(targetMem, memoryBody, "utf8");
  writeFileSync(targetSkill, skillBody, "utf8");

  const afterMem = listTopLevelMdFiles(memDir);
  const afterSkills = listTopLevelMdFiles(skillsDir);
  if (afterMem.length > beforeMem.length || afterSkills.length > beforeSkills.length) {
    logLine(`Fallback induction wrote ${stem}.md memory + skill`);
  }
}

function scriptsRootDir(): string | null {
  if (app.isPackaged) {
    const bundled = join(process.resourcesPath, "scripts");
    if (existsSync(join(bundled, EXPORT_SCRIPT_REL)) && existsSync(join(bundled, INDUCE_SCRIPT_REL))) {
      return bundled;
    }
    return null;
  }
  const dev = join(app.getAppPath(), "scripts");
  if (existsSync(join(dev, EXPORT_SCRIPT_REL)) && existsSync(join(dev, INDUCE_SCRIPT_REL))) {
    return dev;
  }
  return null;
}

function pythonExecutable(): string {
  return process.platform === "win32" ? "python" : "python3";
}

function spawnClosed(proc: ChildProcess, label: string): Promise<void> {
  return new Promise((resolve, reject) => {
    let errBuf = "";
    proc.stderr?.on("data", (chunk: Buffer) => {
      errBuf += chunk.toString();
    });
    proc.on("error", (err) => {
      reject(err);
    });
    proc.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`${label} exit ${code}: ${errBuf.slice(-900)}`));
      } else {
        resolve();
      }
    });
  });
}

/** Serialize per-session export jobs (induction, tinker upload) so they do not overlap. */
const sessionExportChains = new Map<string, Promise<void>>();

export function enqueueSessionJob(sessionId: string, run: () => Promise<void>): void {
  const prev = sessionExportChains.get(sessionId) ?? Promise.resolve();
  const next = prev.catch(() => {}).then(() => run());
  sessionExportChains.set(sessionId, next);
}

export function runWithInductionNotifier(
  sessionId: string,
  inner: () => Promise<void>,
  options?: { trainingUpload?: boolean },
): Promise<void> {
  inductionNotifier?.({ kind: "start", sessionId });
  let ok = false;
  return inner()
    .then(() => {
      ok = true;
    })
    .catch((e) => {
      logLine(`Session job failed (${sessionId}): ${e instanceof Error ? e.message : String(e)}`);
    })
    .finally(() => {
      inductionNotifier?.({ kind: "end", sessionId, ok, trainingUpload: options?.trainingUpload });
    });
}

/** Export one session from SQLite to a JSON file; returns path or null on skip/failure. */
export async function exportSessionJsonFile(sessionId: string): Promise<string | null> {
  const root = scriptsRootDir();
  if (!root) {
    logLine("Skip export: scripts/ not found.");
    return null;
  }
  const exportScript = join(root, EXPORT_SCRIPT_REL);
  if (!existsSync(exportScript)) {
    logLine(`Skip export: missing script under ${root}`);
    return null;
  }

  const userData = app.getPath("userData");
  const dbPath = join(userData, "sessions.db");
  const tasksDir = join(userData, "tasks");
  const fullJsonPath = join(tasksDir, `${sessionId}-workflow-full.json`);
  if (!existsSync(dbPath)) {
    logLine("Skip export: sessions.db missing.");
    return null;
  }
  mkdirSync(tasksDir, { recursive: true });

  const py = pythonExecutable();
  logLine(`export_task_sessions session=${sessionId}`);
  const exportProc = spawn(
    py,
    [exportScript, "--db", dbPath, "--session-id", sessionId, "--output", fullJsonPath],
    { cwd: root, stdio: ["ignore", "pipe", "pipe"], env: process.env }
  );
  await spawnClosed(exportProc, "export_task_sessions");
  return fullJsonPath;
}

/**
 * Export the current session from SQLite and run induce.py (memories + flat skills).
 * Called after each completed workflow step when auto-induction is on.
 */
export function runFullSessionExportAndExtract(sessionId: string): void {
  const root = scriptsRootDir();
  if (!root) {
    logLine("Skip (full session): scripts/ not found.");
    return;
  }
  const induceScript = join(root, INDUCE_SCRIPT_REL);
  if (!existsSync(induceScript)) {
    logLine(`Skip (full session): missing induce script under ${root}`);
    return;
  }

  const userData = app.getPath("userData");
  enqueueSessionJob(sessionId, () =>
    runWithInductionNotifier(sessionId, async () => {
      const fullJsonPath = await exportSessionJsonFile(sessionId);
      if (!fullJsonPath) return;

      const py = pythonExecutable();
      logLine("induce.py starting (full session)");
      const induceProc = spawn(
        py,
        [induceScript, "--data_path", fullJsonPath, "--output_dir", userData],
        { cwd: root, stdio: ["ignore", "pipe", "pipe"], env: process.env }
      );
      await spawnClosed(induceProc, "induce.py");
      writeFallbackInductionOutputs(userData, fullJsonPath);
      logLine("induce.py finished OK (full session)");
    }),
  );
}

/** Export session JSON and POST to the local training proxy (brain click when auto-induction is off). */
export function uploadSessionForTinkerTraining(sessionId: string): void {
  if (isTrainingProxyDisabled()) {
    throw new Error('Training proxy is disabled (set AGENT_COWORK_PROXY_URL, not "disabled").');
  }
  const baseUrl = getTrainingProxyBaseUrl();
  if (!baseUrl) throw new Error(TRAINING_PROXY_START_HINT);

  startTinkerAutoUpdateWatcher();
  enqueueSessionJob(sessionId, () =>
    runWithInductionNotifier(
      sessionId,
      async () => {
        const fullJsonPath = await exportSessionJsonFile(sessionId);
        if (!fullJsonPath) throw new Error("Session export failed");

        const parsed = JSON.parse(readFileSync(fullJsonPath, "utf8")) as unknown;
        const body = Array.isArray(parsed) ? parsed : [parsed];
        const res = await fetch(`${baseUrl}/session`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const text = await res.text().catch(() => "");
          throw new Error(`Training proxy HTTP ${res.status}: ${text.slice(0, 500)}`);
        }
      },
      { trainingUpload: true },
    ),
  );
}
