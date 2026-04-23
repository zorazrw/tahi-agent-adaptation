import { spawn, type ChildProcess } from "child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "fs";
import { join } from "path";
import { app } from "electron";

const BACKEND_BASE_URL = "http://localhost:8000";

const LOG_BASENAME = "context-export.log";

export type ContextInductionNotifierEvent =
  | { kind: "start"; sessionId: string }
  | { kind: "end"; sessionId: string; ok: boolean };

let inductionNotifier: ((ev: ContextInductionNotifierEvent) => void) | null = null;

/** Optional UI hook (e.g. brain icon): first export spawn through induce.py exit. */
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

function scriptsRootDir(): string | null {
  if (app.isPackaged) {
    const bundled = join(process.resourcesPath, "scripts");
    if (existsSync(join(bundled, "export_task_sessions.py"))) return bundled;
    return null;
  }
  const dev = join(app.getAppPath(), "scripts");
  return existsSync(join(dev, "export_task_sessions.py")) ? dev : null;
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

/** Serialize export+induction per session so auto-chained steps don't overlap SQLite / Python runs. */
const sessionExportChains = new Map<string, Promise<void>>();

function enqueueSessionExport(sessionId: string, run: () => Promise<void>): void {
  const prev = sessionExportChains.get(sessionId) ?? Promise.resolve();
  const next = prev.catch(() => {}).then(() => run());
  sessionExportChains.set(sessionId, next);
}

function inductionWrap(sessionId: string, inner: () => Promise<void>): Promise<void> {
  inductionNotifier?.({ kind: "start", sessionId });
  let ok = false;
  return inner()
    .then(() => {
      ok = true;
    })
    .catch((e) => {
      logLine(`Induction failed (session ${sessionId}): ${e instanceof Error ? e.message : String(e)}`);
    })
    .finally(() => {
      inductionNotifier?.({ kind: "end", sessionId, ok });
    });
}

/**
 * When every workflow step is done, export the full session once and run induce.py.
 * Queued after any prior per-step jobs so the DB holds the complete trajectory.
 * Full-session file uses the same `{ uuid, name, trajectory }` schema (full trajectory in one object).
 */
export function runFullSessionExportAndExtract(sessionId: string): void {
  const root = scriptsRootDir();
  if (!root) {
    logLine("Skip (full session): scripts/ not found.");
    return;
  }
  const exportScript = join(root, "export_task_sessions.py");
  const induceScript = join(root, "induce.py");
  if (!existsSync(exportScript) || !existsSync(induceScript)) {
    logLine(`Skip (full session): missing export or induce script under ${root}`);
    return;
  }

  const userData = app.getPath("userData");
  const dbPath = join(userData, "sessions.db");
  const tasksDir = join(userData, "tasks");
  const fullJsonPath = join(tasksDir, `${sessionId}-workflow-full.json`);
  if (!existsSync(dbPath)) {
    logLine("Skip (full session): sessions.db missing.");
    return;
  }
  mkdirSync(tasksDir, { recursive: true });

  const py = pythonExecutable();
  enqueueSessionExport(sessionId, () =>
    inductionWrap(sessionId, async () => {
      logLine(`export_task_sessions full session=${sessionId}`);
      const exportProc = spawn(
        py,
        [
          exportScript,
          "--db",
          dbPath,
          "--session-id",
          sessionId,
          "--output",
          fullJsonPath,
        ],
        { cwd: root, stdio: ["ignore", "pipe", "pipe"], env: process.env }
      );
      await spawnClosed(exportProc, "export_task_sessions (full)");
      logLine("induce.py starting (full session)");
      const induceProc = spawn(
        py,
        [induceScript, "--data_path", fullJsonPath, "--output_dir", userData],
        { cwd: root, stdio: ["ignore", "pipe", "pipe"], env: process.env }
      );
      await spawnClosed(induceProc, "induce.py");
      logLine("induce.py finished OK (full session)");

      try {
        const sessionJson = readFileSync(fullJsonPath, "utf-8");
        const res = await fetch(`${BACKEND_BASE_URL}/session`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: sessionJson,
        });
        if (!res.ok) {
          logLine(`POST /session failed: ${res.status} ${res.statusText}`);
        } else {
          logLine("POST /session succeeded");
        }
      } catch (e) {
        logLine(`POST /session error: ${e instanceof Error ? e.message : String(e)}`);
      }
    })
  );
}
