import { spawn } from "child_process";
import { appendFileSync, existsSync } from "fs";
import { join } from "path";
import { app } from "electron";

const LOG_BASENAME = "context-export.log";

export type ContextInductionNotifierEvent =
  | { kind: "start"; sessionId: string }
  | { kind: "end"; sessionId: string; ok: boolean };

let inductionNotifier: ((ev: ContextInductionNotifierEvent) => void) | null = null;

/** Optional UI hook (e.g. brain icon): first export spawn through extract_context exit. */
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

/**
 * Export the completed workflow node to userData/tasks/{taskUnitId}.json, then run
 * extract_context.py on that file. Memories/skills go under userData.
 * Runs asynchronously; failures are logged only.
 */
export function runExportAndExtractContext(sessionId: string, taskUnitId: string): void {
  const root = scriptsRootDir();
  if (!root) {
    logLine("Skip: scripts/ not found (dev: open repo root; packaged: extraResources).");
    return;
  }
  const exportScript = join(root, "export_task_sessions.py");
  const extractScript = join(root, "extract_context.py");
  if (!existsSync(exportScript) || !existsSync(extractScript)) {
    logLine(`Skip: missing export or extract script under ${root}`);
    return;
  }

  const userData = app.getPath("userData");
  const dbPath = join(userData, "sessions.db");
  const tasksDir = join(userData, "tasks");
  const taskJsonPath = join(tasksDir, `${taskUnitId}.json`);
  if (!existsSync(dbPath)) {
    logLine("Skip: sessions.db missing.");
    return;
  }

  let finished = false;
  const finish = (ok: boolean) => {
    if (finished) return;
    finished = true;
    inductionNotifier?.({ kind: "end", sessionId, ok });
  };

  inductionNotifier?.({ kind: "start", sessionId });

  const py = pythonExecutable();
  const exportArgs = [
    exportScript,
    "--db",
    dbPath,
    "--session-id",
    sessionId,
    "--tasks-dir",
    tasksDir,
    "--task-unit-id",
    taskUnitId,
    "--pretty",
  ];

  logLine(`export_task_sessions session=${sessionId} taskUnit=${taskUnitId}`);

  const exportProc = spawn(py, exportArgs, {
    cwd: root,
    stdio: ["ignore", "pipe", "pipe"],
    env: process.env,
  });

  let exportErr = "";
  exportProc.stderr?.on("data", (chunk: Buffer) => {
    exportErr += chunk.toString();
  });

  exportProc.on("error", (err) => {
    logLine(`export spawn failed: ${err.message}`);
    finish(false);
  });

  exportProc.on("close", (code) => {
    if (code !== 0) {
      logLine(`export_task_sessions exit ${code}: ${exportErr.slice(-600)}`);
      finish(false);
      return;
    }
    logLine("extract_context starting");

    const extractProc = spawn(
      py,
      [extractScript, "--data_path", taskJsonPath, "--output_dir", userData],
      { cwd: root, stdio: ["ignore", "pipe", "pipe"], env: process.env }
    );

    let extractErr = "";
    extractProc.stderr?.on("data", (chunk: Buffer) => {
      extractErr += chunk.toString();
    });
    extractProc.on("error", (err) => {
      logLine(`extract spawn failed: ${err.message}`);
      finish(false);
    });
    extractProc.on("close", (c2) => {
      if (c2 !== 0) {
        logLine(`extract_context exit ${c2}: ${extractErr.slice(-900)}`);
        finish(false);
        return;
      }
      logLine("extract_context finished OK");
      finish(true);
    });
  });
}
