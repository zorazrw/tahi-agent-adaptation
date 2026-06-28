import { spawn, type ChildProcess } from "child_process";
import { existsSync } from "fs";
import { join } from "path";
import { app } from "electron";
import {
  getTrainingProxyBaseUrl,
  isFetchConnectionError,
  isTrainingProxyDisabled,
} from "./training-proxy.js";

const SERVER_SCRIPT = "server_online.py";
const CONFIG_FILE = "config_online.yaml";
const HEALTH_TIMEOUT_MS = 2_000;
const STARTUP_POLL_MS = 500;
const STARTUP_TIMEOUT_MS = 45_000;

let child: ChildProcess | null = null;
let startedByApp = false;
let startPromise: Promise<void> | null = null;
let weightUpdateModeActive = false;

function scriptsRootDir(): string | null {
  const hasServer = (dir: string) =>
    existsSync(join(dir, SERVER_SCRIPT)) && existsSync(join(dir, CONFIG_FILE));

  if (app.isPackaged) {
    const bundled = join(process.resourcesPath, "scripts");
    if (hasServer(bundled)) return bundled;
    return null;
  }
  const dev = join(app.getAppPath(), "scripts");
  if (hasServer(dev)) return dev;
  return null;
}

function pythonExecutable(): string {
  return process.platform === "win32" ? "python" : "python3";
}

async function isTrainingServerHealthy(baseUrl: string): Promise<boolean> {
  try {
    const res = await fetch(`${baseUrl}/healthz`, {
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    if (!res.ok) return false;
    const data = (await res.json()) as { ok?: boolean };
    return data.ok === true;
  } catch {
    return false;
  }
}

async function waitForHealthy(baseUrl: string): Promise<boolean> {
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (await isTrainingServerHealthy(baseUrl)) return true;
    await new Promise((resolve) => setTimeout(resolve, STARTUP_POLL_MS));
  }
  return false;
}

function spawnTrainingServer(scriptsDir: string): void {
  const py = pythonExecutable();
  const proc = spawn(py, [SERVER_SCRIPT, "--config", CONFIG_FILE], {
    cwd: scriptsDir,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child = proc;
  startedByApp = true;

  proc.stdout?.on("data", (chunk: Buffer) => {
    for (const line of chunk.toString().split(/\r?\n/)) {
      if (line.trim()) console.log(`[training-server] ${line}`);
    }
  });
  proc.stderr?.on("data", (chunk: Buffer) => {
    for (const line of chunk.toString().split(/\r?\n/)) {
      if (line.trim()) console.error(`[training-server] ${line}`);
    }
  });
  proc.on("error", (err) => {
    console.error(`[training-server] process error: ${err.message}`);
  });
  proc.on("close", (code, signal) => {
    if (startedByApp) {
      console.log(`[training-server] exited code=${code ?? "null"} signal=${signal ?? "null"}`);
    }
    if (child === proc) {
      child = null;
      startedByApp = false;
    }
  });

  console.log(
    `[training-server] started ${py} ${SERVER_SCRIPT} --config ${CONFIG_FILE} (cwd=${scriptsDir})`,
  );
}

async function ensureTrainingServerRunning(): Promise<void> {
  if (!weightUpdateModeActive || isTrainingProxyDisabled()) return;

  const baseUrl = getTrainingProxyBaseUrl();
  if (!baseUrl) return;

  if (await isTrainingServerHealthy(baseUrl)) {
    if (!weightUpdateModeActive) return;
    console.log(`[training-server] already running at ${baseUrl}`);
    return;
  }

  const scriptsDir = scriptsRootDir();
  if (!scriptsDir) {
    console.warn("[training-server] scripts/ not found; cannot auto-start training server");
    return;
  }

  if (child && startedByApp) {
    const healthy = await waitForHealthy(baseUrl);
    if (!weightUpdateModeActive) {
      stopTrainingServerIfStarted();
      return;
    }
    if (healthy) return;
  }

  if (!weightUpdateModeActive) return;

  spawnTrainingServer(scriptsDir);
  const healthy = await waitForHealthy(baseUrl);
  if (!weightUpdateModeActive) {
    stopTrainingServerIfStarted();
    return;
  }
  if (!healthy) {
    console.warn(
      `[training-server] started but ${baseUrl}/healthz did not become ready within ${STARTUP_TIMEOUT_MS}ms`,
    );
  } else {
    console.log(`[training-server] ready at ${baseUrl}`);
  }
}

/** Start/stop the bundled online training server when entering/leaving Weight Update mode. */
export function syncTrainingServer(weightUpdateMode: boolean): void {
  weightUpdateModeActive = weightUpdateMode;

  if (!weightUpdateMode) {
    startPromise = null;
    stopTrainingServerIfStarted();
    return;
  }

  if (isTrainingProxyDisabled()) return;

  if (startPromise) return;
  startPromise = ensureTrainingServerRunning()
    .catch((err) => {
      const message = err instanceof Error ? err.message : String(err);
      if (!isFetchConnectionError(err)) {
        console.warn(`[training-server] failed to start: ${message}`);
      }
    })
    .finally(() => {
      startPromise = null;
    });
}

export function stopTrainingServerIfStarted(): void {
  if (!child || !startedByApp) return;
  const proc = child;
  child = null;
  startedByApp = false;
  console.log("[training-server] stopping (Weight Update mode disabled)");
  proc.kill("SIGTERM");
}
