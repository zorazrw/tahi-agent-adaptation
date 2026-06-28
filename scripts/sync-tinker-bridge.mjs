#!/usr/bin/env node
/**
 * Postinstall Python setup: scripts/.venv (induce.py, training server) and optional Tinker bridge.
 * Invoked from package.json postinstall after electron-rebuild.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const scriptsVenv = path.join(root, "scripts", ".venv");
const bridgeDir = path.join(root, "tinker-bridge");
const requirements = path.join(root, "scripts", "requirements.txt");

function hasCommand(cmd, args = ["--version"]) {
  return spawnSync(cmd, args, { encoding: "utf8" }).status === 0;
}

function pythonCmd() {
  return process.platform === "win32" ? "python" : "python3";
}

function venvPython() {
  if (process.platform === "win32") {
    return path.join(scriptsVenv, "Scripts", "python.exe");
  }
  const py3 = path.join(scriptsVenv, "bin", "python3");
  if (existsSync(py3)) return py3;
  return path.join(scriptsVenv, "bin", "python");
}

function syncScriptsDeps() {
  const py = pythonCmd();
  if (!hasCommand(py)) {
    console.warn(
      "[postinstall] Python not on PATH — skipped scripts deps (required for induce.py and training server).\n" +
        "  Install Python 3, then run: bun run sync:tinker-bridge"
    );
    return 0;
  }

  if (!existsSync(requirements)) {
    console.error("[postinstall] scripts/requirements.txt not found; cannot sync Python deps.");
    return 1;
  }

  if (hasCommand("uv")) {
    const venvStatus = spawnSync("uv", ["venv", scriptsVenv], { cwd: root, stdio: "inherit" }).status ?? 1;
    if (venvStatus !== 0) return venvStatus;
    return spawnSync("uv", ["pip", "install", "-r", requirements, "--python", venvPython()], {
      cwd: root,
      stdio: "inherit",
    }).status ?? 1;
  }

  if (!existsSync(venvPython())) {
    const venvStatus = spawnSync(py, ["-m", "venv", scriptsVenv], { cwd: root, stdio: "inherit" }).status ?? 1;
    if (venvStatus !== 0) return venvStatus;
  }
  return spawnSync(venvPython(), ["-m", "pip", "install", "-r", requirements], {
    cwd: root,
    stdio: "inherit",
  }).status ?? 1;
}

function syncTinkerBridge() {
  if (!existsSync(path.join(bridgeDir, "pyproject.toml"))) {
    console.error("[postinstall] tinker-bridge/pyproject.toml not found; cannot sync Tinker bridge.");
    return 1;
  }

  if (!hasCommand("uv")) {
    console.warn(
      "[postinstall] uv not on PATH — skipped tinker-bridge (optional unless you use the Tinker provider).\n" +
        "  Install uv: https://docs.astral.sh/uv/\n" +
        "  Then run: bun run sync:tinker-bridge"
    );
    return 0;
  }

  return spawnSync("uv", ["sync", "--project", "tinker-bridge"], { cwd: root, stdio: "inherit" }).status ?? 1;
}

const scriptsStatus = syncScriptsDeps();
if (scriptsStatus !== 0) process.exit(scriptsStatus);

const bridgeStatus = syncTinkerBridge();
process.exit(bridgeStatus);
