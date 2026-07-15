#!/usr/bin/env node
/**
 * Postinstall Python setup: scripts/.venv (induce.py, training server) and Tinker bridge.
 * Invoked from package.json postinstall after electron-rebuild.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { ensureUv } from "./resolve-uv.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const scriptsVenv = path.join(root, "scripts", ".venv");
const scriptsDir = path.join(root, "scripts");
const bridgeDir = path.join(root, "tinker-bridge");
const REQUIREMENTS = path.join(scriptsDir, "requirements.txt");

function venvPython() {
  if (process.platform === "win32") {
    return path.join(scriptsVenv, "Scripts", "python.exe");
  }
  const py3 = path.join(scriptsVenv, "bin", "python3");
  if (existsSync(py3)) return py3;
  return path.join(scriptsVenv, "bin", "python");
}

function syncScriptsDeps(uv) {
  if (!existsSync(REQUIREMENTS)) {
    console.error(`[postinstall] ${REQUIREMENTS} not found; cannot sync scripts deps.`);
    return 1;
  }

  const venvStatus =
    spawnSync(uv, ["venv", "--python", "3.11", scriptsVenv], { cwd: root, stdio: "inherit" }).status ?? 1;
  if (venvStatus !== 0) return venvStatus;

  return (
    spawnSync(uv, ["pip", "install", "-r", REQUIREMENTS, "--python", venvPython()], {
      cwd: root,
      stdio: "inherit",
    }).status ?? 1
  );
}

function syncTinkerBridge(uv) {
  if (!existsSync(path.join(bridgeDir, "pyproject.toml"))) {
    console.error("[postinstall] tinker-bridge/pyproject.toml not found; cannot sync Tinker bridge.");
    return 1;
  }

  return spawnSync(uv, ["sync", "--project", "tinker-bridge"], { cwd: root, stdio: "inherit" }).status ?? 1;
}

const uv = ensureUv(root);

const scriptsStatus = syncScriptsDeps(uv);
if (scriptsStatus !== 0) process.exit(scriptsStatus);

const bridgeStatus = syncTinkerBridge(uv);
process.exit(bridgeStatus);
