/**
 * Locate or install uv for postinstall and runtime (Tinker bridge).
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import os from "node:os";
import path from "node:path";

export function hasCommand(cmd, args = ["--version"]) {
  return spawnSync(cmd, args, { encoding: "utf8" }).status === 0;
}

function pythonCmd() {
  return process.platform === "win32" ? "python" : "python3";
}

function uvBinaryName() {
  return process.platform === "win32" ? "uv.exe" : "uv";
}

function localUvDir(root) {
  return path.join(root, ".uv-bin");
}

function uvCandidates(root) {
  const home = process.env.HOME || process.env.USERPROFILE || os.homedir();
  const name = uvBinaryName();
  const local = localUvDir(root);
  return [
    path.join(local, name),
    path.join(local, "bin", name),
    path.join(local, "Scripts", name),
    path.join(home, ".local", "bin", name),
    path.join(home, ".cargo", "bin", name),
  ];
}

/** Return an absolute path to uv, or null if not found. */
export function findUv(root) {
  if (hasCommand("uv")) return "uv";
  for (const candidate of uvCandidates(root)) {
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

function installUvStandalone(root) {
  const installDir = localUvDir(root);
  mkdirSync(installDir, { recursive: true });
  const env = {
    ...process.env,
    UV_INSTALL_DIR: installDir,
    UV_NO_MODIFY_PATH: "1",
  };

  if (process.platform === "win32") {
    const status =
      spawnSync(
        "powershell",
        [
          "-ExecutionPolicy",
          "Bypass",
          "-NoProfile",
          "-Command",
          "irm https://astral.sh/uv/install.ps1 | iex",
        ],
        { env, stdio: "inherit" }
      ).status ?? 1;
    return status === 0;
  }

  if (hasCommand("curl")) {
    const status =
      spawnSync("sh", ["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], {
        env,
        stdio: "inherit",
      }).status ?? 1;
    return status === 0;
  }

  if (hasCommand("wget")) {
    const status =
      spawnSync("sh", ["-c", "wget -qO- https://astral.sh/uv/install.sh | sh"], {
        env,
        stdio: "inherit",
      }).status ?? 1;
    return status === 0;
  }

  return false;
}

function brewPythonPaths() {
  return ["/opt/homebrew/bin/python3.11", "/usr/local/bin/python3.11", "/opt/homebrew/bin/python3", "/usr/local/bin/python3"];
}

function ensurePython() {
  const py = pythonCmd();
  if (hasCommand(py)) return py;

  for (const candidate of brewPythonPaths()) {
    if (existsSync(candidate)) return candidate;
  }

  if (process.platform === "darwin" && hasCommand("brew")) {
    console.log("[postinstall] Python not found; installing via Homebrew...");
    const status = spawnSync("brew", ["install", "python@3.11"], { stdio: "inherit" }).status ?? 1;
    if (status === 0) {
      for (const candidate of brewPythonPaths()) {
        if (existsSync(candidate)) return candidate;
      }
      if (hasCommand("python3")) return "python3";
    }
  }

  return null;
}

function installUvViaPip(root, py) {
  const installDir = localUvDir(root);
  mkdirSync(installDir, { recursive: true });

  spawnSync(py, ["-m", "ensurepip", "--upgrade"], { stdio: "inherit" });

  const status =
    spawnSync(py, ["-m", "pip", "install", "--upgrade", "uv", "--target", installDir], {
      cwd: root,
      stdio: "inherit",
    }).status ?? 1;
  return status === 0;
}

/**
 * Find uv on PATH or in known locations; install if missing.
 * Exits the process on failure.
 */
export function ensureUv(root) {
  const existing = findUv(root);
  if (existing) return existing;

  console.log("[postinstall] uv not found; installing automatically...");

  if (installUvStandalone(root)) {
    const installed = findUv(root);
    if (installed) {
      console.log(`[postinstall] uv installed at ${installed}`);
      return installed;
    }
  }

  console.log("[postinstall] Standalone uv install failed; trying pip...");
  const py = ensurePython();
  if (!py) {
    console.error(
      "[postinstall] Could not install uv: Python 3 is required for the pip fallback.\n" +
        "  Install Python 3 from https://www.python.org/downloads/ then re-run: bun run sync:tinker-bridge"
    );
    process.exit(1);
  }

  if (!installUvViaPip(root, py)) {
    console.error(
      "[postinstall] Could not install uv.\n" +
        "  Install manually: https://docs.astral.sh/uv/getting-started/installation/\n" +
        "  Then re-run: bun run sync:tinker-bridge"
    );
    process.exit(1);
  }

  const installed = findUv(root);
  if (!installed) {
    console.error("[postinstall] uv was installed but could not be located under .uv-bin/");
    process.exit(1);
  }

  console.log(`[postinstall] uv installed at ${installed}`);
  return installed;
}
