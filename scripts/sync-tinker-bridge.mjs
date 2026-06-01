#!/usr/bin/env node
/**
 * Install Python deps for the Tinker bridge (`uv sync --project tinker-bridge`).
 * Invoked from package.json postinstall after electron-rebuild.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const bridgeDir = path.join(root, "tinker-bridge");

if (!existsSync(path.join(bridgeDir, "pyproject.toml"))) {
  console.error("[postinstall] tinker-bridge/pyproject.toml not found; cannot sync Python deps.");
  process.exit(1);
}

function hasUv() {
  return spawnSync("uv", ["--version"], { encoding: "utf8" }).status === 0;
}

if (!hasUv()) {
  console.warn(
    "[postinstall] uv not on PATH — skipped tinker-bridge (optional unless you use the Tinker provider).\n" +
      "  Install uv: https://docs.astral.sh/uv/\n" +
      "  Then run: bun run sync:tinker-bridge"
  );
  process.exit(0);
}

// Same as manual: `uv sync --project tinker-bridge` from repo root (matches tinker-provider.ts).
const result = spawnSync("uv", ["sync", "--project", "tinker-bridge"], {
  cwd: root,
  stdio: "inherit",
});

process.exit(result.status ?? 1);
