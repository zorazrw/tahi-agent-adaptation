import { app } from "electron";
import Database from "better-sqlite3";
import { spawn } from "child_process";
import { randomUUID } from "crypto";
import { existsSync } from "fs";
import { cp, mkdir, readdir, rm, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join, relative } from "path";

/** Never include API keys or auth tokens in research bundles. */
const EXCLUDED_REL_PATHS = new Set([
  "api-config.json",
  "pi-agent/auth.json",
]);

function shouldExclude(relPosix: string): boolean {
  if (EXCLUDED_REL_PATHS.has(relPosix)) return true;
  const base = relPosix.split("/").pop() ?? relPosix;
  if (base === ".DS_Store") return true;
  return false;
}

function checkpointSessionsDb(dbPath: string): void {
  if (!existsSync(dbPath)) return;
  const db = new Database(dbPath);
  try {
    db.pragma("wal_checkpoint(TRUNCATE)");
  } finally {
    db.close();
  }
}

async function copyFilteredTree(srcRoot: string, destRoot: string): Promise<void> {
  async function walk(srcDir: string, destDir: string): Promise<void> {
    await mkdir(destDir, { recursive: true });
    const entries = await readdir(srcDir, { withFileTypes: true });
    for (const ent of entries) {
      const srcPath = join(srcDir, ent.name);
      const rel = relative(srcRoot, srcPath).replace(/\\/g, "/");
      if (shouldExclude(rel)) continue;
      const destPath = join(destDir, ent.name);
      if (ent.isDirectory()) {
        await walk(srcPath, destPath);
      } else if (ent.isFile()) {
        await cp(srcPath, destPath);
      }
    }
  }
  await walk(srcRoot, destRoot);
}

function runZip(stagingDir: string, zipPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (process.platform === "win32") {
      const ps = [
        "-NoProfile",
        "-Command",
        `Compress-Archive -Path '${stagingDir.replace(/'/g, "''")}\\*' -DestinationPath '${zipPath.replace(/'/g, "''")}' -Force`,
      ];
      const proc = spawn("powershell.exe", ps, { stdio: ["ignore", "pipe", "pipe"] });
      let err = "";
      proc.stderr?.on("data", (c) => {
        err += c.toString();
      });
      proc.on("error", reject);
      proc.on("close", (code) => {
        if (code === 0) resolve();
        else reject(new Error(err.trim() || `Compress-Archive exit ${code}`));
      });
      return;
    }

    const proc = spawn("zip", ["-r", "-q", zipPath, "."], { cwd: stagingDir, stdio: ["ignore", "pipe", "pipe"] });
    let err = "";
    proc.stderr?.on("data", (c) => {
      err += c.toString();
    });
    proc.on("error", (e) => {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") {
        reject(new Error("zip command not found. Install zip or use macOS/Linux."));
      } else {
        reject(e);
      }
    });
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(err.trim() || `zip exit ${code}`));
    });
  });
}

export type ExportRecordingsResult =
  | { success: true; path: string }
  | { success: false; canceled?: boolean; error?: string };

/**
 * Zip app userData (sessions DB + WAL, memories, skills, pi-agent, tasks, …).
 * Excludes api-config.json and pi-agent/auth.json.
 */
export async function exportRecordingsBundleToZip(zipPath: string): Promise<void> {
  const userData = app.getPath("userData");
  if (!existsSync(userData)) {
    throw new Error("App data folder not found.");
  }

  const dbPath = join(userData, "sessions.db");
  checkpointSessionsDb(dbPath);

  const staging = join(tmpdir(), `agent-cowork-export-${randomUUID()}`);
  const bundleRoot = join(staging, "agent-cowork-data");
  await mkdir(bundleRoot, { recursive: true });

  try {
    await copyFilteredTree(userData, bundleRoot);

    const manifest = {
      exportedAt: new Date().toISOString(),
      appVersion: app.getVersion(),
      platform: process.platform,
      note: "Research bundle: sessions.db, messages, memories, skills, pi-agent sessions. API keys excluded.",
    };
    await writeFile(join(bundleRoot, "export-manifest.json"), JSON.stringify(manifest, null, 2), "utf8");

    await runZip(staging, zipPath);
  } finally {
    await rm(staging, { recursive: true, force: true }).catch(() => {});
  }
}

/** Default filename for the save dialog. */
export function defaultRecordingsZipName(): string {
  const d = new Date().toISOString().slice(0, 10);
  return `agent-cowork-recordings-${d}.zip`;
}
