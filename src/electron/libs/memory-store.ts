import {
  readFileSync,
  writeFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  unlinkSync,
  renameSync,
} from "fs";
import { join } from "path";
import { app } from "electron";

const LEGACY_MEM_FILENAME = "memory.md";
/** Any safe single-segment name ending in .md (no slashes, no leading dot). */
const MEMORY_MD_FILE_RE = /^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.md$/;

export type MemorySectionFile = {
  fileName: string;
  title: string;
  content: string;
};

/** e.g. coding-style.md → "Coding Style" */
export function titleFromMemoryFileName(fileName: string): string {
  const base = fileName.replace(/\.md$/i, "");
  if (!base) return fileName;
  return base
    .split(/[._-]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

export function getMemoriesDir(): string {
  return join(app.getPath("userData"), "memories");
}

/** Legacy single-file path (userData root). */
function getLegacyMemoryPath(): string {
  return join(app.getPath("userData"), LEGACY_MEM_FILENAME);
}

export function isValidMemoryFileName(name: string): boolean {
  return MEMORY_MD_FILE_RE.test(name) && !name.includes("/") && !name.includes("\\");
}

/** Create memories/ and migrate legacy root memory.md if present. */
export function ensureMemoriesDir(): void {
  const dir = getMemoriesDir();
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  const legacy = getLegacyMemoryPath();
  if (existsSync(legacy)) {
    const existing = listMemorySectionFiles();
    if (existing.length === 0) {
      const target = join(dir, "memory.md");
      try {
        renameSync(legacy, target);
      } catch {
        try {
          const text = readFileSync(legacy, "utf8");
          writeFileSync(target, text, "utf8");
          unlinkSync(legacy);
        } catch {
          /* ignore */
        }
      }
    }
  }
}

function listMemorySectionFiles(): string[] {
  const dir = getMemoriesDir();
  if (!existsSync(dir)) return [];
  try {
    return readdirSync(dir).filter((n) => MEMORY_MD_FILE_RE.test(n));
  } catch {
    return [];
  }
}

/** All memory sections for UI and IPC (sorted by file name). */
export function readAllMemorySections(): MemorySectionFile[] {
  ensureMemoriesDir();
  const names = listMemorySectionFiles().sort((a, b) => a.localeCompare(b));
  return names.map((fileName) => {
    const path = join(getMemoriesDir(), fileName);
    let content = "";
    try {
      content = readFileSync(path, "utf8");
    } catch {
      /* keep empty */
    }
    return { fileName, title: titleFromMemoryFileName(fileName), content };
  });
}

export type MemoryWriteSection = { fileName: string; content: string };

/** Write sections and optionally remove files. Validates file names. */
export function writeMemorySections(
  sections: MemoryWriteSection[],
  deletedFileNames?: string[]
): void {
  const dir = getMemoriesDir();
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  for (const name of deletedFileNames ?? []) {
    if (!isValidMemoryFileName(name)) continue;
    const p = join(dir, name);
    if (existsSync(p)) {
      try {
        unlinkSync(p);
      } catch {
        /* ignore */
      }
    }
  }

  for (const { fileName, content } of sections) {
    if (!isValidMemoryFileName(fileName)) continue;
    const body = content == null ? "" : String(content);
    writeFileSync(join(dir, fileName), body, "utf8");
  }

}

/** Non-empty prefix for LM: each file as its own block. */
export function readMemoryForPrompt(): string {
  ensureMemoriesDir();
  const sections = readAllMemorySections();
  const blocks: string[] = [];
  for (const { fileName, title, content } of sections) {
    const text = content.trim();
    if (!text) continue;
    blocks.push(`### Memory: ${title} (${fileName})\n\n${text}`);
  }
  if (blocks.length === 0) return "";
  return (
    "The following is persistent context from the operator's memory files (files in the memories folder). Apply when relevant:\n\n" +
    blocks.join("\n\n---\n\n") +
    "\n\n---\n\n"
  );
}
