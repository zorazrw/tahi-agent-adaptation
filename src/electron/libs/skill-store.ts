import { app } from "electron";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  symlinkSync,
  unlinkSync,
  lstatSync,
  readlinkSync,
  readFileSync,
  writeFileSync,
  rmSync,
} from "fs";
import { join } from "path";
import { homedir } from "os";

/** Top-level skill markdown files in the app skills dir (same rules as memory *.md). */
const FLAT_SKILL_MD_RE = /^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.md$/;

/** Sync mirror: each flat foo.md → _flat/foo/SKILL.md for Claude SDK directory layout. */
const FLAT_SYNC_ROOT = "_flat";

/** Prefix on app skill ids for flat files: flat_<stem> where stem is basename without .md */
const FLAT_SKILL_ID_PREFIX = "flat_";

/** Prefix for symlinks created in ~/.claude/skills/ to identify app-managed skills. */
const APP_SKILL_PREFIX = "agent-cowork--";

/** App-specific skills directory inside Electron userData. */
export function getAppSkillsDir(): string {
  return join(app.getPath("userData"), "skills");
}

/** Ensure the app skills directory exists. */
export function ensureAppSkillsDir(): void {
  const dir = getAppSkillsDir();
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
}

/** Parse YAML frontmatter from SKILL.md content. */
function parseFrontmatter(content: string): Record<string, string> {
  const match = content.match(/^---\s*\n([\s\S]*?)\n---/);
  if (!match) return {};
  const result: Record<string, string> = {};
  for (const line of match[1].split("\n")) {
    const colonIdx = line.indexOf(":");
    if (colonIdx > 0) {
      const key = line.slice(0, colonIdx).trim();
      const value = line.slice(colonIdx + 1).trim().replace(/^["']|["']$/g, "");
      if (key && value) result[key] = value;
    }
  }
  return result;
}

function flatSkillStemToId(stem: string): string {
  return `${FLAT_SKILL_ID_PREFIX}${stem}`;
}

function flatSkillIdToStem(id: string): string | null {
  if (!id.startsWith(FLAT_SKILL_ID_PREFIX)) return null;
  const stem = id.slice(FLAT_SKILL_ID_PREFIX.length);
  return stem || null;
}

export function isValidFlatSkillMdFileName(name: string): boolean {
  return FLAT_SKILL_MD_RE.test(name) && !name.includes("/") && !name.includes("\\");
}

function titleFromFlatSkillFileName(fileName: string): string {
  const base = fileName.replace(/\.md$/i, "");
  if (!base) return fileName;
  return base
    .split(/[._-]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

function listFlatSkillMdNames(appSkillsDir: string): string[] {
  if (!existsSync(appSkillsDir)) return [];
  try {
    return readdirSync(appSkillsDir, { withFileTypes: true })
      .filter((e) => e.isFile() && isValidFlatSkillMdFileName(e.name))
      .map((e) => e.name)
      .sort((a, b) => a.localeCompare(b));
  } catch {
    return [];
  }
}

export type FlatSkillSection = {
  fileName: string;
  title: string;
  content: string;
};

/** Top-level *.md files in the app skills folder (brain UI + SDK sync via _flat/). */
export function readAllFlatSkillSections(): FlatSkillSection[] {
  ensureAppSkillsDir();
  const appSkillsDir = getAppSkillsDir();
  const names = listFlatSkillMdNames(appSkillsDir);
  return names.map((fileName) => {
    const fp = join(appSkillsDir, fileName);
    let content = "";
    try {
      content = readFileSync(fp, "utf8");
    } catch {
      /* empty */
    }
    const fm = parseFrontmatter(content);
    const title = fm.name || titleFromFlatSkillFileName(fileName);
    return { fileName, title, content };
  });
}

export type FlatSkillWriteSection = { fileName: string; content: string };

export function writeFlatSkillSections(sections: FlatSkillWriteSection[], deletedFileNames?: string[]): void {
  const appSkillsDir = getAppSkillsDir();
  if (!existsSync(appSkillsDir)) mkdirSync(appSkillsDir, { recursive: true });

  const userSkillsDir = join(homedir(), ".claude", "skills");

  for (const name of deletedFileNames ?? []) {
    if (!isValidFlatSkillMdFileName(name)) continue;
    const fp = join(appSkillsDir, name);
    const stem = name.replace(/\.md$/i, "");
    if (existsSync(fp)) {
      try {
        unlinkSync(fp);
      } catch {
        /* ignore */
      }
    }
    try {
      rmSync(join(appSkillsDir, FLAT_SYNC_ROOT, stem), { recursive: true });
    } catch {
      /* ignore */
    }
    const linkPath = join(userSkillsDir, `${APP_SKILL_PREFIX}${flatSkillStemToId(stem)}`);
    try {
      if (existsSync(linkPath) && lstatSync(linkPath).isSymbolicLink()) unlinkSync(linkPath);
    } catch {
      /* ignore */
    }
  }

  for (const { fileName, content } of sections) {
    if (!isValidFlatSkillMdFileName(fileName)) continue;
    const body = content == null ? "" : String(content);
    writeFileSync(join(appSkillsDir, fileName), body, "utf8");
  }
}

/**
 * List all skills from both the app skills dir and user skills dir.
 * Parses YAML frontmatter from SKILL.md for name and description.
 */
export function listSkills(): SkillInfo[] {
  const appSkillsDir = getAppSkillsDir();
  const userSkillsDir = join(homedir(), ".claude", "skills");
  const skills: SkillInfo[] = [];

  // Scan app skills directory (folder skills + top-level *.md)
  if (existsSync(appSkillsDir)) {
    try {
      for (const entry of readdirSync(appSkillsDir, { withFileTypes: true })) {
        if (entry.name === FLAT_SYNC_ROOT) continue;
        if (!entry.isDirectory()) continue;
        const skillMdPath = join(appSkillsDir, entry.name, "SKILL.md");
        if (!existsSync(skillMdPath)) continue;
        try {
          const content = readFileSync(skillMdPath, "utf8");
          const fm = parseFrontmatter(content);
          skills.push({
            name: fm.name || entry.name,
            description: fm.description || "",
            dirName: entry.name,
            source: "app",
            path: join(appSkillsDir, entry.name),
          });
        } catch { /* skip unreadable */ }
      }
      for (const fileName of listFlatSkillMdNames(appSkillsDir)) {
        try {
          const fp = join(appSkillsDir, fileName);
          const content = readFileSync(fp, "utf8");
          const fm = parseFrontmatter(content);
          skills.push({
            name: fm.name || titleFromFlatSkillFileName(fileName),
            description: fm.description || "",
            dirName: fileName,
            source: "app",
            path: fp,
            isFlatMd: true,
          });
        } catch { /* skip */ }
      }
    } catch { /* directory unreadable */ }
  }

  // Scan user skills directory, skip app-managed symlinks
  if (existsSync(userSkillsDir)) {
    try {
      for (const entry of readdirSync(userSkillsDir, { withFileTypes: true })) {
        if (entry.name.startsWith(APP_SKILL_PREFIX)) continue;
        const entryPath = join(userSkillsDir, entry.name);
        const skillMdPath = join(entryPath, "SKILL.md");
        if (!existsSync(skillMdPath)) continue;
        try {
          const content = readFileSync(skillMdPath, "utf8");
          const fm = parseFrontmatter(content);
          skills.push({
            name: fm.name || entry.name,
            description: fm.description || "",
            dirName: entry.name,
            source: "user",
            path: entryPath,
          });
        } catch { /* skip unreadable */ }
      }
    } catch { /* directory unreadable */ }
  }

  return skills;
}

/** Remove an app-managed skill: delete from app dir and clean up its symlink. */
export function removeAppSkill(dirName: string): { success: boolean; error?: string } {
  try {
    const appSkillsDir = getAppSkillsDir();
    const skillPath = join(appSkillsDir, dirName);
    if (!existsSync(skillPath)) {
      return { success: false, error: "Skill not found" };
    }
    const stat = lstatSync(skillPath);
    if (stat.isFile() && isValidFlatSkillMdFileName(dirName)) {
      unlinkSync(skillPath);
      const stem = dirName.replace(/\.md$/i, "");
      try {
        rmSync(join(appSkillsDir, FLAT_SYNC_ROOT, stem), { recursive: true });
      } catch {
        /* ignore */
      }
      const linkPath = join(homedir(), ".claude", "skills", `${APP_SKILL_PREFIX}${flatSkillStemToId(stem)}`);
      try {
        if (existsSync(linkPath) && lstatSync(linkPath).isSymbolicLink()) unlinkSync(linkPath);
      } catch {
        /* ignore */
      }
      return { success: true };
    }

    rmSync(skillPath, { recursive: true });

    const linkPath = join(homedir(), ".claude", "skills", `${APP_SKILL_PREFIX}${dirName}`);
    try {
      if (lstatSync(linkPath).isSymbolicLink()) {
        unlinkSync(linkPath);
      }
    } catch { /* symlink may not exist */ }

    return { success: true };
  } catch (err) {
    return { success: false, error: err instanceof Error ? err.message : String(err) };
  }
}

/** Read SKILL.md from a skill directory, or a flat *.md file path. */
export function getSkillContent(skillPath: string): { content: string } | { error: string } {
  try {
    if (existsSync(skillPath)) {
      const st = lstatSync(skillPath);
      if (st.isFile() && skillPath.toLowerCase().endsWith(".md")) {
        return { content: readFileSync(skillPath, "utf8") };
      }
    }
    const mdPath = join(skillPath, "SKILL.md");
    if (!existsSync(mdPath)) {
      return { error: "SKILL.md not found" };
    }
    return { content: readFileSync(mdPath, "utf8") };
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
}

/**
 * Sync app-specific skills into ~/.claude/skills/ so the Agent SDK can discover them.
 *
 * - Each app folder with SKILL.md → symlink agent-cowork--&lt;dirName&gt;
 * - Each top-level *.md → mirror to _flat/&lt;stem&gt;/SKILL.md, symlink agent-cowork--flat_&lt;stem&gt;
 */
export function syncAppSkills(): void {
  const appSkillsDir = getAppSkillsDir();
  const userSkillsDir = join(homedir(), ".claude", "skills");

  if (!existsSync(appSkillsDir)) {
    mkdirSync(appSkillsDir, { recursive: true });
  }
  if (!existsSync(userSkillsDir)) {
    mkdirSync(userSkillsDir, { recursive: true });
  }

  const appSkills = new Set<string>();

  try {
    for (const entry of readdirSync(appSkillsDir, { withFileTypes: true })) {
      if (entry.name === FLAT_SYNC_ROOT) continue;
      if (
        entry.isDirectory() &&
        existsSync(join(appSkillsDir, entry.name, "SKILL.md"))
      ) {
        appSkills.add(entry.name);
      }
    }
  } catch {
    /* ignore */
  }

  for (const fileName of listFlatSkillMdNames(appSkillsDir)) {
    const stem = fileName.replace(/\.md$/i, "");
    const flatDir = join(appSkillsDir, FLAT_SYNC_ROOT, stem);
    const flatSkillPath = join(flatDir, "SKILL.md");
    try {
      mkdirSync(flatDir, { recursive: true });
      const body = readFileSync(join(appSkillsDir, fileName), "utf8");
      writeFileSync(flatSkillPath, body, "utf8");
      appSkills.add(flatSkillStemToId(stem));
    } catch (err) {
      console.warn(`[skill-store] Failed to mirror flat skill "${fileName}":`, err);
    }
  }

  try {
    for (const entry of readdirSync(userSkillsDir, { withFileTypes: true })) {
      if (!entry.name.startsWith(APP_SKILL_PREFIX)) continue;

      const linkPath = join(userSkillsDir, entry.name);
      const skillName = entry.name.slice(APP_SKILL_PREFIX.length);

      if (!appSkills.has(skillName)) {
        try {
          if (lstatSync(linkPath).isSymbolicLink()) {
            unlinkSync(linkPath);
          }
        } catch {
          /* ignore */
        }
      }
    }
  } catch {
    /* ignore */
  }

  for (const skillName of appSkills) {
    const linkPath = join(userSkillsDir, `${APP_SKILL_PREFIX}${skillName}`);
    const stem = flatSkillIdToStem(skillName);
    const targetPath =
      stem != null
        ? join(appSkillsDir, FLAT_SYNC_ROOT, stem)
        : join(appSkillsDir, skillName);

    try {
      const stat = lstatSync(linkPath);
      if (stat.isSymbolicLink() && readlinkSync(linkPath) === targetPath) {
        continue;
      }
      unlinkSync(linkPath);
    } catch {
      /* doesn't exist */
    }

    try {
      symlinkSync(targetPath, linkPath, "dir");
    } catch (err) {
      console.warn(`[skill-store] Failed to symlink skill "${skillName}":`, err);
    }
  }
}
