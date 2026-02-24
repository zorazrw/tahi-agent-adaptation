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
  rmSync,
} from "fs";
import { join } from "path";
import { homedir } from "os";

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

/**
 * List all skills from both the app skills dir and user skills dir.
 * Parses YAML frontmatter from SKILL.md for name and description.
 */
export function listSkills(): SkillInfo[] {
  const appSkillsDir = getAppSkillsDir();
  const userSkillsDir = join(homedir(), ".claude", "skills");
  const skills: SkillInfo[] = [];

  // Scan app skills directory
  if (existsSync(appSkillsDir)) {
    try {
      for (const entry of readdirSync(appSkillsDir, { withFileTypes: true })) {
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
    rmSync(skillPath, { recursive: true });

    // Remove symlink from user skills dir
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

/** Read the full SKILL.md content from a skill directory. */
export function getSkillContent(skillPath: string): { content: string } | { error: string } {
  try {
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
 * Each app skill directory (containing a SKILL.md) gets symlinked with an
 * "agent-cowork--" prefix. Stale symlinks from removed app skills are cleaned up.
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

  // Discover valid app skills (directories containing SKILL.md)
  const appSkills = new Set<string>();
  try {
    for (const entry of readdirSync(appSkillsDir, { withFileTypes: true })) {
      if (
        entry.isDirectory() &&
        existsSync(join(appSkillsDir, entry.name, "SKILL.md"))
      ) {
        appSkills.add(entry.name);
      }
    }
  } catch {
    // Directory might be empty or unreadable
  }

  // Clean up stale symlinks (app skills that were removed)
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

  // Create/update symlinks for current app skills
  for (const skillName of appSkills) {
    const linkPath = join(userSkillsDir, `${APP_SKILL_PREFIX}${skillName}`);
    const targetPath = join(appSkillsDir, skillName);

    // Skip if symlink already exists and points to the right place
    try {
      const stat = lstatSync(linkPath);
      if (stat.isSymbolicLink() && readlinkSync(linkPath) === targetPath) {
        continue;
      }
      // Wrong target or not a symlink — remove and recreate
      unlinkSync(linkPath);
    } catch {
      // Doesn't exist yet
    }

    try {
      symlinkSync(targetPath, linkPath, "dir");
    } catch (err) {
      console.warn(
        `[skill-store] Failed to symlink skill "${skillName}":`,
        err
      );
    }
  }
}
