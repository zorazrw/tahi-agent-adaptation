import { join } from "path";
import { mkdirSync, existsSync, readFileSync } from "fs";
import type { Session } from "./session-store.js";

export function resolvePlanFilePath(session: Session): string {
  const base = session.cwd ?? process.cwd();
  const planDir = join(base, ".agent-cowork", "plans");
  if (!existsSync(planDir)) {
    mkdirSync(planDir, { recursive: true });
  }
  return join(planDir, `${session.id}.md`);
}

export function planFileExists(path: string): boolean {
  return existsSync(path);
}

export function readPlanFile(path: string): string | null {
  try {
    return readFileSync(path, "utf8");
  } catch {
    return null;
  }
}

export function buildPlanModeSystemPrompt(planPath: string, planExists: boolean): string {
  const fileInfo = planExists
    ? `A plan file already exists at ${planPath}. You can read it and make incremental edits using the edit tool.`
    : `No plan file exists yet. You should create your plan at ${planPath} using the write tool.`;

  return [
    "Plan mode is active. The user indicated that they do not want you to execute yet.",
    "You MUST NOT make any edits to project files, run destructive commands, or otherwise make changes to the system.",
    "You may ONLY edit or create the plan file mentioned below. All other actions should be read-only.",
    "",
    "## Plan File Info:",
    fileInfo,
    "You should build your plan incrementally by writing to or editing this file.",
    "NOTE that this is the only file you are allowed to edit - other than this you are only allowed to take READ-ONLY actions.",
    "",
    "## Plan Workflow",
    "",
    "### Phase 1: Initial Understanding",
    "Goal: Gain a comprehensive understanding of the user's request by reading through code.",
    "1. Focus on understanding the user's request and the code associated with their request.",
    "2. Read relevant files to understand the codebase structure and existing patterns.",
    "",
    "### Phase 2: Design",
    "Goal: Design an implementation approach.",
    "Think through the implementation based on the user's intent and your exploration.",
    "",
    "### Phase 3: Final Plan",
    "Goal: Write your final plan to the plan file (the only file you can edit).",
    "- Include only your recommended approach, not all alternatives.",
    "- Ensure that the plan file is concise enough to scan quickly, but detailed enough to execute effectively.",
    "- Include the paths of critical files to be modified.",
    "- Include a verification section describing how to test the changes.",
    "",
    "### Phase 4: Call plan_approve tool",
    "At the very end of your turn, once you are happy with your final plan file,",
    "you should always call plan_approve to indicate to the user that you are done planning.",
    "This is critical - your turn should only end with calling plan_approve.",
  ].join("\n");
}

export const PLAN_APPROVE_TOOL_DESCRIPTION =
  "Call this tool when you have finished writing the plan and want to ask the user for approval to switch to implementation mode.";
