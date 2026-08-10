/** Slug for task-keyed memory/skill files (matches induce.py ``_TASK_STEM_RE``). */
const TASK_STEM_RE = /^[a-z0-9][a-z0-9_-]{0,99}$/i;

/** Normalize expertise category from the new-task panel to a memory/skill file stem. */
export function normalizeExpertiseTaskStem(value: string | undefined | null): string | null {
  let s = String(value ?? "").trim().toLowerCase();
  // Heldout dropdown keys are `{task}-heldout`; memory/skills live under `{task}`.
  if (s.endsWith("-heldout")) {
    s = s.slice(0, -"-heldout".length);
  }
  if (!s || !TASK_STEM_RE.test(s)) return null;
  return s;
}

export function expertiseTaskMemoryFileName(stem: string): string {
  return `${stem}.md`;
}
