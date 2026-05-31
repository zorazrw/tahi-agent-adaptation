/** Slug for task-keyed memory/skill files (matches induce.py ``_TASK_STEM_RE``). */
const TASK_STEM_RE = /^[a-z0-9][a-z0-9_-]{0,99}$/i;

/** Normalize expertise category from the new-task panel to a memory/skill file stem. */
export function normalizeExpertiseTaskStem(value: string | undefined | null): string | null {
  const s = String(value ?? "").trim().toLowerCase();
  if (!s || !TASK_STEM_RE.test(s)) return null;
  return s;
}

export function expertiseTaskMemoryFileName(stem: string): string {
  return `${stem}.md`;
}
