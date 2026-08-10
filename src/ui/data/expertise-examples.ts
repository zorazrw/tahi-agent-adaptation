export type ExpertiseTaskCategory = string;

/** Fields shared by every expertise-examples task JSON entry. */
export interface ExpertiseTaskInstance {
  id: number;
  type: string;
  instruction: string;
}

interface RawTask {
  id?: unknown;
  type?: unknown;
  instruction?: unknown;
}

function toCategoryLabel(slug: string): string {
  return slug
    .split("-")
    .map((part) => (part ? `${part[0].toUpperCase()}${part.slice(1)}` : part))
    .join(" ");
}

/** Display labels for heldout category keys (override auto Title Case). */
const HELDOUT_CATEGORY_LABELS: Record<string, string> = {
  "abstract-writing-heldout": "Abstract Writing (heldout)",
  "data-viz-html-heldout": "Data Viz HTML (heldout)",
};

function heldoutCategoryKey(slug: string): string {
  return `${slug}-heldout`;
}

function categoryLabel(key: ExpertiseTaskCategory): string {
  if (HELDOUT_CATEGORY_LABELS[key]) return HELDOUT_CATEGORY_LABELS[key];
  if (key.endsWith("-heldout")) {
    return `${toCategoryLabel(key.slice(0, -"-heldout".length))} (heldout)`;
  }
  return toCategoryLabel(key);
}

function unwrapTaskPayload(raw: unknown): unknown {
  if (Array.isArray(raw)) return raw;
  if (!raw || typeof raw !== "object") return raw;
  const record = raw as { default?: unknown; tasks?: unknown };
  if (Array.isArray(record.default)) return record.default;
  if (Array.isArray(record.tasks)) return record.tasks;
  return raw;
}

function normalizeTasks(raw: unknown): ExpertiseTaskInstance[] {
  const unwrapped = unwrapTaskPayload(raw);
  const candidates: unknown[] = Array.isArray(unwrapped)
    ? unwrapped
    : unwrapped && typeof unwrapped === "object" && Array.isArray((unwrapped as { tasks?: unknown }).tasks)
      ? ((unwrapped as { tasks: unknown[] }).tasks ?? [])
      : [];

  const normalized: ExpertiseTaskInstance[] = [];
  for (const item of candidates) {
    if (!item || typeof item !== "object") continue;
    const task = item as RawTask;
    if (typeof task.id !== "number") continue;
    if (typeof task.instruction !== "string") continue;
    normalized.push({
      id: task.id,
      type: typeof task.type === "string" ? task.type : "Task",
      instruction: task.instruction,
    });
  }

  return normalized;
}

const rawTaskJsonModules = import.meta.glob("../../../expertise-examples/*/tasks.json", {
  eager: true,
}) as Record<string, unknown>;

const rawHeldoutJsonModules = import.meta.glob(
  "../../../expertise-examples/*/heldout.json",
  { eager: true },
) as Record<string, unknown>;

const taskEntries = Object.entries(rawTaskJsonModules)
  .map(([path, raw]) => {
    const match = path.match(/\/expertise-examples\/([^/]+)\/tasks\.json$/);
    if (!match) return null;
    const category = match[1];
    return [category, normalizeTasks(raw)] as const;
  })
  .filter((entry): entry is readonly [ExpertiseTaskCategory, ExpertiseTaskInstance[]] =>
    Boolean(entry),
  );

const heldoutEntries = Object.entries(rawHeldoutJsonModules)
  .map(([path, raw]) => {
    const match = path.match(/\/expertise-examples\/([^/]+)\/heldout\.json$/);
    if (!match) return null;
    const category = heldoutCategoryKey(match[1]);
    return [category, normalizeTasks(raw)] as const;
  })
  .filter((entry): entry is readonly [ExpertiseTaskCategory, ExpertiseTaskInstance[]] =>
    Boolean(entry),
  );

const allEntries = [...taskEntries, ...heldoutEntries].sort(([a], [b]) =>
  a.localeCompare(b),
);

const TASKS_BY_CATEGORY: Record<ExpertiseTaskCategory, ExpertiseTaskInstance[]> =
  Object.fromEntries(allEntries);

export const EXPERTISE_TASK_CATEGORIES: ReadonlyArray<{
  key: ExpertiseTaskCategory;
  label: string;
}> = allEntries.map(([key]) => ({ key, label: categoryLabel(key) }));

export function getExpertiseTasks(category: ExpertiseTaskCategory): ExpertiseTaskInstance[] {
  return TASKS_BY_CATEGORY[category] ?? [];
}

export function getExpertiseTask(
  category: ExpertiseTaskCategory,
  id: number,
): ExpertiseTaskInstance | undefined {
  return TASKS_BY_CATEGORY[category]?.find((t) => t.id === id);
}

export function formatInstructionOptionLabel(task: ExpertiseTaskInstance): string {
  const firstLine = task.instruction.split("\n")[0]?.trim() ?? "";
  const preview =
    firstLine.length > 72 ? `${firstLine.slice(0, 72)}…` : firstLine;
  return preview ? `Task ${task.id}: ${preview}` : `Task ${task.id}`;
}
