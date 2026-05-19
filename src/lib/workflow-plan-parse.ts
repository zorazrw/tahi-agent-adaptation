export type WorkflowPlanTaskInput = {
  description: string;
  outputFiles: string[];
  verifiers: string[];
  children?: WorkflowPlanTaskInput[];
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function arrayOfStrings(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => String(v ?? "").trim()).filter(Boolean);
}

function tryParseJson(text: string): unknown | null {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

/** Unwrap JSON strings nested up to a few levels (some models double-encode tool args). */
export function unwrapJsonLike(value: unknown): unknown {
  let current = value;
  for (let i = 0; i < 4; i++) {
    if (typeof current !== "string") break;
    const trimmed = current.trim();
    if (!trimmed) break;
    const parsed = tryParseJson(trimmed);
    if (parsed === null) break;
    current = parsed;
  }
  return current;
}

function coerceTask(item: unknown): WorkflowPlanTaskInput | null {
  const row = asRecord(item);
  if (!row) return null;
  const description = row.description;
  if (typeof description !== "string" || !description.trim()) return null;
  const childrenRaw = row.children;
  const children = Array.isArray(childrenRaw)
    ? childrenRaw
        .map(coerceTask)
        .filter((c): c is WorkflowPlanTaskInput => c !== null)
    : undefined;
  return {
    description: description.trim(),
    outputFiles: arrayOfStrings(row.outputFiles ?? row.output_files),
    verifiers: arrayOfStrings(row.verifiers ?? row.verifier ?? row.criteria),
    children: children && children.length > 0 ? children : undefined,
  };
}

function coerceTasksArray(value: unknown): WorkflowPlanTaskInput[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const tasks = value.map(coerceTask).filter((t): t is WorkflowPlanTaskInput => t !== null);
  return tasks.length > 0 ? tasks : null;
}

/** Parse workflow plan ``tasks`` from tool args, text, or nested JSON strings. */
export function parseWorkflowPlanPayload(payload: unknown): WorkflowPlanTaskInput[] | null {
  const unwrapped = unwrapJsonLike(payload);
  if (Array.isArray(unwrapped)) return coerceTasksArray(unwrapped);
  const record = asRecord(unwrapped);
  if (!record) return null;
  const direct = coerceTasksArray(record.tasks);
  if (direct) return direct;
  const input = asRecord(record.input);
  if (input) return coerceTasksArray(input.tasks);
  return null;
}

export function isWorkflowPlanToolName(name: string): boolean {
  const n = name.trim();
  if (!n) return false;
  if (n === "workflow_plan" || n === "workflow" || n === "plan") return true;
  if (n.includes("WorkflowPlan")) return true;
  const lower = n.toLowerCase();
  return lower.includes("workflow") && lower.includes("plan");
}

/** Normalize tool args for display / storage when the model stringifies JSON. */
export function normalizeWorkflowPlanToolInput(args: unknown): Record<string, unknown> {
  const tasks = parseWorkflowPlanPayload(args);
  if (tasks) return { tasks };
  const record = asRecord(unwrapJsonLike(args));
  if (record) return record;
  return {};
}
