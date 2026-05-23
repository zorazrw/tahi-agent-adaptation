import abstractWritingTasks from "../../../expertise-examples/abstract-writing/tasks.json";
import literatureWritingTasks from "../../../expertise-examples/literature-writing/tasks.json";
import dataVizTasks from "../../../expertise-examples/data-viz/tasks.json";
import dataVizHtmlTasks from "../../../expertise-examples/data-viz-html/tasks.json";

export type ExpertiseTaskCategory =
  | "abstract-writing"
  | "literature-writing"
  | "data-viz"
  | "data-viz-html";

/** Fields shared by every expertise-examples task JSON entry. */
export interface ExpertiseTaskInstance {
  id: number;
  type: string;
  instruction: string;
}

export const EXPERTISE_TASK_CATEGORIES: ReadonlyArray<{
  key: ExpertiseTaskCategory;
  label: string;
}> = [
  { key: "abstract-writing", label: "Abstract Writing" },
  { key: "literature-writing", label: "Literature Writing" },
  { key: "data-viz", label: "Data Viz" },
  { key: "data-viz-html", label: "Data Viz HTML" },
] as const;

function toExpertiseTasks(
  raw: ReadonlyArray<{ id: number; type: string; instruction: string }>,
): ExpertiseTaskInstance[] {
  return raw.map(({ id, type, instruction }) => ({ id, type, instruction }));
}

const TASKS_BY_CATEGORY: Record<ExpertiseTaskCategory, ExpertiseTaskInstance[]> = {
  "abstract-writing": toExpertiseTasks(abstractWritingTasks),
  "literature-writing": toExpertiseTasks(literatureWritingTasks),
  "data-viz": toExpertiseTasks(dataVizTasks),
  "data-viz-html": toExpertiseTasks(dataVizHtmlTasks),
};

export function getExpertiseTasks(category: ExpertiseTaskCategory): ExpertiseTaskInstance[] {
  return TASKS_BY_CATEGORY[category];
}

export function getExpertiseTask(
  category: ExpertiseTaskCategory,
  id: number,
): ExpertiseTaskInstance | undefined {
  return TASKS_BY_CATEGORY[category].find((t) => t.id === id);
}

export function formatInstructionOptionLabel(task: ExpertiseTaskInstance): string {
  const firstLine = task.instruction.split("\n")[0]?.trim() ?? "";
  const preview =
    firstLine.length > 72 ? `${firstLine.slice(0, 72)}…` : firstLine;
  return preview ? `Task ${task.id}: ${preview}` : `Task ${task.id}`;
}
