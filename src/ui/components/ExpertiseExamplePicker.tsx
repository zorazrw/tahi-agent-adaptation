import { useCallback, useMemo, useState } from "react";
import { useAppStore } from "../store/useAppStore";
import {
  EXPERTISE_TASK_CATEGORIES,
  formatInstructionOptionLabel,
  getExpertiseTask,
  getExpertiseTasks,
  type ExpertiseTaskCategory,
} from "../data/expertise-examples";

const selectClass =
  "h-9 w-full min-w-0 rounded-md border border-ink-900/15 bg-white px-3 py-1 text-sm text-ink-800 shadow-xs outline-none focus:border-primary/40 focus:ring-[3px] focus:ring-primary/10 disabled:cursor-not-allowed disabled:opacity-50";

export function ExpertiseExamplePicker() {
  const setPrompt = useAppStore((s) => s.setPrompt);

  const [taskCategory, setTaskCategory] = useState<ExpertiseTaskCategory | "">("");
  const [instructionId, setInstructionId] = useState<number | "">("");

  const instructions = useMemo(
    () => (taskCategory ? getExpertiseTasks(taskCategory) : []),
    [taskCategory],
  );

  const applyInstruction = useCallback(
    (category: ExpertiseTaskCategory, id: number) => {
      const task = getExpertiseTask(category, id);
      if (task) setPrompt(task.instruction);
    },
    [setPrompt],
  );

  const handleTaskChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const next = e.target.value as ExpertiseTaskCategory | "";
    setTaskCategory(next);
    setInstructionId("");
  };

  const handleInstructionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const raw = e.target.value;
    if (!raw || !taskCategory) {
      setInstructionId("");
      return;
    }
    const id = Number(raw);
    setInstructionId(id);
    applyInstruction(taskCategory, id);
  };

  return (
    <div className="flex flex-col gap-4 text-left">
      <div className="grid gap-1.5">
        <label htmlFor="expertise-task" className="text-xs font-medium text-muted-foreground">
          Task
        </label>
        <select
          id="expertise-task"
          className={selectClass}
          value={taskCategory}
          onChange={handleTaskChange}
        >
          <option value="">Select a task…</option>
          {EXPERTISE_TASK_CATEGORIES.map(({ key, label }) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>
      </div>

      <div className="grid gap-1.5">
        <label htmlFor="expertise-instruction" className="text-xs font-medium text-muted-foreground">
          Instruction
        </label>
        <select
          id="expertise-instruction"
          className={selectClass}
          value={instructionId === "" ? "" : String(instructionId)}
          onChange={handleInstructionChange}
          disabled={!taskCategory || instructions.length === 0}
        >
          <option value="">
            {!taskCategory
              ? "Select a task first…"
              : instructions.length === 0
                ? "No instructions available"
                : "Select an instruction…"}
          </option>
          {instructions.map((task) => (
            <option key={task.id} value={task.id}>
              {formatInstructionOptionLabel(task)}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
