export const AUTO_INDUCTION_KEY = "agent-cowork-auto-context-induction";

/** Settings → Mode → Context Update: brain click runs induce.py when true (weight update when false). */
export function readStoredAutoInduction(): boolean {
  try {
    const v = localStorage.getItem(AUTO_INDUCTION_KEY);
    if (v === "false") return false;
    if (v === "true") return true;
  } catch {
    /* ignore */
  }
  return true;
}
