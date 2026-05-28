export const AUTO_INDUCTION_KEY = "agent-cowork-auto-context-induction";

/** Settings → Skills: auto-run induce.py after each workflow step when true. */
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
