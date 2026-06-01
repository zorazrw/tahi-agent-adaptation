export const AUTO_INDUCTION_KEY = "agent-cowork-auto-context-induction";

/** Settings → Mode → Context Update (true) vs Weight Update (false) for brain single-click. */
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
