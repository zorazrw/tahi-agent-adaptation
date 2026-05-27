import { useAppStore } from "../store/useAppStore";
import type { PredictionAssistMode } from "../store/useAppStore";

function modeLabel(mode: PredictionAssistMode): string {
  switch (mode) {
    case "off":
      return "Off";
    case "suggestion":
      return "Suggestion";
    case "autofill":
      return "Autofill";
  }
}

/** Toolbar control styled like the “Work in a folder” button in `HomePromptInput`. */
export function PredictionAssistOverrideControl() {
  const predictionAssistMode = useAppStore((s) => s.predictionAssistMode);
  const setPredictionAssistMode = useAppStore((s) => s.setPredictionAssistMode);

  return (
    <label
      className="group inline-flex h-8 min-h-8 cursor-pointer items-center gap-2 rounded-lg px-2 text-sm leading-none text-muted-foreground transition-colors hover:bg-ink-900/5 hover:text-ink-700"
      title="User simulator: Off (no prediction), Suggestion (Tab to accept), or Autofill (send predicted message once)."
    >
      <svg viewBox="0 0 24 24" className="h-4 w-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
      <span className="shrink-0 whitespace-nowrap">User Simulator</span>
      <span className="inline-flex min-w-[7.5rem] items-center justify-center self-center rounded-md border border-ink-900/10 bg-transparent px-2 py-0.5 transition-colors group-hover:border-ink-900/15">
        <select
          value={predictionAssistMode}
          onChange={(e) => setPredictionAssistMode(e.target.value as PredictionAssistMode)}
          className="w-full min-w-0 cursor-pointer border-0 bg-transparent py-0 text-center text-sm font-normal leading-none text-ink-800 focus:outline-none focus:ring-0 group-hover:text-ink-900"
          aria-label="User simulator mode"
        >
          <option value="off">{modeLabel("off")}</option>
          <option value="suggestion">{modeLabel("suggestion")}</option>
          <option value="autofill">{modeLabel("autofill")}</option>
        </select>
      </span>
    </label>
  );
}
