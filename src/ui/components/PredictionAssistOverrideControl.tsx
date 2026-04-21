import { useAppStore } from "../store/useAppStore";
import type { PredictionAssistMode, PredictionAssistOverride } from "../store/useAppStore";

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

export function PredictionAssistOverrideControl({ compact = false }: { compact?: boolean }) {
  const predictionAssistMode = useAppStore((s) => s.predictionAssistMode);
  const predictionAssistOverride = useAppStore((s) => s.predictionAssistOverride);
  const setPredictionAssistOverride = useAppStore((s) => s.setPredictionAssistOverride);

  return (
    <label
      className={`inline-flex items-center gap-2 rounded-lg border border-ink-900/10 bg-surface px-2.5 text-muted-foreground transition-colors hover:border-ink-900/20 hover:bg-ink-900/5 ${
        compact ? "h-8 min-h-8" : "h-8 min-h-8"
      }`}
      title={`Local prediction override. Default currently resolves to ${modeLabel(predictionAssistMode)} from Settings.`}
    >
      <span className="text-xs font-medium text-ink-600 whitespace-nowrap">User Simulator</span>
      <select
        value={predictionAssistOverride}
        onChange={(e) => setPredictionAssistOverride(e.target.value as PredictionAssistOverride)}
        className="bg-transparent text-sm text-ink-700 focus:outline-none pr-5 min-w-0"
        aria-label="Prediction assist override"
      >
        <option value="default">Default ({modeLabel(predictionAssistMode)})</option>
        <option value="off">Off</option>
        <option value="suggestion">Suggestion</option>
        <option value="autofill">Autofill</option>
      </select>
    </label>
  );
}
