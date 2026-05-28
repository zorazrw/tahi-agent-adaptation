export const DEFAULT_TRAINING_PROXY_BASE_URL = "http://localhost:8000";

export function getTrainingProxyBaseUrl(): string {
  const fromEnv = process.env.AGENT_COWORK_PROXY_URL?.trim();
  if (fromEnv === "disabled") return "";
  if (fromEnv) return fromEnv.replace(/\/+$/, "");
  return DEFAULT_TRAINING_PROXY_BASE_URL;
}

export function isTrainingProxyDisabled(): boolean {
  return process.env.AGENT_COWORK_PROXY_URL?.trim() === "disabled";
}

export function isFetchConnectionError(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return /fetch failed|ECONNREFUSED|ENOTFOUND|network|socket/i.test(msg);
}

export const TRAINING_PROXY_START_HINT =
  "Start the training proxy in a separate terminal: python scripts/server.py (default http://localhost:8000)";
