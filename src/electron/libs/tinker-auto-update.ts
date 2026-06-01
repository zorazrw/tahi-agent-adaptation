import { BrowserWindow } from "electron";
import {
  getAgentSettings,
  getOpenAICompatibleProviderConfig,
  getTinkerProviderConfig,
  saveAgentSettings,
  saveOpenAICompatibleProviderConfig,
  saveTinkerProviderConfig,
} from "./pi-config.js";
import {
  getTrainingProxyBaseUrl,
  isFetchConnectionError,
  isTrainingProxyDisabled,
  TRAINING_PROXY_START_HINT,
} from "./training-proxy.js";

/** Matches `ModelUpdate.to_event()` in scripts/server.py (`model_path` is the sampler checkpoint). */
export type TinkerModelUpdateEvent = {
  slug: string;
  model_path: string;
  base_model: string | null;
  renderer_name: string | null;
  mode: string;
  updated_at: number;
  state_path?: string | null;
};

function persistTinkerCheckpoint(event: TinkerModelUpdateEvent): boolean {
  const existing = getTinkerProviderConfig();
  if (!existing) return false;

  const samplerPath = event.model_path.trim();
  if (!samplerPath.startsWith("tinker://")) return false;

  const baseModel = event.base_model?.trim() || existing.model.baseModel;
  const slug = event.slug.trim() || existing.model.id;

  saveTinkerProviderConfig({
    baseUrl: existing.baseUrl,
    model: slug,
    baseModel,
    modelPath: samplerPath,
    rendererName: event.renderer_name?.trim() || existing.model.rendererName,
    reasoning: existing.model.reasoning,
    contextWindow: existing.model.contextWindow,
    maxTokens: existing.model.maxTokens,
  });
  return true;
}

const POLL_INTERVAL_MS = 5_000;
const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_DELAY_MS = 30_000;
const OFFLINE_LOG_INTERVAL_MS = 60_000;
const IPC_CHANNEL = "tinker-model-updated" as const;
const OPENAI_COMPATIBLE_PROVIDER = "openai-compatible" as const;
const TINKER_PROVIDER = "tinker" as const;

class TinkerAutoUpdateWatcher {
  private stopped = false;
  private abortController: AbortController | null = null;
  private reconnectDelayMs = RECONNECT_BASE_DELAY_MS;
  private lastAppliedAt: number | null = null;
  private lastAppliedPath: string | null = null;
  private lastOfflineLogAt = 0;

  constructor(
    private readonly getWindows: () => BrowserWindow[],
    private readonly proxyBaseUrl: string,
  ) {}

  async start(): Promise<void> {
    while (!this.stopped) {
      try {
        await this.runOnce();
        this.reconnectDelayMs = RECONNECT_BASE_DELAY_MS;
      } catch (error) {
        if (this.stopped) return;
        if (isFetchConnectionError(error)) {
          this.logProxyOfflineOnce();
        } else {
          const message = error instanceof Error ? error.message : String(error);
          console.warn(
            `[tinker-auto-update] stream error (${message}); falling back to polling for ${POLL_INTERVAL_MS}ms`,
          );
          try {
            await this.pollOnce();
          } catch (pollErr) {
            if (!isFetchConnectionError(pollErr)) {
              const pollMessage = pollErr instanceof Error ? pollErr.message : String(pollErr);
              console.warn(`[tinker-auto-update] poll failed: ${pollMessage}`);
            }
          }
        }
      }
      if (this.stopped) return;
      await sleep(this.reconnectDelayMs);
      this.reconnectDelayMs = Math.min(this.reconnectDelayMs * 2, RECONNECT_MAX_DELAY_MS);
    }
  }

  stop(): void {
    this.stopped = true;
    this.abortController?.abort();
  }

  private logProxyOfflineOnce(): void {
    const now = Date.now();
    if (now - this.lastOfflineLogAt < OFFLINE_LOG_INTERVAL_MS) return;
    this.lastOfflineLogAt = now;
    console.warn(
      `[tinker-auto-update] training proxy not reachable at ${this.proxyBaseUrl}. ${TRAINING_PROXY_START_HINT}`,
    );
  }

  private async runOnce(): Promise<void> {
    this.abortController = new AbortController();
    const url = `${this.proxyBaseUrl}/v1/tinker/events`;
    const res = await fetch(url, {
      method: "GET",
      headers: { Accept: "text/event-stream" },
      signal: this.abortController.signal,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    if (!res.body) throw new Error("Empty SSE response body");

    console.log(`[tinker-auto-update] subscribed to ${url}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (!this.stopped) {
        const { value, done } = await reader.read();
        if (done) return;
        buffer += decoder.decode(value, { stream: true });

        let sepIndex: number;
        while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, sepIndex);
          buffer = buffer.slice(sepIndex + 2);
          await this.handleRawEvent(rawEvent);
        }
      }
    } finally {
      try { reader.cancel(); } catch { /* noop */ }
    }
  }

  private async handleRawEvent(raw: string): Promise<void> {
    let eventName: string | null = null;
    const dataLines: string[] = [];
    for (const line of raw.split("\n")) {
      if (!line || line.startsWith(":")) continue;
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (eventName !== "model-update" || dataLines.length === 0) return;

    let parsed: TinkerModelUpdateEvent;
    try {
      parsed = JSON.parse(dataLines.join("\n")) as TinkerModelUpdateEvent;
    } catch (error) {
      console.warn("[tinker-auto-update] could not parse SSE payload:", error);
      return;
    }
    await this.applyUpdate(parsed);
  }

  private async pollOnce(): Promise<void> {
    const res = await fetch(`${this.proxyBaseUrl}/v1/tinker/current`, {
      headers: { Accept: "application/json" },
    });
    if (res.status === 204) return;
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    await this.applyUpdate((await res.json()) as TinkerModelUpdateEvent);
  }

  private async applyUpdate(event: TinkerModelUpdateEvent): Promise<void> {
    if (!event.slug || !event.model_path) return;
    const modelPath = event.model_path.trim();
    if (
      this.lastAppliedPath === modelPath
      && this.lastAppliedAt !== null
      && event.updated_at <= this.lastAppliedAt
    ) {
      return;
    }

    const providerConfig = getOpenAICompatibleProviderConfig();
    if (providerConfig) {
      try {
        saveOpenAICompatibleProviderConfig({
          baseUrl: providerConfig.baseUrl,
          model: event.slug,
          apiFormat: providerConfig.apiFormat,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[tinker-auto-update] could not persist OpenAI-compatible slug: ${message}`);
      }
    }

    const tinkerUpdated = persistTinkerCheckpoint(event);
    if (tinkerUpdated) {
      console.log(`[tinker-auto-update] updated Tinker checkpoint path=${event.model_path}`);
    }

    try {
      const settings = getAgentSettings();
      const shouldUpdateDefaultModel =
        (settings.defaultProvider === OPENAI_COMPATIBLE_PROVIDER && providerConfig)
        || (settings.defaultProvider === TINKER_PROVIDER && tinkerUpdated);
      if (shouldUpdateDefaultModel) {
        await saveAgentSettings({
          defaultProvider: settings.defaultProvider,
          defaultModel: event.slug,
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[tinker-auto-update] could not persist default model: ${message}`);
    }

    this.lastAppliedAt = event.updated_at;
    this.lastAppliedPath = modelPath;
    console.log(`[tinker-auto-update] applied slug=${event.slug} mode=${event.mode}`);

    for (const win of this.getWindows()) {
      if (win.isDestroyed()) continue;
      try {
        win.webContents.send(IPC_CHANNEL, event);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.warn(`[tinker-auto-update] failed to notify window: ${message}`);
      }
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

let activeWatcher: TinkerAutoUpdateWatcher | null = null;

export function startTinkerAutoUpdateWatcher(): void {
  if (activeWatcher || isTrainingProxyDisabled()) return;
  const baseUrl = getTrainingProxyBaseUrl();
  if (!baseUrl) return;
  activeWatcher = new TinkerAutoUpdateWatcher(() => BrowserWindow.getAllWindows(), baseUrl);
  console.log(`[tinker-auto-update] starting watcher against ${baseUrl}`);
  void activeWatcher.start();
}

export function stopTinkerAutoUpdateWatcher(): void {
  activeWatcher?.stop();
  activeWatcher = null;
}
