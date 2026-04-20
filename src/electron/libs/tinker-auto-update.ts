import { BrowserWindow } from "electron";
import {
  getAgentSettings,
  getOpenAICompatibleProviderConfig,
  saveAgentSettings,
  saveOpenAICompatibleProviderConfig,
} from "./pi-config.js";

/**
 * Payload matching the server-side broadcast schema in
 * {@link ../../../scripts/server.py} (`_broadcast_model_update`).
 */
export type TinkerModelUpdateEvent = {
  slug: string;
  model_path: string;
  base_model: string | null;
  renderer_name: string | null;
  mode: string;
  updated_at: number;
};

const DEFAULT_PROXY_BASE_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 5_000;
const RECONNECT_BASE_DELAY_MS = 1_000;
const RECONNECT_MAX_DELAY_MS = 30_000;
const IPC_CHANNEL = "tinker-model-updated" as const;
const OPENAI_COMPATIBLE_PROVIDER = "openai-compatible" as const;

function getProxyBaseUrl(): string {
  const fromEnv = process.env.AGENT_COWORK_PROXY_URL?.trim();
  if (fromEnv) return fromEnv.replace(/\/+$/, "");
  return DEFAULT_PROXY_BASE_URL;
}

class TinkerAutoUpdateWatcher {
  private stopped = false;
  private abortController: AbortController | null = null;
  private reconnectDelayMs = RECONNECT_BASE_DELAY_MS;
  private lastAppliedAt: number | null = null;

  constructor(
    private readonly getWindows: () => BrowserWindow[],
    private readonly proxyBaseUrl: string,
  ) {}

  async start(): Promise<void> {
    while (!this.stopped) {
      try {
        await this.runOnce();
        // Clean disconnect – reset backoff and reconnect promptly.
        this.reconnectDelayMs = RECONNECT_BASE_DELAY_MS;
      } catch (error) {
        if (this.stopped) return;
        const message = error instanceof Error ? error.message : String(error);
        console.warn(
          `[tinker-auto-update] stream error (${message}); falling back to polling for ${POLL_INTERVAL_MS}ms`,
        );
        try {
          await this.pollOnce();
        } catch (pollErr) {
          const pollMessage = pollErr instanceof Error ? pollErr.message : String(pollErr);
          console.warn(`[tinker-auto-update] poll failed: ${pollMessage}`);
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

  /**
   * Open an SSE connection to the proxy and dispatch every `model-update`
   * event. Returns when the stream closes (network error, proxy restart, etc.).
   */
  private async runOnce(): Promise<void> {
    this.abortController = new AbortController();
    const url = `${this.proxyBaseUrl}/v1/tinker/events`;
    const res = await fetch(url, {
      method: "GET",
      headers: { Accept: "text/event-stream" },
      signal: this.abortController.signal,
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`);
    }
    if (!res.body) {
      throw new Error("Empty SSE response body");
    }

    console.log(`[tinker-auto-update] subscribed to ${url}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (!this.stopped) {
        const { value, done } = await reader.read();
        if (done) return;
        buffer += decoder.decode(value, { stream: true });

        // Events are separated by blank lines per the SSE spec.
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
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
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

  /**
   * Fallback path when SSE is unavailable. We hit `/v1/tinker/current` and
   * apply the latest update if it's newer than the last one we applied. This
   * is only reached when the SSE handshake fails; normal operation is push-only.
   */
  private async pollOnce(): Promise<void> {
    const url = `${this.proxyBaseUrl}/v1/tinker/current`;
    const res = await fetch(url, { method: "GET", headers: { Accept: "application/json" } });
    if (res.status === 204) return;
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    const event = (await res.json()) as TinkerModelUpdateEvent;
    await this.applyUpdate(event);
  }

  /**
   * Idempotently apply a model update:
   *
   * 1. Rewrite the OpenAI-compatible provider's single-model slot to the new
   *    slug. This is the provider that points at the agent-cowork proxy, so
   *    it's the one that actually routes slugs to fine-tuned checkpoints.
   * 2. If the user's current default provider is OpenAI-compatible, update
   *    their `defaultModel` to the new slug too. We intentionally do NOT
   *    touch `defaultProvider` or the Tinker provider config – users who are
   *    using a different provider keep their setup, and the Tinker direct
   *    bridge (which doesn't go through our proxy) is left alone.
   * 3. Notify every open renderer window so the Settings UI can refresh.
   */
  private async applyUpdate(event: TinkerModelUpdateEvent): Promise<void> {
    if (!event.slug || !event.model_path) {
      console.warn("[tinker-auto-update] ignoring malformed event:", event);
      return;
    }
    if (this.lastAppliedAt !== null && event.updated_at <= this.lastAppliedAt) {
      return;
    }

    const providerConfig = getOpenAICompatibleProviderConfig();
    if (!providerConfig) {
      console.warn(
        `[tinker-auto-update] OpenAI-compatible provider not configured; cannot surface slug ${event.slug}. ` +
        `Configure Settings → API → OpenAI-Compatible Endpoint pointing at the agent-cowork proxy to receive updates.`,
      );
      return;
    }

    try {
      saveOpenAICompatibleProviderConfig({
        baseUrl: providerConfig.baseUrl,
        model: event.slug,
        apiFormat: providerConfig.apiFormat,
        // apiKey intentionally omitted so the existing auth.json entry is preserved.
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[tinker-auto-update] could not persist new slug ${event.slug}: ${message}`);
      return;
    }

    try {
      const settings = getAgentSettings();
      if (settings.defaultProvider === OPENAI_COMPATIBLE_PROVIDER) {
        // Preserve defaultProvider explicitly; only the model slug changes.
        await saveAgentSettings({
          defaultProvider: settings.defaultProvider,
          defaultModel: event.slug,
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[tinker-auto-update] could not update default model: ${message}`);
    }

    this.lastAppliedAt = event.updated_at;
    console.log(
      `[tinker-auto-update] applied slug=${event.slug} base_model=${event.base_model ?? "?"} mode=${event.mode}`,
    );

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

/**
 * Start a singleton watcher that keeps the OpenAI-compatible provider config /
 * default model in sync with the training proxy. Safe to call multiple times;
 * extra invocations are no-ops.
 *
 * The watcher is long-lived: it runs for the lifetime of the Electron process
 * and reconnects with exponential backoff if the proxy is down.
 */
export function startTinkerAutoUpdateWatcher(): void {
  if (activeWatcher) return;
  // Opt-out for users who don't run the training proxy locally.
  if (process.env.AGENT_COWORK_PROXY_URL === "disabled") {
    console.log("[tinker-auto-update] disabled via AGENT_COWORK_PROXY_URL=disabled");
    return;
  }
  const baseUrl = getProxyBaseUrl();
  const watcher = new TinkerAutoUpdateWatcher(() => BrowserWindow.getAllWindows(), baseUrl);
  activeWatcher = watcher;
  console.log(`[tinker-auto-update] starting watcher against ${baseUrl}`);
  void watcher.start();
}

export function stopTinkerAutoUpdateWatcher(): void {
  activeWatcher?.stop();
  activeWatcher = null;
}

/** Exported for tests. */
export const __tinkerAutoUpdateInternals = {
  IPC_CHANNEL,
};
