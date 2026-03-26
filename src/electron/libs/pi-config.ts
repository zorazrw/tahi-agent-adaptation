import { BrowserWindow, app, dialog, shell } from "electron";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import {
  AuthStorage,
  DefaultResourceLoader,
  ModelRegistry,
  SessionManager,
  SettingsManager,
} from "@mariozechner/pi-coding-agent";
import type { OAuthProviderId } from "@mariozechner/pi-ai/oauth";
import type {
  AgentSettings,
  AvailableModel,
  OpenAICompatibleProviderConfig,
  OpenAICompatibleProviderInput,
  ProviderAuthStatus,
} from "../types.js";
import { getAppSkillsDir } from "./skill-store.js";
import { loadApiConfig } from "./config-store.js";

const PI_AGENT_DIR_NAME = "pi-agent";
const DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com";
const OPENAI_COMPATIBLE_PROVIDER = "openai-compatible" as const;
const OPENAI_COMPATIBLE_API_KEY_PLACEHOLDER = "${OPENAI_COMPATIBLE_API_KEY}";

let bootstrapComplete = false;

type LegacyBootstrapConfig = {
  apiKey?: string;
  model?: string;
  baseURL?: string;
};

export function getPiAgentDir(): string {
  return join(app.getPath("userData"), PI_AGENT_DIR_NAME);
}

export function getPiSessionsDir(appSessionId: string): string {
  return join(getPiAgentDir(), "sessions", appSessionId);
}

function ensureDir(path: string): void {
  if (!existsSync(path)) {
    mkdirSync(path, { recursive: true });
  }
}

function getAuthPath(): string {
  return join(getPiAgentDir(), "auth.json");
}

function getModelsPath(): string {
  return join(getPiAgentDir(), "models.json");
}

type ModelsJsonConfig = {
  providers?: Record<string, Record<string, unknown>>;
};

function readLegacyBootstrapConfig(): LegacyBootstrapConfig | null {
  const uiConfig = loadApiConfig();
  if (uiConfig?.apiKey) {
    return {
      apiKey: uiConfig.apiKey,
      model: uiConfig.model,
      baseURL: uiConfig.baseURL,
    };
  }

  try {
    const settingsPath = join(homedir(), ".claude", "settings.json");
    const raw = readFileSync(settingsPath, "utf8");
    const parsed = JSON.parse(raw) as { env?: Record<string, unknown> };
    const env = parsed.env ?? {};
    const apiKey = env.ANTHROPIC_AUTH_TOKEN;
    if (!apiKey) return null;
    return {
      apiKey: String(apiKey),
      model: env.ANTHROPIC_MODEL ? String(env.ANTHROPIC_MODEL) : undefined,
      baseURL: env.ANTHROPIC_BASE_URL ? String(env.ANTHROPIC_BASE_URL) : undefined,
    };
  } catch {
    return null;
  }
}

function hasPiAuthOrSettings(): boolean {
  return existsSync(getAuthPath()) || existsSync(join(getPiAgentDir(), "settings.json"));
}

function readModelsConfig(): ModelsJsonConfig {
  const modelsPath = getModelsPath();
  if (!existsSync(modelsPath)) {
    return { providers: {} };
  }
  try {
    const parsed = JSON.parse(readFileSync(modelsPath, "utf8")) as ModelsJsonConfig;
    return {
      ...parsed,
      providers: parsed.providers ?? {},
    };
  } catch {
    return { providers: {} };
  }
}

function writeModelsConfig(config: ModelsJsonConfig): void {
  writeFileSync(
    getModelsPath(),
    JSON.stringify(
      {
        ...config,
        providers: config.providers ?? {},
      },
      null,
      2
    ),
    "utf8"
  );
}

function writeAnthropicBaseUrlOverride(baseURL: string): void {
  const normalized = baseURL.trim();
  if (!normalized || normalized === DEFAULT_ANTHROPIC_BASE_URL) {
    return;
  }

  const nextConfig = readModelsConfig();
  const providers = nextConfig.providers ?? {};
  const anthropic = providers.anthropic ?? {};
  nextConfig.providers = {
    ...providers,
    anthropic: {
      ...anthropic,
      api: (anthropic.api as string | undefined) ?? "anthropic",
      baseUrl: normalized,
    },
  };
  writeModelsConfig(nextConfig);
}

export function ensurePiBootstrap(): void {
  if (bootstrapComplete) return;

  const agentDir = getPiAgentDir();
  ensureDir(agentDir);

  if (!hasPiAuthOrSettings()) {
    const legacy = readLegacyBootstrapConfig();
    if (legacy?.apiKey) {
      const authStorage = AuthStorage.create(getAuthPath());
      authStorage.set("anthropic", { type: "api_key", key: legacy.apiKey });

      const settingsManager = SettingsManager.create(process.cwd(), agentDir);
      settingsManager.setDefaultProvider("anthropic");
      if (legacy.model) {
        settingsManager.setDefaultModel(legacy.model);
      }
      if (legacy.baseURL) {
        writeAnthropicBaseUrlOverride(legacy.baseURL);
      }
    }
  }

  bootstrapComplete = true;
}

export function createPiManagers(cwd: string) {
  ensurePiBootstrap();
  const agentDir = getPiAgentDir();
  const authStorage = AuthStorage.create(getAuthPath());
  const modelRegistry = new ModelRegistry(authStorage, getModelsPath());
  const settingsManager = SettingsManager.create(cwd, agentDir);
  return {
    agentDir,
    authStorage,
    modelRegistry,
    settingsManager,
  };
}

export async function createPiResourceLoader(
  cwd: string,
  options?: {
    appendSystemPrompt?: string;
  }
) {
  const { agentDir, settingsManager } = createPiManagers(cwd);
  const resourceLoader = new DefaultResourceLoader({
    cwd,
    agentDir,
    settingsManager,
    additionalSkillPaths: [getAppSkillsDir()],
    appendSystemPrompt: options?.appendSystemPrompt,
  });
  await resourceLoader.reload();
  return resourceLoader;
}

export function createPiSessionManager(appSessionId: string, cwd: string, piSessionFile?: string) {
  const sessionDir = getPiSessionsDir(appSessionId);
  ensureDir(sessionDir);
  if (piSessionFile && existsSync(piSessionFile)) {
    return SessionManager.open(piSessionFile, sessionDir);
  }
  return SessionManager.create(cwd, sessionDir);
}

export function getAgentSettings(): AgentSettings {
  const { settingsManager } = createPiManagers(process.cwd());
  return {
    defaultProvider: settingsManager.getDefaultProvider(),
    defaultModel: settingsManager.getDefaultModel(),
    defaultThinkingLevel: settingsManager.getDefaultThinkingLevel(),
  };
}

export async function saveAgentSettings(settings: AgentSettings): Promise<void> {
  const { settingsManager } = createPiManagers(process.cwd());
  if (settings.defaultProvider && settings.defaultModel) {
    settingsManager.setDefaultModelAndProvider(settings.defaultProvider, settings.defaultModel);
  } else {
    if (settings.defaultProvider) settingsManager.setDefaultProvider(settings.defaultProvider);
    if (settings.defaultModel) settingsManager.setDefaultModel(settings.defaultModel);
  }
  if (settings.defaultThinkingLevel) {
    settingsManager.setDefaultThinkingLevel(settings.defaultThinkingLevel);
  }
  await settingsManager.flush();
}

export function listAvailableModels(): AvailableModel[] {
  const { modelRegistry } = createPiManagers(process.cwd());
  return modelRegistry
    .getAll()
    .map((model) => ({
      provider: model.provider,
      id: model.id,
      label: `${model.provider}/${model.id}`,
      reasoning: Boolean(model.reasoning),
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
}

export function getProviderAuthStatus(provider: string): ProviderAuthStatus {
  const { authStorage } = createPiManagers(process.cwd());
  const cred = authStorage.get(provider);
  const oauthProvider = authStorage.getOAuthProviders().find((item) => item.id === provider);

  return {
    provider,
    hasAuth: authStorage.hasAuth(provider),
    authType: cred?.type ?? (authStorage.hasAuth(provider) ? "env" : undefined),
    supportsOAuth: Boolean(oauthProvider),
    oauthName: oauthProvider?.name,
  };
}

export function saveProviderApiKey(provider: string, apiKey: string): void {
  const { authStorage } = createPiManagers(process.cwd());
  authStorage.set(provider, { type: "api_key", key: apiKey.trim() });
}

export function getOpenAICompatibleProviderConfig(): OpenAICompatibleProviderConfig | null {
  const config = readModelsConfig();
  const providerConfig = config.providers?.[OPENAI_COMPATIBLE_PROVIDER];
  if (!providerConfig) {
    return null;
  }

  const models = Array.isArray(providerConfig.models) ? providerConfig.models : [];
  const modelConfig = models.find((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"));
  const baseUrl = typeof providerConfig.baseUrl === "string" ? providerConfig.baseUrl.trim() : "";
  const model = typeof modelConfig?.id === "string" ? modelConfig.id.trim() : "";
  if (!baseUrl || !model) {
    return null;
  }

  const { authStorage } = createPiManagers(process.cwd());
  return {
    provider: OPENAI_COMPATIBLE_PROVIDER,
    baseUrl,
    model,
    apiFormat: providerConfig.api === "openai-responses" ? "openai-responses" : "openai-completions",
    hasApiKey: authStorage.hasAuth(OPENAI_COMPATIBLE_PROVIDER),
  };
}

export function saveOpenAICompatibleProviderConfig(input: OpenAICompatibleProviderInput): void {
  const baseUrl = input.baseUrl.trim().replace(/\/+$/, "");
  const model = input.model.trim();
  const apiFormat = input.apiFormat;
  const apiKey = input.apiKey?.trim();

  if (!baseUrl) {
    throw new Error("Base URL is required");
  }
  if (!model) {
    throw new Error("Model slug is required");
  }

  const config = readModelsConfig();
  const providers = config.providers ?? {};
  providers[OPENAI_COMPATIBLE_PROVIDER] = {
    ...(providers[OPENAI_COMPATIBLE_PROVIDER] ?? {}),
    api: apiFormat,
    baseUrl,
    apiKey: OPENAI_COMPATIBLE_API_KEY_PLACEHOLDER,
    models: [
      {
        id: model,
        name: model,
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 128000,
        maxTokens: 16384,
      },
    ],
  };
  config.providers = providers;
  writeModelsConfig(config);

  if (apiKey) {
    const { authStorage } = createPiManagers(process.cwd());
    authStorage.set(OPENAI_COMPATIBLE_PROVIDER, { type: "api_key", key: apiKey });
  }
}

export function removeOpenAICompatibleProviderConfig(): void {
  const config = readModelsConfig();
  if (config.providers?.[OPENAI_COMPATIBLE_PROVIDER]) {
    delete config.providers[OPENAI_COMPATIBLE_PROVIDER];
    writeModelsConfig(config);
  }

  const { authStorage } = createPiManagers(process.cwd());
  if (authStorage.has(OPENAI_COMPATIBLE_PROVIDER)) {
    authStorage.remove(OPENAI_COMPATIBLE_PROVIDER);
  }
}

async function promptInRenderer(title: string, message: string, placeholder = ""): Promise<string> {
  const win = BrowserWindow.getAllWindows()[0];
  if (!win) {
    throw new Error("No active window");
  }
  const promptScript = `window.prompt(${JSON.stringify(`${title}\n\n${message}`)}, ${JSON.stringify(placeholder)})`;
  const value = await win.webContents.executeJavaScript(promptScript, true);
  if (typeof value !== "string" || !value.trim()) {
    throw new Error("Login cancelled");
  }
  return value.trim();
}

export async function loginProvider(provider: string): Promise<void> {
  const { authStorage } = createPiManagers(process.cwd());
  const providerId = provider as OAuthProviderId;

  await authStorage.login(providerId, {
    onAuth: (info) => {
      shell.openExternal(info.url).catch(() => {
        // Ignore browser launch failures; prompt still contains the URL.
      });
      const win = BrowserWindow.getAllWindows()[0];
      if (win) {
        dialog.showMessageBox(win, {
          type: "info",
          title: `Login: ${provider}`,
          message: `Continue login for ${provider} in your browser.`,
          detail: [info.url, info.instructions].filter(Boolean).join("\n\n"),
        }).catch(() => {});
      }
    },
    onPrompt: async (prompt) => promptInRenderer(`Login: ${provider}`, prompt.message, prompt.placeholder ?? ""),
    onManualCodeInput: async () =>
      promptInRenderer(`Login: ${provider}`, "Paste the redirect URL or code from the browser."),
  });
}

export function logoutProvider(provider: string): void {
  const { authStorage } = createPiManagers(process.cwd());
  authStorage.logout(provider);
}
