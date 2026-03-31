import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "fs";
import { dirname } from "path";

export const TINKER_PROVIDER = "tinker" as const;
export const TINKER_API = "tinker-python";
export const TINKER_BASE_URL_PLACEHOLDER = "https://tinker.local";
export const TINKER_ENV_API_KEY_COMMAND = "!printenv TINKER_API_KEY";

export type TinkerModelConfig = {
  id: string;
  baseModel: string;
  modelPath?: string;
  rendererName?: string;
  reasoning: boolean;
  contextWindow: number;
  maxTokens: number;
};

export type TinkerProviderConfig = {
  provider: typeof TINKER_PROVIDER;
  baseUrl?: string;
  hasApiKey: boolean;
  model: TinkerModelConfig;
};

export type TinkerProviderInput = {
  baseUrl?: string;
  apiKey?: string;
  model: string;
  baseModel: string;
  modelPath?: string;
  rendererName?: string;
  reasoning?: boolean;
  contextWindow?: number;
  maxTokens?: number;
};

export type StoredTinkerProviderConfig = {
  version: 1;
  provider: typeof TINKER_PROVIDER;
  baseUrl?: string;
  model: TinkerModelConfig;
};

function ensureParentDir(path: string): void {
  mkdirSync(dirname(path), { recursive: true });
}

function requireNonEmpty(value: string, fieldName: string): string {
  const normalized = value.trim();
  if (!normalized) {
    throw new Error(`${fieldName} is required`);
  }
  return normalized;
}

function normalizeOptional(value?: string): string | undefined {
  const normalized = value?.trim();
  return normalized ? normalized : undefined;
}

function normalizePositiveInt(value: number | undefined, fieldName: string, fallback: number): number {
  const normalized = value ?? fallback;
  if (!Number.isFinite(normalized) || normalized <= 0) {
    throw new Error(`${fieldName} must be a positive number`);
  }
  return Math.floor(normalized);
}

export function normalizeTinkerProviderInput(input: TinkerProviderInput): StoredTinkerProviderConfig {
  return {
    version: 1,
    provider: TINKER_PROVIDER,
    baseUrl: normalizeOptional(input.baseUrl),
    model: {
      id: requireNonEmpty(input.model, "Model slug"),
      baseModel: requireNonEmpty(input.baseModel, "Base model"),
      modelPath: normalizeOptional(input.modelPath),
      rendererName: normalizeOptional(input.rendererName),
      reasoning: input.reasoning ?? true,
      contextWindow: normalizePositiveInt(input.contextWindow, "Context window", 128000),
      maxTokens: normalizePositiveInt(input.maxTokens, "Max tokens", 16384),
    },
  };
}

export function readStoredTinkerProviderConfig(configPath: string): StoredTinkerProviderConfig | null {
  if (!existsSync(configPath)) {
    return null;
  }

  try {
    const parsed = JSON.parse(readFileSync(configPath, "utf8")) as Partial<StoredTinkerProviderConfig>;
    if (parsed.provider !== TINKER_PROVIDER || !parsed.model) {
      return null;
    }
    return normalizeTinkerProviderInput({
      baseUrl: parsed.baseUrl,
      model: String(parsed.model.id ?? ""),
      baseModel: String(parsed.model.baseModel ?? ""),
      modelPath: parsed.model.modelPath,
      rendererName: parsed.model.rendererName,
      reasoning: parsed.model.reasoning,
      contextWindow: parsed.model.contextWindow,
      maxTokens: parsed.model.maxTokens,
    });
  } catch {
    return null;
  }
}

export function writeStoredTinkerProviderConfig(
  configPath: string,
  input: TinkerProviderInput,
): StoredTinkerProviderConfig {
  const normalized = normalizeTinkerProviderInput(input);
  ensureParentDir(configPath);
  writeFileSync(configPath, JSON.stringify(normalized, null, 2), "utf8");
  return normalized;
}

export function removeStoredTinkerProviderConfig(configPath: string): void {
  if (existsSync(configPath)) {
    unlinkSync(configPath);
  }
}

export function toPublicTinkerProviderConfig(
  config: StoredTinkerProviderConfig,
  hasApiKey: boolean,
): TinkerProviderConfig {
  return {
    provider: TINKER_PROVIDER,
    baseUrl: config.baseUrl,
    hasApiKey,
    model: config.model,
  };
}

export function buildRegisteredTinkerModels(config: StoredTinkerProviderConfig) {
  return [
    {
      id: config.model.id,
      name: config.model.id,
      reasoning: config.model.reasoning,
      input: ["text"] as Array<"text" | "image">,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: config.model.contextWindow,
      maxTokens: config.model.maxTokens,
    },
  ];
}

export function getRegisteredTinkerBaseUrl(config: StoredTinkerProviderConfig): string {
  return config.baseUrl ?? TINKER_BASE_URL_PLACEHOLDER;
}
