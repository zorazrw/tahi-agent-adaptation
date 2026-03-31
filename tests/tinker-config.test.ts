import { afterEach, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync } from "fs";
import { rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import {
  TINKER_BASE_URL_PLACEHOLDER,
  buildRegisteredTinkerModels,
  getRegisteredTinkerBaseUrl,
  normalizeTinkerProviderInput,
  readStoredTinkerProviderConfig,
  removeStoredTinkerProviderConfig,
  writeStoredTinkerProviderConfig,
} from "../src/electron/libs/tinker-config";

const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function createTempConfigPath(): string {
  const dir = mkdtempSync(join(tmpdir(), "tinker-config-"));
  tempDirs.push(dir);
  return join(dir, "tinker-provider.json");
}

describe("tinker-config", () => {
  test("normalizes and persists a Tinker provider config", () => {
    const configPath = createTempConfigPath();

    const stored = writeStoredTinkerProviderConfig(configPath, {
      baseUrl: " https://tinker.example.com ",
      model: " qwen-tooling ",
      baseModel: " Qwen/Qwen3-30B-A3B-Instruct-2507 ",
      modelPath: " tinker://weights/qwen-tooling ",
      rendererName: " qwen3_instruct ",
      reasoning: false,
      contextWindow: 262144,
      maxTokens: 8192,
    });

    expect(stored.baseUrl).toBe("https://tinker.example.com");
    expect(stored.model.id).toBe("qwen-tooling");
    expect(stored.model.baseModel).toBe("Qwen/Qwen3-30B-A3B-Instruct-2507");
    expect(stored.model.modelPath).toBe("tinker://weights/qwen-tooling");
    expect(stored.model.rendererName).toBe("qwen3_instruct");
    expect(stored.model.reasoning).toBe(false);

    const reloaded = readStoredTinkerProviderConfig(configPath);
    expect(reloaded).toEqual(stored);
  });

  test("builds a model registration with text-only defaults", () => {
    const stored = normalizeTinkerProviderInput({
      model: "qwen-tooling",
      baseModel: "Qwen/Qwen3-30B-A3B-Instruct-2507",
      reasoning: true,
      contextWindow: 128000,
      maxTokens: 4096,
    });

    expect(getRegisteredTinkerBaseUrl(stored)).toBe(TINKER_BASE_URL_PLACEHOLDER);
    expect(buildRegisteredTinkerModels(stored)).toEqual([
      {
        id: "qwen-tooling",
        name: "qwen-tooling",
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 128000,
        maxTokens: 4096,
      },
    ]);
  });

  test("removes the stored config file", () => {
    const configPath = createTempConfigPath();
    writeStoredTinkerProviderConfig(configPath, {
      model: "qwen-tooling",
      baseModel: "Qwen/Qwen3-30B-A3B-Instruct-2507",
    });

    expect(existsSync(configPath)).toBe(true);
    removeStoredTinkerProviderConfig(configPath);
    expect(existsSync(configPath)).toBe(false);
  });
});
