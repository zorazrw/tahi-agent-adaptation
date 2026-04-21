import { existsSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import {
  AuthStorage,
  DefaultResourceLoader,
  ModelRegistry,
  SessionManager,
  SettingsManager,
  createAgentSession,
} from "@mariozechner/pi-coding-agent";
import { registerTinkerProvider } from "./tinker-provider.js";

function extractAssistantText(message: unknown): string {
  if (!message || typeof message !== "object" || !("role" in message) || message.role !== "assistant") {
    return "";
  }
  const candidateContent = (message as { content?: unknown }).content;
  const content = Array.isArray(candidateContent) ? candidateContent : [];
  return content
    .map((block) => {
      if (!block || typeof block !== "object" || !("type" in block)) return "";
      if (block.type === "text" && "text" in block) return String(block.text ?? "");
      return "";
    })
    .filter(Boolean)
    .join("\n")
    .trim();
}

function resolvePiAgentDir(): string {
  const candidates = [
    process.env.PI_AGENT_DIR?.trim(),
    join(homedir(), "Library", "Application Support", "agent-cowork", "pi-agent"),
    join(homedir(), ".pi-agent"),
  ].filter(Boolean) as string[];

  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }

  return candidates[0] ?? join(homedir(), ".pi-agent");
}

function createGenericPiManagers(cwd: string) {
  const agentDir = resolvePiAgentDir();
  const authPath = join(agentDir, "auth.json");
  const modelsPath = join(agentDir, "models.json");
  const tinkerConfigPath = join(agentDir, "tinker-provider.json");
  const authStorage = AuthStorage.create(authPath);
  const ModelRegistryCtor = ModelRegistry as unknown as {
    new (authStorage: AuthStorage, modelsPath: string): ModelRegistry;
  };
  const modelRegistry = new ModelRegistryCtor(authStorage, modelsPath);
  registerTinkerProvider(modelRegistry, tinkerConfigPath);
  const settingsManager = SettingsManager.create(cwd, agentDir);
  return {
    agentDir,
    authStorage,
    modelRegistry,
    settingsManager,
  };
}

export async function runPiTextPrompt(options: {
  cwd: string;
  prompt: string;
  appendSystemPrompt?: string;
}): Promise<string> {
  const cwd = options.cwd;
  const { agentDir, authStorage, modelRegistry, settingsManager } = createGenericPiManagers(cwd);
  const resourceLoader = new DefaultResourceLoader({
    cwd,
    agentDir,
    settingsManager,
    appendSystemPrompt: options.appendSystemPrompt,
  });
  await resourceLoader.reload();

  const { session, modelFallbackMessage } = await createAgentSession({
    cwd,
    agentDir,
    authStorage,
    modelRegistry,
    settingsManager,
    resourceLoader,
    tools: [],
    sessionManager: SessionManager.inMemory(cwd),
  });

  if (modelFallbackMessage && !session.model) {
    session.dispose();
    throw new Error(modelFallbackMessage);
  }

  try {
    await session.prompt(options.prompt);
    const assistantMessage = [...session.messages].reverse().find((message) => {
      return typeof message === "object" && message !== null && "role" in message && message.role === "assistant";
    });
    return extractAssistantText(assistantMessage);
  } finally {
    session.dispose();
  }
}
