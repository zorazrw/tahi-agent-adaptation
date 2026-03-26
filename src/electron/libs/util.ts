import { SessionManager, createAgentSession } from "@mariozechner/pi-coding-agent";
import { createPiManagers, createPiResourceLoader } from "./pi-config.js";

export function getEnhancedEnv(): Record<string, string | undefined> {
  return {
    ...process.env,
  };
}

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

export const generateSessionTitle = async (userIntent: string | null) => {
  if (!userIntent) return "New Session";

  try {
    const cwd = process.cwd();
    const { agentDir, authStorage, modelRegistry, settingsManager } = createPiManagers(cwd);
    const resourceLoader = await createPiResourceLoader(cwd);

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
      throw new Error(modelFallbackMessage);
    }

    await session.prompt(
      [
        "Generate a short, clear title for this conversation.",
        "Return only the title with no prefix or explanation.",
        "",
        userIntent,
      ].join("\n")
    );

    const assistantMessage = [...session.messages].reverse().find((message) => {
      return typeof message === "object" && message !== null && "role" in message && message.role === "assistant";
    });
    const title = extractAssistantText(assistantMessage);
    session.dispose();

    if (title) {
      const afterColon = title.includes(":") ? title.slice(title.indexOf(":") + 1).trim() : title;
      if (afterColon) return afterColon;
    }
  } catch (error) {
    console.error("Failed to generate session title with pi:", error);
  }

  const words = userIntent.trim().split(/\s+/).slice(0, 5);
  return words.join(" ").toUpperCase() + (userIntent.trim().split(/\s+/).length > 5 ? "..." : "");
};
