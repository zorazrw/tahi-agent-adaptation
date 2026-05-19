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

const MAX_TITLE_WORDS = 8;

/** Strip labels/quotes and cap at MAX_TITLE_WORDS (matches fallback). */
function normalizeGeneratedTitle(raw: string): string {
  let t = raw.trim().replace(/^["'`]+|["'`]+$/g, "").trim();
  const labelMatch = /^(?:title|session title)\s*:\s*/i.exec(t);
  if (labelMatch) t = t.slice(labelMatch[0].length).trim();
  const words = t.split(/\s+/).filter(Boolean);
  if (words.length === 0) return "";
  if (words.length <= MAX_TITLE_WORDS) return words.join(" ");
  return `${words.slice(0, MAX_TITLE_WORDS).join(" ")}...`;
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
        "Generate a concise session title for the task below.",
        `Rules: at most ${MAX_TITLE_WORDS} words; no punctuation-only titles; no quotes; no colon labels (do not write "Title:").`,
        "Return only the title text—no explanation, no markdown, no restating the task.",
        "",
        userIntent,
      ].join("\n")
    );

    const assistantMessage = [...session.messages].reverse().find((message) => {
      return typeof message === "object" && message !== null && "role" in message && message.role === "assistant";
    });
    const title = extractAssistantText(assistantMessage);
    session.dispose();

    const normalized = normalizeGeneratedTitle(title);
    if (normalized) return normalized;
  } catch (error) {
    console.error("Failed to generate session title with pi:", error);
  }

  const words = userIntent.trim().split(/\s+/).slice(0, MAX_TITLE_WORDS);
  return words.join(" ").toUpperCase() + (userIntent.trim().split(/\s+/).length > MAX_TITLE_WORDS ? "..." : "");
};
