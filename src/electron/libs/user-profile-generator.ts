import { existsSync, readFileSync } from "fs";
import { writeFile } from "fs/promises";
import { resolve } from "path";
import type { SessionStore } from "./session-store.js";
import type { StreamMessage } from "../types.js";
import { runPiTextPrompt } from "./pi-prompt.js";
import { getUserProfileRepoPath } from "./user-predict.js";

const DEFAULT_LAST_N = 10;
const MAX_LAST_N = 200;
const MAX_TOTAL_TRANSCRIPT_CHARS = 60_000;

function clampLastN(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return DEFAULT_LAST_N;
  return Math.min(Math.max(Math.floor(n), 1), MAX_LAST_N);
}

function extractUserPromptsFromSession(messages: StreamMessage[]): string[] {
  const prompts: string[] = [];
  for (const message of messages) {
    if (message.type === "user_prompt" && typeof message.prompt === "string") {
      const trimmed = message.prompt.trim();
      if (trimmed) prompts.push(trimmed);
    }
  }
  return prompts;
}

type ChatBundle = {
  title: string;
  createdAt: number;
  prompts: string[];
};

function collectRecentChatBundles(sessionStore: SessionStore, lastN: number): ChatBundle[] {
  const stored = sessionStore.listSessions();
  const bundles: ChatBundle[] = [];
  for (const session of stored) {
    const history = sessionStore.getSessionHistory(session.id);
    if (!history) continue;
    const prompts = extractUserPromptsFromSession(history.messages);
    if (prompts.length === 0) continue;
    bundles.push({
      title: history.session.title || "Untitled chat",
      createdAt: history.session.createdAt,
      prompts,
    });
    if (bundles.length >= lastN) break;
  }
  return bundles;
}

function formatChatsForPrompt(bundles: ChatBundle[]): string {
  const sections: string[] = [];
  let totalChars = 0;
  bundles.forEach((bundle, index) => {
    const header = `--- Chat ${index + 1} (${bundle.title}) ---`;
    const lines = [header];
    for (const prompt of bundle.prompts) {
      lines.push(`User: ${prompt}`);
    }
    const block = lines.join("\n");
    if (totalChars + block.length > MAX_TOTAL_TRANSCRIPT_CHARS) {
      sections.push("(...older chats truncated to keep prompt within size budget...)");
      return;
    }
    sections.push(block);
    totalChars += block.length + 2;
  });
  return sections.join("\n\n");
}

function buildGenerationPrompt(args: {
  existingProfile: string;
  chatsBlock: string;
  chatCount: number;
}): string {
  return [
    "You are generating a USER_PROFILE.md for use by an agent that predicts the user's next move.",
    "Goal: capture how this specific user works — preferences, recurring intents, communication style, common interventions, and what triggers them.",
    "",
    "Output rules:",
    "- Return ONLY the markdown body of USER_PROFILE.md. No code fences, no surrounding commentary.",
    "- Keep it dense and editable; use short paragraphs and bullet lists.",
    "- Prefer observations that are concretely supported by the chats below. Mark uncertain inferences as tentative.",
    "- Do not include personally identifying info beyond what appears in the chats (no phone, address, etc.).",
    "- Do not invent facts. If there is not enough signal for a section, write a brief note instead of fabricating.",
    "- Use second-or-third person consistent with the existing profile if one is provided; otherwise, third person.",
    "",
    "Recommended sections (omit any you cannot ground in the data):",
    "1. High-level identity",
    "2. How they interact with the agent",
    "3. What triggers their intervention",
    "4. Style of feedback (with short example phrasings drawn from the chats)",
    "5. Stable preferences (visualization, formatting, interaction, etc.)",
    "6. Likely next actions by situation",
    "7. What the agent should do proactively",
    "8. Caveats",
    "",
    "EXISTING USER_PROFILE.md (treat as prior; refine rather than fully discard unless contradicted):",
    args.existingProfile.trim() || "(none)",
    "",
    `RECENT USER CHATS (${args.chatCount} most-recent sessions; only the user's prompts are included):`,
    args.chatsBlock || "(no user prompts found in recent chats)",
    "",
    "Now produce the updated USER_PROFILE.md markdown.",
  ].join("\n");
}

function stripFences(text: string): string {
  const trimmed = text.trim();
  const fence = trimmed.match(/^```(?:markdown|md)?\s*([\s\S]*?)```\s*$/i);
  if (fence?.[1]) return fence[1].trim();
  return trimmed;
}

export type UserProfileGenerationResult = {
  success: boolean;
  profilePath?: string;
  markdown?: string;
  chatCount?: number;
  promptCount?: number;
  error?: string;
};

function resolveProfileTargetPath(cwd?: string): string {
  if (cwd && cwd.trim()) {
    return resolve(cwd, "USER_PROFILE.md");
  }
  return getUserProfileRepoPath();
}

export async function generateUserProfileMarkdown(args: {
  cwd?: string;
  lastN?: number;
  sessionStore: SessionStore;
  writeToDisk?: boolean;
  /** Test seam — defaults to `runPiTextPrompt`. */
  runPrompt?: (input: { cwd: string; prompt: string }) => Promise<string>;
}): Promise<UserProfileGenerationResult> {
  const lastN = clampLastN(args.lastN ?? DEFAULT_LAST_N);
  const profilePath = resolveProfileTargetPath(args.cwd);
  const existingProfile = existsSync(profilePath) ? readFileSync(profilePath, "utf8") : "";

  const bundles = collectRecentChatBundles(args.sessionStore, lastN);
  if (bundles.length === 0) {
    return {
      success: false,
      profilePath,
      error: "No prior chats with user prompts were found. Send at least one message in a session first.",
    };
  }

  const chatsBlock = formatChatsForPrompt(bundles);
  const promptCount = bundles.reduce((sum, bundle) => sum + bundle.prompts.length, 0);

  const promptCwd = args.cwd && args.cwd.trim() ? args.cwd : process.cwd();
  const runPrompt = args.runPrompt ?? runPiTextPrompt;
  let raw: string;
  try {
    raw = await runPrompt({
      cwd: promptCwd,
      prompt: buildGenerationPrompt({
        existingProfile,
        chatsBlock,
        chatCount: bundles.length,
      }),
    });
  } catch (error) {
    return {
      success: false,
      profilePath,
      chatCount: bundles.length,
      promptCount,
      error: error instanceof Error ? error.message : String(error),
    };
  }

  const markdown = stripFences(raw);
  if (!markdown) {
    return {
      success: false,
      profilePath,
      chatCount: bundles.length,
      promptCount,
      error: "Model returned an empty profile. Try again or increase the chat window.",
    };
  }

  if (args.writeToDisk !== false) {
    try {
      await writeFile(profilePath, markdown.endsWith("\n") ? markdown : `${markdown}\n`, "utf8");
    } catch (error) {
      return {
        success: false,
        profilePath,
        chatCount: bundles.length,
        promptCount,
        markdown,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  return {
    success: true,
    profilePath,
    markdown,
    chatCount: bundles.length,
    promptCount,
  };
}

export function readUserProfile(cwd?: string): { markdown: string; profilePath: string } {
  const profilePath = resolveProfileTargetPath(cwd);
  const markdown = existsSync(profilePath) ? readFileSync(profilePath, "utf8") : "";
  return { markdown, profilePath };
}

export async function writeUserProfile(args: { cwd?: string; markdown: string }): Promise<{ profilePath: string }> {
  const profilePath = resolveProfileTargetPath(args.cwd);
  const content = args.markdown.endsWith("\n") ? args.markdown : `${args.markdown}\n`;
  await writeFile(profilePath, content, "utf8");
  return { profilePath };
}
