import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import type { SessionStore, SessionHistory, StoredSession } from "../src/electron/libs/session-store";
import { generateUserProfileMarkdown, readUserProfile } from "../src/electron/libs/user-profile-generator";
import type { StreamMessage } from "../src/lib/runtime-types";

const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function mkCwd(): string {
  const cwd = mkdtempSync(join(tmpdir(), "user-profile-gen-"));
  tempDirs.push(cwd);
  return cwd;
}

/**
 * Minimal SessionStore stub — generator only reads listSessions() + getSessionHistory().
 * Sessions are returned in insertion order, mirroring the production "newest first" sort.
 */
function makeStubStore(
  rows: Array<{ id?: string; title: string; cwd?: string; prompts: string[]; createdAt?: number }>
): SessionStore {
  const stored: StoredSession[] = rows.map((row, i) => ({
    id: row.id ?? `s-${i}`,
    title: row.title,
    status: "completed",
    engine: "pi",
    cwd: row.cwd,
    workflowTree: [],
    verificationDepth: 0,
    autoContextInduction: true,
    createdAt: row.createdAt ?? 1_000_000 + i,
    updatedAt: 1_000_000 + i,
  }));
  const histories = new Map<string, SessionHistory>();
  rows.forEach((row, i) => {
    const id = stored[i].id;
    const messages: StreamMessage[] = row.prompts.map((prompt) => ({ type: "user_prompt", prompt }));
    histories.set(id, { session: stored[i], messages });
  });
  const store: Pick<SessionStore, "listSessions" | "getSessionHistory"> = {
    listSessions: () => stored,
    getSessionHistory: (id: string) => histories.get(id) ?? null,
  };
  return store as unknown as SessionStore;
}

describe("generateUserProfileMarkdown", () => {
  test("writes the model output to USER_PROFILE.md at the target cwd", async () => {
    const cwd = mkCwd();
    const store = makeStubStore([
      { title: "Plot a chart", cwd, prompts: ["make a scatter plot", "axes bigger"] },
      { title: "Refactor", cwd, prompts: ["clean up the helpers"] },
    ]);

    const fakeMarkdown = "# Profile\n\n- user wants larger axis labels\n";
    const result = await generateUserProfileMarkdown({
      cwd,
      lastN: 10,
      sessionStore: store,
      runPrompt: async () => fakeMarkdown,
    });

    expect(result.success).toBe(true);
    expect(result.chatCount).toBe(2);
    expect(result.promptCount).toBe(3);
    expect(result.markdown?.trim()).toBe(fakeMarkdown.trim());
    expect(result.profilePath).toBe(join(cwd, "USER_PROFILE.md"));

    const written = readFileSync(join(cwd, "USER_PROFILE.md"), "utf8");
    expect(written).toContain("user wants larger axis labels");
  });

  test("collects only the last N sessions that contain user prompts", async () => {
    const cwd = mkCwd();
    const store = makeStubStore([
      { title: "newest", cwd, prompts: ["two", "three"] },
      { title: "no-prompts", cwd, prompts: [] },
      { title: "old", cwd, prompts: ["one"] },
    ]);

    let capturedPrompt = "";
    const result = await generateUserProfileMarkdown({
      cwd,
      lastN: 2,
      sessionStore: store,
      writeToDisk: false,
      runPrompt: async ({ prompt }) => {
        capturedPrompt = prompt;
        return "ok";
      },
    });

    expect(result.success).toBe(true);
    expect(result.chatCount).toBe(2);
    expect(result.promptCount).toBe(3);
    expect(capturedPrompt).toContain("--- Chat 1 (newest) ---");
    expect(capturedPrompt).toContain("User: two");
    expect(capturedPrompt).toContain("User: three");
    expect(capturedPrompt).toContain("--- Chat 2 (old) ---");
    expect(capturedPrompt).toContain("User: one");
    expect(capturedPrompt).not.toContain("no-prompts");
  });

  test("does not truncate long individual prompts (per-prompt cap removed)", async () => {
    const cwd = mkCwd();
    const longPrompt = "x".repeat(5_000);
    const store = makeStubStore([{ title: "long", cwd, prompts: [longPrompt] }]);

    let captured = "";
    await generateUserProfileMarkdown({
      cwd,
      lastN: 5,
      sessionStore: store,
      writeToDisk: false,
      runPrompt: async ({ prompt }) => {
        captured = prompt;
        return "ok";
      },
    });

    expect(captured).toContain(`User: ${longPrompt}`);
    // ellipsis only appears if the per-prompt cap re-emerges
    expect(captured.includes(`User: ${"x".repeat(599)}…`)).toBe(false);
  });

  test("returns a clean failure when no chats have user prompts", async () => {
    const cwd = mkCwd();
    const store = makeStubStore([{ title: "empty", cwd, prompts: [] }]);

    let called = false;
    const result = await generateUserProfileMarkdown({
      cwd,
      sessionStore: store,
      runPrompt: async () => {
        called = true;
        return "should not be invoked";
      },
    });

    expect(called).toBe(false);
    expect(result.success).toBe(false);
    expect(result.error).toMatch(/no prior chats/i);
  });

  test("strips a ```markdown fence wrapper from the model response", async () => {
    const cwd = mkCwd();
    const store = makeStubStore([{ title: "s", cwd, prompts: ["hi"] }]);

    const result = await generateUserProfileMarkdown({
      cwd,
      sessionStore: store,
      runPrompt: async () => "```markdown\n# Body\n\nfoo\n```\n",
    });

    expect(result.success).toBe(true);
    expect(result.markdown).toBe("# Body\n\nfoo");
  });

  test("readUserProfile returns empty markdown but a resolved path when file is absent", () => {
    const cwd = mkCwd();
    const { markdown, profilePath } = readUserProfile(cwd);
    expect(markdown).toBe("");
    expect(profilePath).toBe(join(cwd, "USER_PROFILE.md"));
  });
});
