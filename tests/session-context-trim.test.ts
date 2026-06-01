import { describe, expect, test } from "bun:test";
import { isContextLengthExceededError } from "../src/electron/libs/session-context-trim";

describe("isContextLengthExceededError", () => {
  test("detects prompt + max_tokens context window errors", () => {
    expect(
      isContextLengthExceededError(
        "Error code: 400 - {'detail': \"Prompt length plus max_tokens exceeds the model's context window: 18202 prompt tokens + 16384 max_tokens > 32768.\"}"
      )
    ).toBe(true);
  });

  test("detects generic context window overflow messages", () => {
    expect(isContextLengthExceededError("prompt is too long: 100 tokens > 50 maximum")).toBe(true);
    expect(isContextLengthExceededError("Your input exceeds the context window of this model")).toBe(
      true
    );
  });

  test("ignores unrelated API errors", () => {
    expect(isContextLengthExceededError("Error code: 401 - Unauthorized")).toBe(false);
    expect(isContextLengthExceededError("rate limit exceeded")).toBe(false);
  });
});
