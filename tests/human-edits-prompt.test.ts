import { describe, expect, test } from "bun:test";
import {
  appendHumanEditsToContinuePrompt,
  compactVerifierLinesAnnotation,
  findHumanEditsWindowEnd,
  findHumanEditsWindowStart,
  flattenWorkflowVerifierLines,
  gatherVerifierEditDiffSinceAgentRound,
} from "../src/electron/libs/human-edits-prompt.js";
import { compactFileEditAnnotation } from "../src/electron/libs/text-diff.js";

function wfSnapshot(
  verifiers: Array<{ criterion: string; status: "success" | "failure" | "unchecked" }>,
  nodeId = "n1",
  description = "step"
) {
  return {
    workflow: [
      {
        id: nodeId,
        description,
        outputFiles: [],
        verifiers,
        status: "pending",
        children: [],
      },
    ],
    file: [],
    memory: {},
    skill: {},
  };
}

function twoNodeWf(
  nodes: Array<{
    id: string;
    description: string;
    verifiers: Array<{ criterion: string; status: "success" | "failure" | "unchecked" }>;
  }>
) {
  return {
    workflow: nodes.map((n) => ({
      id: n.id,
      description: n.description,
      outputFiles: [],
      verifiers: n.verifiers,
      status: "pending",
      children: [],
    })),
    file: [],
    memory: {},
    skill: {},
  };
}


describe("compactFileEditAnnotation", () => {
  test("code files: line-aligned - / + on changed lines only", () => {
    const out = compactFileEditAnnotation(
      "h1 { font-size: 24px; }\n.title { color: blue; }",
      "h1 { font-size: 36px; }\n.title { color: blue; }",
      "chart.html"
    );
    expect(out).toContain("path=chart.html");
    expect(out).toContain("-");
    expect(out).toContain("+");
    expect(out).toContain("24px");
    expect(out).toContain("36px");
    expect(out).not.toContain("color: blue");
  });

  test("prose files: sentence-aligned compact bullets", () => {
    const out = compactFileEditAnnotation(
      "First sentence here. Second sentence stays.\n\nThird paragraph.",
      "First sentence here. Second sentence changed.\n\nThird paragraph.",
      "notes.md"
    );
    expect(out).toContain("path=notes.md");
    expect(out).toContain("•");
    expect(out).not.toContain("First sentence here");
  });
});

describe("flattenWorkflowVerifierLines", () => {
  test("flattens criterion and status", () => {
    const lines = flattenWorkflowVerifierLines(
      wfSnapshot([{ criterion: "Has title", status: "unchecked" }]).workflow
    );
    expect(lines).toEqual(["Has title: unchecked"]);
  });
});

describe("compactVerifierLinesAnnotation", () => {
  test("reports added and removed criteria", () => {
    const out = compactVerifierLinesAnnotation(
      ["Old rule: unchecked"],
      ["New rule: unchecked"]
    );
    expect(out).toContain("removed:");
    expect(out).toContain("added:");
  });
});

describe("findHumanEditsWindowStart", () => {
  test("starts after last run_result before the latest user_prompt", () => {
    const rows = [
      { message: { type: "run_result" } },
      { message: { type: "file_edit" } },
      { message: { type: "user_prompt" } },
    ];
    expect(findHumanEditsWindowStart(rows)).toBe(1);
    expect(findHumanEditsWindowEnd(rows)).toBe(2);
  });
});

describe("gatherVerifierEditDiffSinceAgentRound", () => {
  test("single diff for active node from run_result to user_prompt", () => {
    const before = wfSnapshot([{ criterion: "A", status: "unchecked" }]);
    const after = wfSnapshot([{ criterion: "B", status: "unchecked" }]);
    const rows = [
      { message: { type: "run_result" }, snapshot: before },
      { message: { type: "edit_verifier" }, snapshot: after },
      { message: { type: "user_prompt" }, snapshot: after },
    ];
    const diffs = gatherVerifierEditDiffSinceAgentRound(rows, 1, 2, "n1");
    expect(diffs).toHaveLength(1);
    expect(diffs[0]).toContain("verifiers (step)");
    expect(diffs[0]).toContain("removed:");
    expect(diffs[0]).toContain("added:");
  });

  test("skips when no edit_verifier in current human-edit window", () => {
    const before = wfSnapshot([{ criterion: "A", status: "unchecked" }]);
    const after = wfSnapshot([{ criterion: "B", status: "unchecked" }]);
    const rows = [
      { message: { type: "run_result" }, snapshot: before },
      { message: { type: "file_edit" }, snapshot: after },
      { message: { type: "user_prompt" }, snapshot: after },
    ];
    expect(gatherVerifierEditDiffSinceAgentRound(rows, 1, 2, "n1")).toHaveLength(0);
  });

  test("only includes verifiers from the requested node", () => {
    const before = twoNodeWf([
      { id: "n1", description: "step one", verifiers: [{ criterion: "A", status: "unchecked" }] },
      { id: "n2", description: "step two", verifiers: [{ criterion: "Other", status: "unchecked" }] },
    ]);
    const after = twoNodeWf([
      { id: "n1", description: "step one", verifiers: [{ criterion: "B", status: "unchecked" }] },
      { id: "n2", description: "step two", verifiers: [{ criterion: "Other changed", status: "unchecked" }] },
    ]);
    const rows = [
      { message: { type: "run_result" }, snapshot: before },
      { message: { type: "edit_verifier" }, snapshot: after },
      { message: { type: "user_prompt" }, snapshot: after },
    ];
    const diffs = gatherVerifierEditDiffSinceAgentRound(rows, 1, 2, "n1");
    expect(diffs).toHaveLength(1);
    expect(diffs[0]).toContain("step one");
    expect(diffs[0]).not.toContain("Other");
  });

  test("endpoint diff ignores intermediate edit_verifier snapshots", () => {
    const mid = wfSnapshot([{ criterion: "X", status: "unchecked" }]);
    const before = wfSnapshot([{ criterion: "A", status: "unchecked" }]);
    const after = wfSnapshot([{ criterion: "B", status: "unchecked" }]);
    const rows = [
      { message: { type: "run_result" }, snapshot: before },
      { message: { type: "edit_verifier" }, snapshot: mid },
      { message: { type: "edit_verifier" }, snapshot: after },
      { message: { type: "user_prompt" }, snapshot: after },
    ];
    const diffs = gatherVerifierEditDiffSinceAgentRound(rows, 1, 3, "n1");
    expect(diffs).toHaveLength(1);
    expect(diffs[0]).toContain("A");
    expect(diffs[0]).toContain("B");
    expect(diffs[0]).not.toContain("X");
  });
});

describe("endpoint file diff (run_result → user_prompt)", () => {
  test("compact annotation compares endpoints, not each intermediate save", () => {
    const before = "small";
    const after = "large";
    const out = compactFileEditAnnotation(before, after, "out.md");
    expect(out).toContain("path=out.md");
    expect(out).toContain("small");
    expect(out).toContain("large");
  });
});

describe("appendHumanEditsToContinuePrompt", () => {
  test("returns plain prompt when there are no edits", () => {
    expect(appendHumanEditsToContinuePrompt("make it bigger", [], [])).toBe("make it bigger");
  });

  test("appends file and verifier sections once each", () => {
    const prompt = appendHumanEditsToContinuePrompt(
      "fix title",
      ["path=chart.html\n• font-size: 24px → 36px"],
      ["path=verifiers\n• added: Large title: unchecked"]
    );
    expect(prompt.split("path=chart.html").length - 1).toBe(1);
    expect(prompt.split("path=verifiers").length - 1).toBe(1);
    expect(prompt).toContain("localized line changes");
  });
});
