import { describe, expect, test } from "bun:test";
import {
  applyWorkflowPatch,
  executeAction,
  executableActionSchema,
} from "../src/lib/executable-actions";
import type { ClientEvent, WorkflowNode } from "../src/lib/runtime-types";

function node(overrides: Partial<WorkflowNode> = {}): WorkflowNode {
  const verifiers = overrides.verifiers ?? ["kept verifier"];
  return {
    id: overrides.id ?? "node-1",
    description: overrides.description ?? "Original step",
    outputFiles: overrides.outputFiles ?? ["out.md"],
    verifiers,
    verifierMarks: overrides.verifierMarks ?? verifiers.map(() => undefined),
    children: overrides.children ?? [],
    status: overrides.status ?? "pending",
    depth: overrides.depth ?? 0,
  };
}

describe("workflow executable actions", () => {
  test("edit_workflow uses patches that preserve omitted verifier fields", () => {
    const current = [node({ verifierMarks: ["check"] })];

    const next = applyWorkflowPatch(current, [
      {
        op: "update_node",
        nodeId: "node-1",
        description: "Renamed step",
      },
    ]);

    expect(next[0].description).toBe("Renamed step");
    expect(next[0].verifiers).toEqual(["kept verifier"]);
    expect(next[0].verifierMarks).toEqual(["check"]);
    expect(current[0].description).toBe("Original step");
  });

  test("edit_verifier executable validates and resets verifier marks on replacement", async () => {
    const parsed = executableActionSchema.parse({
      type: "edit_verifier",
      nodeId: "node-1",
      verifiers: ["new verifier", "second verifier"],
    });
    expect(parsed.type).toBe("edit_verifier");

    const events: ClientEvent[] = [];
    await executeAction(parsed, {
      sessionId: "session-1",
      currentWorkflowTree: [node({ verifierMarks: ["check"] })],
      sendEvent: (event) => events.push(event),
    });

    expect(events[0].type).toBe("session.updateWorkflowTree");
    if (events[0].type !== "session.updateWorkflowTree") {
      throw new Error("expected workflow update event");
    }
    const next = events[0].payload.workflowTree;
    expect(next[0].verifiers).toEqual(["new verifier", "second verifier"]);
    expect(next[0].verifierMarks).toEqual([undefined, undefined]);
  });
});
