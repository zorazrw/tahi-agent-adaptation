import { useMemo, useState, type ReactNode } from "react";
import {
  BrainIcon,
  CircleStopIcon,
  FileTextIcon,
  GitBranchIcon,
  ListChecksIcon,
  MessageSquareTextIcon,
  XIcon,
} from "lucide-react";
import type { ExecutableAction } from "../../lib/executable-actions";
import type { PredictedUserActionSuggestion, WorkflowNode } from "../types";
import { useAppStore } from "../store/useAppStore";

type DebugActionKey =
  | "message"
  | "edit_workflow"
  | "edit_verifier"
  | "file_edit"
  | "brain_edit_memory"
  | "brain_edit_skill"
  | "stop";

type DebugResult = {
  text: string;
};

type DebugActionConfig = {
  key: DebugActionKey;
  title: string;
  actionType: ExecutableAction["type"];
  description: string;
  disabledReason?: string;
  icon: ReactNode;
  accent: string;
  build: () => ExecutableAction;
};

function firstWorkflowNode(nodes: WorkflowNode[] | undefined): WorkflowNode | undefined {
  for (const node of nodes ?? []) {
    return node;
  }
  return undefined;
}

function findWorkflowNode(nodes: WorkflowNode[] | undefined, nodeId: string | null): WorkflowNode | undefined {
  if (!nodeId) return undefined;
  for (const node of nodes ?? []) {
    if (node.id === nodeId) return node;
    const found = findWorkflowNode(node.children, nodeId);
    if (found) return found;
  }
  return undefined;
}

function debugWorkflowNode(): WorkflowNode {
  return {
    id: `debug-${crypto.randomUUID()}`,
    description: "Debug prediction workflow step",
    outputFiles: [],
    verifiers: ["Debug step can be added from the prediction debug panel"],
    verifierMarks: [undefined],
    children: [],
    status: "pending",
    depth: 0,
  };
}

export function PredictionDebugPanel({
  onStageSuggestion,
  onClose,
}: {
  onStageSuggestion: (suggestion: PredictedUserActionSuggestion) => void;
  onClose: () => void;
}) {
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const sessions = useAppStore((s) => s.sessions);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const setGlobalError = useAppStore((s) => s.setGlobalError);
  const [lastResult, setLastResult] = useState<DebugResult | null>(null);

  const activeSession = activeSessionId ? sessions[activeSessionId] : undefined;
  const workflowTree = activeSession?.workflowTree ?? [];
  const selectedNode = findWorkflowNode(workflowTree, selectedNodeId);
  const fallbackNode = firstWorkflowNode(workflowTree);
  const verifierTarget = selectedNode ?? fallbackNode;

  const actions = useMemo<DebugActionConfig[]>(() => {
    const timestamp = new Date().toISOString();
    return [
      {
        key: "message",
        title: "Message",
        actionType: "message",
        description: "Send a normal predicted user prompt.",
        icon: <MessageSquareTextIcon className="size-4" aria-hidden />,
        accent: "text-primary bg-primary-subtle border-primary/20",
        build: () => ({
          type: "message",
          text: "Debug prediction action: continue the session with a short test prompt.",
          ...(selectedNodeId ? { verificationNodeId: selectedNodeId } : {}),
        }),
      },
      {
        key: "edit_workflow",
        title: "Workflow Edit",
        actionType: "edit_workflow",
        description: "Append a root workflow step through a workflow patch.",
        icon: <GitBranchIcon className="size-4" aria-hidden />,
        accent: "text-info bg-info-light border-info/20",
        build: () => ({
          type: "edit_workflow",
          patch: [
            {
              op: "add_node",
              parentId: null,
              node: debugWorkflowNode(),
            },
          ],
        }),
      },
      {
        key: "edit_verifier",
        title: "Verifier Edit",
        actionType: "edit_verifier",
        description: "Replace the selected step's verifier list.",
        disabledReason: verifierTarget ? undefined : "No workflow node",
        icon: <ListChecksIcon className="size-4" aria-hidden />,
        accent: "text-success bg-success-light border-success/20",
        build: () => ({
          type: "edit_verifier",
          nodeId: verifierTarget?.id ?? "",
          verifiers: [
            `Debug verifier written at ${timestamp}`,
            "Prediction debug panel can update verifier criteria",
          ],
        }),
      },
      {
        key: "file_edit",
        title: "File Edit",
        actionType: "file_edit",
        description: "Write a small text file in the session cwd.",
        icon: <FileTextIcon className="size-4" aria-hidden />,
        accent: "text-ink-700 bg-ink-900/5 border-ink-900/10",
        build: () => ({
          type: "file_edit",
          path: "prediction-debug-output.txt",
          contents: [
            "Prediction debug file edit",
            `sessionId: ${activeSessionId ?? "(none)"}`,
            `timestamp: ${timestamp}`,
            "",
          ].join("\n"),
        }),
      },
      {
        key: "brain_edit_memory",
        title: "Brain Edit: Memory",
        actionType: "brain_edit",
        description: "Write a debug memory section and record a brain edit.",
        icon: <BrainIcon className="size-4" aria-hidden />,
        accent: "text-[#7A4E20] bg-[#F1E3D1] border-[#D8B98F]",
        build: () => ({
          type: "brain_edit",
          kind: "memory",
          sections: [
            {
              fileName: "prediction-debug.md",
              content: `# Prediction Debug\n\nLast memory debug action: ${timestamp}\n`,
            },
          ],
        }),
      },
      {
        key: "brain_edit_skill",
        title: "Brain Edit: Skill",
        actionType: "brain_edit",
        description: "Write a debug skill section and record a brain edit.",
        icon: <BrainIcon className="size-4" aria-hidden />,
        accent: "text-[#7A4E20] bg-[#F1E3D1] border-[#D8B98F]",
        build: () => ({
          type: "brain_edit",
          kind: "skill",
          sections: [
            {
              fileName: "prediction-debug-skill.md",
              content: `# Prediction Debug Skill\n\nLast skill debug action: ${timestamp}\n`,
            },
          ],
        }),
      },
      {
        key: "stop",
        title: "Stop",
        actionType: "stop",
        description: "Accept a predicted stop. This is intentionally a no-op.",
        disabledReason: undefined,
        icon: <CircleStopIcon className="size-4" aria-hidden />,
        accent: "text-error bg-error-light border-error/20",
        build: () => ({ type: "stop" }),
      },
    ];
  }, [activeSessionId, selectedNodeId, verifierTarget]);

  const stageDebugSuggestion = (config: DebugActionConfig) => {
    if (!activeSessionId || !activeSession) {
      setGlobalError("Open a session before staging prediction debug suggestions.");
      return;
    }
    if (config.disabledReason) return;

    const action = config.build();
    const draftText =
      action.type === "message"
        ? action.text
        : action.type === "file_edit"
          ? action.path
          : "";
    const suggestion: PredictedUserActionSuggestion = {
      actionType: action.type,
      draftText,
      confidence: 1,
      rationale: `Debug suggestion staged for ${config.actionType}. Click Accept in the chat suggestion popup to execute it.`,
      rawResponse: JSON.stringify({
        debugPrediction: true,
        source: "PredictionDebugPanel",
        key: config.key,
        actionType: action.type,
      }),
      executable: action,
    };

    onStageSuggestion(suggestion);
    setLastResult({
      text: `${config.title} staged in the suggestion popup`,
    });
  };

  return (
    <div className="border-b border-ink-900/10 bg-white/90 px-4 py-3 shadow-soft">
      <div className="mb-3 flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <span className="rounded-md border border-primary/20 bg-primary-subtle px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-primary">
              Prediction Debug
            </span>
            <span className="text-[11px] text-muted-foreground">Shift+Cmd+D</span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            Stage executable prediction actions in the chat suggestion popup. Click Accept there to run one.
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground hover:bg-ink-900/5 hover:text-ink-800"
          aria-label="Close prediction debug panel"
          title="Close"
        >
          <XIcon className="size-4" aria-hidden />
        </button>
      </div>

      <div className="grid grid-cols-1 gap-2 xl:grid-cols-2">
        {actions.map((action) => (
          <button
            key={action.key}
            type="button"
            disabled={Boolean(action.disabledReason)}
            onClick={() => stageDebugSuggestion(action)}
            className="group flex min-w-0 items-start gap-3 rounded-lg border border-ink-900/10 bg-surface px-3 py-2.5 text-left transition-colors hover:border-ink-900/20 hover:bg-white disabled:cursor-not-allowed disabled:opacity-50"
            title={action.disabledReason ?? `Run ${action.actionType}`}
          >
            <span className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border ${action.accent}`}>
              {action.icon}
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-semibold text-ink-900">{action.title}</span>
              <span className="mt-0.5 block text-xs leading-snug text-muted-foreground">
                {action.disabledReason ?? action.description}
              </span>
            </span>
          </button>
        ))}
      </div>

      {lastResult && (
        <div
          className="mt-3 rounded-lg border border-success/20 bg-success-light px-3 py-2 text-xs text-success"
        >
          {lastResult.text}
        </div>
      )}
    </div>
  );
}
