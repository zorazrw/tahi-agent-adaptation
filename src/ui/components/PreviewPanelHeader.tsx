import type { FC } from "react";
import { XIcon } from "lucide-react";
import { useAppStore } from "../store/useAppStore";
import { getPreviewFileForNode } from "./FilePreview";
import type { WorkflowNode } from "../types";

function findNode(tree: WorkflowNode[], id: string): WorkflowNode | undefined {
  for (const node of tree) {
    if (node.id === id) return node;
    const found = findNode(node.children, id);
    if (found) return found;
  }
  return undefined;
}

export const PreviewPanelHeader: FC = () => {
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const sessions = useAppStore((s) => s.sessions);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const setPreviewPanelOpen = useAppStore((s) => s.setPreviewPanelOpen);

  const session = activeSessionId ? sessions[activeSessionId] : undefined;
  const tree = session?.workflowTree ?? [];
  const selectedNode = selectedNodeId ? findNode(tree, selectedNodeId) : undefined;
  const currentFile = selectedNode ? getPreviewFileForNode(selectedNode.outputFiles) : null;

  return (
    <div className="shrink-0 flex items-center gap-2 px-3 py-2 border-b border-ink-900/10">
      <span className="text-xs font-medium text-ink-700 truncate">
        {selectedNode ? selectedNode.description : "Preview"}
      </span>

      {currentFile && (
        <span className="text-xs text-muted-foreground truncate ml-auto mr-2" title={currentFile}>
          {currentFile.split("/").pop()}
        </span>
      )}

      <button
        onClick={() => setPreviewPanelOpen(false)}
        className="shrink-0 p-1 rounded hover:bg-ink-900/5 text-muted-foreground hover:text-ink-700 transition-colors"
        aria-label="Close preview panel"
      >
        <XIcon className="size-4" />
      </button>
    </div>
  );
};
