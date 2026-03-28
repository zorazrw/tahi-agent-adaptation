import { useEffect, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { useAppStore } from "../store/useAppStore";
import type { WorkflowRunMode } from "../store/useAppStore";
import { Spinner } from "./Spinner";

interface SettingsModalProps {
  onClose: () => void;
}

type Tab = "api" | "workflow" | "skills";

function WorkflowPanel() {
  const workflowRunMode = useAppStore((s) => s.workflowRunMode);
  const setWorkflowRunMode = useAppStore((s) => s.setWorkflowRunMode);

  const row = (mode: WorkflowRunMode, title: string, description: string) => (
    <label
      key={mode}
      className="flex cursor-pointer gap-3 rounded-xl border border-ink-900/10 bg-surface p-4 hover:border-ink-900/20 transition-colors has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-primary/30"
    >
      <input
        type="radio"
        name="workflowRunMode"
        className="mt-1 border-ink-900/20 text-primary focus:ring-primary/30"
        checked={workflowRunMode === mode}
        onChange={() => setWorkflowRunMode(mode)}
      />
      <div>
        <div className="text-sm font-medium text-ink-800">{title}</div>
        <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{description}</p>
      </div>
    </label>
  );

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground leading-relaxed">
        Controls how the app advances through the workflow tree after a step completes successfully.
      </p>
      <div className="space-y-2">
        {row(
          "manual",
          "Wait",
          "Pause after each step. Start the next step yourself with Run in the sidebar."
        )}
        {row(
          "auto",
          "Auto",
          "Automatically start the next incomplete step until every step in the plan is finished (default for new installs)."
        )}
      </div>
    </div>
  );
}

export function SettingsModal({ onClose }: SettingsModalProps) {
  const [tab, setTab] = useState<Tab>("api");

  return (
    <Dialog.Root open onOpenChange={(open) => { if (!open) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-ink-900/20 backdrop-blur-sm animate-fade-in" />
        <Dialog.Content className="fixed inset-0 z-50 flex items-center justify-center px-4 py-8">
          <div className="w-full max-w-lg rounded-2xl border border-ink-900/5 bg-surface shadow-elevated animate-scale-in flex flex-col max-h-[80vh]">
            {/* Header */}
            <div className="flex items-center justify-between px-6 pt-6 pb-0 shrink-0">
              <Dialog.Title className="text-base font-semibold text-ink-800">Settings</Dialog.Title>
              <Dialog.Close asChild>
                <button
                  className="rounded-full p-1.5 text-muted-foreground hover:bg-surface-tertiary hover:text-ink-700 transition-colors"
                  aria-label="Close"
                >
                  <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              </Dialog.Close>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 px-6 pt-4 pb-0 shrink-0 flex-wrap">
              <button
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                  tab === "api"
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:text-ink-700 hover:bg-ink-900/5"
                }`}
                onClick={() => setTab("api")}
              >
                API
              </button>
              <button
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                  tab === "workflow"
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:text-ink-700 hover:bg-ink-900/5"
                }`}
                onClick={() => setTab("workflow")}
              >
                Workflow
              </button>
              <button
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                  tab === "skills"
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:text-ink-700 hover:bg-ink-900/5"
                }`}
                onClick={() => setTab("skills")}
              >
                Skills
              </button>
            </div>

            {/* Tab content */}
            <div className="flex-1 min-h-0 overflow-y-auto px-6 pb-6 pt-4">
              {tab === "api" ? (
                <ApiPanel onClose={onClose} />
              ) : tab === "workflow" ? (
                <WorkflowPanel />
              ) : (
                <SkillsPanel />
              )}
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

/* ---------- API Config Panel ---------- */

function ApiPanel({ onClose }: { onClose: () => void }) {
  const [apiKey, setApiKey] = useState("");
  const [baseURL, setBaseURL] = useState("");
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    setLoading(true);
    window.electron.getApiConfig()
      .then((config) => {
        if (config) {
          setApiKey(config.apiKey);
          setBaseURL(config.baseURL);
          setModel(config.model);
        }
      })
      .catch((err) => {
        console.error("Failed to load API config:", err);
        setError("Failed to load configuration");
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const handleSave = async () => {
    if (!apiKey.trim()) { setError("API Key is required"); return; }
    if (!baseURL.trim()) { setError("Base URL is required"); return; }
    if (!model.trim()) { setError("Model is required"); return; }
    try { new URL(baseURL); } catch { setError("Invalid Base URL format"); return; }

    setError(null);
    setSaving(true);

    try {
      const result = await window.electron.saveApiConfig({
        apiKey: apiKey.trim(),
        baseURL: baseURL.trim(),
        model: model.trim(),
        apiType: "anthropic"
      });

      if (result.success) {
        setSuccess(true);
        setTimeout(() => {
          setSuccess(false);
          onClose();
        }, 1000);
      } else {
        setError(result.error || "Failed to save configuration");
      }
    } catch (err) {
      console.error("Failed to save API config:", err);
      setError("Failed to save configuration");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Spinner className="w-6 h-6 text-primary" color="currentColor" />
      </div>
    );
  }

  return (
    <>
      <p className="text-sm text-muted-foreground">Supports Anthropic's official API as well as third-party APIs compatible with the Anthropic format.</p>
      <div className="mt-4 grid gap-4">
        <label className="grid gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">Base URL</span>
          <input
            type="url"
            className="rounded-xl border border-ink-900/10 bg-surface-secondary px-4 py-2.5 text-sm text-ink-800 placeholder:text-placeholder focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20 transition-colors"
            placeholder="https://..."
            value={baseURL}
            onChange={(e) => setBaseURL(e.target.value)}
            required
          />
        </label>
        <label className="grid gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">API Key</span>
          <input
            type="password"
            className="rounded-xl border border-ink-900/10 bg-surface-secondary px-4 py-2.5 text-sm text-ink-800 placeholder:text-placeholder focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20 transition-colors"
            placeholder="sk-..."
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            required
          />
        </label>
        <label className="grid gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">Model Name</span>
          <input
            type="text"
            className="rounded-xl border border-ink-900/10 bg-surface-secondary px-4 py-2.5 text-sm text-ink-800 placeholder:text-placeholder focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20 transition-colors"
            placeholder="claude-3-5-sonnet-20241022"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            required
          />
        </label>

        {error && (
          <div className="rounded-xl border border-error/20 bg-error-light px-4 py-2.5 text-sm text-error">
            {error}
          </div>
        )}
        {success && (
          <div className="rounded-xl border border-success/20 bg-success-light px-4 py-2.5 text-sm text-success">
            Configuration saved successfully!
          </div>
        )}

        <div className="flex gap-3">
          <button
            className="flex-1 rounded-xl border border-ink-900/10 bg-surface px-4 py-2.5 text-sm font-medium text-ink-700 hover:bg-surface-tertiary transition-colors"
            onClick={onClose}
            disabled={saving}
          >
            Cancel
          </button>
          <button
            className="flex-1 rounded-xl bg-primary px-4 py-2.5 text-sm font-medium text-white shadow-soft hover:bg-primary-hover transition-colors disabled:cursor-not-allowed disabled:opacity-50"
            onClick={handleSave}
            disabled={saving || !apiKey.trim() || !baseURL.trim() || !model.trim()}
          >
            {saving ? <Spinner className="mx-auto w-5 h-5" /> : "Save"}
          </button>
        </div>
      </div>
    </>
  );
}

/* ---------- Skills Panel ---------- */

function SkillsPanel() {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedPath, setExpandedPath] = useState<string | null>(null);
  const [skillContent, setSkillContent] = useState<string | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [removeConfirm, setRemoveConfirm] = useState<string | null>(null);

  useEffect(() => {
    window.electron.listSkills().then((result) => {
      setSkills(result);
      setLoading(false);
    });
  }, []);

  const handleToggleSkill = async (skill: SkillInfo) => {
    if (expandedPath === skill.path) {
      setExpandedPath(null);
      setSkillContent(null);
      return;
    }
    setExpandedPath(skill.path);
    setSkillContent(null);
    setLoadingContent(true);
    const result = await window.electron.getSkillContent(skill.path);
    setSkillContent("content" in result ? result.content : "Failed to load content.");
    setLoadingContent(false);
  };

  const handleRemoveSkill = async (dirName: string) => {
    const result = await window.electron.removeSkill(dirName);
    if (result.success) {
      setSkills((prev) => prev.filter((s) => !(s.dirName === dirName && s.source === "app")));
      if (skills.find((s) => s.dirName === dirName)?.path === expandedPath) {
        setExpandedPath(null);
        setSkillContent(null);
      }
    }
    setRemoveConfirm(null);
  };

  const handleOpenFolder = () => {
    window.electron.getSkillsDir();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Spinner className="w-6 h-6 text-primary" color="currentColor" />
      </div>
    );
  }

  return (
    <>
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-muted-foreground">Manage installed skills.</p>
        <button
          onClick={handleOpenFolder}
          className="rounded-lg border border-ink-900/10 bg-surface px-3 py-1.5 text-xs font-medium text-ink-600 hover:bg-surface-tertiary hover:border-ink-900/20 transition-colors shrink-0"
        >
          Open Folder
        </button>
      </div>

      {skills.length === 0 ? (
        <div className="rounded-xl border border-dashed border-ink-900/15 bg-surface/50 px-6 py-10 text-center">
          <svg viewBox="0 0 24 24" className="mx-auto h-8 w-8 text-ink-300 mb-3" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
          </svg>
          <p className="text-sm text-muted-foreground">
            No skills found. Add a <code className="text-xs bg-ink-900/5 px-1 py-0.5 rounded">*.md</code> file in the skills
            folder, or a subfolder containing{" "}
            <code className="text-xs bg-ink-900/5 px-1 py-0.5 rounded">SKILL.md</code>.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {skills.map((skill) => (
            <div
              key={`${skill.source}-${skill.dirName}`}
              className={`rounded-xl border transition-colors ${
                expandedPath === skill.path
                  ? "border-primary/30 bg-primary-subtle/30"
                  : "border-ink-900/10 bg-surface hover:border-ink-900/20 hover:bg-surface-tertiary"
              }`}
            >
              <div
                className="flex items-center gap-3 px-4 py-3 cursor-pointer"
                onClick={() => handleToggleSkill(skill)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleToggleSkill(skill); } }}
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-ink-800 truncate">{skill.name}</div>
                  {skill.description && (
                    <div className="text-xs text-muted-foreground truncate mt-0.5">{skill.description}</div>
                  )}
                </div>
                <span
                  className={`shrink-0 inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium ${
                    skill.source === "app"
                      ? "bg-amber-100 text-amber-700"
                      : "bg-ink-900/8 text-ink-500"
                  }`}
                >
                  {skill.source === "app" ? "App" : "User"}
                </span>
                {skill.source === "app" && (
                  <button
                    onClick={(e) => { e.stopPropagation(); setRemoveConfirm(skill.dirName); }}
                    className="shrink-0 rounded-lg p-1.5 text-ink-400 hover:text-red-500 hover:bg-red-50 transition-colors"
                    aria-label={`Remove ${skill.name}`}
                  >
                    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                  </button>
                )}
                <svg
                  viewBox="0 0 24 24"
                  className={`h-3.5 w-3.5 shrink-0 text-ink-400 transition-transform ${expandedPath === skill.path ? "rotate-180" : ""}`}
                  fill="none" stroke="currentColor" strokeWidth="2"
                >
                  <path d="M6 9l6 6 6-6" />
                </svg>
              </div>
              {expandedPath === skill.path && (
                <div className="border-t border-ink-900/10 px-4 py-3">
                  {loadingContent ? (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground py-2">
                      <Spinner className="w-3.5 h-3.5" color="currentColor" />
                      Loading...
                    </div>
                  ) : (
                    <pre className="text-xs text-ink-700 whitespace-pre-wrap break-words max-h-64 overflow-y-auto leading-relaxed">
                      {skillContent}
                    </pre>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {removeConfirm && (
        <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4">
          <p className="text-sm text-ink-700">Remove this skill? The skill directory and its symlink will be deleted.</p>
          <div className="mt-3 flex gap-2">
            <button
              onClick={() => setRemoveConfirm(null)}
              className="flex-1 rounded-lg border border-ink-900/10 bg-white px-3 py-2 text-xs font-medium text-ink-700 hover:bg-surface-tertiary transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => handleRemoveSkill(removeConfirm)}
              className="flex-1 rounded-lg bg-red-500 px-3 py-2 text-xs font-medium text-white hover:bg-red-600 transition-colors"
            >
              Remove
            </button>
          </div>
        </div>
      )}
    </>
  );
}
