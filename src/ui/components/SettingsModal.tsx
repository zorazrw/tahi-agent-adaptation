import { useEffect, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { useAppStore } from "../store/useAppStore";
import type { WorkflowRunMode } from "../store/useAppStore";
import { Spinner } from "./Spinner";
import type { AvailableModel, OpenAICompatibleApiFormat, ProviderAuthStatus } from "../../lib/runtime-types";

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
          <div
            className={`w-full rounded-2xl border border-ink-900/5 bg-surface shadow-elevated animate-scale-in flex flex-col max-h-[84vh] ${
              tab === "api" ? "max-w-5xl" : "max-w-3xl"
            }`}
          >
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
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [thinkingLevel, setThinkingLevel] = useState<"off" | "minimal" | "low" | "medium" | "high" | "xhigh">("medium");
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [providerStatuses, setProviderStatuses] = useState<Record<string, ProviderAuthStatus>>({});
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savingApiKey, setSavingApiKey] = useState(false);
  const [authBusy, setAuthBusy] = useState(false);
  const [customBaseUrl, setCustomBaseUrl] = useState("");
  const [customModel, setCustomModel] = useState("");
  const [customApiFormat, setCustomApiFormat] = useState<OpenAICompatibleApiFormat>("openai-completions");
  const [customApiKey, setCustomApiKey] = useState("");
  const [customConfigured, setCustomConfigured] = useState(false);
  const [savingCustomProvider, setSavingCustomProvider] = useState(false);
  const [removingCustomProvider, setRemovingCustomProvider] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const loadStatuses = async (providers: string[]) => {
    const entries = await Promise.all(
      providers.map(async (name) => [name, await window.electron.getProviderAuthStatus(name)] as const)
    );
    setProviderStatuses(Object.fromEntries(entries));
  };

  const syncModelState = async (
    availableModels: AvailableModel[],
    preferred?: { provider?: string; model?: string },
    defaults?: { defaultProvider?: string; defaultModel?: string }
  ) => {
    setModels(availableModels);
    const providers = [...new Set(availableModels.map((item) => item.provider))];
    const nextProvider =
      preferred?.provider && providers.includes(preferred.provider)
        ? preferred.provider
        : defaults?.defaultProvider && providers.includes(defaults.defaultProvider)
          ? defaults.defaultProvider
          : providers[0] || "";
    const providerModels = availableModels.filter((item) => item.provider === nextProvider);
    const nextModel =
      preferred?.model && providerModels.some((item) => item.id === preferred.model)
        ? preferred.model
        : defaults?.defaultModel && providerModels.some((item) => item.id === defaults.defaultModel)
          ? defaults.defaultModel
          : providerModels[0]?.id || "";
    setProvider(nextProvider);
    setModel(nextModel);
    await loadStatuses(providers);
  };

  const loadOpenAICompatibleProvider = async () => {
    const config = await window.electron.getOpenAICompatibleProvider();
    setCustomConfigured(Boolean(config));
    setCustomBaseUrl(config?.baseUrl ?? "");
    setCustomModel(config?.model ?? "");
    setCustomApiFormat(config?.apiFormat ?? "openai-completions");
    setCustomApiKey("");
    return config;
  };

  useEffect(() => {
    setLoading(true);
    Promise.all([
      window.electron.getAgentSettings(),
      window.electron.listAvailableModels(),
      loadOpenAICompatibleProvider(),
    ])
      .then(async ([settings, availableModels]) => {
        setThinkingLevel(settings.defaultThinkingLevel ?? "medium");
        await syncModelState(availableModels, undefined, settings);
      })
      .catch((err) => {
        console.error("Failed to load agent settings:", err);
        setError("Failed to load agent settings");
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const handleSave = async () => {
    if (!provider.trim()) { setError("Provider is required"); return; }
    if (!model.trim()) { setError("Model is required"); return; }
    setError(null);
    setSaving(true);

    try {
      const result = await window.electron.saveAgentSettings({
        defaultProvider: provider,
        defaultModel: model,
        defaultThinkingLevel: thinkingLevel,
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

  const handleSaveOpenAICompatibleProvider = async () => {
    if (!customBaseUrl.trim()) {
      setError("Custom base URL is required");
      return;
    }
    if (!customModel.trim()) {
      setError("Custom model slug is required");
      return;
    }

    setError(null);
    setSavingCustomProvider(true);
    try {
      const result = await window.electron.saveOpenAICompatibleProvider({
        baseUrl: customBaseUrl.trim(),
        model: customModel.trim(),
        apiFormat: customApiFormat,
        apiKey: customApiKey.trim() || undefined,
      });
      if (!result.success) {
        setError(result.error || "Failed to save custom provider");
        return;
      }

      const [availableModels] = await Promise.all([
        window.electron.listAvailableModels(),
        loadOpenAICompatibleProvider(),
      ]);
      await syncModelState(availableModels, {
        provider: "openai-compatible",
        model: customModel.trim(),
      });
      setSuccess(true);
      setTimeout(() => setSuccess(false), 1200);
    } catch (err) {
      console.error("Failed to save OpenAI-compatible provider:", err);
      setError("Failed to save custom provider");
    } finally {
      setSavingCustomProvider(false);
    }
  };

  const handleRemoveOpenAICompatibleProvider = async () => {
    setError(null);
    setRemovingCustomProvider(true);
    try {
      const result = await window.electron.removeOpenAICompatibleProvider();
      if (!result.success) {
        setError(result.error || "Failed to remove custom provider");
        return;
      }

      setCustomConfigured(false);
      setCustomBaseUrl("");
      setCustomModel("");
      setCustomApiFormat("openai-completions");
      setCustomApiKey("");

      const settings = await window.electron.getAgentSettings();
      const availableModels = await window.electron.listAvailableModels();
      await syncModelState(
        availableModels,
        provider === "openai-compatible" ? undefined : { provider, model },
        settings
      );
      setSuccess(true);
      setTimeout(() => setSuccess(false), 1200);
    } catch (err) {
      console.error("Failed to remove OpenAI-compatible provider:", err);
      setError("Failed to remove custom provider");
    } finally {
      setRemovingCustomProvider(false);
    }
  };

  const handleSaveApiKey = async () => {
    if (!provider.trim() || !apiKey.trim()) {
      setError("Provider and API key are required");
      return;
    }
    setError(null);
    setSavingApiKey(true);
    try {
      const result = await window.electron.saveProviderApiKey(provider, apiKey.trim());
      if (!result.success) {
        setError(result.error || "Failed to save API key");
        return;
      }
      await loadStatuses([...new Set(models.map((item) => item.provider))]);
      setApiKey("");
      setSuccess(true);
      setTimeout(() => setSuccess(false), 1200);
    } catch (err) {
      console.error("Failed to save provider API key:", err);
      setError("Failed to save provider API key");
    } finally {
      setSavingApiKey(false);
    }
  };

  const handleAuthAction = async (action: "login" | "logout") => {
    if (!provider.trim()) return;
    setError(null);
    setAuthBusy(true);
    try {
      const result =
        action === "login"
          ? await window.electron.loginProvider(provider)
          : await window.electron.logoutProvider(provider);
      if (!result.success) {
        setError(result.error || `Failed to ${action}`);
        return;
      }
      await loadStatuses([...new Set(models.map((item) => item.provider))]);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 1200);
    } catch (err) {
      console.error(`Failed to ${action} provider:`, err);
      setError(`Failed to ${action} provider`);
    } finally {
      setAuthBusy(false);
    }
  };

  const providerOptions = [...new Set(models.map((item) => item.provider))];
  const modelOptions = models.filter((item) => item.provider === provider);
  const currentStatus = provider ? providerStatuses[provider] : undefined;
  const fieldClass =
    "w-full min-w-0 rounded-xl border border-ink-900/10 bg-white px-4 py-2.5 text-sm text-ink-800 placeholder:text-placeholder focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/20 transition-colors";
  const cardClass = "rounded-[24px] border border-ink-900/8 bg-surface-secondary/75 p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.7)]";
  const secondaryButtonClass =
    "rounded-xl border border-ink-900/10 bg-white px-4 py-2.5 text-sm font-medium text-ink-700 hover:bg-surface-tertiary transition-colors disabled:cursor-not-allowed disabled:opacity-50";

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Spinner className="w-6 h-6 text-primary" color="currentColor" />
      </div>
    );
  }

  return (
    <>
      <div className="rounded-[28px] border border-ink-900/6 bg-[linear-gradient(180deg,rgba(255,255,255,0.96),rgba(249,246,241,0.88))] p-5">
        <div className="flex flex-col gap-1">
          <div className="text-[11px] font-semibold uppercase tracking-[0.24em] text-primary/80">Runtime Configuration</div>
          <p className="text-sm text-muted-foreground">
            Configure the default Pi provider on the left and manage any custom OpenAI-compatible endpoint on the right.
          </p>
        </div>

        <div className="mt-5 grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,0.95fr)]">
          <section className={cardClass}>
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="text-base font-semibold text-ink-800">Default Provider</div>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  Choose the provider and model used for new Pi sessions, then keep credentials in sync for that provider.
                </p>
              </div>
              <div className="max-w-full truncate rounded-full bg-white px-3 py-1 text-[11px] font-medium text-ink-600 shadow-sm">
                {provider || "No provider"}
              </div>
            </div>

            <div className="mt-5 grid gap-4 md:grid-cols-2">
              <label className="grid min-w-0 gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">Provider</span>
                <select
                  className={fieldClass}
                  value={provider}
                  onChange={(e) => {
                    const nextProvider = e.target.value;
                    setProvider(nextProvider);
                    const nextModel = models.find((item) => item.provider === nextProvider)?.id || "";
                    setModel(nextModel);
                  }}
                >
                  {providerOptions.map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
              </label>
              <label className="grid min-w-0 gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">Model</span>
                <select
                  className={`${fieldClass} truncate`}
                  title={model}
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  {modelOptions.map((item) => (
                    <option key={item.id} value={item.id}>{item.label}</option>
                  ))}
                </select>
              </label>
            </div>

            <div className="mt-4 grid gap-4 md:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">Default Thinking Level</span>
                <select
                  className={fieldClass}
                  value={thinkingLevel}
                  onChange={(e) => setThinkingLevel(e.target.value as typeof thinkingLevel)}
                >
                  {["off", "minimal", "low", "medium", "high", "xhigh"].map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
              </label>

              <div className="rounded-2xl border border-ink-900/8 bg-white/80 px-4 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-500">Auth Status</div>
                <div className="mt-2 text-sm text-ink-700">
                  {currentStatus?.hasAuth
                    ? `Configured via ${currentStatus.authType === "env" ? "environment" : currentStatus.authType?.replace("_", " ") || "credentials"}`
                    : "Not configured"}
                </div>
                {currentStatus?.supportsOAuth && (
                  <div className="mt-1 text-xs text-muted-foreground">
                    OAuth supported: {currentStatus.oauthName || provider}
                  </div>
                )}
              </div>
            </div>

            <div className="mt-5 grid gap-4">
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">Provider API Key</span>
                <input
                  type="password"
                  className={fieldClass}
                  placeholder="sk-..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
              </label>

              <div className={`grid gap-3 ${currentStatus?.supportsOAuth ? "md:grid-cols-3" : "md:grid-cols-1"}`}>
                <button
                  className={secondaryButtonClass}
                  onClick={handleSaveApiKey}
                  disabled={savingApiKey || !provider.trim() || !apiKey.trim()}
                >
                  {savingApiKey ? <Spinner className="mx-auto w-5 h-5" /> : "Save API Key"}
                </button>
                {currentStatus?.supportsOAuth && (
                  <>
                    <button
                      className={secondaryButtonClass}
                      onClick={() => handleAuthAction("login")}
                      disabled={authBusy || !provider.trim()}
                    >
                      Login OAuth
                    </button>
                    <button
                      className={secondaryButtonClass}
                      onClick={() => handleAuthAction("logout")}
                      disabled={authBusy || !provider.trim()}
                    >
                      Logout
                    </button>
                  </>
                )}
              </div>
            </div>
          </section>

          <section className={`${cardClass} bg-[linear-gradient(180deg,rgba(255,255,255,0.92),rgba(250,242,236,0.94))]`}>
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-base font-semibold text-ink-800">OpenAI-Compatible Endpoint</div>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  Save a custom `/v1`-style endpoint with its own base URL, model slug, and API protocol.
                </p>
              </div>
              <div className={`rounded-full px-3 py-1 text-[11px] font-medium shadow-sm ${
                customConfigured ? "bg-primary/10 text-primary" : "bg-white text-ink-500"
              }`}>
                {customConfigured ? "Configured" : "Not Configured"}
              </div>
            </div>

            <div className="mt-5 grid gap-4">
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">API Format</span>
                <select
                  className={fieldClass}
                  value={customApiFormat}
                  onChange={(e) => setCustomApiFormat(e.target.value as OpenAICompatibleApiFormat)}
                >
                  <option value="openai-completions">OpenAI Completions</option>
                  <option value="openai-responses">OpenAI Responses</option>
                </select>
              </label>
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">Base URL</span>
                <input
                  type="text"
                  className={`${fieldClass} font-mono text-[13px]`}
                  placeholder="https://your-endpoint.example.com/v1"
                  value={customBaseUrl}
                  onChange={(e) => setCustomBaseUrl(e.target.value)}
                />
              </label>
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">Model Slug</span>
                <input
                  type="text"
                  className={`${fieldClass} font-mono text-[13px]`}
                  placeholder="gpt-4.1-mini or local-model"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                />
              </label>
              <label className="grid gap-1.5">
                <span className="text-xs font-medium text-muted-foreground">Endpoint API Key</span>
                <input
                  type="password"
                  className={`${fieldClass} font-mono text-[13px]`}
                  placeholder={customConfigured ? "Leave blank to keep the saved key" : "sk-..."}
                  value={customApiKey}
                  onChange={(e) => setCustomApiKey(e.target.value)}
                />
              </label>

              <div className="rounded-2xl border border-primary/10 bg-white/80 px-4 py-3 text-xs leading-5 text-muted-foreground">
                {customConfigured
                  ? "This endpoint is already stored. Re-saving with a blank key preserves the existing credential."
                  : "Once saved, this endpoint appears in the default provider picker as `openai-compatible`."}
              </div>

              <div className="grid gap-3">
                <button
                  className="rounded-xl bg-primary px-4 py-2.5 text-sm font-medium text-white shadow-soft hover:bg-primary-hover transition-colors disabled:cursor-not-allowed disabled:opacity-50"
                  onClick={handleSaveOpenAICompatibleProvider}
                  disabled={savingCustomProvider || !customBaseUrl.trim() || !customModel.trim()}
                >
                  {savingCustomProvider ? <Spinner className="mx-auto w-5 h-5" /> : "Save Endpoint"}
                </button>
                <button
                  className={secondaryButtonClass}
                  onClick={handleRemoveOpenAICompatibleProvider}
                  disabled={removingCustomProvider || !customConfigured}
                >
                  {removingCustomProvider ? <Spinner className="mx-auto w-5 h-5" /> : "Remove Endpoint"}
                </button>
              </div>
            </div>
          </section>
        </div>

        {(error || success) && (
          <div className="mt-4 grid gap-3">
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
          </div>
        )}

        <div className="mt-5 flex flex-col-reverse gap-3 border-t border-ink-900/6 pt-5 sm:flex-row sm:justify-end">
          <button
            className={`${secondaryButtonClass} sm:min-w-36`}
            onClick={onClose}
            disabled={saving}
          >
            Cancel
          </button>
          <button
            className="rounded-xl bg-primary px-4 py-2.5 text-sm font-medium text-white shadow-soft hover:bg-primary-hover transition-colors disabled:cursor-not-allowed disabled:opacity-50 sm:min-w-36"
            onClick={handleSave}
            disabled={saving || !provider.trim() || !model.trim()}
          >
            {saving ? <Spinner className="mx-auto w-5 h-5" /> : "Save Defaults"}
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
