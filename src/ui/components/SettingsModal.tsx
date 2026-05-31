import { useEffect, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { useAppStore } from "../store/useAppStore";
import type { WorkflowRunMode } from "../store/useAppStore";
import { Spinner } from "./Spinner";
import type {
  AvailableModel,
  OpenAICompatibleApiFormat,
  ProviderAuthStatus,
} from "../../lib/runtime-types";
import { AUTO_INDUCTION_KEY, readStoredAutoInduction } from "../lib/auto-induction";

interface SettingsModalProps {
  onClose: () => void;
}

type Tab = "api" | "mode" | "data";

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
      <div className="grid grid-cols-2 gap-2">
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

function deriveTinkerSlug(baseModel: string): string {
  const tail = baseModel.trim().split("/").pop() ?? baseModel.trim();
  const normalized = tail.trim().toLowerCase().replace(/\s+/g, "-");
  return normalized || "tinker";
}

export function SettingsModal({ onClose }: SettingsModalProps) {
  const [tab, setTab] = useState<Tab>("api");

  return (
    <Dialog.Root open onOpenChange={(open) => { if (!open) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-ink-900/20 backdrop-blur-sm animate-fade-in" />
        <Dialog.Content className="fixed inset-0 z-50 flex items-center justify-center px-4 py-8">
          <div
            className="w-full max-w-3xl rounded-2xl border border-ink-900/5 bg-surface shadow-elevated animate-scale-in flex flex-col max-h-[84vh]"
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
                  tab === "mode"
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:text-ink-700 hover:bg-ink-900/5"
                }`}
                onClick={() => setTab("mode")}
              >
                Mode
              </button>
              <button
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                  tab === "data"
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:text-ink-700 hover:bg-ink-900/5"
                }`}
                onClick={() => setTab("data")}
              >
                Data
              </button>
            </div>

            {/* Tab content */}
            <div className="flex-1 min-h-0 overflow-y-auto px-6 pb-6 pt-4">
              {tab === "api" ? (
                <ApiPanel onClose={onClose} />
              ) : tab === "mode" ? (
                <ModePanel />
              ) : (
                <DataPanel />
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
  const [authBusy, setAuthBusy] = useState(false);
  const [customBaseUrl, setCustomBaseUrl] = useState("");
  const [customModel, setCustomModel] = useState("");
  const [customApiFormat, setCustomApiFormat] = useState<OpenAICompatibleApiFormat>("openai-completions");
  const [customApiKey, setCustomApiKey] = useState("");
  const [customConfigured, setCustomConfigured] = useState(false);
  const [removingCustomProvider, setRemovingCustomProvider] = useState(false);
  const [tinkerBaseUrl, setTinkerBaseUrl] = useState("");
  const [tinkerBaseModel, setTinkerBaseModel] = useState("");
  const [tinkerModelPath, setTinkerModelPath] = useState("");
  const [tinkerRendererName, setTinkerRendererName] = useState("");
  const [tinkerReasoning, setTinkerReasoning] = useState(true);
  const [tinkerContextWindow, setTinkerContextWindow] = useState("128000");
  const [tinkerMaxTokens, setTinkerMaxTokens] = useState("16384");
  const [tinkerApiKey, setTinkerApiKey] = useState("");
  const [tinkerConfigured, setTinkerConfigured] = useState(false);
  const [removingTinkerProvider, setRemovingTinkerProvider] = useState(false);
  const [tinkerAdvancedOpen, setTinkerAdvancedOpen] = useState(false);
  const [tinkerResolving, setTinkerResolving] = useState(false);
  const [tinkerResolveError, setTinkerResolveError] = useState<string | null>(null);
  const [tinkerBaseModelResolved, setTinkerBaseModelResolved] = useState(false);
  const [authMethod, setAuthMethod] = useState<"oauth" | "api_key">("api_key");
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
    const fallbackModel = providerModels.find((item) => item.id.includes("claude-sonnet-4-6"))?.id
      ?? providerModels[0]?.id ?? "";
    const nextModel =
      preferred?.model && providerModels.some((item) => item.id === preferred.model)
        ? preferred.model
        : defaults?.defaultModel && providerModels.some((item) => item.id === defaults.defaultModel)
          ? defaults.defaultModel
          : fallbackModel;
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

  const loadTinkerProvider = async () => {
    const config = await window.electron.getTinkerProvider();
    setTinkerConfigured(Boolean(config));
    setTinkerBaseUrl(config?.baseUrl ?? "");
    setTinkerBaseModel(config?.model.baseModel ?? "");
    setTinkerModelPath(config?.model.modelPath ?? "");
    setTinkerRendererName(config?.model.rendererName ?? "");
    setTinkerReasoning(config?.model.reasoning ?? true);
    setTinkerContextWindow(String(config?.model.contextWindow ?? 128000));
    setTinkerMaxTokens(String(config?.model.maxTokens ?? 16384));
    setTinkerApiKey("");
    setTinkerBaseModelResolved(Boolean(config?.model.modelPath?.startsWith("tinker://") && config?.model.baseModel));
    setTinkerResolveError(null);
    setTinkerResolving(false);
    return config;
  };

  useEffect(() => {
    setLoading(true);
    Promise.all([
      window.electron.getAgentSettings(),
      window.electron.listAvailableModels(),
      loadOpenAICompatibleProvider(),
      loadTinkerProvider(),
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

  // After training completes, main persists the new sampler_path into the Tinker provider file.
  useEffect(() => {
    const unsubscribe = window.electron.onTinkerModelUpdated(async (event) => {
      try {
        const [settings, availableModels] = await Promise.all([
          window.electron.getAgentSettings(),
          window.electron.listAvailableModels(),
          loadOpenAICompatibleProvider(),
          loadTinkerProvider(),
        ]);

        // Prefer the live SSE event over disk in case the reload raced the write.
        if (event.model_path?.startsWith("tinker://")) {
          setTinkerModelPath(event.model_path);
          setTinkerResolveError(null);
        }
        if (event.base_model) {
          setTinkerBaseModel(event.base_model);
          setTinkerBaseModelResolved(Boolean(event.model_path?.startsWith("tinker://")));
        }
        if (event.renderer_name) {
          setTinkerRendererName(event.renderer_name);
        }

        await syncModelState(availableModels, { model: event.slug }, settings);
      } catch (err) {
        console.error("Failed to apply tinker model update in UI:", err);
      }
    });
    return unsubscribe;
  }, []);

  // Sync authMethod when provider or statuses change.
  useEffect(() => {
    const status = providerStatuses[provider];
    if (!status) return;
    if (status.supportsOAuth && status.authType === "oauth") {
      setAuthMethod("oauth");
    } else if (status.authType === "api_key" || status.authType === "env" || !status.supportsOAuth) {
      setAuthMethod("api_key");
    } else {
      // Provider supports OAuth but no auth configured yet – default to OAuth for Anthropic.
      setAuthMethod(status.supportsOAuth ? "oauth" : "api_key");
    }
  }, [provider, providerStatuses]);

  useEffect(() => {
    if (provider !== "tinker") {
      setTinkerResolving(false);
      setTinkerResolveError(null);
      return;
    }

    const nextPath = tinkerModelPath.trim();
    if (!nextPath || !nextPath.startsWith("tinker://")) {
      setTinkerResolving(false);
      setTinkerResolveError(null);
      setTinkerBaseModelResolved(false);
      return;
    }

    let cancelled = false;
    setTinkerResolving(true);
    setTinkerResolveError(null);

    const timeoutId = window.setTimeout(async () => {
      try {
        const result = await window.electron.resolveTinkerCheckpoint(
          nextPath,
          tinkerApiKey.trim() || undefined,
          tinkerBaseUrl.trim() || undefined,
        );
        if (cancelled) return;
        if (!result.ok) {
          setTinkerBaseModelResolved(false);
          setTinkerResolveError(result.error);
          return;
        }
        setTinkerBaseModel(result.base_model);
        setTinkerBaseModelResolved(true);
        setTinkerResolveError(null);
      } catch (err) {
        if (cancelled) return;
        setTinkerBaseModelResolved(false);
        setTinkerResolveError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) {
          setTinkerResolving(false);
        }
      }
    }, 500);

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [provider, tinkerModelPath, tinkerApiKey, tinkerBaseUrl]);

  const handleSave = async () => {
    setError(null);
    setSaving(true);
    try {
      if (provider === "tinker") {
        if (!tinkerBaseModel.trim()) { setError("Base model is required"); return; }
        const contextWindow = Number.parseInt(tinkerContextWindow, 10);
        const maxTokens = Number.parseInt(tinkerMaxTokens, 10);
        if (!Number.isFinite(contextWindow) || contextWindow <= 0) { setError("Context window must be a positive number"); return; }
        if (!Number.isFinite(maxTokens) || maxTokens <= 0) { setError("Max tokens must be a positive number"); return; }
        const tinkerSlug = deriveTinkerSlug(tinkerBaseModel);

        const tinkerResult = await window.electron.saveTinkerProvider({
          baseUrl: tinkerBaseUrl.trim() || undefined,
          apiKey: tinkerApiKey.trim() || undefined,
          model: tinkerSlug,
          baseModel: tinkerBaseModel.trim(),
          modelPath: tinkerModelPath.trim() || undefined,
          rendererName: tinkerRendererName.trim() || undefined,
          reasoning: tinkerReasoning,
          contextWindow,
          maxTokens,
        });
        if (!tinkerResult.success) { setError(tinkerResult.error || "Failed to save Tinker provider"); return; }

        const settingsResult = await window.electron.saveAgentSettings({
          defaultProvider: "tinker",
          defaultModel: tinkerSlug,
          defaultThinkingLevel: thinkingLevel,
        });
        if (!settingsResult.success) { setError(settingsResult.error || "Failed to save settings"); return; }

        const [availableModels] = await Promise.all([
          window.electron.listAvailableModels(),
          loadTinkerProvider(),
        ]);
        await syncModelState(availableModels, { provider: "tinker", model: tinkerSlug });
      } else if (provider === "openai-compatible") {
        if (!customBaseUrl.trim()) { setError("Base URL is required"); return; }
        if (!customModel.trim()) { setError("Model slug is required"); return; }

        const customResult = await window.electron.saveOpenAICompatibleProvider({
          baseUrl: customBaseUrl.trim(),
          model: customModel.trim(),
          apiFormat: customApiFormat,
          apiKey: customApiKey.trim() || undefined,
        });
        if (!customResult.success) { setError(customResult.error || "Failed to save endpoint"); return; }

        const settingsResult = await window.electron.saveAgentSettings({
          defaultProvider: "openai-compatible",
          defaultModel: customModel.trim(),
          defaultThinkingLevel: thinkingLevel,
        });
        if (!settingsResult.success) { setError(settingsResult.error || "Failed to save settings"); return; }

        const [availableModels] = await Promise.all([
          window.electron.listAvailableModels(),
          loadOpenAICompatibleProvider(),
        ]);
        await syncModelState(availableModels, { provider: "openai-compatible", model: customModel.trim() });
      } else {
        if (!provider.trim()) { setError("Provider is required"); return; }
        if (!model.trim()) { setError("Model is required"); return; }

        if (apiKey.trim()) {
          const keyResult = await window.electron.saveProviderApiKey(provider, apiKey.trim());
          if (!keyResult.success) { setError(keyResult.error || "Failed to save API key"); return; }
          await loadStatuses([...new Set(models.map((item) => item.provider))]);
          setApiKey("");
        }

        const settingsResult = await window.electron.saveAgentSettings({
          defaultProvider: provider,
          defaultModel: model,
          defaultThinkingLevel: thinkingLevel,
        });
        if (!settingsResult.success) { setError(settingsResult.error || "Failed to save settings"); return; }
      }

      setSuccess(true);
      setTimeout(() => {
        setSuccess(false);
        onClose();
      }, 1000);
    } catch (err) {
      console.error("Failed to save:", err);
      setError("Failed to save configuration");
    } finally {
      setSaving(false);
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

  const handleRemoveTinkerProvider = async () => {
    setError(null);
    setRemovingTinkerProvider(true);
    try {
      const result = await window.electron.removeTinkerProvider();
      if (!result.success) {
        setError(result.error || "Failed to remove Tinker provider");
        return;
      }

      setTinkerConfigured(false);
      setTinkerBaseUrl("");
      setTinkerBaseModel("");
      setTinkerModelPath("");
      setTinkerRendererName("");
      setTinkerReasoning(true);
      setTinkerContextWindow("128000");
      setTinkerMaxTokens("16384");
      setTinkerApiKey("");
      setTinkerBaseModelResolved(false);
      setTinkerResolveError(null);
      setTinkerResolving(false);

      const settings = await window.electron.getAgentSettings();
      const availableModels = await window.electron.listAvailableModels();
      await syncModelState(
        availableModels,
        provider === "tinker" ? undefined : { provider, model },
        settings
      );
      setSuccess(true);
      setTimeout(() => setSuccess(false), 1200);
    } catch (err) {
      console.error("Failed to remove Tinker provider:", err);
      setError("Failed to remove Tinker provider");
    } finally {
      setRemovingTinkerProvider(false);
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

  // Priority order for provider selector; unlisted providers sort alphabetically after these.
  const ALLOWED_PROVIDERS = new Set(["anthropic", "openai", "openai-compatible", "openrouter", "tinker"]);
  const PROVIDER_PRIORITY: string[] = ["anthropic", "openai", "tinker", "openai-compatible", "openrouter"];
  const PROVIDER_LABELS: Record<string, string> = {
    anthropic: "Anthropic",
    openai: "OpenAI",
    openrouter: "OpenRouter",
    "openai-compatible": "OpenAI-Compatible Endpoint",
    tinker: "Tinker",
  };

  const providerOptions = [...new Set([...models.map((item) => item.provider), "openai-compatible", "tinker"])]
    .filter((p) => ALLOWED_PROVIDERS.has(p))
    .sort((a, b) => {
      const ai = PROVIDER_PRIORITY.indexOf(a);
      const bi = PROVIDER_PRIORITY.indexOf(b);
      if (ai !== -1 && bi !== -1) return ai - bi;
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      return a.localeCompare(b);
    });
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
            Configure the default provider and model. Use Anthropic or OpenAI directly, or select &ldquo;OpenAI-Compatible Endpoint&rdquo; or &ldquo;Tinker&rdquo; to set up a custom provider.
          </p>
        </div>

        <div className="mt-5">
          <section className={cardClass}>
            <div className="flex items-start gap-4">
              <div className="min-w-0">
                <div className="text-base font-semibold text-ink-800">Default Provider</div>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  Choose the provider and model used for new Pi sessions, then keep credentials in sync for that provider.
                </p>
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
                    const providerModels = models.filter((item) => item.provider === nextProvider);
                    const nextModel = providerModels.find((item) => item.id.includes("claude-sonnet-4-6"))?.id
                      ?? providerModels[0]?.id ?? "";
                    setModel(nextModel);
                  }}
                >
                  {providerOptions.map((item) => (
                    <option key={item} value={item}>{PROVIDER_LABELS[item] ?? item}</option>
                  ))}
                </select>
              </label>
              {provider === "tinker" ? (
                <label className="grid min-w-0 gap-1.5">
                  <span className="text-xs font-medium text-muted-foreground">Thinking</span>
                  <button
                    type="button"
                    className={`${fieldClass} flex items-center justify-between cursor-pointer`}
                    onClick={() => setThinkingLevel(thinkingLevel === "off" ? "medium" : "off")}
                  >
                    <span>{thinkingLevel === "off" ? "Off" : "On"}</span>
                    <span className={`inline-block h-4 w-8 rounded-full transition-colors ${thinkingLevel === "off" ? "bg-ink-900/15" : "bg-primary"}`}>
                      <span className={`block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${thinkingLevel === "off" ? "translate-x-0" : "translate-x-4"}`} />
                    </span>
                  </button>
                </label>
              ) : provider === "openai-compatible" ? (
                <label className="grid min-w-0 gap-1.5">
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
              ) : (
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
              )}
            </div>

            {provider === "openai-compatible" ? (
              <div className="mt-4 grid gap-4">
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
                <div className="grid gap-4 md:grid-cols-2">
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
                </div>
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
                {customConfigured && (
                  <button
                    className={`${secondaryButtonClass} text-red-600 hover:text-red-700 hover:bg-red-50`}
                    onClick={handleRemoveOpenAICompatibleProvider}
                    disabled={removingCustomProvider}
                  >
                    {removingCustomProvider ? <Spinner className="mx-auto w-5 h-5" /> : "Remove Endpoint"}
                  </button>
                )}
              </div>
            ) : provider === "tinker" ? (
              <div className="mt-4 grid gap-4">
                <label className="grid gap-1.5">
                  <span className="text-xs font-medium text-muted-foreground">Tinker API Key</span>
                  <input
                    type="password"
                    className={`${fieldClass} font-mono text-[13px]`}
                    placeholder={tinkerConfigured ? "Leave blank to keep the saved key" : "tk-..."}
                    value={tinkerApiKey}
                    onChange={(e) => setTinkerApiKey(e.target.value)}
                  />
                </label>
                <label className="grid gap-1.5">
                  <span className="text-xs font-medium text-muted-foreground">Checkpoint / Model Path</span>
                  <input
                    type="text"
                    className={`${fieldClass} font-mono text-[13px]`}
                    placeholder="Paste a tinker:// checkpoint to auto-resolve the base model"
                    value={tinkerModelPath}
                    onChange={(e) => setTinkerModelPath(e.target.value)}
                  />
                </label>
                {(tinkerResolving || tinkerResolveError || tinkerBaseModelResolved) && (
                  <div className="rounded-2xl border border-ink-900/8 bg-white/80 px-4 py-3 text-xs text-muted-foreground">
                    {tinkerResolving && (
                      <div className="flex items-center gap-2">
                        <Spinner className="h-4 w-4" color="currentColor" />
                        Resolving base model from checkpoint...
                      </div>
                    )}
                    {!tinkerResolving && tinkerBaseModelResolved && (
                      <div className="space-y-1">
                        <div>
                          Resolved base model: <span className="font-medium text-ink-700">{tinkerBaseModel}</span>
                        </div>
                        {tinkerModelPath.startsWith("tinker://") && (
                          <div className="break-all font-mono text-[11px] text-ink-600">
                            Active checkpoint: {tinkerModelPath}
                          </div>
                        )}
                      </div>
                    )}
                    {!tinkerResolving && tinkerResolveError && (
                      <div className="text-error">{tinkerResolveError}</div>
                    )}
                  </div>
                )}
                <label className="grid gap-1.5">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-xs font-medium text-muted-foreground">Base Model</span>
                    {tinkerBaseModelResolved && (
                      <span className="rounded-full bg-primary/10 px-2.5 py-1 text-[10px] font-medium text-primary">
                        Auto-resolved
                      </span>
                    )}
                  </div>
                  <input
                    type="text"
                    className={`${fieldClass} font-mono text-[13px]`}
                    placeholder="Qwen/Qwen3-30B-A3B-Instruct-2507"
                    value={tinkerBaseModel}
                    onChange={(e) => {
                      setTinkerBaseModel(e.target.value);
                      setTinkerBaseModelResolved(false);
                    }}
                    readOnly={tinkerBaseModelResolved}
                  />
                </label>

                <button
                  type="button"
                  className="flex items-center gap-2 text-xs font-medium text-muted-foreground hover:text-ink-700 transition-colors py-1"
                  onClick={() => setTinkerAdvancedOpen(!tinkerAdvancedOpen)}
                >
                  <svg
                    viewBox="0 0 24 24"
                    className={`h-3.5 w-3.5 transition-transform ${tinkerAdvancedOpen ? "rotate-90" : ""}`}
                    fill="none" stroke="currentColor" strokeWidth="2"
                  >
                    <path d="M9 18l6-6-6-6" />
                  </svg>
                  Advanced Settings
                </button>

                {tinkerAdvancedOpen && (
                  <div className="grid gap-4 rounded-2xl border border-ink-900/6 bg-white/60 p-4">
                    <label className="grid gap-1.5">
                      <span className="text-xs font-medium text-muted-foreground">Tinker Base URL</span>
                      <input
                        type="text"
                        className={`${fieldClass} font-mono text-[13px]`}
                        placeholder="Optional override for ServiceClient(base_url=...)"
                        value={tinkerBaseUrl}
                        onChange={(e) => setTinkerBaseUrl(e.target.value)}
                      />
                    </label>
                    <label className="grid gap-1.5">
                      <span className="text-xs font-medium text-muted-foreground">Renderer Override</span>
                      <input
                        type="text"
                        className={`${fieldClass} font-mono text-[13px]`}
                        placeholder="Optional; defaults to the recommended renderer"
                        value={tinkerRendererName}
                        onChange={(e) => setTinkerRendererName(e.target.value)}
                      />
                    </label>
                    <div className="grid gap-4 md:grid-cols-3">
                      <label className="grid gap-1.5">
                        <span className="text-xs font-medium text-muted-foreground">Reasoning</span>
                        <select
                          className={fieldClass}
                          value={tinkerReasoning ? "on" : "off"}
                          onChange={(e) => setTinkerReasoning(e.target.value === "on")}
                        >
                          <option value="on">Enabled</option>
                          <option value="off">Disabled</option>
                        </select>
                      </label>
                      <label className="grid gap-1.5">
                        <span className="text-xs font-medium text-muted-foreground">Context Window</span>
                        <input
                          type="number"
                          min="1"
                          className={`${fieldClass} font-mono text-[13px]`}
                          value={tinkerContextWindow}
                          onChange={(e) => setTinkerContextWindow(e.target.value)}
                        />
                      </label>
                      <label className="grid gap-1.5">
                        <span className="text-xs font-medium text-muted-foreground">Max Tokens</span>
                        <input
                          type="number"
                          min="1"
                          className={`${fieldClass} font-mono text-[13px]`}
                          value={tinkerMaxTokens}
                          onChange={(e) => setTinkerMaxTokens(e.target.value)}
                        />
                      </label>
                    </div>
                  </div>
                )}

                {tinkerConfigured && (
                  <button
                    className={`${secondaryButtonClass} text-red-600 hover:text-red-700 hover:bg-red-50`}
                    onClick={handleRemoveTinkerProvider}
                    disabled={removingTinkerProvider}
                  >
                    {removingTinkerProvider ? <Spinner className="mx-auto w-5 h-5" /> : "Remove Tinker Provider"}
                  </button>
                )}
              </div>
            ) : (
              <>
                <div className="mt-4 grid gap-4 md:grid-cols-2">
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

                  {currentStatus?.supportsOAuth && (
                    <label className="grid gap-1.5">
                      <span className="text-xs font-medium text-muted-foreground">Authentication Method</span>
                      <select
                        className={fieldClass}
                        value={authMethod}
                        onChange={(e) => setAuthMethod(e.target.value as "oauth" | "api_key")}
                      >
                        <option value="oauth">OAuth (Claude Pro / Max)</option>
                        <option value="api_key">API Key</option>
                      </select>
                    </label>
                  )}
                </div>

                <div className="mt-4">
                  {currentStatus?.supportsOAuth && authMethod === "oauth" ? (
                    <div className="rounded-2xl border border-ink-900/8 bg-white/80 px-4 py-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-ink-500">Login Status</div>
                          <div className="mt-1.5 flex items-center gap-2">
                            <span className={`inline-block h-2 w-2 rounded-full ${currentStatus.hasAuth && currentStatus.authType === "oauth" ? "bg-green-500" : "bg-ink-300"}`} />
                            <span className="text-sm text-ink-700">
                              {currentStatus.hasAuth && currentStatus.authType === "oauth"
                                ? "Logged in"
                                : "Not logged in"}
                            </span>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          {currentStatus.hasAuth && currentStatus.authType === "oauth" ? (
                            <button
                              className={`${secondaryButtonClass} text-red-600 hover:text-red-700 hover:bg-red-50`}
                              onClick={() => handleAuthAction("logout")}
                              disabled={authBusy}
                            >
                              {authBusy ? <Spinner className="mx-auto w-4 h-4" /> : "Logout"}
                            </button>
                          ) : (
                            <button
                              className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-white shadow-soft hover:bg-primary-hover transition-colors disabled:cursor-not-allowed disabled:opacity-50"
                              onClick={() => handleAuthAction("login")}
                              disabled={authBusy}
                            >
                              {authBusy ? <Spinner className="mx-auto w-4 h-4" /> : "Login with OAuth"}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="grid gap-4">
                      <label className="grid gap-1.5">
                        <span className="text-xs font-medium text-muted-foreground">Provider API Key</span>
                        <div className="flex items-center gap-2">
                          <input
                            type="password"
                            className={`${fieldClass} flex-1`}
                            placeholder="sk-..."
                            value={apiKey}
                            onChange={(e) => setApiKey(e.target.value)}
                          />
                          {currentStatus?.hasAuth && currentStatus.authType === "api_key" && !apiKey && (
                            <span className="text-green-600 text-sm">✓</span>
                          )}
                        </div>
                      </label>
                      {currentStatus?.hasAuth && currentStatus.authType === "env" && (
                        <div className="text-xs text-muted-foreground">
                          Key detected from environment variable.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </>
            )}
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
            disabled={saving || !provider.trim() || (
              provider === "tinker" ? (tinkerResolving || !tinkerBaseModel.trim()) :
              provider === "openai-compatible" ? (!customBaseUrl.trim() || !customModel.trim()) :
              !model.trim()
            )}
          >
            {saving ? <Spinner className="mx-auto w-5 h-5" /> : "Save"}
          </button>
        </div>
      </div>
    </>
  );
}

/* ---------- Data / export panel ---------- */

function DataPanel() {
  const [exporting, setExporting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async () => {
    setExporting(true);
    setMessage(null);
    setError(null);
    try {
      const result = await window.electron.exportRecordingsBundle();
      if (result.success) {
        setMessage(`Saved to ${result.path}`);
      } else if (result.canceled) {
        setMessage(null);
      } else {
        setError(result.error ?? "Export failed.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground leading-relaxed">
        Package your local app data (database, memories, skills, session files) into a zip you can email or upload.
        API keys are not included.
      </p>
      <button
        type="button"
        onClick={() => void handleExport()}
        disabled={exporting}
        className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
      >
        {exporting ? (
          <span className="inline-flex items-center gap-2">
            <Spinner className="w-4 h-4" color="currentColor" />
            Preparing zip…
          </span>
        ) : (
          "Export recordings (zip)"
        )}
      </button>
      {message ? <p className="text-xs text-ink-700 break-all">{message}</p> : null}
      {error ? <p className="text-xs text-red-600">{error}</p> : null}
      <p className="text-xs text-muted-foreground leading-relaxed">
        Includes <code className="bg-ink-900/5 px-1 rounded">sessions.db</code> and related files from your app data
        folder. Choose where to save in the dialog, then send that zip back to the study team.
      </p>
    </div>
  );
}

/* ---------- Mode Panel (skills + execution) ---------- */

function ModePanel() {
  return (
    <div className="space-y-6">
      <SkillsPanel />
      <div className="border-t border-ink-900/10 pt-6">
        <WorkflowPanel />
      </div>
    </div>
  );
}

/* ---------- Skills Panel ---------- */

function SkillsPanel() {
  const activeSessionId = useAppStore((s) => s.activeSessionId);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedPath, setExpandedPath] = useState<string | null>(null);
  const [skillContent, setSkillContent] = useState<string | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [removeConfirm, setRemoveConfirm] = useState<string | null>(null);
  const [autoContextInduction, setAutoContextInduction] = useState<boolean>(readStoredAutoInduction);

  useEffect(() => {
    try {
      localStorage.setItem(AUTO_INDUCTION_KEY, autoContextInduction ? "true" : "false");
    } catch {
      /* ignore */
    }
  }, [autoContextInduction]);

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

  const setUpdateMode = (contextUpdate: boolean) => {
    setAutoContextInduction(contextUpdate);
    if (activeSessionId && typeof window !== "undefined" && window.electron?.sendClientEvent) {
      window.electron.sendClientEvent({
        type: "session.setAutoContextInduction",
        payload: { sessionId: activeSessionId, autoContextInduction: contextUpdate },
      });
    }
  };

  const updateModeRow = (contextUpdate: boolean, title: string, description: string) => (
    <label
      key={title}
      className="flex cursor-pointer gap-3 rounded-xl border border-ink-900/10 bg-surface p-4 hover:border-ink-900/20 transition-colors has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-primary/30"
    >
      <input
        type="radio"
        name="updateMode"
        className="mt-1 border-ink-900/20 text-primary focus:ring-primary/30"
        checked={autoContextInduction === contextUpdate}
        onChange={() => setUpdateMode(contextUpdate)}
      />
      <div>
        <div className="text-sm font-medium text-ink-800">{title}</div>
        <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{description}</p>
      </div>
    </label>
  );

  return (
    <>
      <div className="mb-4 grid grid-cols-2 gap-2">
        {updateModeRow(
          true,
          "Context Update",
          "A Brain click triggers memory and skill induction based on the current task session data. Double-click Brain anytime to view and edit the content."
        )}
        {updateModeRow(
          false,
          "Weight Update",
          "A Brain click triggers weight update of the agent backbone language model."
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8">
          <Spinner className="w-6 h-6 text-primary" color="currentColor" />
        </div>
      ) : (
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
      )}
    </>
  );
}
