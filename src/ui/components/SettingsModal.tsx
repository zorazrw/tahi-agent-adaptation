import { useEffect, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { Spinner } from "./Spinner";

interface SettingsModalProps {
  onClose: () => void;
}

export function SettingsModal({ onClose }: SettingsModalProps) {
  const [apiKey, setApiKey] = useState("");
  const [baseURL, setBaseURL] = useState("");
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    // Load current configuration
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
    // Validate input
    if (!apiKey.trim()) {
      setError("API Key is required");
      return;
    }
    if (!baseURL.trim()) {
      setError("Base URL is required");
      return;
    }
    if (!model.trim()) {
      setError("Model is required");
      return;
    }

    // Validate URL format
    try {
      new URL(baseURL);
    } catch {
      setError("Invalid Base URL format");
      return;
    }

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

  return (
    <Dialog.Root open onOpenChange={(open) => { if (!open) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-ink-900/20 backdrop-blur-sm animate-fade-in" />
        <Dialog.Content className="fixed inset-0 z-50 flex items-center justify-center px-4 py-8">
          <div className="w-full max-w-lg rounded-2xl border border-ink-900/5 bg-surface p-6 shadow-elevated animate-scale-in">
            <div className="flex items-center justify-between">
              <Dialog.Title className="text-base font-semibold text-ink-800">API Configuration</Dialog.Title>
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
            <p className="mt-2 text-sm text-muted-foreground">Supports Anthropic's official API as well as third-party APIs compatible with the Anthropic format.</p>

            {loading ? (
              <div className="mt-5 flex items-center justify-center py-8">
                <Spinner className="w-6 h-6 text-primary" color="currentColor" />
              </div>
            ) : (
              <div className="mt-5 grid gap-4">
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
                    {saving ? (
                      <Spinner className="mx-auto w-5 h-5" />
                    ) : "Save"}
                  </button>
                </div>
              </div>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
