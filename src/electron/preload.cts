import electron from "electron";

electron.contextBridge.exposeInMainWorld("electron", {
    subscribeStatistics: (callback) =>
        ipcOn("statistics", stats => {
            callback(stats);
        }),
    getStaticData: () => ipcInvoke("getStaticData"),
    
    // Claude Agent IPC APIs
    sendClientEvent: (event: any) => {
        electron.ipcRenderer.send("client-event", event);
    },
    onServerEvent: (callback: (event: any) => void) => {
        const cb = (_: Electron.IpcRendererEvent, payload: string) => {
            try {
                const event = JSON.parse(payload);
                callback(event);
            } catch (error) {
                console.error("Failed to parse server event:", error);
            }
        };
        electron.ipcRenderer.on("server-event", cb);
        return () => electron.ipcRenderer.off("server-event", cb);
    },
    generateSessionTitle: (userInput: string | null) => 
        ipcInvoke("generate-session-title", userInput),
    getRecentCwds: (limit?: number) => 
        ipcInvoke("get-recent-cwds", limit),
    selectDirectory: () => 
        ipcInvoke("select-directory"),
    getAgentSettings: () =>
        ipcInvoke("get-agent-settings"),
    saveAgentSettings: (settings: any) =>
        ipcInvoke("save-agent-settings", settings),
    listAvailableModels: () =>
        ipcInvoke("list-available-models"),
    getOpenAICompatibleProvider: () =>
        ipcInvoke("get-openai-compatible-provider"),
    getTinkerProvider: () =>
        ipcInvoke("get-tinker-provider"),
    saveOpenAICompatibleProvider: (config: any) =>
        ipcInvoke("save-openai-compatible-provider", config),
    saveTinkerProvider: (config: any) =>
        ipcInvoke("save-tinker-provider", config),
    removeOpenAICompatibleProvider: () =>
        ipcInvoke("remove-openai-compatible-provider"),
    listOpenAICompatibleModels: (baseUrl: string, apiKey?: string) =>
        ipcInvoke("list-openai-compatible-models", baseUrl, apiKey),
    removeTinkerProvider: () =>
        ipcInvoke("remove-tinker-provider"),
    resolveTinkerCheckpoint: (tinkerPath: string, apiKey?: string, baseUrl?: string) =>
        ipcInvoke("resolve-tinker-checkpoint", tinkerPath, apiKey, baseUrl),
    getProviderAuthStatus: (provider: string) =>
        ipcInvoke("get-provider-auth-status", provider),
    saveProviderApiKey: (provider: string, apiKey: string) =>
        ipcInvoke("save-provider-api-key", provider, apiKey),
    loginProvider: (provider: string) =>
        ipcInvoke("login-provider", provider),
    logoutProvider: (provider: string) =>
        ipcInvoke("logout-provider", provider),
    previewFile: (filePath: string, cwd?: string | null) =>
        ipcInvoke("preview-file", filePath, cwd ?? undefined),
    writeFile: (filePath: string, cwd?: string | null, content?: string, sessionId?: string | null) =>
        ipcInvoke("write-file", filePath, cwd ?? undefined, content ?? "", sessionId ?? undefined),
    listSkills: () =>
        ipcInvoke("list-skills"),
    removeSkill: (dirName: string) =>
        ipcInvoke("remove-skill", dirName),
    getSkillContent: (path: string) =>
        ipcInvoke("get-skill-content", path),
    getSkillsDir: () =>
        ipcInvoke("get-skills-dir"),
    showItemInFolder: (filePath: string, cwd?: string | null) =>
        ipcInvoke("show-item-in-folder", filePath, cwd ?? undefined),
    createTempSessionDir: () =>
        ipcInvoke("create-temp-session-dir"),
    copyFilesToDir: (filePaths: string[], targetDir: string) =>
        ipcInvoke("copy-files-to-dir", filePaths, targetDir),
    selectFiles: () =>
        ipcInvoke("select-files"),
    getPathForFile: (file: File) =>
        electron.webUtils.getPathForFile(file),
    getMemoryMd: () =>
        ipcInvoke("get-memory-md"),
    saveMemoryMd: (payload: { sections: { fileName: string; content: string }[]; deletedFileNames?: string[] }) =>
        ipcInvoke("save-memory-md", payload),
    saveSkillMd: (payload: { sections: { fileName: string; content: string }[]; deletedFileNames?: string[] }) =>
        ipcInvoke("save-skill-md", payload),
    postSessionToTrainer: (sessionId: string) =>
        ipcInvoke("post-session-to-trainer", sessionId),
    onTinkerModelUpdated: (callback: (event: TinkerModelUpdateEvent) => void) => {
        const cb = (_: Electron.IpcRendererEvent, payload: TinkerModelUpdateEvent) => {
            try {
                callback(payload);
            } catch (error) {
                console.error("Failed to handle tinker model update:", error);
            }
        };
        electron.ipcRenderer.on("tinker-model-updated", cb);
        return () => electron.ipcRenderer.off("tinker-model-updated", cb);
    },
} satisfies Window['electron'])

// Intercept drop events in the preload context where webUtils has direct
// access to native File handles (avoids contextBridge serialization issues).
document.addEventListener("drop", (e) => {
    if (!e.dataTransfer?.files?.length) return;
    const paths: string[] = [];
    for (const file of Array.from(e.dataTransfer.files)) {
        try {
            const p = electron.webUtils.getPathForFile(file);
            if (p) paths.push(p);
        } catch { /* skip */ }
    }
    if (paths.length > 0) {
        window.dispatchEvent(new CustomEvent("electron-drop-paths", { detail: paths }));
    }
}, true); // capture phase so it fires before renderer handlers

function ipcInvoke<Key extends keyof EventPayloadMapping>(key: Key, ...args: any[]): Promise<EventPayloadMapping[Key]> {
    return electron.ipcRenderer.invoke(key, ...args);
}

function ipcOn<Key extends keyof EventPayloadMapping>(key: Key, callback: (payload: EventPayloadMapping[Key]) => void) {
    const cb = (_: Electron.IpcRendererEvent, payload: any) => callback(payload)
    electron.ipcRenderer.on(key, cb);
    return () => electron.ipcRenderer.off(key, cb)
}
