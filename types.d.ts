type Statistics = {
    cpuUsage: number;
    ramUsage: number;
    storageData: number;
}

type StaticData = {
    totalStorage: number;
    cpuModel: string;
    totalMemoryGB: number;
}

type UnsubscribeFunction = () => void;

type PreviewFileResult =
    | { kind: "txt"; content: string }
    | { kind: "xlsx"; sheets: { name: string; html: string }[] }
    | { kind: "docx"; data: string }
    | { kind: "image"; dataUrl: string }
    | { kind: "pdf"; data: string }
    | { kind: "md"; content: string }
    | { kind: "code"; content: string; language: string }
    | { kind: "csv"; content: string }
    | { kind: "json"; content: string }
    | { kind: "html"; content: string }
    | { kind: "video"; dataUrl: string }
    | { kind: "audio"; dataUrl: string }
    | { error: string };

type SkillInfo = {
    name: string;
    description: string;
    dirName: string;
    source: "app" | "user";
    path: string;
    /** True when path is a top-level *.md file in the app skills folder (not a SKILL.md directory). */
    isFlatMd?: boolean;
}

type MemorySectionDto = {
    fileName: string;
    title: string;
    content: string;
}

type MemorySavePayload = {
    sections: { fileName: string; content: string }[];
    deletedFileNames?: string[];
}

type AgentSettings = {
    defaultProvider?: string;
    defaultModel?: string;
    defaultThinkingLevel?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
}

type OpenAICompatibleApiFormat = "openai-completions" | "openai-responses";

type OpenAICompatibleProviderConfig = {
    provider: "openai-compatible";
    baseUrl: string;
    model: string;
    apiFormat: OpenAICompatibleApiFormat;
    hasApiKey: boolean;
}

type OpenAICompatibleProviderInput = {
    baseUrl: string;
    model: string;
    apiFormat: OpenAICompatibleApiFormat;
    apiKey?: string;
}

type TinkerModelConfig = {
    id: string;
    baseModel: string;
    modelPath?: string;
    rendererName?: string;
    reasoning: boolean;
    contextWindow: number;
    maxTokens: number;
}

type TinkerProviderConfig = {
    provider: "tinker";
    baseUrl?: string;
    hasApiKey: boolean;
    model: TinkerModelConfig;
}

type TinkerProviderInput = {
    baseUrl?: string;
    apiKey?: string;
    model: string;
    baseModel: string;
    modelPath?: string;
    rendererName?: string;
    reasoning?: boolean;
    contextWindow?: number;
    maxTokens?: number;
}

type AvailableModel = {
    provider: string;
    id: string;
    label: string;
    reasoning: boolean;
}

type ProviderAuthStatus = {
    provider: string;
    hasAuth: boolean;
    authType?: "api_key" | "oauth" | "env";
    supportsOAuth: boolean;
    oauthName?: string;
}

type ResolveTinkerCheckpointResult =
    | { ok: true; base_model: string }
    | { ok: false; error: string };

type EventPayloadMapping = {
    statistics: Statistics;
    getStaticData: StaticData;
    "generate-session-title": string;
    "get-recent-cwds": string[];
    "select-directory": string | null;
    "get-agent-settings": AgentSettings;
    "save-agent-settings": { success: boolean; error?: string };
    "list-available-models": AvailableModel[];
    "get-openai-compatible-provider": OpenAICompatibleProviderConfig | null;
    "get-tinker-provider": TinkerProviderConfig | null;
    "save-openai-compatible-provider": { success: boolean; error?: string };
    "save-tinker-provider": { success: boolean; error?: string };
    "remove-openai-compatible-provider": { success: boolean; error?: string };
    "remove-tinker-provider": { success: boolean; error?: string };
    "resolve-tinker-checkpoint": ResolveTinkerCheckpointResult;
    "get-provider-auth-status": ProviderAuthStatus;
    "save-provider-api-key": { success: boolean; error?: string };
    "login-provider": { success: boolean; error?: string };
    "logout-provider": { success: boolean; error?: string };
    "preview-file": PreviewFileResult;
    "write-file": { success: boolean; error?: string };
    "list-skills": SkillInfo[];
    "remove-skill": { success: boolean; error?: string };
    "get-skill-content": { content: string } | { error: string };
    "get-skills-dir": string;
    "show-item-in-folder": void;
    "create-temp-session-dir": string;
    "copy-files-to-dir": string[];
    "select-files": string[];
    "get-memory-md": { dir: string; sections: MemorySectionDto[]; skillsDir: string; skillSections: MemorySectionDto[] };
    "save-memory-md": { success: boolean; error?: string };
    "save-skill-md": { success: boolean; error?: string };
    "export-recordings-bundle":
        | { success: true; path: string }
        | { success: false; canceled?: boolean; error?: string };
}

interface Window {
    electron: {
        subscribeStatistics: (callback: (statistics: Statistics) => void) => UnsubscribeFunction;
        getStaticData: () => Promise<StaticData>;
        // Claude Agent IPC APIs
        sendClientEvent: (event: any) => void;
        onServerEvent: (callback: (event: any) => void) => UnsubscribeFunction;
        generateSessionTitle: (userInput: string | null) => Promise<string>;
        getRecentCwds: (limit?: number) => Promise<string[]>;
        selectDirectory: () => Promise<string | null>;
        getAgentSettings: () => Promise<AgentSettings>;
        saveAgentSettings: (settings: AgentSettings) => Promise<{ success: boolean; error?: string }>;
        listAvailableModels: () => Promise<AvailableModel[]>;
        getOpenAICompatibleProvider: () => Promise<OpenAICompatibleProviderConfig | null>;
        getTinkerProvider: () => Promise<TinkerProviderConfig | null>;
        saveOpenAICompatibleProvider: (config: OpenAICompatibleProviderInput) => Promise<{ success: boolean; error?: string }>;
        saveTinkerProvider: (config: TinkerProviderInput) => Promise<{ success: boolean; error?: string }>;
        removeOpenAICompatibleProvider: () => Promise<{ success: boolean; error?: string }>;
        removeTinkerProvider: () => Promise<{ success: boolean; error?: string }>;
        resolveTinkerCheckpoint: (tinkerPath: string, apiKey?: string, baseUrl?: string) => Promise<ResolveTinkerCheckpointResult>;
        getProviderAuthStatus: (provider: string) => Promise<ProviderAuthStatus>;
        saveProviderApiKey: (provider: string, apiKey: string) => Promise<{ success: boolean; error?: string }>;
        loginProvider: (provider: string) => Promise<{ success: boolean; error?: string }>;
        logoutProvider: (provider: string) => Promise<{ success: boolean; error?: string }>;
        previewFile: (filePath: string, cwd?: string | null) => Promise<PreviewFileResult>;
        writeFile: (
            filePath: string,
            cwd?: string | null,
            content?: string,
            sessionId?: string | null
        ) => Promise<{ success: boolean; error?: string }>;
        listSkills: () => Promise<SkillInfo[]>;
        removeSkill: (dirName: string) => Promise<{ success: boolean; error?: string }>;
        getSkillContent: (path: string) => Promise<{ content: string } | { error: string }>;
        getSkillsDir: () => Promise<string>;
        showItemInFolder: (filePath: string, cwd?: string | null) => Promise<void>;
        createTempSessionDir: () => Promise<string>;
        copyFilesToDir: (filePaths: string[], targetDir: string) => Promise<string[]>;
        selectFiles: () => Promise<string[]>;
        getPathForFile: (file: File) => string;
        getMemoryMd: () => Promise<{ dir: string; sections: MemorySectionDto[]; skillsDir: string; skillSections: MemorySectionDto[] }>;
        saveMemoryMd: (payload: MemorySavePayload) => Promise<{ success: boolean; error?: string }>;
        saveSkillMd: (payload: MemorySavePayload) => Promise<{ success: boolean; error?: string }>;
        exportRecordingsBundle: () => Promise<
            | { success: true; path: string }
            | { success: false; canceled?: boolean; error?: string }
        >;
    }
}
