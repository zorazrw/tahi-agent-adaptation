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
    | { kind: "xlsx"; data: unknown[][] }
    | { kind: "docx"; html: string }
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
}

type EventPayloadMapping = {
    statistics: Statistics;
    getStaticData: StaticData;
    "generate-session-title": string;
    "get-recent-cwds": string[];
    "select-directory": string | null;
    "get-api-config": { apiKey: string; baseURL: string; model: string; apiType?: "anthropic" } | null;
    "save-api-config": { success: boolean; error?: string };
    "check-api-config": { hasConfig: boolean; config: { apiKey: string; baseURL: string; model: string; apiType?: "anthropic" } | null };
    "preview-file": PreviewFileResult;
    "list-skills": SkillInfo[];
    "remove-skill": { success: boolean; error?: string };
    "get-skill-content": { content: string } | { error: string };
    "get-skills-dir": string;
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
        getApiConfig: () => Promise<{ apiKey: string; baseURL: string; model: string; apiType?: "anthropic" } | null>;
        saveApiConfig: (config: { apiKey: string; baseURL: string; model: string; apiType?: "anthropic" }) => Promise<{ success: boolean; error?: string }>;
        checkApiConfig: () => Promise<{ hasConfig: boolean; config: { apiKey: string; baseURL: string; model: string; apiType?: "anthropic" } | null }>;
        previewFile: (filePath: string, cwd?: string | null) => Promise<PreviewFileResult>;
        listSkills: () => Promise<SkillInfo[]>;
        removeSkill: (dirName: string) => Promise<{ success: boolean; error?: string }>;
        getSkillContent: (path: string) => Promise<{ content: string } | { error: string }>;
        getSkillsDir: () => Promise<string>;
    }
}
