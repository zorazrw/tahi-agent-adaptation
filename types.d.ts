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
    }
}
