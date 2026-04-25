import { app, BrowserWindow, ipcMain, dialog, globalShortcut, Menu, shell } from "electron"
import { execSync } from "child_process";
import { readFile, writeFile, mkdir, copyFile } from "fs/promises";
import { resolve, isAbsolute, basename, join } from "path";
import { randomUUID } from "crypto";
import { homedir } from "os";
import * as XLSX from "xlsx";

import { ipcMainHandle, isDev, DEV_PORT } from "./util.js";
import { getPreloadPath, getUIPath, getIconPath } from "./pathResolver.js";
import { getStaticData, pollResources, stopPolling } from "./test.js";
import { handleClientEvent, sessions, cleanupAllSessions, recordFileEditAfterPreviewSave } from "./ipc-handlers.js";
import { generateSessionTitle } from "./libs/util.js";
import type { ClientEvent } from "./types.js";
import {
    ensureAppSkillsDir,
    listSkills,
    removeAppSkill,
    getSkillContent,
    getAppSkillsDir,
    readAllFlatSkillSections,
    writeFlatSkillSections,
    syncAppSkills,
    isValidFlatSkillMdFileName,
} from "./libs/skill-store.js";
import { ensureMemoriesDir, readAllMemorySections, writeMemorySections, getMemoriesDir } from "./libs/memory-store.js";
import {
    createPiManagers,
    ensurePiBootstrap,
    getAgentSettings,
    getOpenAICompatibleProviderConfig,
    getProviderAuthStatus,
    getTinkerProviderConfig,
    listAvailableModels,
    listOpenAICompatibleModels,
    loginProvider,
    logoutProvider,
    removeOpenAICompatibleProviderConfig,
    removeTinkerProviderConfig,
    saveAgentSettings,
    saveOpenAICompatibleProviderConfig,
    saveProviderApiKey,
    saveTinkerProviderConfig,
} from "./libs/pi-config.js";
import { resolveTinkerCheckpoint, shutdownTinkerBridge } from "./libs/tinker-provider.js";
import { startTinkerAutoUpdateWatcher, stopTinkerAutoUpdateWatcher } from "./libs/tinker-auto-update.js";
import { postSessionToTrainer } from "./libs/context-export.js";

type SaveMemoryParseResult =
    | { ok: true; sections: { fileName: string; content: string }[]; deletedFileNames: string[] | undefined }
    | { ok: false; error: string };

/**
 * Accepts current shape { sections, deletedFileNames }, legacy string (single file),
 * or older builds that expected a string and returned "Invalid content" when given an object.
 */
function parseSaveMemoryPayload(payload: unknown): SaveMemoryParseResult {
    if (payload == null) {
        return { ok: false, error: "Invalid payload" };
    }
    if (typeof payload === "string") {
        return {
            ok: true,
            sections: [{ fileName: "general.md", content: payload }],
            deletedFileNames: undefined,
        };
    }
    if (typeof payload !== "object") {
        return { ok: false, error: "Invalid payload" };
    }
    const p = payload as Record<string, unknown>;

    let deleted: string[] | undefined;
    if (Array.isArray(p.deletedFileNames)) {
        deleted = p.deletedFileNames.filter((n): n is string => typeof n === "string");
        if (deleted.length === 0) deleted = undefined;
    }

    const rawSections = p.sections;
    if (rawSections === undefined && typeof p.content === "string") {
        const name = typeof p.fileName === "string" && p.fileName.trim() ? p.fileName.trim() : "general.md";
        return { ok: true, sections: [{ fileName: name, content: p.content }], deletedFileNames: deleted };
    }
    if (!Array.isArray(rawSections)) {
        return { ok: false, error: "Invalid sections: expected an array" };
    }

    const sections: { fileName: string; content: string }[] = [];
    for (const row of rawSections) {
        if (!row || typeof row !== "object") continue;
        const r = row as Record<string, unknown>;
        const fileName = r.fileName;
        if (typeof fileName !== "string" || !fileName.trim()) continue;
        const c = r.content;
        const content = c == null ? "" : typeof c === "string" ? c : String(c);
        sections.push({ fileName: fileName.trim(), content });
    }

    if (rawSections.length > 0 && sections.length === 0) {
        return {
            ok: false,
            error: "Could not save memory: each section needs a file name and editable body.",
        };
    }

    return { ok: true, sections, deletedFileNames: deleted };
}

function parseSaveSkillPayload(payload: unknown): SaveMemoryParseResult {
    if (payload == null) {
        return { ok: false, error: "Invalid payload" };
    }
    if (typeof payload === "string") {
        return {
            ok: true,
            sections: [{ fileName: "skill.md", content: payload }],
            deletedFileNames: undefined,
        };
    }
    if (typeof payload !== "object") {
        return { ok: false, error: "Invalid payload" };
    }
    const p = payload as Record<string, unknown>;

    let deleted: string[] | undefined;
    if (Array.isArray(p.deletedFileNames)) {
        deleted = p.deletedFileNames.filter(
            (n): n is string => typeof n === "string" && isValidFlatSkillMdFileName(n)
        );
        if (deleted.length === 0) deleted = undefined;
    }

    const rawSections = p.sections;
    if (rawSections === undefined && typeof p.content === "string") {
        const name =
            typeof p.fileName === "string" && p.fileName.trim() && isValidFlatSkillMdFileName(p.fileName.trim())
                ? p.fileName.trim()
                : "skill.md";
        return { ok: true, sections: [{ fileName: name, content: p.content }], deletedFileNames: deleted };
    }
    if (!Array.isArray(rawSections)) {
        return { ok: false, error: "Invalid sections: expected an array" };
    }

    const sections: { fileName: string; content: string }[] = [];
    for (const row of rawSections) {
        if (!row || typeof row !== "object") continue;
        const r = row as Record<string, unknown>;
        const fileName = r.fileName;
        if (typeof fileName !== "string" || !isValidFlatSkillMdFileName(fileName.trim())) continue;
        const c = r.content;
        const content = c == null ? "" : typeof c === "string" ? c : String(c);
        sections.push({ fileName: fileName.trim(), content });
    }

    if (rawSections.length > 0 && sections.length === 0) {
        return {
            ok: false,
            error: "Could not save skills: each file must be a valid *.md name in the skills folder.",
        };
    }

    return { ok: true, sections, deletedFileNames: deleted };
}

let cleanupComplete = false;
let mainWindow: BrowserWindow | null = null;

function killViteDevServer(): void {
    if (!isDev()) return;
    try {
        if (process.platform === 'win32') {
            execSync(`for /f "tokens=5" %a in ('netstat -ano ^| findstr :${DEV_PORT}') do taskkill /PID %a /F`, { stdio: 'ignore', shell: 'cmd.exe' });
        } else {
            execSync(`lsof -ti:${DEV_PORT} | xargs kill -9 2>/dev/null || true`, { stdio: 'ignore' });
        }
    } catch {
        // Process may already be dead
    }
}

function cleanup(): void {
    if (cleanupComplete) return;
    cleanupComplete = true;

    globalShortcut.unregisterAll();
    stopPolling();
    stopTinkerAutoUpdateWatcher();
    cleanupAllSessions();
    shutdownTinkerBridge("app-shutdown");
    killViteDevServer();
}

function handleSignal(): void {
    cleanup();
    app.quit();
}

// Initialize everything when app is ready
app.on("ready", () => {
    Menu.setApplicationMenu(null);
    ensureAppSkillsDir();
    ensureMemoriesDir();
    ensurePiBootstrap();
    // Setup event handlers
    app.on("before-quit", cleanup);
    app.on("will-quit", cleanup);
    app.on("window-all-closed", () => {
        cleanup();
        app.quit();
    });

    process.on("SIGTERM", handleSignal);
    process.on("SIGINT", handleSignal);
    process.on("SIGHUP", handleSignal);

    // Create main window
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        minWidth: 900,
        minHeight: 600,
        webPreferences: {
            preload: getPreloadPath(),
        },
        icon: getIconPath(),
        titleBarStyle: "hiddenInset",
        backgroundColor: "#FAF9F6",
        trafficLightPosition: { x: 15, y: 18 }
    });

    if (isDev()) mainWindow.loadURL(`http://localhost:${DEV_PORT}`)
    else mainWindow.loadFile(getUIPath());

    globalShortcut.register('CommandOrControl+Q', () => {
        cleanup();
        app.quit();
    });

    // Enable DevTools via Cmd+Option+I
    globalShortcut.register('CommandOrControl+Alt+I', () => {
        mainWindow?.webContents.toggleDevTools();
    });

    pollResources(mainWindow);
    startTinkerAutoUpdateWatcher();

    ipcMainHandle("getStaticData", () => {
        return getStaticData();
    });

    // Handle client events
    ipcMain.on("client-event", (_: any, event: ClientEvent) => {
        handleClientEvent(event);
    });

    // Handle session title generation
    ipcMainHandle("generate-session-title", async (_: any, userInput: string | null) => {
        return await generateSessionTitle(userInput);
    });

    // Handle recent cwds request
    ipcMainHandle("get-recent-cwds", (_: any, limit?: number) => {
        const boundedLimit = limit ? Math.min(Math.max(limit, 1), 20) : 8;
        return sessions.listRecentCwds(boundedLimit);
    });

    // Handle directory selection
    ipcMainHandle("select-directory", async () => {
        const result = await dialog.showOpenDialog(mainWindow!, {
            properties: ['openDirectory']
        });

        if (result.canceled) {
            return null;
        }

        return result.filePaths[0];
    });

    // Create temp session directory
    ipcMainHandle("create-temp-session-dir", async () => {
        const id = randomUUID();
        const dir = join(homedir(), "agent-cowork-sessions", id);
        await mkdir(dir, { recursive: true });
        return dir;
    });

    // Copy files into a target directory
    ipcMainHandle("copy-files-to-dir", async (_: any, filePaths: string[], targetDir: string) => {
        const names: string[] = [];
        for (const src of filePaths) {
            const name = basename(src);
            await copyFile(src, join(targetDir, name));
            names.push(name);
        }
        return names;
    });

    // Select files dialog
    ipcMainHandle("select-files", async () => {
        const result = await dialog.showOpenDialog(mainWindow!, {
            properties: ['openFile', 'multiSelections']
        });
        if (result.canceled) return [];
        return result.filePaths;
    });

    ipcMainHandle("get-agent-settings", () => {
        return getAgentSettings();
    });

    ipcMainHandle("save-agent-settings", async (_: any, settings: any) => {
        try {
            await saveAgentSettings(settings);
            shutdownTinkerBridge("settings-changed");
            return { success: true };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    });

    ipcMainHandle("list-available-models", () => {
        return listAvailableModels();
    });

    ipcMainHandle("get-openai-compatible-provider", () => {
        return getOpenAICompatibleProviderConfig();
    });

    ipcMainHandle("get-tinker-provider", () => {
        return getTinkerProviderConfig();
    });

    ipcMainHandle("save-openai-compatible-provider", (_: any, config: any) => {
        try {
            saveOpenAICompatibleProviderConfig(config);
            return { success: true };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    });

    ipcMainHandle("list-openai-compatible-models", async (_: any, baseUrl: string, apiKey?: string) => {
        try {
            return await listOpenAICompatibleModels(baseUrl, apiKey);
        } catch (error) {
            return {
                ok: false,
                error: error instanceof Error ? error.message : String(error),
            };
        }
    });

    ipcMainHandle("remove-openai-compatible-provider", () => {
        try {
            removeOpenAICompatibleProviderConfig();
            return { success: true };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    });

    ipcMainHandle("save-tinker-provider", (_: any, config: any) => {
        try {
            saveTinkerProviderConfig(config);
            shutdownTinkerBridge("settings-changed");
            return { success: true };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    });

    ipcMainHandle("remove-tinker-provider", () => {
        try {
            removeTinkerProviderConfig();
            shutdownTinkerBridge("settings-changed");
            return { success: true };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    });

    ipcMainHandle("resolve-tinker-checkpoint", async (_: any, tinkerPath: string, apiKey?: string, baseUrl?: string) => {
        try {
            const { authStorage } = createPiManagers(process.cwd());
            const savedCredential = authStorage.get("tinker");
            const effectiveApiKey =
                typeof apiKey === "string" && apiKey.trim()
                    ? apiKey.trim()
                    : savedCredential?.type === "api_key"
                        ? savedCredential.key
                        : undefined;
            return await resolveTinkerCheckpoint(tinkerPath, effectiveApiKey, baseUrl);
        } catch (error) {
            return {
                ok: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    });

    ipcMainHandle("get-provider-auth-status", (_: any, provider: string) => {
        return getProviderAuthStatus(provider);
    });

    ipcMainHandle("save-provider-api-key", (_: any, provider: string, apiKey: string) => {
        try {
            saveProviderApiKey(provider, apiKey);
            return { success: true };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    });

    ipcMainHandle("login-provider", async (_: any, provider: string) => {
        try {
            await loginProvider(provider);
            return { success: true };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    });

    ipcMainHandle("logout-provider", (_: any, provider: string) => {
        try {
            logoutProvider(provider);
            return { success: true };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error)
            };
        }
    });

    // Skills management
    ipcMainHandle("list-skills", () => {
        return listSkills();
    });

    ipcMainHandle("remove-skill", (_: any, dirName: string) => {
        return removeAppSkill(dirName);
    });

    ipcMainHandle("get-skill-content", (_: any, skillPath: string) => {
        return getSkillContent(skillPath);
    });

    ipcMainHandle("get-skills-dir", () => {
        const dir = getAppSkillsDir();
        shell.openPath(dir);
        return dir;
    });

    ipcMainHandle("get-memory-md", () => {
        const dir = getMemoriesDir();
        const sections = readAllMemorySections();
        const skillsDir = getAppSkillsDir();
        const skillSections = readAllFlatSkillSections();
        return { dir, sections, skillsDir, skillSections };
    });

    ipcMainHandle("save-memory-md", (_: any, payload: unknown) => {
        try {
            const parsed = parseSaveMemoryPayload(payload);
            if (!parsed.ok) {
                return { success: false, error: parsed.error };
            }
            writeMemorySections(parsed.sections, parsed.deletedFileNames);
            return { success: true };
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            return { success: false, error: message };
        }
    });

    ipcMainHandle("save-skill-md", (_: any, payload: unknown) => {
        try {
            const parsed = parseSaveSkillPayload(payload);
            if (!parsed.ok) {
                return { success: false, error: parsed.error };
            }
            writeFlatSkillSections(parsed.sections, parsed.deletedFileNames);
            syncAppSkills();
            return { success: true };
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            return { success: false, error: message };
        }
    });

    const IMAGE_MIME: Record<string, string> = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp", ".svg": "image/svg+xml",
    };
    const VIDEO_MIME: Record<string, string> = { ".mp4": "video/mp4", ".webm": "video/webm" };
    const AUDIO_MIME: Record<string, string> = { ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg" };
    const CODE_LANG: Record<string, string> = {
        ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".py": "python", ".rb": "ruby", ".rs": "rust", ".go": "go",
        ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".cs": "csharp",
        ".css": "css", ".scss": "scss", ".less": "less",
        ".php": "php", ".swift": "swift", ".kt": "kotlin",
        ".sh": "bash", ".bash": "bash", ".zsh": "bash",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "ini", ".xml": "xml",
        ".sql": "sql", ".r": "r", ".lua": "lua", ".dart": "dart", ".scala": "scala",
        ".ex": "elixir", ".exs": "elixir", ".hs": "haskell", ".ml": "ocaml",
    };
    const TEXT_KINDS = new Set([".txt", ".md", ".csv", ".tsv", ".json", ".html", ".htm"]);
    const ALLOWED_EXTS = new Set([
        ...Object.keys(IMAGE_MIME), ...Object.keys(VIDEO_MIME), ...Object.keys(AUDIO_MIME),
        ...Object.keys(CODE_LANG), ...TEXT_KINDS,
        ".xlsx", ".xls", ".docx", ".pdf",
    ]);

    ipcMainHandle("preview-file", async (_: any, filePath: string, cwd?: string | null) => {
        try {
            if (!filePath || typeof filePath !== "string") {
                return { error: "Invalid file path" };
            }
            const ext = filePath.toLowerCase().slice(filePath.lastIndexOf("."));
            if (!ALLOWED_EXTS.has(ext)) {
                return { error: `File type "${ext}" is not supported for preview` };
            }
            const base = (cwd && typeof cwd === "string") ? cwd : process.cwd();
            const resolved = isAbsolute(filePath) ? filePath : resolve(base, filePath);

            // --- Text-based formats (read as UTF-8) ---
            if (ext === ".txt") {
                return { kind: "txt", content: await readFile(resolved, "utf8") };
            }
            if (ext === ".md") {
                return { kind: "md", content: await readFile(resolved, "utf8") };
            }
            if (ext === ".csv" || ext === ".tsv") {
                return { kind: "csv", content: await readFile(resolved, "utf8") };
            }
            if (ext === ".json") {
                return { kind: "json", content: await readFile(resolved, "utf8") };
            }
            if (ext === ".html" || ext === ".htm") {
                return { kind: "html", content: await readFile(resolved, "utf8") };
            }
            if (CODE_LANG[ext]) {
                return { kind: "code", content: await readFile(resolved, "utf8"), language: CODE_LANG[ext] };
            }

            // --- Binary formats (read as buffer) ---
            const buffer = await readFile(resolved);

            if (IMAGE_MIME[ext]) {
                const dataUrl = `data:${IMAGE_MIME[ext]};base64,${buffer.toString("base64")}`;
                return { kind: "image", dataUrl };
            }
            if (VIDEO_MIME[ext]) {
                const dataUrl = `data:${VIDEO_MIME[ext]};base64,${buffer.toString("base64")}`;
                return { kind: "video", dataUrl };
            }
            if (AUDIO_MIME[ext]) {
                const dataUrl = `data:${AUDIO_MIME[ext]};base64,${buffer.toString("base64")}`;
                return { kind: "audio", dataUrl };
            }
            if (ext === ".pdf") {
                return { kind: "pdf", data: buffer.toString("base64") };
            }
            if (ext === ".xlsx" || ext === ".xls") {
                const workbook = XLSX.read(buffer, { type: "buffer" });
                const sheets = workbook.SheetNames.map((name) => ({
                    name,
                    html: XLSX.utils.sheet_to_html(workbook.Sheets[name]!),
                }));
                return { kind: "xlsx", sheets };
            }
            if (ext === ".docx") {
                return { kind: "docx", data: buffer.toString("base64") };
            }

            return { error: "Unsupported file type" };
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            return { error: message };
        }
    });

    ipcMainHandle(
        "write-file",
        async (
            _: any,
            filePath: string,
            cwd?: string | null,
            content?: string,
            sessionId?: string | null
        ) => {
            try {
                if (!filePath || typeof filePath !== "string") {
                    return { success: false, error: "Invalid file path" };
                }
                if (typeof content !== "string") {
                    return { success: false, error: "Invalid content" };
                }
                const base = cwd && typeof cwd === "string" ? cwd : process.cwd();
                const resolved = isAbsolute(filePath) ? filePath : resolve(base, filePath);
                await writeFile(resolved, content, "utf8");
                const sid = typeof sessionId === "string" ? sessionId.trim() : "";
                if (sid) {
                    /** Always record resolved path so DB/export matches workflow ``outputFiles`` (often absolute). */
                    const pathForRecord = resolved.replace(/\\/g, "/");
                    recordFileEditAfterPreviewSave(sid, pathForRecord, content);
                }
                return { success: true };
            } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                return { success: false, error: message };
            }
        }
    );

    ipcMainHandle("show-item-in-folder", async (_: any, filePath: string, cwd?: string | null) => {
        try {
            if (!filePath || typeof filePath !== "string") return;
            const resolved = isAbsolute(filePath) ? filePath : resolve(cwd ?? process.cwd(), filePath);
            shell.showItemInFolder(resolved);
        } catch {
            // silently ignore
        }
    });

    ipcMainHandle("post-session-to-trainer", async (_: any, sessionId: string) => {
        if (!sessionId || typeof sessionId !== "string") {
            return { success: false, error: "sessionId required" };
        }
        const result = await postSessionToTrainer(sessionId);
        return result.ok
            ? { success: true }
            : { success: false, error: result.error };
    });
})
