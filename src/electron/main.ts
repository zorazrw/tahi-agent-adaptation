import { app, BrowserWindow, ipcMain, dialog, globalShortcut, Menu, shell } from "electron"
import { execSync } from "child_process";
import { readFile } from "fs/promises";
import { resolve, isAbsolute } from "path";
import * as XLSX from "xlsx";
import mammoth from "mammoth";
import { ipcMainHandle, isDev, DEV_PORT } from "./util.js";
import { getPreloadPath, getUIPath, getIconPath } from "./pathResolver.js";
import { getStaticData, pollResources, stopPolling } from "./test.js";
import { handleClientEvent, sessions, cleanupAllSessions } from "./ipc-handlers.js";
import { generateSessionTitle } from "./libs/util.js";
import { saveApiConfig } from "./libs/config-store.js";
import { getCurrentApiConfig } from "./libs/claude-settings.js";
import type { ClientEvent } from "./types.js";
import "./libs/claude-settings.js";
import { ensureAppSkillsDir, listSkills, removeAppSkill, getSkillContent, getAppSkillsDir } from "./libs/skill-store.js";

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
    cleanupAllSessions();
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

    pollResources(mainWindow);

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

    // Handle API config
    ipcMainHandle("get-api-config", () => {
        return getCurrentApiConfig();
    });

    ipcMainHandle("check-api-config", () => {
        const config = getCurrentApiConfig();
        return { hasConfig: config !== null, config };
    });

    ipcMainHandle("save-api-config", (_: any, config: any) => {
        try {
            saveApiConfig(config);
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
                const firstSheetName = workbook.SheetNames[0];
                const worksheet = firstSheetName ? workbook.Sheets[firstSheetName] : null;
                if (!worksheet) {
                    return { error: "Workbook has no sheets" };
                }
                const data = XLSX.utils.sheet_to_json<unknown[]>(worksheet, { header: 1, defval: "" });
                return { kind: "xlsx", data };
            }
            if (ext === ".docx") {
                const result = await mammoth.convertToHtml({ buffer });
                return { kind: "docx", html: result.value };
            }

            return { error: "Unsupported file type" };
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            return { error: message };
        }
    });

    ipcMainHandle("show-item-in-folder", async (_: any, filePath: string, cwd?: string | null) => {
        try {
            if (!filePath || typeof filePath !== "string") return;
            const resolved = isAbsolute(filePath) ? filePath : resolve(cwd ?? process.cwd(), filePath);
            shell.showItemInFolder(resolved);
        } catch {
            // silently ignore
        }
    });
})
