import { app } from "electron";
import { spawn } from "child_process";
import { appendFileSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, writeFileSync } from "fs";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";
import { config as loadDotenv } from "dotenv";

import type { ServerEvent, StreamMessage, WorkflowNode } from "./types.js";
import { runClaude, buildPromptForNode } from "./libs/runner.js";
import { SessionStore, type Session } from "./libs/session-store.js";
import { buildExportEnvironmentSnapshot } from "./libs/message-state-snapshot.js";
import { saveAgentSettings, saveTinkerProviderConfig } from "./libs/pi-config.js";
import { ensureTinkerBridgeWarm } from "./libs/tinker-provider.js";
import {
  completeNodeAndDescendants,
  findNodeById,
  findParentNode,
  getNodePath,
} from "./libs/workflow-tree-utils.js";

type TaskSpec = {
  id: number | string;
  type?: string;
  instruction: string;
  human_output?: string | null;
};

type Args = {
  tasks: string;
  limit: number;
  taskIds?: string[];
  workplaceTemplate: string;
  modelPath: string;
  baseModel: string;
  rendererName?: string;
  out: string;
  eval: boolean;
  evalBackend?: string;
  evalModel?: string;
  evalBaseUrl?: string;
  evalApiKey?: string;
  evalRequestTimeout?: number;
  evalMaxRetries?: number;
  verifiersJson?: string;
  force: boolean;
  resume: boolean;
  python: string;
  tinkerBaseUrl?: string;
  tinkerApiKey?: string;
};

type RunOutcome = {
  status: "idle" | "completed" | "error";
  error?: string;
};

type TaskSummary = {
  task_id: number | string;
  instruction: string;
  session_id?: string;
  status: "completed" | "error" | "skipped";
  workdir: string;
  session_json: string;
  ratings_json?: string;
  score?: number | null;
  output_files: string[];
  workflow_nodes: Array<{ id: string; description: string; status: string; outputFiles: string[] }>;
  error?: string;
  eval_error?: string;
};

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const PI_AGENT_DIR_NAME = "pi-agent";
const TINKER_CONFIG_FILE = "tinker-provider.json";
const AUTH_FILE = "auth.json";
const HEADLESS_EXECUTION_NOTE =
  "Headless execution only: save outputs to files in the working directory. Do not open GUI windows, interactive plot viewers, browser tabs, or commands that wait for manual closing.";

function defaultPythonInterpreter(): string {
  const unixVenv = join(repoRoot, ".venv", "bin", "python");
  if (existsSync(unixVenv)) return unixVenv;
  const winVenv = join(repoRoot, ".venv", "Scripts", "python.exe");
  if (existsSync(winVenv)) return winVenv;
  return "python3";
}

function parseArgs(argv: string[]): Args {
  const out: Partial<Args> = { limit: 18, eval: false, force: false, resume: false, python: defaultPythonInterpreter() };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      const v = argv[++i];
      if (!v) throw new Error(`Missing value for ${a}`);
      return v;
    };
    if (a === "--tasks") out.tasks = next();
    else if (a === "--limit") out.limit = Number(next());
    else if (a === "--task-ids") out.taskIds = next().split(",").map((part) => part.trim()).filter(Boolean);
    else if (a === "--workplace-template") out.workplaceTemplate = next();
    else if (a === "--model-path") out.modelPath = next();
    else if (a === "--base-model") out.baseModel = next();
    else if (a === "--renderer-name") out.rendererName = next();
    else if (a === "--out") out.out = next();
    else if (a === "--eval") out.eval = true;
    else if (a === "--no-eval") out.eval = false;
    else if (a === "--eval-backend") out.evalBackend = next();
    else if (a === "--eval-model") out.evalModel = next();
    else if (a === "--eval-base-url") out.evalBaseUrl = next();
    else if (a === "--eval-api-key") out.evalApiKey = next();
    else if (a === "--eval-request-timeout") out.evalRequestTimeout = Number(next());
    else if (a === "--eval-max-retries") out.evalMaxRetries = Number(next());
    else if (a === "--verifiers-json" || a === "--verifier-json") out.verifiersJson = next();
    else if (a === "--force") out.force = true;
    else if (a === "--resume") out.resume = true;
    else if (a === "--python") out.python = next();
    else if (a === "--tinker-base-url") out.tinkerBaseUrl = next();
    else if (a === "--tinker-api-key") out.tinkerApiKey = next();
    else if (a === "--help" || a === "-h") {
      printHelp();
      process.exit(0);
    }
  }
  for (const key of ["tasks", "workplaceTemplate", "modelPath", "baseModel", "out"] as const) {
    if (!out[key]) throw new Error(`Missing required --${key.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`)}`);
  }
  const limit = out.limit ?? 18;
  if (!Number.isFinite(limit) || limit <= 0) throw new Error("--limit must be a positive number");
  out.limit = limit;
  if (out.evalBackend && !["anthropic", "openai"].includes(out.evalBackend)) {
    throw new Error("--eval-backend must be one of: anthropic, openai");
  }
  if (out.evalRequestTimeout !== undefined && (!Number.isFinite(out.evalRequestTimeout) || out.evalRequestTimeout <= 0)) {
    throw new Error("--eval-request-timeout must be a positive number");
  }
  if (out.evalMaxRetries !== undefined && (!Number.isFinite(out.evalMaxRetries) || out.evalMaxRetries < 0)) {
    throw new Error("--eval-max-retries must be zero or a positive number");
  }
  if (out.force && out.resume) throw new Error("--force and --resume are mutually exclusive");
  return out as Args;
}

function printHelp(): void {
  console.log(`Usage:
  bun run headless:tasks -- --tasks tasks.json --limit 18 \\
    --workplace-template trash/workplace-set/<DIR> \\
    --model-path tinker://... --base-model Qwen/Qwen3.5-35B-A3B \\
    --renderer-name qwen3_5 --out runs/headless_v2_eval [--eval] [--force|--resume]

Simple run example:
  bun run headless:tasks -- --tasks tasks.json --limit 18 \\
    --workplace-template trash/workplace-set/test-0527/dpo \\
    --model-path tinker://... --base-model Qwen/Qwen3.5-35B-A3B \\
    --renderer-name qwen3_5 --out runs/headless_dpo_eval --resume

Run only specific task ids:
  bun run headless:tasks -- --tasks tasks.json --limit 18 --task-ids 7,13 \\
    --workplace-template trash/workplace-set/test-0527/dpo \\
    --model-path tinker://... --base-model Qwen/Qwen3.5-35B-A3B \\
    --renderer-name qwen3_5 --out runs/headless_dpo_eval --force --eval

Run with eval:
  bun run headless:tasks -- --tasks tasks.json --limit 18 \\
    --workplace-template trash/workplace-set/test-0527/dpo \\
    --model-path tinker://... --base-model Qwen/Qwen3.5-35B-A3B \\
    --renderer-name qwen3_5 --out runs/headless_dpo_eval \\
    --resume --eval --eval-backend openai --eval-model gpt-4.1-mini

Optional verifier override:
  --verifiers-json scripts/verifiers.json

Eval defaults to --eval-backend openai --eval-model gpt-4.1-mini via scripts/.env or ./.env.
Verifier source defaults to scripts/verifiers.json.
 `);
}

function piAgentDir(userDataDir: string): string {
  return join(userDataDir, PI_AGENT_DIR_NAME);
}

function tinkerConfigPath(userDataDir: string): string {
  return join(piAgentDir(userDataDir), TINKER_CONFIG_FILE);
}

function authPath(userDataDir: string): string {
  return join(piAgentDir(userDataDir), AUTH_FILE);
}

function addHeadlessExecutionNote(prompt: string): string {
  const trimmed = prompt.trim();
  if (!trimmed) return HEADLESS_EXECUTION_NOTE;
  return `${trimmed}\n\n${HEADLESS_EXECUTION_NOTE}`;
}

function flattenNodes(tree: WorkflowNode[]): WorkflowNode[] {
  const out: WorkflowNode[] = [];
  for (const node of tree) {
    out.push(node);
    if (node.children?.length) out.push(...flattenNodes(node.children));
  }
  return out;
}

function findNextRunnableNodeId(tree: WorkflowNode[], verificationDepth: number): string | null {
  for (const node of tree) {
    if (node.status === "completed") continue;
    if (node.children.length > 0 && node.depth < verificationDepth) {
      const nested = findNextRunnableNodeId(node.children, verificationDepth);
      if (nested) return nested;
      continue;
    }
    return node.id;
  }
  return null;
}

class HeadlessSessionRuntime {
  private currentNodeId: string | null = null;
  private waiter: ((outcome: RunOutcome) => void) | null = null;
  private lastError: string | undefined;

  constructor(
    private readonly store: SessionStore,
    private readonly eventsPath: string,
    private readonly bashEnv: Record<string, string>,
  ) {}

  emit = (event: ServerEvent): void => {
    appendFileSync(this.eventsPath, JSON.stringify(event) + "\n", "utf8");

    if (event.type === "permission.request") {
      const session = this.store.getSession(event.payload.sessionId);
      const pending = session?.pendingPermissions.get(event.payload.toolUseId);
      pending?.resolve({ behavior: "deny", message: "Headless mode does not provide human intervention." });
      return;
    }

    if (event.type === "runner.error") {
      this.lastError = event.payload.message;
      this.finish({ status: "error", error: event.payload.message });
      return;
    }

    if (event.type === "workflow.plan") {
      const { sessionId, workflowTree } = event.payload;
      this.store.updateSession(sessionId, {
        workflowTree,
        verificationDepth: 0,
        status: "idle",
      });
      return;
    }

    if (event.type === "stream.user_prompt") {
      const rowId = this.store.recordMessage(event.payload.sessionId, {
        type: "user_prompt",
        prompt: event.payload.prompt,
      });
      const session = this.store.getSession(event.payload.sessionId);
      if (session) this.store.writeMessageSnapshot(rowId, buildExportEnvironmentSnapshot(session));
      return;
    }

    if (event.type === "stream.message") {
      const { sessionId, message } = event.payload;
      const rowId = this.store.recordMessage(sessionId, message);
      const session = this.store.getSession(sessionId);
      if (session) this.store.writeMessageSnapshot(rowId, buildExportEnvironmentSnapshot(session));
      this.maybeMarkNodeComplete(sessionId, message);
      return;
    }

    if (event.type === "session.messagesReset") {
      this.store.replaceMessages(event.payload.sessionId, event.payload.messages);
      return;
    }

    if (event.type === "session.status") {
      this.store.updateSession(event.payload.sessionId, { status: event.payload.status });
      if (event.payload.status === "error") {
        this.finish({ status: "error", error: event.payload.error });
      } else if (event.payload.status !== "running") {
        this.finish({ status: event.payload.status });
      }
    }
  };

  async runPrompt(prompt: string, session: Session, branchEntryId?: string): Promise<RunOutcome> {
    this.lastError = undefined;
    const result = new Promise<RunOutcome>((resolve) => {
      this.waiter = resolve;
    });
    await runClaude({
      prompt,
      session,
      branchEntryId,
      bashEnv: this.bashEnv,
      onEvent: this.emit,
      onSessionUpdate: (updates) => this.store.updateSession(session.id, updates),
    });
    return result;
  }

  async solveNode(session: Session, nodeId: string): Promise<RunOutcome> {
    const node = findNodeById(session.workflowTree ?? [], nodeId);
    if (!node) return { status: "error", error: `Node ${nodeId} not found` };
    node.status = "running";
    this.store.updateSession(session.id, { workflowTree: session.workflowTree, status: "running" });
    this.currentNodeId = nodeId;
    const prompt = buildPromptForNode(
      node.description,
      getNodePath(session.workflowTree ?? [], nodeId),
      node.outputFiles,
      undefined,
      session.cwd,
    );
    const headlessPrompt = addHeadlessExecutionNote(prompt);
    this.emit({ type: "stream.user_prompt", payload: { sessionId: session.id, prompt: headlessPrompt } });
    return await this.runPrompt(headlessPrompt, session);
  }

  private finish(outcome: RunOutcome): void {
    const waiter = this.waiter;
    if (!waiter) return;
    this.waiter = null;
    waiter(outcome.error ? outcome : { ...outcome, error: this.lastError });
  }

  private maybeMarkNodeComplete(sessionId: string, message: StreamMessage): void {
    if (!this.currentNodeId) return;
    if (message.type !== "run_result") return;
    const nodeId = this.currentNodeId;
    this.currentNodeId = null;
    if (message.status !== "success") return;

    const session = this.store.getSession(sessionId);
    if (!session?.workflowTree) return;
    const completedNode = findNodeById(session.workflowTree, nodeId);
    if (!completedNode) return;

    if (!completedNode.originalOutputs && completedNode.outputFiles.length > 0) {
      const originals: { path: string; content: string }[] = [];
      for (const relPath of completedNode.outputFiles) {
        try {
          originals.push({
            path: relPath,
            content: readFileSync(join(session.cwd ?? process.cwd(), relPath), "utf8"),
          });
        } catch {
          // Missing output files are captured later by the session snapshot.
        }
      }
      if (originals.length > 0) completedNode.originalOutputs = originals;
    }

    completeNodeAndDescendants(completedNode);
    let parent = findParentNode(session.workflowTree, nodeId);
    while (parent && parent.children.every((child) => child.status === "completed")) {
      parent.status = "completed";
      parent = findParentNode(session.workflowTree, parent.id);
    }
    this.store.updateSession(sessionId, { workflowTree: session.workflowTree });
  }
}

function streamMessageToAction(message: StreamMessage): { actor: "user" | "agent"; action: string; message?: string } {
  if (message.type === "user_prompt") {
    return { actor: "user", action: `message(${JSON.stringify(message.prompt ?? "")})` };
  }
  if (message.type === "assistant") {
    const blocks = (message as { blocks?: Array<Record<string, unknown>> }).blocks ?? [];
    const text = blocks
      .map((block) => (
        typeof block.text === "string"
          ? block.text
          : typeof block.name === "string"
            ? `${block.name}()`
            : ""
      ))
      .filter(Boolean)
      .join(" | ");
    return { actor: "agent", action: text ? `message(${JSON.stringify(text)})` : "assistant()" };
  }
  if (message.type === "tool_result") {
    return { actor: "agent", action: `${message.toolName || "tool"}()` };
  }
  if (message.type === "run_result") {
    return { actor: "agent", action: `result(${JSON.stringify(message.status)})` };
  }
  if (message.type === "verifier_label") {
    return { actor: "agent", action: `verify(${JSON.stringify(message.nodeId)})` };
  }
  return { actor: "agent", action: `${message.type}()` };
}

function writeTrajectorySession(store: SessionStore, session: Session, task: TaskSpec, outPath: string): void {
  const rows = store.getMessageRowsWithSnapshots(session.id);
  const trajectory = rows.map((row) => {
    const base = streamMessageToAction(row.message);
    return row.snapshot ? { ...base, environment: row.snapshot } : base;
  });
  trajectory.push({
    actor: "agent",
    action: "final_snapshot()",
    environment: buildExportEnvironmentSnapshot(session),
  });
  const payload = {
    uuid: session.id,
    name: session.title,
    task: task.instruction,
    model: "headless-tinker",
    trajectory,
  };
  writeFileSync(outPath, JSON.stringify(payload, null, 2) + "\n", "utf8");
}

function collectOutputFiles(session: Session): string[] {
  const cwd = session.cwd ?? process.cwd();
  const files = new Set<string>();
  for (const node of flattenNodes(session.workflowTree ?? [])) {
    for (const rel of node.outputFiles ?? []) {
      if (existsSync(join(cwd, rel))) files.add(rel);
    }
  }
  return [...files].sort();
}

function copyArtifacts(session: Session, artifactDir: string): string[] {
  mkdirSync(artifactDir, { recursive: true });
  const cwd = session.cwd ?? process.cwd();
  const copied: string[] = [];
  for (const rel of collectOutputFiles(session)) {
    const src = join(cwd, rel);
    const dst = join(artifactDir, rel);
    mkdirSync(dirname(dst), { recursive: true });
    cpSync(src, dst);
    copied.push(rel);
  }
  return copied;
}

function runChild(command: string, args: string[], cwd: string, logPath: string): Promise<void> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { cwd, env: process.env });
    const log = (chunk: Buffer) => appendFileSync(logPath, chunk);
    child.stdout.on("data", log);
    child.stderr.on("data", log);
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolvePromise();
      else reject(new Error(`${command} exited with ${code}; see ${logPath}`));
    });
  });
}

function scoreFromRatings(path: string): number | null {
  if (!existsSync(path)) return null;
  const report = JSON.parse(readFileSync(path, "utf8")) as {
    average_success_rate?: number;
    tasks?: Array<{ versions?: Array<{ average_success_pct?: number }> }>;
  };
  if (typeof report.average_success_rate === "number") {
    return report.average_success_rate * 100.0;
  }
  const scores: number[] = [];
  for (const task of report.tasks ?? []) {
    const versions = task.versions ?? [];
    for (let i = versions.length - 1; i >= 0; i--) {
      const score = versions[i]?.average_success_pct;
      if (typeof score === "number") {
        scores.push(score);
        break;
      }
    }
  }
  if (scores.length === 0) return null;
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

function taskSummaryPath(taskDir: string): string {
  return join(taskDir, "task_summary.json");
}

function loadTaskSummary(taskDir: string): TaskSummary | null {
  const path = taskSummaryPath(taskDir);
  if (!existsSync(path)) return null;
  const raw = JSON.parse(readFileSync(path, "utf8")) as TaskSummary;
  return raw;
}

function writeTaskSummary(taskDir: string, summary: TaskSummary): void {
  writeFileSync(taskSummaryPath(taskDir), JSON.stringify(summary, null, 2) + "\n", "utf8");
}

function shouldSkipExistingTask(args: Args, summary: TaskSummary | null): boolean {
  if (!summary || summary.status !== "completed") return false;
  if (!existsSync(summary.session_json)) return false;
  return true;
}

function backupDirectory(srcDir: string, backupRoot: string): string {
  mkdirSync(backupRoot, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const baseName = `${srcDir.split(/[\\/]/).pop()}-${stamp}`;
  let target = join(backupRoot, baseName);
  let suffix = 1;
  while (existsSync(target)) {
    target = join(backupRoot, `${baseName}-${suffix++}`);
  }
  renameSync(srcDir, target);
  return target;
}

function backupTaskDirBeforeRerun(taskDir: string): string {
  return backupDirectory(taskDir, join(dirname(taskDir), ".rerun-backups"));
}

function backupOutDirBeforeForce(outDir: string): string {
  return backupDirectory(outDir, join(dirname(outDir), ".headless-run-backups"));
}

function collectEvaluationArtifactNames(taskDir: string, workdir: string, copied: string[]): string[] {
  const names: string[] = [];
  const seen = new Set<string>();
  for (const name of copied) {
    if (name && !seen.has(name)) {
      names.push(name);
      seen.add(name);
    }
  }
  for (const root of [join(taskDir, "artifacts"), workdir]) {
    if (!existsSync(root)) continue;
    for (const entry of readdirSync(root, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      if (!seen.has(entry.name)) {
        names.push(entry.name);
        seen.add(entry.name);
      }
    }
  }
  return names;
}

async function runEvaluation(
  args: Args,
  taskDir: string,
  sessionJson: string,
  runLog: string,
  artifactNames: string[],
  artifactRoots: string[],
): Promise<{ ratingsPath: string; score: number | null }> {
  const ratingsPath = join(taskDir, "ratings.json");
  const evalBackend = args.evalBackend ?? "openai";
  await runChild(
    args.python,
    [
      "scripts/tools/grade_redo.py",
      "--session-json",
      sessionJson,
      "--verifiers",
      args.verifiersJson ? resolve(args.verifiersJson) : join(repoRoot, "scripts", "verifiers.json"),
      "--json-out",
      ratingsPath,
      "--backend",
      evalBackend,
      ...artifactNames.flatMap((name) => ["--artifact-name", name]),
      ...artifactRoots.flatMap((root) => ["--artifact-root", root]),
      ...(args.evalModel ? ["--model", args.evalModel] : []),
      ...(args.evalBaseUrl ? ["--base-url", args.evalBaseUrl] : []),
      ...(args.evalApiKey ? ["--api-key", args.evalApiKey] : []),
      ...(args.evalRequestTimeout !== undefined ? ["--request-timeout", String(args.evalRequestTimeout)] : []),
      ...(args.evalMaxRetries !== undefined ? ["--max-retries", String(args.evalMaxRetries)] : []),
    ],
    repoRoot,
    runLog,
  );
  return { ratingsPath, score: scoreFromRatings(ratingsPath) };
}

function copyUiAuthIfAvailable(sourceUserDataDir: string, targetUserDataDir: string): boolean {
  const source = authPath(sourceUserDataDir);
  if (!existsSync(source)) return false;
  const target = authPath(targetUserDataDir);
  mkdirSync(dirname(target), { recursive: true });
  cpSync(source, target);
  return true;
}

async function runTask(args: Args, store: SessionStore, task: TaskSpec, index: number): Promise<TaskSummary> {
  const taskDir = join(resolve(args.out), `task_${String(index + 1).padStart(3, "0")}`);
  const workdir = join(taskDir, "workdir");
  const logsDir = join(taskDir, "logs");
  const artifactsDir = join(taskDir, "artifacts");
  const matplotlibConfigDir = join(taskDir, ".matplotlib");
  const sessionJson = join(taskDir, "session.json");
  const eventsPath = join(logsDir, "events.jsonl");
  const runLog = join(logsDir, "run.log");
  mkdirSync(logsDir, { recursive: true });
  if (existsSync(taskDir) && args.force) backupTaskDirBeforeRerun(taskDir);
  mkdirSync(logsDir, { recursive: true });
  cpSync(resolve(args.workplaceTemplate), workdir, { recursive: true });
  mkdirSync(matplotlibConfigDir, { recursive: true });

  const headlessBashEnv = {
    MPLBACKEND: "Agg",
    MPLCONFIGDIR: matplotlibConfigDir,
    QT_QPA_PLATFORM: "offscreen",
    PYTHONUNBUFFERED: "1",
  };

  const runtime = new HeadlessSessionRuntime(store, eventsPath, headlessBashEnv);
  const session = store.createSession({
    cwd: workdir,
    title: `Task ${task.id}`,
    prompt: task.instruction,
    engine: "pi",
    autoContextInduction: false,
  });

  try {
    const planningPrompt = addHeadlessExecutionNote(task.instruction);
    runtime.emit({ type: "stream.user_prompt", payload: { sessionId: session.id, prompt: planningPrompt } });
    store.updateSession(session.id, { status: "running", lastPrompt: planningPrompt });
    const plan = await runtime.runPrompt(planningPrompt, session);
    if (plan.status === "error") throw new Error(plan.error || "Planning failed");

    let guard = 0;
    while (guard++ < 100) {
      const fresh = store.getSession(session.id)!;
      const nextId = findNextRunnableNodeId(fresh.workflowTree ?? [], fresh.verificationDepth ?? 0);
      if (!nextId) break;
      const outcome = await runtime.solveNode(fresh, nextId);
      if (outcome.status === "error") throw new Error(outcome.error || `Node ${nextId} failed`);
    }

    const finalSession = store.getSession(session.id)!;
    writeTrajectorySession(store, finalSession, task, sessionJson);
    const copied = copyArtifacts(finalSession, artifactsDir);

    let ratingsPath: string | undefined;
    let score: number | null | undefined;
    let evalError: string | undefined;
    if (args.eval) {
      const evalArtifactNames = collectEvaluationArtifactNames(taskDir, workdir, copied);
      try {
        ({ ratingsPath, score } = await runEvaluation(
          args,
          taskDir,
          sessionJson,
          runLog,
          evalArtifactNames,
          [artifactsDir, workdir],
        ));
      } catch (error) {
        ratingsPath = join(taskDir, "ratings.json");
        score = null;
        evalError = error instanceof Error ? error.message : String(error);
        appendFileSync(runLog, `${evalError}\n`, "utf8");
      }
    }

    const nodes = flattenNodes(finalSession.workflowTree ?? []);
    const summary: TaskSummary = {
      task_id: task.id,
      instruction: task.instruction,
      session_id: session.id,
      status: "completed",
      workdir,
      session_json: sessionJson,
      ratings_json: ratingsPath,
      score,
      output_files: copied,
      workflow_nodes: nodes.map((node) => ({
        id: node.id,
        description: node.description,
        status: node.status,
        outputFiles: node.outputFiles,
      })),
      eval_error: evalError,
    };
    writeTaskSummary(taskDir, summary);
    return summary;
  } catch (error) {
    const finalSession = store.getSession(session.id) ?? session;
    writeTrajectorySession(store, finalSession, task, sessionJson);
    appendFileSync(runLog, `${error instanceof Error ? error.stack || error.message : String(error)}\n`, "utf8");
    const summary: TaskSummary = {
      task_id: task.id,
      instruction: task.instruction,
      session_id: session.id,
      status: "error",
      workdir,
      session_json: sessionJson,
      output_files: collectOutputFiles(finalSession),
      workflow_nodes: flattenNodes(finalSession.workflowTree ?? []).map((node) => ({
        id: node.id,
        description: node.description,
        status: node.status,
        outputFiles: node.outputFiles,
      })),
      error: error instanceof Error ? error.message : String(error),
    };
    writeTaskSummary(taskDir, summary);
    return summary;
  }
}

function writeSummary(outDir: string, summaries: TaskSummary[]): void {
  const scored = summaries
    .map((item) => item.score)
    .filter((score): score is number => typeof score === "number" && Number.isFinite(score));
  const zeroScoreTaskIds = summaries.filter((item) => item.score === 0).map((item) => item.task_id);
  const unscoredTaskIds = summaries.filter((item) => item.score == null).map((item) => item.task_id);
  const overall = scored.length > 0 ? scored.reduce((a, b) => a + b, 0) / scored.length : null;
  writeFileSync(
    join(outDir, "summary.json"),
    JSON.stringify(
      {
        overall_score: overall,
        scored_task_count: scored.length,
        unscored_task_count: summaries.length - scored.length,
        total_task_count: summaries.length,
        zero_score_task_ids: zeroScoreTaskIds,
        unscored_task_ids: unscoredTaskIds,
        tasks: summaries,
      },
      null,
      2,
    ) + "\n",
    "utf8",
  );
  const rows = ["task_id,status,score,session_id,error"];
  for (const item of summaries) {
    rows.push([
      JSON.stringify(item.task_id),
      item.status,
      item.score ?? "",
      item.session_id ?? "",
      JSON.stringify(item.error ?? ""),
    ].join(","));
  }
  writeFileSync(join(outDir, "scores.csv"), rows.join("\n") + "\n", "utf8");
}

function loadAllTaskSummaries(outDir: string): TaskSummary[] {
  if (!existsSync(outDir)) return [];
  return readdirSync(outDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^task_\d+$/.test(entry.name))
    .map((entry) => join(outDir, entry.name))
    .sort((a, b) => a.localeCompare(b))
    .map((taskDir) => loadTaskSummary(taskDir))
    .filter((summary): summary is TaskSummary => summary !== null);
}

async function main(): Promise<void> {
  loadDotenv({ path: join(repoRoot, "scripts", ".env") });
  loadDotenv({ path: join(repoRoot, ".env") });

  const args = parseArgs(process.argv.slice(2));
  const outDir = resolve(args.out);
  if (args.eval) {
    const verifiersPath = args.verifiersJson ? resolve(args.verifiersJson) : join(repoRoot, "scripts", "verifiers.json");
    if (!existsSync(verifiersPath)) {
      throw new Error(
        `Verifier catalog not found: ${verifiersPath}. Create it first, for example: ` +
          "python scripts/tools/extract_verifiers.py out.json -o scripts/verifiers.json",
      );
    }
  }
  const uiUserDataDir = app.getPath("userData");
  if (existsSync(outDir) && args.force && !args.taskIds?.length) {
    const backupPath = backupOutDirBeforeForce(outDir);
    console.log(`[headless] backed up previous output dir to ${backupPath}`);
  }
  if (existsSync(outDir) && !args.force && !args.resume) {
    throw new Error(`Output directory already exists: ${outDir}. Pass --force to replace it.`);
  }
  mkdirSync(outDir, { recursive: true });
  const headlessUserDataDir = join(outDir, ".electron-user-data");
  app.setPath("userData", headlessUserDataDir);
  await app.whenReady();

  const hasExplicitTinkerKey = Boolean(args.tinkerApiKey?.trim() || process.env.TINKER_API_KEY?.trim());
  const copiedUiAuth = hasExplicitTinkerKey ? false : copyUiAuthIfAvailable(uiUserDataDir, headlessUserDataDir);

  saveTinkerProviderConfig({
    model: "headless-tinker",
    baseModel: args.baseModel,
    modelPath: args.modelPath,
    rendererName: args.rendererName,
    apiKey: args.tinkerApiKey ?? process.env.TINKER_API_KEY,
    baseUrl: args.tinkerBaseUrl,
  });
  await saveAgentSettings({ defaultProvider: "tinker", defaultModel: "headless-tinker" });
  if (!hasExplicitTinkerKey && !copiedUiAuth) {
    console.warn(
      "[headless] no explicit Tinker key found. Pass --tinker-api-key, export TINKER_API_KEY, or save a Tinker key in the UI settings."
    );
  }
  await ensureTinkerBridgeWarm(tinkerConfigPath(headlessUserDataDir));

  const tasks = JSON.parse(readFileSync(resolve(args.tasks), "utf8")) as TaskSpec[];
  const limited = tasks.slice(0, args.limit);
  const wantedIds = args.taskIds ? new Set(args.taskIds.map(String)) : null;
  const selected = limited
    .map((task, index) => ({ task, index }))
    .filter(({ task }) => !wantedIds || wantedIds.has(String(task.id)));
  const store = new SessionStore(join(app.getPath("userData"), "sessions.db"));
  try {
    for (let i = 0; i < selected.length; i++) {
      const { task, index } = selected[i]!;
      const taskDir = join(outDir, `task_${String(index + 1).padStart(3, "0")}`);
      if (args.resume && existsSync(taskDir)) {
        const existing = loadTaskSummary(taskDir);
        if (shouldSkipExistingTask(args, existing)) {
          console.log(`[headless] skip completed task ${i + 1}/${selected.length}: ${task.id}`);
          writeSummary(outDir, loadAllTaskSummaries(outDir));
          continue;
        }
        console.log(`[headless] rerun incomplete task ${i + 1}/${selected.length}: ${task.id}`);
        const backupPath = backupTaskDirBeforeRerun(taskDir);
        console.log(`[headless] backed up previous task dir to ${backupPath}`);
      }
      console.log(`[headless] task ${i + 1}/${selected.length}: ${task.id}`);
      await runTask(args, store, task, index);
      writeSummary(outDir, loadAllTaskSummaries(outDir));
    }
  } finally {
    store.close();
    app.quit();
  }
}

main().catch((error) => {
  console.error(error);
  app.exit(1);
});
