import { app } from "electron";
import { config as loadDotenv } from "dotenv";
import { randomUUID } from "crypto";
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { homedir } from "os";
import { basename, dirname, join, resolve } from "path";

import {
  DefaultResourceLoader,
  SessionManager,
  createAgentSession,
  type CreateAgentSessionOptions,
} from "@mariozechner/pi-coding-agent";
import type { Model } from "@mariozechner/pi-ai";

import { createPiManagers, saveAgentSettings } from "./libs/pi-config.js";
import type { ServerEvent, StreamMessage, WorkflowNode } from "./types.js";
import { runClaude, buildPromptForNode } from "./libs/runner.js";
import { SessionStore, type Session } from "./libs/session-store.js";
import { buildExportEnvironmentSnapshot } from "./libs/message-state-snapshot.js";
import {
  completeNodeAndDescendants,
  findNodeById,
  findParentNode,
  getNodePath,
} from "./libs/workflow-tree-utils.js";

type TaskSpec = {
  task: string;
  name?: string;
  output_path?: string;
  outputPath?: string;
  rubrics?: string[];
  verifiers?: string[];
  hidden_user_profile?: string;
  persona?: string;
  [key: string]: unknown;
};

type PreparedTaskSpec = {
  task: string;
  name: string;
  outputPath: string;
  rubrics: string[];
  hiddenUserProfile: string;
  metadata: Record<string, unknown>;
};

type Args = {
  task: string[];
  tasks?: string;
  generateTasks: number;
  taskDomain: string;
  output: string;
  append: boolean;
  rounds: number;
  minRevisions: number;
  offlineDemo: boolean;
  provider?: string;
  model?: string;
  agentProvider?: string;
  agentModel?: string;
  userProvider?: string;
  userModel?: string;
  taskProvider?: string;
  taskModel?: string;
  sourceUserDataDir?: string;
  userDataDir?: string;
  useUiUserData: boolean;
};

type ModelSelection = {
  provider?: string;
  model?: string;
};

type RunOutcome = {
  status: "idle" | "completed" | "error";
  error?: string;
};

type AgentAttempt = {
  assistantMessage: string;
  files: OutputFile[];
  messages: OpenAIMessage[];
  modelLabel: string;
};

type SimulatorResult = {
  satisfied: boolean;
  feedback: string;
  rubric_results: RubricResult[];
};

type RubricResult = {
  criterion: string;
  passed: boolean;
  reason?: string;
};

type OutputFile = {
  path: string;
  content: string;
};

type OpenAIMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  thinking?: string;
  environment?: unknown;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: {
      name: string;
      arguments: string;
    };
  }>;
  tool_call_id?: string;
  name?: string;
};

type SessionJson = Record<string, unknown>;

const DEFAULT_OUTPUT = ".artifacts/synthetic_sessions.json";
const PI_AGENT_DIR_NAME = "pi-agent";
const PI_CONFIG_FILES = ["auth.json", "models.json", "settings.json", "tinker-provider.json"];

const TRAINING_SYSTEM_PROMPT = [
  "You are Agent Cowork, a task-solving assistant.",
  "Create or revise concrete artifacts for the user's task.",
  "Prefer clear, useful, complete files over explanation-only responses.",
].join("\n");

const HEADLESS_EXECUTION_NOTE =
  "Headless execution only: save outputs to files in the working directory. Do not open GUI windows, interactive plot viewers, browser tabs, or commands that wait for manual closing.";

const TASK_PREP_SYSTEM_PROMPT = [
  "You design realistic synthetic training tasks for a task-solving agent.",
  "Return compact JSON only.",
].join("\n");

const USER_SIM_SYSTEM_PROMPT = [
  "You are a realistic user simulator for synthetic training.",
  "You have hidden preferences and rubrics. Inspect the latest agent artifact,",
  "decide whether a real user would accept it, and provide concise follow-up",
  "feedback when it is not yet good enough.",
  "",
  "Return JSON only with this shape:",
  "{",
  '  "satisfied": false,',
  '  "feedback": "one actionable user follow-up, or empty string if satisfied",',
  '  "rubric_results": [',
  '    {"criterion": "exact rubric text", "passed": true, "reason": "short reason"}',
  "  ]",
  "}",
  "",
  "Rules:",
  "- Do not reveal that you are a simulator.",
  "- Feedback should be a natural user message, not a rubric report.",
  "- Prefer one specific revision request over a long list.",
  "- If minimum revision pressure is still active, ask for a meaningful refinement",
  "  even if the artifact is mostly acceptable.",
].join("\n");

const WORKFLOW_PLAN_INSTRUCTION = "Create a workflow plan for this task.\n\nTask instruction:\n";

const GENERIC_RUBRICS = [
  "The artifact directly satisfies the user's stated task.",
  "The artifact is specific, complete, and ready to use.",
  "The artifact is concise and easy to scan.",
  "The artifact follows all explicit constraints in the prompt.",
];

const TOOL_SCHEMAS = [
  functionSchema("read", "Read the contents of a file.", {
    type: "object",
    properties: {
      path: { type: "string" },
      offset: { type: "number" },
      limit: { type: "number" },
    },
    required: ["path"],
  }),
  functionSchema("write", "Write full content to a relative path.", {
    type: "object",
    properties: {
      path: { type: "string" },
      content: { type: "string" },
    },
    required: ["path", "content"],
    additionalProperties: false,
  }),
  functionSchema("edit", "Edit a single file using exact text replacement.", {
    type: "object",
    properties: {
      path: { type: "string" },
      edits: {
        type: "array",
        items: {
          type: "object",
          properties: {
            oldText: { type: "string" },
            newText: { type: "string" },
          },
          required: ["oldText", "newText"],
          additionalProperties: false,
        },
      },
    },
    required: ["path", "edits"],
    additionalProperties: false,
  }),
  functionSchema("bash", "Execute a bash command in the current working directory.", {
    type: "object",
    properties: {
      command: { type: "string" },
      timeout: { type: "number" },
    },
    required: ["command"],
  }),
  functionSchema("grep", "Search file contents for a pattern.", {
    type: "object",
    properties: {
      pattern: { type: "string" },
      path: { type: "string" },
      glob: { type: "string" },
      ignoreCase: { type: "boolean" },
      literal: { type: "boolean" },
      context: { type: "number" },
      limit: { type: "number" },
    },
    required: ["pattern"],
  }),
  functionSchema("find", "Search for files by glob pattern.", {
    type: "object",
    properties: {
      pattern: { type: "string" },
      path: { type: "string" },
      limit: { type: "number" },
    },
    required: ["pattern"],
  }),
  functionSchema("ls", "List directory contents.", {
    type: "object",
    properties: {
      path: { type: "string" },
      limit: { type: "number" },
    },
    required: [],
  }),
  functionSchema("workflow_plan", "Register a hierarchical workflow plan.", {
    type: "object",
    properties: {
      tasks: {
        type: "array",
        items: { $ref: "#/$defs/WorkflowNode" },
      },
    },
    required: ["tasks"],
    $defs: {
      WorkflowNode: {
        type: "object",
        properties: {
          description: { type: "string" },
          outputFiles: { type: "array", items: { type: "string" } },
          verifiers: { type: "array", items: { type: "string" } },
          children: { type: "array", items: { $ref: "#/$defs/WorkflowNode" } },
        },
        required: ["description", "outputFiles", "verifiers"],
        additionalProperties: false,
      },
    },
  }),
  functionSchema("ask_user_question", "Ask the operator a structured question and wait for an answer.", {
    type: "object",
    properties: {
      questions: {
        type: "array",
        items: {
          type: "object",
          properties: {
            question: { type: "string" },
            header: { type: "string" },
            options: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  label: { type: "string" },
                  description: { type: "string" },
                },
                required: ["label"],
              },
            },
            multiSelect: { type: "boolean" },
          },
          required: ["question"],
        },
      },
      answers: {
        type: "object",
        additionalProperties: { type: "string" },
      },
    },
    required: ["questions"],
  }),
];

function functionSchema(name: string, description: string, parameters: Record<string, unknown>) {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters,
    },
  };
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    task: [],
    generateTasks: 0,
    taskDomain: "general knowledge work artifacts",
    output: DEFAULT_OUTPUT,
    append: false,
    rounds: 3,
    minRevisions: 1,
    offlineDemo: false,
    useUiUserData: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    const next = () => {
      const v = argv[++i];
      if (!v) throw new Error(`Missing value for ${a}`);
      return v;
    };
    if (a === "--task") args.task.push(next());
    else if (a === "--tasks") args.tasks = next();
    else if (a === "--generate-tasks") args.generateTasks = Number(next());
    else if (a === "--task-domain") args.taskDomain = next();
    else if (a === "--output" || a === "--out") args.output = next();
    else if (a === "--append") args.append = true;
    else if (a === "--rounds") args.rounds = Number(next());
    else if (a === "--min-revisions") args.minRevisions = Number(next());
    else if (a === "--offline-demo") args.offlineDemo = true;
    else if (a === "--provider") args.provider = next();
    else if (a === "--model") args.model = next();
    else if (a === "--agent-provider") args.agentProvider = next();
    else if (a === "--agent-model") args.agentModel = next();
    else if (a === "--user-provider" || a === "--sim-provider") args.userProvider = next();
    else if (a === "--user-model" || a === "--sim-model") args.userModel = next();
    else if (a === "--task-provider") args.taskProvider = next();
    else if (a === "--task-model") args.taskModel = next();
    else if (a === "--source-user-data-dir") args.sourceUserDataDir = next();
    else if (a === "--user-data-dir") args.userDataDir = next();
    else if (a === "--use-ui-user-data") args.useUiUserData = true;
    else if (a === "--help" || a === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${a}`);
    }
  }

  if (!Number.isFinite(args.rounds) || args.rounds < 1) throw new Error("--rounds must be at least 1");
  if (!Number.isFinite(args.minRevisions) || args.minRevisions < 0) {
    throw new Error("--min-revisions must be non-negative");
  }
  if (!Number.isFinite(args.generateTasks) || args.generateTasks < 0) {
    throw new Error("--generate-tasks must be non-negative");
  }
  return args;
}

function printHelp(): void {
  console.log(`Usage:
  bun run synth:interactions -- --task "Create a concise project status update" \\
    --output .artifacts/synthetic_sessions.json

  bun run synth:interactions -- --tasks tasks.jsonl --rounds 3 \\
    --output .artifacts/synthetic_sessions.json

  printf '%s\\n' '{"task":"Write a short onboarding note"}' | \\
    bun run synth:interactions -- --tasks - --offline-demo

  bun run synth:interactions -- --generate-tasks 10 \\
    --task-domain "spreadsheet and document editing tasks" \\
    --output .artifacts/synthetic_sessions.json

  bun run synth:interactions -- --offline-demo \\
    --output .artifacts/synthetic_sessions_demo.json

Model-backed runs use Pi / Agent Cowork provider settings copied from the app's
pi-agent config. Optional --provider/--model, --agent-model, --user-model, and
--task-model select Pi models by provider/id. No OPENAI_API_KEY is read by this
script directly.

Use --source-user-data-dir to override the app userData directory copied from,
and --user-data-dir to override the isolated synthetic userData directory.
`);
}

function slugify(value: string, fallback = "task"): string {
  const slug = value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return slug.slice(0, 60) || fallback;
}

function coerceStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean);
}

function taskFromUnknown(value: unknown, index: number): TaskSpec {
  if (typeof value === "string") return { task: value };
  if (!value || typeof value !== "object") throw new Error(`Task #${index} must be a string or object`);
  const obj = value as Record<string, unknown>;
  const task = String(obj.task ?? obj.instruction ?? obj.prompt ?? "").trim();
  if (!task) throw new Error(`Task #${index} is missing task/instruction/prompt`);
  return { ...obj, task };
}

function loadTasksFile(path: string): TaskSpec[] {
  const text = path === "-" ? readFileSync(0, "utf8") : readFileSync(resolve(path), "utf8");
  const parseJsonl = () => {
    const out: TaskSpec[] = [];
    for (const [lineIndex, rawLine] of text.split(/\r?\n/).entries()) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) continue;
      let parsed: unknown = line;
      try {
        parsed = JSON.parse(line);
      } catch {
        // Plain line tasks are valid in JSONL mode.
      }
      out.push(taskFromUnknown(parsed, lineIndex + 1));
    }
    return out;
  };

  if (path.endsWith(".jsonl")) return parseJsonl();
  try {
    const parsed = JSON.parse(text) as unknown;
    const items =
      parsed && typeof parsed === "object" && Array.isArray((parsed as { tasks?: unknown }).tasks)
        ? (parsed as { tasks: unknown[] }).tasks
        : Array.isArray(parsed)
          ? parsed
          : [parsed];
    return items.map((item, i) => taskFromUnknown(item, i + 1));
  } catch {
    return parseJsonl();
  }
}

function taskMetadata(raw: TaskSpec): Record<string, unknown> {
  const known = new Set([
    "task",
    "instruction",
    "prompt",
    "name",
    "title",
    "output_path",
    "outputPath",
    "rubrics",
    "verifiers",
    "hidden_user_profile",
    "persona",
  ]);
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (!known.has(key)) out[key] = value;
  }
  return out;
}

function safeOutputPath(path: string, name: string): string {
  const parts = path.trim().replaceAll("\\", "/").replace(/^\.\//, "").split("/").filter((part) => {
    return Boolean(part) && part !== "." && part !== "..";
  });
  const cleaned = parts.join("/");
  return cleaned && !cleaned.startsWith("/") ? cleaned : `${slugify(name)}.md`;
}

function addHeadlessExecutionNote(prompt: string): string {
  const trimmed = prompt.trim();
  if (!trimmed) return HEADLESS_EXECUTION_NOTE;
  return `${trimmed}\n\n${HEADLESS_EXECUTION_NOTE}`;
}

function appPrimaryOutputPath(spec: PreparedTaskSpec): string {
  return basename(spec.outputPath.replace(/\\/g, "/"));
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

class AppHarnessRuntime {
  private currentNodeId: string | null = null;
  private waiter: ((outcome: RunOutcome) => void) | null = null;
  private lastError: string | undefined;
  readonly events: ServerEvent[] = [];

  constructor(
    private readonly store: SessionStore,
    private readonly bashEnv: Record<string, string>,
  ) {}

  emit = (event: ServerEvent): void => {
    this.events.push(event);

    if (event.type === "permission.request") {
      const session = this.store.getSession(event.payload.sessionId);
      const pending = session?.pendingPermissions.get(event.payload.toolUseId);
      pending?.resolve({ behavior: "deny", message: "Synthetic headless mode does not provide human intervention." });
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
    const result = new Promise<RunOutcome>((resolveOutcome) => {
      this.waiter = resolveOutcome;
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
      node,
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
          // Missing output files remain visible through the environment snapshot.
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

async function prepareTaskSpec(raw: TaskSpec, model: ModelSelection, offline: boolean): Promise<PreparedTaskSpec> {
  const rawName = String(raw.name ?? raw.title ?? "").trim();
  const rawPath = String(raw.output_path ?? raw.outputPath ?? "").trim();
  const rawRubrics = coerceStringList(raw.rubrics ?? raw.verifiers);
  const rawProfile = String(raw.hidden_user_profile ?? raw.persona ?? "").trim();
  if (rawName && rawPath && rawRubrics.length > 0 && rawProfile) {
    return {
      task: raw.task,
      name: rawName,
      outputPath: safeOutputPath(rawPath, rawName),
      rubrics: rawRubrics,
      hiddenUserProfile: rawProfile,
      metadata: taskMetadata(raw),
    };
  }

  if (offline) {
    const name = rawName || raw.task.split(/\r?\n/)[0]?.slice(0, 80) || "Synthetic task";
    return {
      task: raw.task,
      name,
      outputPath: safeOutputPath(rawPath || `${slugify(name)}.md`, name),
      rubrics: rawRubrics.length > 0 ? rawRubrics : [...GENERIC_RUBRICS],
      hiddenUserProfile:
        rawProfile || "Prefers concise, concrete work with clear structure, accurate details, and no generic filler.",
      metadata: taskMetadata(raw),
    };
  }

  const filled = await runPiJson(model, TASK_PREP_SYSTEM_PROMPT, {
    task: raw.task,
    current_name: rawName,
    current_output_path: rawPath,
    current_rubrics: rawRubrics,
    current_hidden_user_profile: rawProfile,
    instructions:
      "Fill missing fields for one synthetic task. Use 4-6 rubrics. Choose one primary relative output path.",
    return_schema: {
      name: "short task title",
      output_path: "relative path",
      rubrics: ["criterion"],
      hidden_user_profile: "private simulator preferences",
    },
  });

  const name = rawName || String(filled.name ?? raw.task.split(/\r?\n/)[0]?.slice(0, 80) ?? "Synthetic task").trim();
  const rubrics = rawRubrics.length > 0 ? rawRubrics : coerceStringList(filled.rubrics) || [...GENERIC_RUBRICS];
  return {
    task: raw.task,
    name,
    outputPath: safeOutputPath(rawPath || String(filled.output_path ?? ""), name),
    rubrics: rubrics.length > 0 ? rubrics : [...GENERIC_RUBRICS],
    hiddenUserProfile:
      rawProfile ||
      String(filled.hidden_user_profile ?? "").trim() ||
      "Prefers concise, concrete, accurate work that follows the prompt.",
    metadata: taskMetadata(raw),
  };
}

async function generateTaskSpecs(model: ModelSelection, count: number, domain: string): Promise<TaskSpec[]> {
  const result = await runPiJson(model, TASK_PREP_SYSTEM_PROMPT, {
    count,
    domain,
    instructions:
      "Generate varied task-solving prompts suitable for an artifact-writing agent. Prefer tasks that can be completed by writing one markdown/html/csv/json/text file and can naturally receive follow-ups.",
    return_schema: {
      tasks: [
        {
          name: "short title",
          task: "user-facing initial instruction",
          output_path: "relative path",
          rubrics: ["4-6 criteria"],
          hidden_user_profile: "private simulator preferences",
        },
      ],
    },
  });
  if (!Array.isArray(result.tasks)) throw new Error("Task generator response did not include tasks[]");
  return result.tasks.map((item: unknown, i: number) => taskFromUnknown(item, i + 1));
}

function modelSelection(args: Args, role: "agent" | "user" | "task"): ModelSelection {
  if (role === "agent") {
    return { provider: args.agentProvider ?? args.provider, model: args.agentModel ?? args.model };
  }
  if (role === "user") {
    return { provider: args.userProvider ?? args.provider, model: args.userModel ?? args.model };
  }
  return { provider: args.taskProvider ?? args.provider, model: args.taskModel ?? args.model };
}

function resolvePiModel(selection: ModelSelection): Model<any> | undefined {
  if (!selection.provider && !selection.model) return undefined;
  const { modelRegistry } = createPiManagers(process.cwd());
  const all = modelRegistry.getAll();
  if (selection.provider && selection.model) {
    const found = modelRegistry.find(selection.provider, selection.model);
    if (!found) throw new Error(`Pi model not found: ${selection.provider}/${selection.model}`);
    return found;
  }
  if (selection.model) {
    const matches = all.filter((candidate) => candidate.id === selection.model);
    if (matches.length === 1) return matches[0];
    if (matches.length > 1) {
      throw new Error(`Model id ${selection.model} is ambiguous; pass --provider.`);
    }
    throw new Error(`Pi model not found: ${selection.model}`);
  }
  const providerMatches = all.filter((candidate) => candidate.provider === selection.provider);
  if (providerMatches.length === 0) throw new Error(`Pi provider has no registered models: ${selection.provider}`);
  return undefined;
}

async function applyAgentModelSelection(selection: ModelSelection): Promise<void> {
  if (!selection.provider && !selection.model) return;
  const resolved = resolvePiModel(selection);
  await saveAgentSettings({
    defaultProvider: resolved?.provider ?? selection.provider,
    defaultModel: resolved?.id ?? selection.model,
  });
}

async function createPiSession(
  cwd: string,
  options: {
    model: ModelSelection;
    systemPrompt?: string;
    appendSystemPrompt?: string;
    tools?: CreateAgentSessionOptions["tools"];
  },
) {
  const { agentDir, authStorage, modelRegistry, settingsManager } = createPiManagers(cwd);
  settingsManager.applyOverrides({ compaction: { enabled: false } });
  const resourceLoader = new DefaultResourceLoader({
    cwd,
    agentDir,
    settingsManager,
    noExtensions: true,
    systemPrompt: options.systemPrompt,
    appendSystemPrompt: options.appendSystemPrompt,
  });
  await resourceLoader.reload();
  const { session, modelFallbackMessage } = await createAgentSession({
    cwd,
    agentDir,
    authStorage,
    modelRegistry,
    settingsManager,
    resourceLoader,
    sessionManager: SessionManager.inMemory(cwd),
    tools: options.tools ?? [],
    model: resolvePiModel(options.model),
  });
  if (modelFallbackMessage && !session.model) {
    session.dispose();
    throw new Error(modelFallbackMessage);
  }
  return session;
}

async function runPiJson(model: ModelSelection, systemPrompt: string, payload: unknown): Promise<Record<string, unknown>> {
  const session = await createPiSession(process.cwd(), {
    model,
    systemPrompt,
    tools: [],
  });
  try {
    await session.prompt(JSON.stringify(payload, null, 2));
    const text = lastAssistantText(session.messages);
    return parseJsonObject(text);
  } finally {
    session.dispose();
  }
}

function buildInitialAgentPrompt(spec: PreparedTaskSpec): string {
  const primary = appPrimaryOutputPath(spec);
  return addHeadlessExecutionNote([
    spec.task,
    "",
    `Primary output file: ${primary}`,
    "Create or revise the concrete artifact in that file. Keep any final chat note brief.",
  ].join("\n"));
}

function buildFollowupAgentPrompt(spec: PreparedTaskSpec, feedback: string): string {
  const primary = appPrimaryOutputPath(spec);
  return addHeadlessExecutionNote([
    feedback,
    "",
    `Revise the current artifact in ${primary}.`,
    "Read the current file first if useful, then edit or rewrite it with the requested improvement.",
  ].join("\n"));
}

async function solveAllWorkflowNodes(
  runtime: AppHarnessRuntime,
  store: SessionStore,
  sessionId: string,
): Promise<void> {
  let guard = 0;
  while (guard++ < 100) {
    const fresh = store.getSession(sessionId);
    if (!fresh) throw new Error(`Session ${sessionId} not found`);
    const nextId = findNextRunnableNodeId(fresh.workflowTree ?? [], fresh.verificationDepth ?? 0);
    if (!nextId) return;
    const outcome = await runtime.solveNode(fresh, nextId);
    if (outcome.status === "error") throw new Error(outcome.error || `Workflow node ${nextId} failed`);
  }
  throw new Error("Workflow execution exceeded 100 node runs");
}

async function runInitialHarnessRound(
  spec: PreparedTaskSpec,
  runtime: AppHarnessRuntime,
  store: SessionStore,
  session: Session,
): Promise<{ planningUnit: Record<string, unknown>; attempt: AgentAttempt }> {
  const planningPrompt = buildInitialAgentPrompt(spec);
  runtime.emit({ type: "stream.user_prompt", payload: { sessionId: session.id, prompt: planningPrompt } });
  store.updateSession(session.id, { status: "running", lastPrompt: planningPrompt });
  const plan = await runtime.runPrompt(planningPrompt, session);
  if (plan.status === "error") throw new Error(plan.error || "Workflow planning failed");

  const afterPlanningSession = store.getSession(session.id);
  if (!afterPlanningSession) throw new Error(`Session ${session.id} not found after planning`);
  const planningRows = store.getMessageRowsWithSnapshots(session.id);
  const planningMessages = rowsToOpenAI(planningRows);
  const workflowTree = workflowTreeToLlmNative(afterPlanningSession.workflowTree ?? []);
  const planningUnit = {
    intent: "planning",
    agent_trajectories: [
      {
        prompt: planningPrompt,
        messages: planningMessages,
      },
    ],
    human_trajectories: [],
    verifiers: [],
    workflow_tree_generated: workflowTree,
    workflow_tree_final: workflowTree,
  };

  await solveAllWorkflowNodes(runtime, store, session.id);
  const finalSession = store.getSession(session.id);
  if (!finalSession) throw new Error(`Session ${session.id} not found after execution`);
  const allRows = store.getMessageRowsWithSnapshots(session.id);
  const executionRows = allRows.slice(planningRows.length);
  const files = collectHarnessOutputFiles(finalSession, spec);
  return {
    planningUnit,
    attempt: {
      assistantMessage: lastStreamAssistantText(executionRows.map((row) => row.message)) || "I wrote the artifact.",
      files: files.length > 0 ? files : fallbackOutputFiles(finalSession.cwd ?? process.cwd(), spec, executionRows.map((row) => row.message)),
      messages: rowsToOpenAI(executionRows),
      modelLabel: modelLabelFromRows(allRows),
    },
  };
}

async function runFollowupHarnessRound(
  spec: PreparedTaskSpec,
  runtime: AppHarnessRuntime,
  store: SessionStore,
  sessionId: string,
  prompt: string,
): Promise<AgentAttempt> {
  const beforeRows = store.getMessageRowsWithSnapshots(sessionId);
  const session = store.getSession(sessionId);
  if (!session) throw new Error(`Session ${sessionId} not found`);
  const followupPrompt = buildFollowupAgentPrompt(spec, prompt);
  runtime.emit({ type: "stream.user_prompt", payload: { sessionId, prompt: followupPrompt } });
  store.updateSession(session.id, { status: "running", lastPrompt: followupPrompt });
  const outcome = await runtime.runPrompt(followupPrompt, session);
  if (outcome.status === "error") throw new Error(outcome.error || "Follow-up run failed");

  const finalSession = store.getSession(sessionId);
  if (!finalSession) throw new Error(`Session ${sessionId} not found after follow-up`);
  const allRows = store.getMessageRowsWithSnapshots(sessionId);
  const deltaRows = allRows.slice(beforeRows.length);
  const files = collectHarnessOutputFiles(finalSession, spec);
  return {
    assistantMessage: lastStreamAssistantText(deltaRows.map((row) => row.message)) || "I revised the artifact.",
    files: files.length > 0 ? files : fallbackOutputFiles(finalSession.cwd ?? process.cwd(), spec, deltaRows.map((row) => row.message)),
    messages: rowsToOpenAI(deltaRows),
    modelLabel: modelLabelFromRows(allRows),
  };
}

function offlineAgentRound(
  spec: PreparedTaskSpec,
  prompt: string,
  roundIndex: number,
  priorFiles: OutputFile[],
): AgentAttempt {
  let content: string;
  let message: string;
  if (roundIndex === 0) {
    content = [
      `# ${spec.name}`,
      "",
      "## Draft",
      spec.task,
      "",
      "This first pass covers the core request, but it is intentionally brief.",
      "",
      "## Next Steps",
      "- Confirm the target audience.",
      "- Tighten the final wording after feedback.",
    ].join("\n");
    message = "I drafted the initial artifact.";
  } else {
    const previous = priorFiles.at(-1)?.content ?? "";
    content = [
      `# ${spec.name}`,
      "",
      "## Final",
      spec.task,
      "",
      "Key improvements requested by the user:",
      `- ${prompt.trim()}`,
      "",
      "The artifact is now more specific, easier to scan, and ready to use.",
      "",
      "## Deliverable",
      previous.slice(0, 600).trim(),
      "",
      "## Polished Notes",
      "- Clear structure with explicit headings.",
      "- Concrete language instead of generic filler.",
      "- Constraints from the original task and follow-up are reflected.",
    ].join("\n");
    message = "I incorporated the feedback and rewrote the artifact.";
  }
  const file = { path: spec.outputPath, content };
  return {
    assistantMessage: message,
    files: [file],
    messages: [
      { role: "user", content: prompt },
      assistantWriteMessage(message, [file]),
      {
        role: "tool",
        tool_call_id: "call_write_0",
        name: "write",
        content: `Wrote ${file.path} (${file.content.length} chars).`,
      },
    ],
    modelLabel: "offline-demo",
  };
}

async function runUserSimulator(
  spec: PreparedTaskSpec,
  model: ModelSelection,
  roundIndex: number,
  maxRounds: number,
  minRevisions: number,
  completedRevisions: number,
  files: OutputFile[],
  offline: boolean,
): Promise<SimulatorResult> {
  if (offline) return offlineUserSimulator(spec, roundIndex, minRevisions, completedRevisions);
  const result = await runPiJson(model, USER_SIM_SYSTEM_PROMPT, {
    task: spec.task,
    hidden_user_profile: spec.hiddenUserProfile,
    rubrics: spec.rubrics,
    round_index: roundIndex,
    max_rounds: maxRounds,
    completed_revisions: completedRevisions,
    min_revisions_before_acceptance: minRevisions,
    latest_artifacts: files,
  });
  return normalizeSimulatorResult(result, spec, roundIndex, maxRounds, minRevisions, completedRevisions);
}

function offlineUserSimulator(
  spec: PreparedTaskSpec,
  roundIndex: number,
  minRevisions: number,
  completedRevisions: number,
): SimulatorResult {
  const forceRevision = completedRevisions < minRevisions;
  const satisfied = roundIndex > 0 && !forceRevision;
  const passedCount = satisfied ? spec.rubrics.length : Math.max(0, spec.rubrics.length - 2);
  return {
    satisfied,
    feedback: satisfied ? "" : "Make this more specific, more polished, and easier to use without extra back-and-forth.",
    rubric_results: spec.rubrics.map((criterion, i) => ({
      criterion,
      passed: i < passedCount,
      reason: "Synthetic offline demo rubric label.",
    })),
  };
}

function normalizeSimulatorResult(
  raw: Record<string, unknown>,
  spec: PreparedTaskSpec,
  roundIndex: number,
  maxRounds: number,
  minRevisions: number,
  completedRevisions: number,
): SimulatorResult {
  const rawRows = Array.isArray(raw.rubric_results) ? raw.rubric_results : [];
  const byCriterion = new Map<string, RubricResult>();
  const byIndex: RubricResult[] = [];
  for (const row of rawRows) {
    if (!row || typeof row !== "object") continue;
    const obj = row as Record<string, unknown>;
    const criterion = String(obj.criterion ?? "").trim();
    const normalized = {
      criterion,
      passed: Boolean(obj.passed),
      reason: String(obj.reason ?? "").trim(),
    };
    byIndex.push(normalized);
    if (criterion) byCriterion.set(criterion, normalized);
  }

  const rubricResults = spec.rubrics.map((criterion, i) => {
    const positional = byIndex[i] ? { ...byIndex[i], criterion } : undefined;
    return byCriterion.get(criterion) ?? positional ?? {
      criterion,
      passed: false,
      reason: "No simulator judgment returned.",
    };
  });

  let satisfied = Boolean(raw.satisfied);
  let feedback = String(raw.feedback ?? "").trim();
  if (completedRevisions < minRevisions && roundIndex < maxRounds - 1) {
    satisfied = false;
    feedback ||= "Can you make this more specific and closer to my stated preferences?";
    if (rubricResults.length > 0) {
      rubricResults[rubricResults.length - 1] = {
        ...rubricResults[rubricResults.length - 1]!,
        passed: false,
        reason: "Minimum revision pressure requested more feedback.",
      };
    }
  }
  if (!satisfied && !feedback && roundIndex < maxRounds - 1) {
    const failed = rubricResults.find((item) => !item.passed);
    feedback = failed ? `Please improve this against: ${failed.criterion}` : "Please make this more polished and directly useful.";
  }
  return { satisfied, feedback, rubric_results: rubricResults };
}

async function synthesizeSession(
  spec: PreparedTaskSpec,
  selections: { agent: ModelSelection; user: ModelSelection },
  args: Args,
  store: SessionStore,
  outputPath: string,
  taskIndex: number,
): Promise<SessionJson> {
  const workdir = join(dirname(outputPath), ".synth-workdirs", `${String(taskIndex + 1).padStart(3, "0")}-${slugify(spec.name)}`);
  mkdirSync(workdir, { recursive: true });

  const taskUnits: Record<string, unknown>[] = [];
  const agentTrajectories: Record<string, unknown>[] = [];
  const humanTrajectories: Record<string, unknown>[] = [];
  const priorFiles: OutputFile[] = [];
  let prompt = spec.task;
  let finalSimResult: SimulatorResult | undefined;
  let completedRevisions = 0;
  let modelLabel = "offline-demo";
  let sessionId = `synthetic-${randomUUID()}`;
  let runtime: AppHarnessRuntime | undefined;
  let appSession: Session | undefined;
  let firstAttempt: AgentAttempt | undefined;

  if (args.offlineDemo) {
    taskUnits.push(buildPlanningUnit(spec));
  } else {
    const headlessBashEnv = {
      MPLBACKEND: "Agg",
      QT_QPA_PLATFORM: "offscreen",
      PYTHONUNBUFFERED: "1",
    };
    runtime = new AppHarnessRuntime(store, headlessBashEnv);
    appSession = store.createSession({
      cwd: workdir,
      title: spec.name,
      prompt: spec.task,
      engine: "pi",
      autoContextInduction: false,
    });
    sessionId = appSession.id;
    const initial = await runInitialHarnessRound(spec, runtime, store, appSession);
    taskUnits.push(initial.planningUnit);
    firstAttempt = initial.attempt;
  }

  for (let roundIndex = 0; roundIndex < args.rounds; roundIndex++) {
    const attempt = args.offlineDemo
      ? offlineAgentRound(spec, prompt, roundIndex, priorFiles)
      : roundIndex === 0 && firstAttempt
        ? firstAttempt
        : await runFollowupHarnessRound(spec, runtime!, store, sessionId, prompt);
    modelLabel = attempt.modelLabel;
    priorFiles.push(...attempt.files);
    const sim = await runUserSimulator(
      spec,
      selections.user,
      roundIndex,
      args.rounds,
      args.minRevisions,
      completedRevisions,
      attempt.files,
      args.offlineDemo,
    );
    finalSimResult = sim;

    const agentPrompt = args.offlineDemo
      ? prompt
      : roundIndex === 0
        ? buildInitialAgentPrompt(spec)
        : buildFollowupAgentPrompt(spec, prompt);
    agentTrajectories.push({
      prompt: agentPrompt,
      messages: attempt.messages.length > 0 ? attempt.messages : [{ role: "user", content: prompt }],
      output_files: attempt.files,
      verifiers: verifierRows(sim),
    });

    if (sim.satisfied || roundIndex >= args.rounds - 1) break;
    humanTrajectories.push({
      type: "follow_up",
      round_index: roundIndex,
      prompt: sim.feedback,
      simulator_satisfied_before_follow_up: sim.satisfied,
    });
    prompt = sim.feedback;
    completedRevisions += 1;
  }

  const finalVerifiers = verifierRows(finalSimResult);
  taskUnits.push({
    intent: spec.name,
    agent_trajectories: agentTrajectories,
    human_trajectories: humanTrajectories,
    verifiers: finalVerifiers,
    reward: finalVerifiers.map((row) => (row.status ? 1.0 : 0.0)),
  });

  return {
    uuid: sessionId,
    name: spec.name,
    task: spec.task,
    initial_task_instruction: spec.task,
    model: modelLabel,
    system_prompt: TRAINING_SYSTEM_PROMPT,
    tool_schemas: TOOL_SCHEMAS,
    task_units: taskUnits,
    synthetic: {
      kind: "agent_user_simulator",
      created_at_unix: Math.floor(Date.now() / 1000),
      agent_model: modelLabel,
      rounds_requested: args.rounds,
      min_revisions: args.minRevisions,
      hidden_user_profile: spec.hiddenUserProfile,
      workdir,
      metadata: spec.metadata,
    },
  };
}

function buildPlanningUnit(spec: PreparedTaskSpec): Record<string, unknown> {
  const workflowTree = [
    {
      description: "Understand the task and success criteria",
      outputFiles: [],
      verifiers: ["The task requirements and constraints are reflected in the plan."],
      children: [],
    },
    {
      description: "Draft the primary artifact",
      outputFiles: [spec.outputPath],
      verifiers: spec.rubrics.slice(0, 2),
      children: [],
    },
    {
      description: "Revise the artifact using user feedback",
      outputFiles: [spec.outputPath],
      verifiers: spec.rubrics,
      children: [],
    },
  ];
  const prompt = WORKFLOW_PLAN_INSTRUCTION + spec.task;
  return {
    intent: "planning",
    agent_trajectories: [
      {
        prompt,
        messages: [
          { role: "user", content: prompt },
          workflowPlanMessage(workflowTree),
          {
            role: "tool",
            tool_call_id: "call_workflow_plan_0",
            name: "workflow_plan",
            content: "Registered workflow plan.",
          },
          { role: "assistant", content: "I will work through the planned artifact revisions." },
        ],
      },
    ],
    human_trajectories: [],
    verifiers: [],
    workflow_tree_generated: workflowTree,
    workflow_tree_final: workflowTree,
  };
}

function verifierRows(sim?: SimulatorResult): Array<{ criterion: string; status: boolean; reason?: string }> {
  return (sim?.rubric_results ?? []).map((row) => ({
    criterion: row.criterion,
    status: Boolean(row.passed),
    reason: row.reason,
  }));
}

function workflowPlanMessage(workflowTree: unknown[]): OpenAIMessage {
  return {
    role: "assistant",
    content: "",
    tool_calls: [
      {
        id: "call_workflow_plan_0",
        type: "function",
        function: {
          name: "workflow_plan",
          arguments: JSON.stringify({ tasks: workflowTree }),
        },
      },
    ],
  };
}

function assistantWriteMessage(content: string, files: OutputFile[]): OpenAIMessage {
  return {
    role: "assistant",
    content,
    tool_calls: files.map((file, i) => ({
      id: `call_write_${i}`,
      type: "function",
      function: {
        name: "write",
        arguments: JSON.stringify(file),
      },
    })),
  };
}

function fallbackOutputFiles(workdir: string, spec: PreparedTaskSpec, messages: unknown[]): OutputFile[] {
  const abs = join(workdir, spec.outputPath);
  if (existsSync(abs)) {
    try {
      return [{ path: spec.outputPath, content: readFileSync(abs, "utf8") }];
    } catch {
      // Fall through to assistant text.
    }
  }
  const primary = appPrimaryOutputPath(spec);
  const primaryAbs = join(workdir, primary);
  if (existsSync(primaryAbs)) {
    try {
      return [{ path: primary, content: readFileSync(primaryAbs, "utf8") }];
    } catch {
      // Fall through to assistant text.
    }
  }
  return [{
    path: primary,
    content: lastStreamAssistantText(messages) || lastAssistantText(messages) || "No artifact content was produced.",
  }];
}

function rowsToOpenAI(rows: Array<{ message: StreamMessage; snapshot: unknown }>): OpenAIMessage[] {
  return rows
    .map((row) => streamMessageToOpenAI(row.message, row.snapshot))
    .filter((message): message is OpenAIMessage => Boolean(message));
}

function streamMessageToOpenAI(message: StreamMessage, snapshot?: unknown): OpenAIMessage | null {
  if (message.type === "user_prompt") {
    return withEnvironment({ role: "user", content: message.prompt ?? "" }, snapshot);
  }
  if (message.type === "assistant" && "blocks" in message) {
    if (message.stopReason === "error") return null;
    const text: string[] = [];
    const thinking: string[] = [];
    const toolCalls: NonNullable<OpenAIMessage["tool_calls"]> = [];
    for (const block of message.blocks ?? []) {
      if (block.type === "text") text.push(block.text ?? "");
      else if (block.type === "thinking") thinking.push(block.thinking ?? "");
      else if (block.type === "tool_use") {
        toolCalls.push({
          id: block.id || randomUUID(),
          type: "function",
          function: {
            name: block.name || "tool",
            arguments: JSON.stringify(block.input ?? {}),
          },
        });
      }
    }
    const out: OpenAIMessage = { role: "assistant", content: text.filter(Boolean).join("\n") };
    if (thinking.length > 0) out.thinking = thinking.filter(Boolean).join("\n");
    if (toolCalls.length > 0) out.tool_calls = toolCalls;
    return withEnvironment(out, snapshot);
  }
  if (message.type === "tool_result" && "toolUseId" in message) {
    return withEnvironment({
      role: "tool",
      content: String(message.content ?? ""),
      tool_call_id: message.toolUseId ?? "",
      name: message.toolName ?? "tool",
    }, snapshot);
  }
  if (message.type === "assistant" && "message" in message) {
    const blocks = Array.isArray(message.message.content) ? message.message.content : [];
    const text: string[] = [];
    const thinking: string[] = [];
    const toolCalls: NonNullable<OpenAIMessage["tool_calls"]> = [];
    for (const block of blocks) {
      if (!block || typeof block !== "object") continue;
      if (block.type === "text") text.push(String(block.text ?? ""));
      else if (block.type === "thinking") thinking.push(String(block.thinking ?? ""));
      else if (block.type === "tool_use") {
        toolCalls.push({
          id: String(block.id ?? randomUUID()),
          type: "function",
          function: {
            name: String(block.name ?? "tool"),
            arguments: JSON.stringify(block.input ?? {}),
          },
        });
      }
    }
    const out: OpenAIMessage = { role: "assistant", content: text.filter(Boolean).join("\n") };
    if (thinking.length > 0) out.thinking = thinking.filter(Boolean).join("\n");
    if (toolCalls.length > 0) out.tool_calls = toolCalls;
    return withEnvironment(out, snapshot);
  }
  if (message.type === "user" && "message" in message) {
    const toolResults = Array.isArray(message.message.content) ? message.message.content : [];
    const first = toolResults[0];
    if (first && first.type === "tool_result") {
      return withEnvironment({
        role: "tool",
        content: contentToText(first.content),
        tool_call_id: first.tool_use_id ?? "",
        name: "tool",
      }, snapshot);
    }
  }
  return null;
}

function withEnvironment(message: OpenAIMessage, snapshot?: unknown): OpenAIMessage {
  if (snapshot && typeof snapshot === "object") {
    message.environment = snapshot;
  }
  return message;
}

function collectHarnessOutputFiles(session: Session, spec: PreparedTaskSpec): OutputFile[] {
  const cwd = session.cwd ?? process.cwd();
  const candidates = new Set<string>();
  candidates.add(spec.outputPath);
  candidates.add(appPrimaryOutputPath(spec));
  for (const node of flattenNodes(session.workflowTree ?? [])) {
    for (const rel of node.outputFiles ?? []) candidates.add(rel);
  }

  const files: OutputFile[] = [];
  const seen = new Set<string>();
  for (const raw of candidates) {
    const path = safeOutputPath(String(raw ?? ""), spec.name);
    if (!path || seen.has(path)) continue;
    seen.add(path);
    const abs = join(cwd, path);
    if (!existsSync(abs)) continue;
    try {
      files.push({ path, content: readFileSync(abs, "utf8") });
    } catch {
      // Ignore unreadable candidates; snapshots still record filesystem state.
    }
  }
  return files;
}

function workflowTreeToLlmNative(nodes: WorkflowNode[]): Array<Record<string, unknown>> {
  return nodes.map((node) => {
    const out: Record<string, unknown> = {
      description: node.description,
      outputFiles: node.outputFiles ?? [],
      verifiers: node.verifiers ?? [],
    };
    if (node.children?.length) out.children = workflowTreeToLlmNative(node.children);
    return out;
  });
}

function modelLabelFromRows(rows: Array<{ message: StreamMessage }>): string {
  for (const row of rows) {
    const message = row.message;
    if (message.type === "system_init" && "engine" in message && message.engine === "pi") {
      const provider = message.provider ?? "unknown";
      const model = message.model ?? "unknown";
      return `${provider}/${model}`;
    }
  }
  for (const row of rows) {
    const message = row.message;
    if (message.type === "assistant" && "model" in message && message.model) {
      return `${message.provider ?? "unknown"}/${message.model}`;
    }
  }
  return "app-harness-pi";
}

function lastStreamAssistantText(messages: unknown[]): string {
  for (const message of [...messages].reverse()) {
    if (!message || typeof message !== "object") continue;
    const msg = message as Record<string, unknown>;
    if (msg.type !== "assistant") continue;
    if (Array.isArray(msg.blocks)) {
      const text = msg.blocks
        .map((block) => {
          if (!block || typeof block !== "object") return "";
          const row = block as Record<string, unknown>;
          return row.type === "text" ? String(row.text ?? "") : "";
        })
        .filter(Boolean)
        .join("\n")
        .trim();
      if (text) return text;
    }
    const nested = (msg.message as { content?: unknown } | undefined)?.content;
    const text = contentToText(nested).trim();
    if (text) return text;
  }
  return "";
}

function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((part) => {
    if (!part || typeof part !== "object") return "";
    const p = part as Record<string, unknown>;
    if (p.type === "text") return String(p.text ?? "");
    if (p.type === "image") return `[image: ${String(p.mimeType ?? "")}]`;
    return "";
  }).filter(Boolean).join("\n");
}

function lastAssistantText(messages: unknown[]): string {
  for (const message of [...messages].reverse()) {
    if (!message || typeof message !== "object") continue;
    const msg = message as Record<string, unknown>;
    if (msg.role !== "assistant") continue;
    const text = contentToText(msg.content).trim();
    if (text) return text;
  }
  return "";
}

function parseJsonObject(text: string): Record<string, unknown> {
  const trimmed = text.trim().replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed as Record<string, unknown>;
  } catch {
    const start = trimmed.indexOf("{");
    const end = trimmed.lastIndexOf("}");
    if (start >= 0 && end > start) {
      const parsed = JSON.parse(trimmed.slice(start, end + 1)) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed as Record<string, unknown>;
    }
  }
  throw new Error(`Expected JSON object from Pi response, got: ${trimmed.slice(0, 300)}`);
}

function loadExistingOutput(path: string): { sessions: SessionJson[]; wrapper: boolean } {
  if (!existsSync(path)) return { sessions: [], wrapper: false };
  const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
  if (Array.isArray(parsed)) return { sessions: parsed as SessionJson[], wrapper: false };
  if (parsed && typeof parsed === "object" && Array.isArray((parsed as { sessions?: unknown }).sessions)) {
    return { sessions: (parsed as { sessions: SessionJson[] }).sessions, wrapper: true };
  }
  throw new Error(`${path} must contain a JSON array or {"sessions": [...]}`);
}

function writeSessions(path: string, sessions: SessionJson[], wrapper: boolean): void {
  mkdirSync(dirname(path), { recursive: true });
  const payload = wrapper ? { sessions } : sessions;
  writeFileSync(path, JSON.stringify(payload, null, 2) + "\n", "utf8");
}

function copyPiConfig(sourceUserDataDir: string, targetUserDataDir: string): void {
  const sourceDir = join(sourceUserDataDir, PI_AGENT_DIR_NAME);
  const targetDir = join(targetUserDataDir, PI_AGENT_DIR_NAME);
  mkdirSync(targetDir, { recursive: true });
  for (const file of PI_CONFIG_FILES) {
    const src = join(sourceDir, file);
    if (existsSync(src)) cpSync(src, join(targetDir, file));
  }
}

function hasPiConfig(userDataDir: string): boolean {
  const dir = join(userDataDir, PI_AGENT_DIR_NAME);
  return PI_CONFIG_FILES.some((file) => existsSync(join(dir, file)));
}

function discoverSourceUserDataDir(defaultUserDataDir: string, explicit?: string): string {
  if (explicit) {
    const resolved = resolve(explicit);
    if (!hasPiConfig(resolved)) {
      throw new Error(`No pi-agent config found under --source-user-data-dir: ${resolved}`);
    }
    return resolved;
  }

  const candidates = [
    defaultUserDataDir,
    join(homedir(), "Library", "Application Support", "agent-cowork"),
    join(homedir(), "Library", "Application Support", "Agent Cowork"),
  ];
  for (const candidate of candidates) {
    if (hasPiConfig(candidate)) return candidate;
  }
  return defaultUserDataDir;
}

async function main(): Promise<void> {
  loadDotenv({ path: join(process.cwd(), "scripts", ".env") });
  loadDotenv({ path: join(process.cwd(), ".env") });

  const args = parseArgs(process.argv.slice(2));
  const outputPath = resolve(args.output);
  const uiUserDataDir = app.getPath("userData");
  const sourceUserDataDir = discoverSourceUserDataDir(uiUserDataDir, args.sourceUserDataDir);
  const syntheticUserDataDir = args.useUiUserData
    ? sourceUserDataDir
    : resolve(args.userDataDir ?? join(dirname(outputPath), ".synth-pi-user-data"));
  if (!args.useUiUserData) copyPiConfig(sourceUserDataDir, syntheticUserDataDir);
  app.setPath("userData", syntheticUserDataDir);
  await app.whenReady();

  const selections = {
    agent: modelSelection(args, "agent"),
    user: modelSelection(args, "user"),
    task: modelSelection(args, "task"),
  };
  if (!args.offlineDemo) await applyAgentModelSelection(selections.agent);

  let rawSpecs: TaskSpec[] = args.task.map((task) => ({ task }));
  if (args.tasks) rawSpecs = rawSpecs.concat(loadTasksFile(args.tasks));
  if (args.generateTasks > 0) {
    if (args.offlineDemo) throw new Error("--generate-tasks requires model-backed Pi mode");
    rawSpecs = rawSpecs.concat(await generateTaskSpecs(selections.task, args.generateTasks, args.taskDomain));
  }
  if (rawSpecs.length === 0) {
    if (!args.offlineDemo) throw new Error("Provide --task, --tasks, or --generate-tasks.");
    rawSpecs = [
      {
        task: "Create a concise project status update for an engineering lead.",
        name: "Engineering Status Update",
        output_path: "engineering-status-update.md",
      },
    ];
  }

  const existing = args.append ? loadExistingOutput(outputPath) : { sessions: [], wrapper: false };
  const newSessions: SessionJson[] = [];
  mkdirSync(dirname(outputPath), { recursive: true });
  const store = new SessionStore(join(dirname(outputPath), ".synth-sessions.db"));
  try {
    for (let i = 0; i < rawSpecs.length; i++) {
      const prepared = await prepareTaskSpec(rawSpecs[i]!, selections.task, args.offlineDemo);
      console.error(`[${i + 1}/${rawSpecs.length}] synthesizing: ${prepared.name}`);
      newSessions.push(await synthesizeSession(prepared, selections, args, store, outputPath, i));
    }
  } finally {
    store.close();
  }
  writeSessions(outputPath, [...existing.sessions, ...newSessions], existing.wrapper);
  console.error(`Wrote ${outputPath} (${newSessions.length} new sessions, ${existing.sessions.length + newSessions.length} total)`);
}

main()
  .catch((error) => {
    console.error(error instanceof Error ? error.stack || error.message : String(error));
    app.exit(1);
  })
  .finally(() => {
    if (app.isReady()) app.quit();
  });
