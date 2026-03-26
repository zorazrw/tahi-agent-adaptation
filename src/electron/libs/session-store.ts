import Database from "better-sqlite3";
import { z } from "zod";
import type {
  AppPermissionResult,
  SessionEngine,
  SessionStatus,
  StreamMessage,
  WorkflowNode,
} from "../types.js";
import { migrateFromFlatSteps } from "./workflow-tree-utils.js";

// ── Zod schemas for JSON columns ──────────────────────────────────────

const verifierMarkCell = z
  .unknown()
  .transform((v) => (v === "check" || v === "cross" ? v : undefined));

/** Recursive Zod schema for WorkflowNode tree stored as JSON. */
const workflowNodeSchema: z.ZodType<WorkflowNode> = z.lazy(() =>
  z.object({
    id: z.string(),
    description: z.string(),
    outputFiles: z.array(z.string()),
    verifiers: z.array(z.string()),
    verifierMarks: z.array(verifierMarkCell),
    children: z.array(workflowNodeSchema),
    status: z.enum(["pending", "running", "completed", "error"]),
    depth: z.number(),
    resumePoint: z
      .union([
        z.object({ entryId: z.string() }),
        z.object({ uuid: z.string(), claudeSessionId: z.string() }),
      ])
      .optional(),
    originalOutputs: z
      .array(
        z.object({
          path: z.string(),
          content: z.string(),
        })
      )
      .optional(),
  })
);

const workflowTreeSchema = z.array(workflowNodeSchema);

// Legacy schemas for migration
const stepsSchema = z.array(z.string());
const stringGrid = z.array(z.array(z.string()));
const completedStepIndicesSchema = z.array(z.number().int());
const verifierMarksGridSchema = z.array(z.array(verifierMarkCell));

export type VerifierMark = "check" | "cross" | undefined;

/** Parse a DB TEXT column that stores JSON, returning `undefined` on null/invalid. */
function parseJsonColumn<T>(raw: unknown, schema: z.ZodType<T>): T | undefined {
  if (raw == null) return undefined;
  try {
    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    return schema.parse(parsed);
  } catch {
    return undefined;
  }
}

function serializeWorkflowTree(tree: WorkflowNode[] | undefined): string | null {
  if (!tree || !Array.isArray(tree)) return null;
  return JSON.stringify(tree);
}

export type PendingPermission = {
  toolUseId: string;
  toolName: string;
  input: unknown;
  resolve: (result: AppPermissionResult) => void;
};

export type Session = {
  id: string;
  title: string;
  engine: SessionEngine;
  claudeSessionId?: string;
  piSessionFile?: string;
  status: SessionStatus;
  cwd?: string;
  allowedTools?: string;
  lastPrompt?: string;
  workflowTree?: WorkflowNode[];
  verificationDepth?: number;
  pendingPermissions: Map<string, PendingPermission>;
  abortController?: AbortController;
};

export type StoredSession = {
  id: string;
  title: string;
  status: SessionStatus;
  engine: SessionEngine;
  cwd?: string;
  allowedTools?: string;
  lastPrompt?: string;
  claudeSessionId?: string;
  piSessionFile?: string;
  workflowTree?: WorkflowNode[];
  verificationDepth?: number;
  createdAt: number;
  updatedAt: number;
};

export type SessionHistory = {
  session: StoredSession;
  messages: StreamMessage[];
};

export class SessionStore {
  private sessions = new Map<string, Session>();
  private db: Database.Database;

  constructor(dbPath: string) {
    this.db = new Database(dbPath);
    this.initialize();
    this.loadSessions();
  }

  createSession(options: {
    cwd?: string;
    allowedTools?: string;
    prompt?: string;
    title: string;
    engine?: SessionEngine;
  }): Session {
    const id = crypto.randomUUID();
    const now = Date.now();
    const session: Session = {
      id,
      title: options.title,
      engine: options.engine ?? "pi",
      status: "idle",
      cwd: options.cwd,
      allowedTools: options.allowedTools,
      lastPrompt: options.prompt,
      workflowTree: [],
      verificationDepth: 0,
      pendingPermissions: new Map()
    };
    this.sessions.set(id, session);
    this.db
      .prepare(
        `insert into sessions
          (id, title, engine, claude_session_id, pi_session_file, status, cwd, allowed_tools, last_prompt, workflow_tree, verification_depth, created_at, updated_at)
         values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        id,
        session.title,
        session.engine,
        session.claudeSessionId ?? null,
        session.piSessionFile ?? null,
        session.status,
        session.cwd ?? null,
        session.allowedTools ?? null,
        session.lastPrompt ?? null,
        serializeWorkflowTree(session.workflowTree),
        session.verificationDepth ?? 0,
        now,
        now
      );
    return session;
  }

  getSession(id: string): Session | undefined {
    return this.sessions.get(id);
  }

  listSessions(): StoredSession[] {
    const rows = this.db
      .prepare(
        `select id, title, engine, claude_session_id, pi_session_file, status, cwd, allowed_tools, last_prompt,
                workflow_tree, verification_depth, steps, verification_criteria, output_files,
                completed_step_indices, verifier_marks, created_at, updated_at
         from sessions
         order by updated_at desc`
      )
      .all() as Array<Record<string, unknown>>;
    return rows.map((row) => {
      const id = String(row.id);
      const mem = this.sessions.get(id);
      let tree = parseJsonColumn(row.workflow_tree, workflowTreeSchema) ?? mem?.workflowTree;

      // Migration: if no workflow_tree but has steps, convert
      if ((!tree || tree.length === 0) && row.steps) {
        const steps = parseJsonColumn(row.steps, stepsSchema) ?? [];
        if (steps.length > 0) {
          const outputFiles = parseJsonColumn(row.output_files, stringGrid) ?? [];
          const verificationCriteria = parseJsonColumn(row.verification_criteria, stringGrid) ?? [];
          const verifierMarks = parseJsonColumn(row.verifier_marks, verifierMarksGridSchema) ?? [];
          const completedStepIndices = parseJsonColumn(row.completed_step_indices, completedStepIndicesSchema) ?? [];
          tree = migrateFromFlatSteps(steps, outputFiles, verificationCriteria, verifierMarks, completedStepIndices);
        }
      }

      return {
        id,
        title: String(row.title),
        status: row.status as SessionStatus,
        engine: row.engine ? (String(row.engine) as SessionEngine) : "legacy-claude",
        cwd: row.cwd ? String(row.cwd) : undefined,
        allowedTools: row.allowed_tools ? String(row.allowed_tools) : undefined,
        lastPrompt: row.last_prompt ? String(row.last_prompt) : undefined,
        claudeSessionId: row.claude_session_id ? String(row.claude_session_id) : undefined,
        piSessionFile: row.pi_session_file ? String(row.pi_session_file) : undefined,
        workflowTree: tree ?? [],
        verificationDepth: row.verification_depth != null ? Number(row.verification_depth) : 0,
        createdAt: Number(row.created_at),
        updatedAt: Number(row.updated_at)
      };
    });
  }

  listRecentCwds(limit = 8): string[] {
    const rows = this.db
      .prepare(
        `select cwd, max(updated_at) as latest
         from sessions
         where cwd is not null and trim(cwd) != ''
         group by cwd
         order by latest desc
         limit ?`
      )
      .all(limit) as Array<Record<string, unknown>>;
    return rows.map((row) => String(row.cwd));
  }

  getSessionHistory(id: string): SessionHistory | null {
    const sessionRow = this.db
      .prepare(
        `select id, title, engine, claude_session_id, pi_session_file, status, cwd, allowed_tools, last_prompt,
                workflow_tree, verification_depth, steps, verification_criteria, output_files,
                completed_step_indices, verifier_marks, created_at, updated_at
         from sessions
         where id = ?`
      )
      .get(id) as Record<string, unknown> | undefined;
    if (!sessionRow) return null;

    const messages = (this.db
      .prepare(
        `select data from messages where session_id = ? order by created_at asc`
      )
      .all(id) as Array<Record<string, unknown>>)
      .map((row) => JSON.parse(String(row.data)) as StreamMessage);

    const mem = this.sessions.get(id);
    let tree = parseJsonColumn(sessionRow.workflow_tree, workflowTreeSchema) ?? mem?.workflowTree;

    // Migration: if no workflow_tree but has steps, convert
    if ((!tree || tree.length === 0) && sessionRow.steps) {
      const steps = parseJsonColumn(sessionRow.steps, stepsSchema) ?? [];
      if (steps.length > 0) {
        const outputFiles = parseJsonColumn(sessionRow.output_files, stringGrid) ?? [];
        const verificationCriteria = parseJsonColumn(sessionRow.verification_criteria, stringGrid) ?? [];
        const verifierMarks = parseJsonColumn(sessionRow.verifier_marks, verifierMarksGridSchema) ?? [];
        const completedStepIndices = parseJsonColumn(sessionRow.completed_step_indices, completedStepIndicesSchema) ?? [];
        tree = migrateFromFlatSteps(steps, outputFiles, verificationCriteria, verifierMarks, completedStepIndices);
      }
    }

    return {
      session: {
        id: String(sessionRow.id),
        title: String(sessionRow.title),
        status: sessionRow.status as SessionStatus,
        engine: sessionRow.engine ? (String(sessionRow.engine) as SessionEngine) : "legacy-claude",
        cwd: sessionRow.cwd ? String(sessionRow.cwd) : undefined,
        allowedTools: sessionRow.allowed_tools ? String(sessionRow.allowed_tools) : undefined,
        lastPrompt: sessionRow.last_prompt ? String(sessionRow.last_prompt) : undefined,
        claudeSessionId: sessionRow.claude_session_id ? String(sessionRow.claude_session_id) : undefined,
        piSessionFile: sessionRow.pi_session_file ? String(sessionRow.pi_session_file) : undefined,
        workflowTree: tree ?? [],
        verificationDepth: sessionRow.verification_depth != null ? Number(sessionRow.verification_depth) : 0,
        createdAt: Number(sessionRow.created_at),
        updatedAt: Number(sessionRow.updated_at)
      },
      messages
    };
  }

  updateSession(id: string, updates: Partial<Session>): Session | undefined {
    const session = this.sessions.get(id);
    if (!session) return undefined;
    Object.assign(session, updates);
    this.persistSession(id, updates);
    return session;
  }

  setAbortController(id: string, controller: AbortController | undefined): void {
    const session = this.sessions.get(id);
    if (!session) return;
    session.abortController = controller;
  }

  /** Inserts message row; returns the row id for optional snapshot writes. */
  recordMessage(sessionId: string, message: StreamMessage): string {
    const id =
      ("uuid" in message && typeof message.uuid === "string" && message.uuid) ||
      ("id" in message && typeof message.id === "string" && message.id) ||
      crypto.randomUUID();
    this.db
      .prepare(
        `insert or ignore into messages (id, session_id, data, created_at, state_snapshot) values (?, ?, ?, ?, ?)`
      )
      .run(id, sessionId, JSON.stringify(message), Date.now(), null);
    return id;
  }

  /** Attach per-step environment (workflow + files + brain memory/skill maps) for export; overwrites when set again. */
  writeMessageSnapshot(
    messageId: string,
    snapshot: { workflow?: unknown; file?: unknown; verifier?: unknown; memory?: unknown; skill?: unknown }
  ): void {
    this.db
      .prepare(`update messages set state_snapshot = ? where id = ?`)
      .run(JSON.stringify(snapshot), messageId);
  }

  replaceMessages(sessionId: string, messages: StreamMessage[]): void {
    const tx = this.db.transaction((items: StreamMessage[]) => {
      this.db.prepare(`delete from messages where session_id = ?`).run(sessionId);
      const insert = this.db.prepare(
        `insert into messages (id, session_id, data, created_at) values (?, ?, ?, ?)`
      );
      for (const message of items) {
        const id =
          ("uuid" in message && typeof message.uuid === "string" && message.uuid) ||
          ("id" in message && typeof message.id === "string" && message.id) ||
          crypto.randomUUID();
        insert.run(id, sessionId, JSON.stringify(message), Date.now());
      }
    });
    tx(messages);
  }

  deleteSession(id: string): boolean {
    const existing = this.sessions.get(id);
    if (existing) {
      this.sessions.delete(id);
    }
    this.db.prepare(`delete from messages where session_id = ?`).run(id);
    const result = this.db.prepare(`delete from sessions where id = ?`).run(id);
    const removedFromDb = result.changes > 0;
    return removedFromDb || Boolean(existing);
  }

  /** Persist workflow tree to DB. */
  persistWorkflowTree(id: string, workflowTree: WorkflowNode[]): void {
    const json = serializeWorkflowTree(workflowTree);
    this.db
      .prepare(`update sessions set workflow_tree = ?, updated_at = ? where id = ?`)
      .run(json, Date.now(), id);
  }

  /** Last persisted workflow tree from the DB (for classifying user edits vs stale in-memory state). */
  getPersistedWorkflowTree(id: string): WorkflowNode[] | undefined {
    const row = this.db
      .prepare(`select workflow_tree from sessions where id = ?`)
      .get(id) as { workflow_tree: string | null } | undefined;
    if (!row?.workflow_tree) return undefined;
    return parseJsonColumn(row.workflow_tree, workflowTreeSchema);
  }

  /** Persist verification depth to DB. */
  persistVerificationDepth(id: string, verificationDepth: number): void {
    this.db
      .prepare(`update sessions set verification_depth = ?, updated_at = ? where id = ?`)
      .run(verificationDepth, Date.now(), id);
  }

  /** Persist only title to DB. */
  persistTitle(id: string, title: string): void {
    this.db
      .prepare(`update sessions set title = ?, updated_at = ? where id = ?`)
      .run(title, Date.now(), id);
  }

  private persistSession(id: string, updates: Partial<Session>): void {
    const fields: string[] = [];
    const values: Array<string | number | null> = [];
    const updatable = {
      title: "title",
      engine: "engine",
      claudeSessionId: "claude_session_id",
      piSessionFile: "pi_session_file",
      status: "status",
      cwd: "cwd",
      allowedTools: "allowed_tools",
      lastPrompt: "last_prompt",
      workflowTree: "workflow_tree",
      verificationDepth: "verification_depth"
    } as const;

    const jsonKeys = new Set<string>(["workflowTree"]);

    for (const key of Object.keys(updates) as Array<keyof typeof updatable>) {
      const column = updatable[key];
      if (!column) continue;
      fields.push(`${column} = ?`);
      const value = updates[key];
      if (jsonKeys.has(key)) {
        values.push(value != null ? JSON.stringify(value) : null);
      } else if (key === "verificationDepth") {
        values.push(value != null ? Number(value) : 0);
      } else {
        values.push(value === undefined ? null : (value as string));
      }
    }

    if (fields.length === 0) return;
    fields.push("updated_at = ?");
    values.push(Date.now());
    values.push(id);
    this.db
      .prepare(`update sessions set ${fields.join(", ")} where id = ?`)
      .run(...values);
  }

  private initialize(): void {
    this.db.exec(`pragma journal_mode = WAL;`);
    this.db.exec(
      `create table if not exists sessions (
        id text primary key,
        title text,
        engine text,
        claude_session_id text,
        pi_session_file text,
        status text not null,
        cwd text,
        allowed_tools text,
        last_prompt text,
        steps text,
        verification_criteria text,
        created_at integer not null,
        updated_at integer not null
      )`
    );
    // Legacy columns (kept for migration)
    try { this.db.exec(`alter table sessions add column steps text`); } catch { /* exists */ }
    try { this.db.exec(`alter table sessions add column verification_criteria text`); } catch { /* exists */ }
    try { this.db.exec(`alter table sessions add column output_files text`); } catch { /* exists */ }
    try { this.db.exec(`alter table sessions add column completed_step_indices text`); } catch { /* exists */ }
    try { this.db.exec(`alter table sessions add column verifier_marks text`); } catch { /* exists */ }
    try { this.db.exec(`alter table sessions add column step_resume_points text`); } catch { /* exists */ }
    // New columns
    try { this.db.exec(`alter table sessions add column engine text`); } catch { /* exists */ }
    try { this.db.exec(`alter table sessions add column workflow_tree text`); } catch { /* exists */ }
    try { this.db.exec(`alter table sessions add column verification_depth integer default 0`); } catch { /* exists */ }
    try { this.db.exec(`alter table sessions add column pi_session_file text`); } catch { /* exists */ }
    try { this.db.exec(`update sessions set engine = 'legacy-claude' where engine is null or trim(engine) = ''`); } catch { /* ignore */ }

    this.db.exec(
      `create table if not exists messages (
        id text primary key,
        session_id text not null,
        data text not null,
        created_at integer not null,
        foreign key (session_id) references sessions(id)
      )`
    );
    try {
      this.db.exec(`alter table messages add column state_snapshot text`);
    } catch {
      /* column exists */
    }
    this.db.exec(`create index if not exists messages_session_id on messages(session_id)`);
  }

  private loadSessions(): void {
    const rows = this.db
      .prepare(
        `select id, title, engine, claude_session_id, pi_session_file, status, cwd, allowed_tools, last_prompt,
                workflow_tree, verification_depth, steps, verification_criteria, output_files,
                verifier_marks, completed_step_indices
         from sessions`
      )
      .all();
    for (const row of rows as Array<Record<string, unknown>>) {
      let tree = parseJsonColumn(row.workflow_tree, workflowTreeSchema);

      // Migration: if no workflow_tree but has steps, convert
      if ((!tree || tree.length === 0) && row.steps) {
        const steps = parseJsonColumn(row.steps, stepsSchema) ?? [];
        if (steps.length > 0) {
          const outputFiles = parseJsonColumn(row.output_files, stringGrid) ?? [];
          const verificationCriteria = parseJsonColumn(row.verification_criteria, stringGrid) ?? [];
          const verifierMarks = parseJsonColumn(row.verifier_marks, verifierMarksGridSchema) ?? [];
          const completedStepIndices = parseJsonColumn(row.completed_step_indices, completedStepIndicesSchema) ?? [];
          tree = migrateFromFlatSteps(steps, outputFiles, verificationCriteria, verifierMarks, completedStepIndices);
        }
      }

      const session: Session = {
        id: String(row.id),
        title: String(row.title),
        engine: row.engine ? (String(row.engine) as SessionEngine) : "legacy-claude",
        claudeSessionId: row.claude_session_id ? String(row.claude_session_id) : undefined,
        piSessionFile: row.pi_session_file ? String(row.pi_session_file) : undefined,
        status: row.status as SessionStatus,
        cwd: row.cwd ? String(row.cwd) : undefined,
        allowedTools: row.allowed_tools ? String(row.allowed_tools) : undefined,
        lastPrompt: row.last_prompt ? String(row.last_prompt) : undefined,
        workflowTree: tree ?? [],
        verificationDepth: row.verification_depth != null ? Number(row.verification_depth) : 0,
        pendingPermissions: new Map()
      };
      this.sessions.set(session.id, session);
    }
  }

  /**
   * Returns the UUID of the last SDK **assistant** message for a session.
   */
  getLastAssistantMessageUuid(sessionId: string): string | undefined {
    const rows = this.db
      .prepare(`SELECT data FROM messages WHERE session_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 50`)
      .all(sessionId) as Array<Record<string, unknown>>;
    for (const row of rows) {
      const msg = JSON.parse(String(row.data));
      if (msg.uuid && msg.type === "assistant") {
        return String(msg.uuid);
      }
    }
    return undefined;
  }

  /** Deletes all messages for a session that come after the message with the given UUID. */
  deleteMessagesAfter(sessionId: string, afterMessageUuid: string): void {
    const row = this.db
      .prepare(`SELECT rowid FROM messages WHERE id = ? AND session_id = ?`)
      .get(afterMessageUuid, sessionId) as Record<string, unknown> | undefined;
    if (!row) return;
    const rowid = Number(row.rowid);
    this.db
      .prepare(`DELETE FROM messages WHERE session_id = ? AND rowid > ?`)
      .run(sessionId, rowid);
  }

  /** Loads all messages for a session from DB (for reset after truncation). */
  getMessages(sessionId: string): StreamMessage[] {
    return (this.db
      .prepare(`SELECT data FROM messages WHERE session_id = ? ORDER BY created_at ASC, rowid ASC`)
      .all(sessionId) as Array<Record<string, unknown>>)
      .map((row) => JSON.parse(String(row.data)) as StreamMessage);
  }

  close(): void {
    this.db.close();
  }
}
