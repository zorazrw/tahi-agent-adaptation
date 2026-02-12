import Database from "better-sqlite3";
import type { SessionStatus, StreamMessage } from "../types.js";

function parseSteps(raw: unknown): string[] | undefined {
  if (raw == null) return undefined;
  try {
    const arr = typeof raw === "string" ? JSON.parse(raw) : raw;
    return Array.isArray(arr) && arr.every((x) => typeof x === "string") ? arr : undefined;
  } catch {
    return undefined;
  }
}

function parseVerificationCriteria(raw: unknown): string[][] | undefined {
  if (raw == null) return undefined;
  try {
    const arr = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (!Array.isArray(arr)) return undefined;
    const ok = arr.every(
      (row) => Array.isArray(row) && row.every((x) => typeof x === "string")
    );
    return ok ? (arr as string[][]) : undefined;
  } catch {
    return undefined;
  }
}

function parseOutputFiles(raw: unknown): string[][] | undefined {
  if (raw == null) return undefined;
  try {
    const arr = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (!Array.isArray(arr)) return undefined;
    const ok = arr.every(
      (row) => Array.isArray(row) && row.every((x) => typeof x === "string")
    );
    return ok ? (arr as string[][]) : undefined;
  } catch {
    return undefined;
  }
}

function parseCompletedStepIndices(raw: unknown): number[] | undefined {
  if (raw == null) return undefined;
  try {
    const arr = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (!Array.isArray(arr)) return undefined;
    const ok = arr.every((x) => typeof x === "number" && Number.isInteger(x));
    return ok ? (arr as number[]) : undefined;
  } catch {
    return undefined;
  }
}

export type VerifierMark = "check" | "cross" | undefined;

function parseVerifierMarks(raw: unknown): VerifierMark[][] | undefined {
  if (raw == null) return undefined;
  try {
    const arr = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (!Array.isArray(arr)) return undefined;
    const result: VerifierMark[][] = [];
    for (const row of arr) {
      if (!Array.isArray(row)) return undefined;
      result.push(
        row.map((cell) => (cell === "check" || cell === "cross" ? cell : undefined))
      );
    }
    return result;
  } catch {
    return undefined;
  }
}

function serializeVerifierMarks(marks: VerifierMark[][] | undefined): string | null {
  if (!marks || !Array.isArray(marks)) return null;
  const arr = marks.map((row) =>
    Array.isArray(row) ? row.map((m) => (m === "check" || m === "cross" ? m : "")) : []
  );
  return JSON.stringify(arr);
}

export type PendingPermission = {
  toolUseId: string;
  toolName: string;
  input: unknown;
  resolve: (result: { behavior: "allow" | "deny"; updatedInput?: unknown; message?: string }) => void;
};

export type Session = {
  id: string;
  title: string;
  claudeSessionId?: string;
  status: SessionStatus;
  cwd?: string;
  allowedTools?: string;
  lastPrompt?: string;
  steps?: string[];
  completedStepIndices?: number[];
  outputFiles?: string[][];
  verificationCriteria?: string[][];
  verifierMarks?: VerifierMark[][];
  pendingPermissions: Map<string, PendingPermission>;
  abortController?: AbortController;
};

export type StoredSession = {
  id: string;
  title: string;
  status: SessionStatus;
  cwd?: string;
  allowedTools?: string;
  lastPrompt?: string;
  claudeSessionId?: string;
  steps?: string[];
  completedStepIndices?: number[];
  outputFiles?: string[][];
  verificationCriteria?: string[][];
  verifierMarks?: VerifierMark[][];
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

  createSession(options: { cwd?: string; allowedTools?: string; prompt?: string; title: string }): Session {
    const id = crypto.randomUUID();
    const now = Date.now();
    const session: Session = {
      id,
      title: options.title,
      status: "idle",
      cwd: options.cwd,
      allowedTools: options.allowedTools,
      lastPrompt: options.prompt,
      steps: [],
      outputFiles: [],
      verificationCriteria: [],
      verifierMarks: [],
      pendingPermissions: new Map()
    };
    this.sessions.set(id, session);
    const stepsJson = JSON.stringify(session.steps ?? []);
    const outputFilesJson = JSON.stringify(session.outputFiles ?? []);
    const verificationCriteriaJson = JSON.stringify(session.verificationCriteria ?? []);
    const verifierMarksJson = serializeVerifierMarks(session.verifierMarks);
    this.db
      .prepare(
        `insert into sessions
          (id, title, claude_session_id, status, cwd, allowed_tools, last_prompt, steps, verification_criteria, output_files, verifier_marks, created_at, updated_at)
         values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        id,
        session.title,
        session.claudeSessionId ?? null,
        session.status,
        session.cwd ?? null,
        session.allowedTools ?? null,
        session.lastPrompt ?? null,
        stepsJson,
        verificationCriteriaJson,
        outputFilesJson,
        verifierMarksJson,
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
        `select id, title, claude_session_id, status, cwd, allowed_tools, last_prompt, steps, verification_criteria, output_files, verifier_marks, completed_step_indices, created_at, updated_at
         from sessions
         order by updated_at desc`
      )
      .all() as Array<Record<string, unknown>>;
    return rows.map((row) => {
      const id = String(row.id);
      const mem = this.sessions.get(id);
      return {
        id,
        title: String(row.title),
        status: row.status as SessionStatus,
        cwd: row.cwd ? String(row.cwd) : undefined,
        allowedTools: row.allowed_tools ? String(row.allowed_tools) : undefined,
        lastPrompt: row.last_prompt ? String(row.last_prompt) : undefined,
        claudeSessionId: row.claude_session_id ? String(row.claude_session_id) : undefined,
        steps: parseSteps(row.steps) ?? mem?.steps ?? [],
        completedStepIndices: parseCompletedStepIndices(row.completed_step_indices) ?? mem?.completedStepIndices ?? [],
        outputFiles: parseOutputFiles(row.output_files) ?? mem?.outputFiles ?? [],
        verificationCriteria: parseVerificationCriteria(row.verification_criteria) ?? mem?.verificationCriteria ?? [],
        verifierMarks: parseVerifierMarks(row.verifier_marks) ?? mem?.verifierMarks ?? [],
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
        `select id, title, claude_session_id, status, cwd, allowed_tools, last_prompt, steps, verification_criteria, output_files, verifier_marks, completed_step_indices, created_at, updated_at
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
    return {
      session: {
        id: String(sessionRow.id),
        title: String(sessionRow.title),
        status: sessionRow.status as SessionStatus,
        cwd: sessionRow.cwd ? String(sessionRow.cwd) : undefined,
        allowedTools: sessionRow.allowed_tools ? String(sessionRow.allowed_tools) : undefined,
        lastPrompt: sessionRow.last_prompt ? String(sessionRow.last_prompt) : undefined,
        claudeSessionId: sessionRow.claude_session_id ? String(sessionRow.claude_session_id) : undefined,
        steps: parseSteps(sessionRow.steps) ?? mem?.steps ?? [],
        completedStepIndices: parseCompletedStepIndices(sessionRow.completed_step_indices) ?? mem?.completedStepIndices ?? [],
        outputFiles: parseOutputFiles(sessionRow.output_files) ?? mem?.outputFiles ?? [],
        verificationCriteria: parseVerificationCriteria(sessionRow.verification_criteria) ?? mem?.verificationCriteria ?? [],
        verifierMarks: parseVerifierMarks(sessionRow.verifier_marks) ?? mem?.verifierMarks ?? [],
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

  recordMessage(sessionId: string, message: StreamMessage): void {
    const id = ('uuid' in message && message.uuid) ? String(message.uuid) : crypto.randomUUID();
    this.db
      .prepare(
        `insert or ignore into messages (id, session_id, data, created_at) values (?, ?, ?, ?)`
      )
      .run(id, sessionId, JSON.stringify(message), Date.now());
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

  /** Persist only steps to DB (e.g. when session may not be in memory yet). */
  persistSteps(id: string, steps: string[]): void {
    const stepsJson = Array.isArray(steps) ? JSON.stringify(steps) : null;
    this.db
      .prepare(`update sessions set steps = ?, updated_at = ? where id = ?`)
      .run(stepsJson, Date.now(), id);
  }

  /** Persist only title to DB (e.g. when session may not be in memory yet). */
  persistTitle(id: string, title: string): void {
    this.db
      .prepare(`update sessions set title = ?, updated_at = ? where id = ?`)
      .run(title, Date.now(), id);
  }

  /** Persist only verification criteria to DB (e.g. when session may not be in memory yet). */
  persistVerificationCriteria(id: string, verificationCriteria: string[][]): void {
    const json = Array.isArray(verificationCriteria) ? JSON.stringify(verificationCriteria) : null;
    this.db
      .prepare(`update sessions set verification_criteria = ?, updated_at = ? where id = ?`)
      .run(json, Date.now(), id);
  }

  /** Persist only verifier marks to DB. */
  persistVerifierMarks(id: string, verifierMarks: VerifierMark[][]): void {
    const json = serializeVerifierMarks(verifierMarks);
    this.db
      .prepare(`update sessions set verifier_marks = ?, updated_at = ? where id = ?`)
      .run(json, Date.now(), id);
  }

  private persistSession(id: string, updates: Partial<Session>): void {
    const fields: string[] = [];
    const values: Array<string | number | null> = [];
    const updatable = {
      title: "title",
      claudeSessionId: "claude_session_id",
      status: "status",
      cwd: "cwd",
      allowedTools: "allowed_tools",
      lastPrompt: "last_prompt",
      steps: "steps",
      completedStepIndices: "completed_step_indices",
      outputFiles: "output_files",
      verificationCriteria: "verification_criteria",
      verifierMarks: "verifier_marks"
    } as const;

    for (const key of Object.keys(updates) as Array<keyof typeof updatable>) {
      const column = updatable[key];
      if (!column) continue;
      fields.push(`${column} = ?`);
      const value = updates[key];
      if (key === "steps") {
        values.push(Array.isArray(value) ? JSON.stringify(value) : null);
      } else if (key === "completedStepIndices") {
        values.push(Array.isArray(value) && value.every((x) => typeof x === "number") ? JSON.stringify(value) : null);
      } else if (key === "outputFiles" || key === "verificationCriteria") {
        values.push(Array.isArray(value) && value.every(Array.isArray) ? JSON.stringify(value) : null);
      } else if (key === "verifierMarks") {
        values.push(serializeVerifierMarks(value as VerifierMark[][]));
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
        claude_session_id text,
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
    try {
      this.db.exec(`alter table sessions add column steps text`);
    } catch {
      /* column may already exist */
    }
    try {
      this.db.exec(`alter table sessions add column verification_criteria text`);
    } catch {
      /* column may already exist */
    }
    try {
      this.db.exec(`alter table sessions add column output_files text`);
    } catch {
      /* column may already exist */
    }
    try {
      this.db.exec(`alter table sessions add column completed_step_indices text`);
    } catch {
      /* column may already exist */
    }
    try {
      this.db.exec(`alter table sessions add column verifier_marks text`);
    } catch {
      /* column may already exist */
    }
    this.db.exec(
      `create table if not exists messages (
        id text primary key,
        session_id text not null,
        data text not null,
        created_at integer not null,
        foreign key (session_id) references sessions(id)
      )`
    );
    this.db.exec(`create index if not exists messages_session_id on messages(session_id)`);
  }

  private loadSessions(): void {
    const rows = this.db
      .prepare(
        `select id, title, claude_session_id, status, cwd, allowed_tools, last_prompt, steps, verification_criteria, output_files, verifier_marks, completed_step_indices
         from sessions`
      )
      .all();
    for (const row of rows as Array<Record<string, unknown>>) {
      const session: Session = {
        id: String(row.id),
        title: String(row.title),
        claudeSessionId: row.claude_session_id ? String(row.claude_session_id) : undefined,
        status: row.status as SessionStatus,
        cwd: row.cwd ? String(row.cwd) : undefined,
        allowedTools: row.allowed_tools ? String(row.allowed_tools) : undefined,
        lastPrompt: row.last_prompt ? String(row.last_prompt) : undefined,
        steps: parseSteps(row.steps),
        completedStepIndices: parseCompletedStepIndices(row.completed_step_indices),
        outputFiles: parseOutputFiles(row.output_files),
        verificationCriteria: parseVerificationCriteria(row.verification_criteria),
        verifierMarks: parseVerifierMarks(row.verifier_marks),
        pendingPermissions: new Map()
      };
      this.sessions.set(session.id, session);
    }
  }

  close(): void {
    this.db.close();
  }
}
