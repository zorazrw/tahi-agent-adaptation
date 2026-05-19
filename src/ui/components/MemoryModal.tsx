import { useEffect, useState, useCallback, useRef } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { Spinner } from "./Spinner";
import { ViewToggle, useViewToggle } from "./file-renderers/ViewToggle";
import type { ClientEvent, ServerEvent } from "../types";

interface MemoryModalProps {
  onClose: () => void;
  /** Active task session; used to append a ``brain_edit`` timeline row after a successful save. */
  taskSessionId: string | null;
}

type MemorySection = {
  fileName: string;
  title: string;
  content: string;
};

const IPC_TIMEOUT_MS = 15_000;

/** Stem for a new file (becomes stem + ".md"); matches server-safe names. */
const STEM_RE = /^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/;

function titleFromMemoryFileName(fileName: string): string {
  const base = fileName.replace(/\.md$/i, "");
  if (!base) return fileName;
  return base
    .split(/[._-]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}

function BrainMdField({
  content,
  onChange,
  ariaLabel,
}: {
  content: string;
  onChange: (value: string) => void;
  ariaLabel: string;
}) {
  const [mode, setMode] = useViewToggle("preview");
  return (
    <div className="flex flex-col gap-2 min-h-0">
      <div className="flex items-center justify-end gap-2 shrink-0">
        <span className="text-[11px] text-muted-foreground mr-auto">Markdown</span>
        <ViewToggle mode={mode} onChange={setMode} />
      </div>
      {mode === "preview" ? (
        <div
          className="md-prose min-h-0 h-[min(400px,45vh)] overflow-y-auto overscroll-y-contain rounded-lg border border-ink-900/10 bg-surface px-3 py-2"
          tabIndex={0}
          role="region"
          aria-label={`${ariaLabel} preview`}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex, rehypeHighlight, rehypeRaw]}
          >
            {content}
          </ReactMarkdown>
        </div>
      ) : (
        <textarea
          className="min-h-[120px] h-[min(400px,45vh)] max-h-[min(400px,45vh)] w-full overflow-y-auto rounded-lg border border-ink-900/10 bg-surface px-3 py-2 text-sm text-ink-800 font-mono leading-relaxed resize-none focus:outline-none focus:ring-2 focus:ring-primary/25 focus:border-primary/30"
          value={content}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
          aria-label={ariaLabel}
        />
      )}
    </div>
  );
}

type BrainLoadResult = {
  dir: string;
  sections: MemorySection[];
  skillsDir: string;
  skillSections: MemorySection[];
};

function parseSectionRows(arr: unknown): MemorySection[] {
  if (!Array.isArray(arr)) return [];
  const sections: MemorySection[] = [];
  for (const item of arr) {
    if (!item || typeof item !== "object") continue;
    const row = item as Record<string, unknown>;
    const fileName = row.fileName;
    const content = row.content;
    if (typeof fileName !== "string" || typeof content !== "string") continue;
    const title =
      typeof row.title === "string" ? row.title : titleFromMemoryFileName(fileName);
    sections.push({ fileName, title, content });
  }
  return sections;
}

/** Handles current IPC shape, missing arrays, and legacy `{ path, content }` (memory only). */
function normalizeBrainLoad(raw: unknown): BrainLoadResult {
  if (!raw || typeof raw !== "object") {
    return { dir: "", sections: [], skillsDir: "", skillSections: [] };
  }
  const o = raw as Record<string, unknown>;
  if (Array.isArray(o.sections)) {
    const dir = typeof o.dir === "string" ? o.dir : "";
    const skillsDir = typeof o.skillsDir === "string" ? o.skillsDir : "";
    return {
      dir,
      sections: parseSectionRows(o.sections),
      skillsDir,
      skillSections: parseSectionRows(o.skillSections),
    };
  }
  if (typeof o.path === "string" && typeof o.content === "string") {
    const path = o.path;
    const parts = path.split(/[/\\]/);
    const base = parts.pop() ?? "memory.md";
    const dir = parts.length > 0 ? parts.join("/") : "";
    return {
      dir,
      sections: [{ fileName: base, title: titleFromMemoryFileName(base), content: o.content }],
      skillsDir: "",
      skillSections: [],
    };
  }
  return {
    dir: typeof o.dir === "string" ? o.dir : "",
    sections: [],
    skillsDir: typeof o.skillsDir === "string" ? o.skillsDir : "",
    skillSections: [],
  };
}

async function readBrainFromMain(): Promise<BrainLoadResult> {
  if (typeof window.electron.getMemoryMd === "function") {
    const raw = await window.electron.getMemoryMd();
    return normalizeBrainLoad(raw);
  }
  return new Promise((resolve, reject) => {
    const requestId = crypto.randomUUID();
    const timeoutId = window.setTimeout(() => {
      unsub();
      reject(new Error("Timed out loading memory. Fully quit and restart Agent Cowork, then try again."));
    }, IPC_TIMEOUT_MS);
    const unsub = window.electron.onServerEvent((ev: ServerEvent) => {
      if (ev.type === "memory.readResult" && ev.payload.requestId === requestId) {
        window.clearTimeout(timeoutId);
        unsub();
        resolve(
          normalizeBrainLoad({
            dir: ev.payload.dir,
            sections: ev.payload.sections,
            skillsDir: ev.payload.skillsDir,
            skillSections: ev.payload.skillSections,
          })
        );
      }
    });
    window.electron.sendClientEvent({ type: "memory.read", payload: { requestId } });
  });
}

function writeMemoryToMain(payload: {
  sections: { fileName: string; content: string }[];
  deletedFileNames?: string[];
}): Promise<{ success: boolean; error?: string }> {
  if (typeof window.electron.saveMemoryMd === "function") {
    return window.electron.saveMemoryMd(payload);
  }
  return new Promise((resolve, reject) => {
    const requestId = crypto.randomUUID();
    const timeoutId = window.setTimeout(() => {
      unsub();
      reject(new Error("Timed out saving memory. Fully quit and restart Agent Cowork, then try again."));
    }, IPC_TIMEOUT_MS);
    const unsub = window.electron.onServerEvent((ev: ServerEvent) => {
      if (ev.type === "memory.writeResult" && ev.payload.requestId === requestId) {
        window.clearTimeout(timeoutId);
        unsub();
        resolve({ success: ev.payload.success, error: ev.payload.error });
      }
    });
    const msg: ClientEvent = {
      type: "memory.write",
      payload: { requestId, sections: payload.sections, deletedFileNames: payload.deletedFileNames },
    };
    window.electron.sendClientEvent(msg);
  });
}

function writeSkillToMain(payload: {
  sections: { fileName: string; content: string }[];
  deletedFileNames?: string[];
}): Promise<{ success: boolean; error?: string }> {
  if (typeof window.electron.saveSkillMd === "function") {
    return window.electron.saveSkillMd(payload);
  }
  return new Promise((resolve, reject) => {
    const requestId = crypto.randomUUID();
    const timeoutId = window.setTimeout(() => {
      unsub();
      reject(new Error("Timed out saving skills. Fully quit and restart Agent Cowork, then try again."));
    }, IPC_TIMEOUT_MS);
    const unsub = window.electron.onServerEvent((ev: ServerEvent) => {
      if (ev.type === "skills.writeResult" && ev.payload.requestId === requestId) {
        window.clearTimeout(timeoutId);
        unsub();
        resolve({ success: ev.payload.success, error: ev.payload.error });
      }
    });
    const msg: ClientEvent = {
      type: "skills.write",
      payload: { requestId, sections: payload.sections, deletedFileNames: payload.deletedFileNames },
    };
    window.electron.sendClientEvent(msg);
  });
}

function notifyBrainEditRecorded(taskSessionId: string | null | undefined) {
  const sid = taskSessionId?.trim();
  if (!sid) return;
  if (typeof window.electron?.sendClientEvent === "function") {
    window.electron.sendClientEvent({ type: "session.recordBrainEdit", payload: { sessionId: sid } });
  }
}

export function MemoryModal({ onClose, taskSessionId }: MemoryModalProps) {
  const [memoriesDir, setMemoriesDir] = useState<string | null>(null);
  const [skillsDir, setSkillsDir] = useState<string | null>(null);
  const [sections, setSections] = useState<MemorySection[]>([]);
  const [skillSections, setSkillSections] = useState<MemorySection[]>([]);
  const [pendingDeletes, setPendingDeletes] = useState<string[]>([]);
  const [skillPendingDeletes, setSkillPendingDeletes] = useState<string[]>([]);
  /** File names added in this session before save — removing these does not delete on disk. */
  const unstagedNewFilesRef = useRef<Set<string>>(new Set());
  const unstagedNewSkillFilesRef = useRef<Set<string>>(new Set());
  const [newSlug, setNewSlug] = useState("");
  const [newSkillSlug, setNewSkillSlug] = useState("");
  const [addError, setAddError] = useState<string | null>(null);
  const [skillAddError, setSkillAddError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveOk, setSaveOk] = useState(false);

  const load = useCallback(async () => {
    const { dir, sections: initial, skillsDir: sd, skillSections: skInitial } = await readBrainFromMain();
    setMemoriesDir(dir || null);
    setSkillsDir(sd || null);
    setSections((initial ?? []).map((s) => ({ ...s })));
    setSkillSections((skInitial ?? []).map((s) => ({ ...s })));
    setPendingDeletes([]);
    setSkillPendingDeletes([]);
    unstagedNewFilesRef.current = new Set();
    unstagedNewSkillFilesRef.current = new Set();
    setLoadError(null);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await load();
      } catch (e) {
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : "Could not load memory files");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [load]);

  function updateSectionContent(fileName: string, content: string) {
    setSections((prev) =>
      prev.map((s) => (s.fileName === fileName ? { ...s, content } : s))
    );
    setSaveOk(false);
  }

  function updateSkillSectionContent(fileName: string, content: string) {
    setSkillSections((prev) =>
      prev.map((s) => (s.fileName === fileName ? { ...s, content } : s))
    );
    setSaveOk(false);
  }

  function removeSection(fileName: string) {
    setSections((prev) => prev.filter((s) => s.fileName !== fileName));
    if (unstagedNewFilesRef.current.has(fileName)) {
      unstagedNewFilesRef.current.delete(fileName);
    } else {
      setPendingDeletes((prev) => (prev.includes(fileName) ? prev : [...prev, fileName]));
    }
    setSaveOk(false);
  }

  function removeSkillSection(fileName: string) {
    setSkillSections((prev) => prev.filter((s) => s.fileName !== fileName));
    if (unstagedNewSkillFilesRef.current.has(fileName)) {
      unstagedNewSkillFilesRef.current.delete(fileName);
    } else {
      setSkillPendingDeletes((prev) => (prev.includes(fileName) ? prev : [...prev, fileName]));
    }
    setSaveOk(false);
  }

  function handleAddSection() {
    setAddError(null);
    let stem = newSlug.trim();
    if (stem.toLowerCase().endsWith(".md")) {
      stem = stem.slice(0, -3);
    }
    if (!stem) {
      setAddError("Enter a file name without .md (e.g. preferences or coding-style).");
      return;
    }
    if (!STEM_RE.test(stem)) {
      setAddError(
        "Use letters, numbers, dots, underscores, or hyphens. Must start with a letter or number."
      );
      return;
    }
    const fileName = `${stem}.md`;
    if (sections.some((s) => s.fileName === fileName)) {
      setAddError("A section with that name already exists.");
      return;
    }
    const title = titleFromMemoryFileName(fileName);
    unstagedNewFilesRef.current.add(fileName);
    setSections((prev) => [...prev, { fileName, title, content: "" }]);
    setNewSlug("");
    setSaveOk(false);
  }

  function handleAddSkillSection() {
    setSkillAddError(null);
    let stem = newSkillSlug.trim();
    if (stem.toLowerCase().endsWith(".md")) {
      stem = stem.slice(0, -3);
    }
    if (!stem) {
      setSkillAddError("Enter a file name without .md (e.g. run-tests or code-review).");
      return;
    }
    if (!STEM_RE.test(stem)) {
      setSkillAddError(
        "Use letters, numbers, dots, underscores, or hyphens. Must start with a letter or number."
      );
      return;
    }
    const fileName = `${stem}.md`;
    if (skillSections.some((s) => s.fileName === fileName)) {
      setSkillAddError("A skill file with that name already exists.");
      return;
    }
    const title = titleFromMemoryFileName(fileName);
    unstagedNewSkillFilesRef.current.add(fileName);
    setSkillSections((prev) => [...prev, { fileName, title, content: "" }]);
    setNewSkillSlug("");
    setSaveOk(false);
  }

  async function handleSave() {
    setSaving(true);
    setSaveError(null);
    setSaveOk(false);
    try {
      const memResult = await writeMemoryToMain({
        sections: sections.map((s) => ({
          fileName: String(s.fileName ?? "").trim(),
          content: s.content == null ? "" : String(s.content),
        })),
        deletedFileNames: pendingDeletes.length > 0 ? pendingDeletes : undefined,
      });
      if (!memResult.success) {
        setSaveError(memResult.error ?? "Failed to save memory");
        return;
      }
      const skillResult = await writeSkillToMain({
        sections: skillSections.map((s) => ({
          fileName: String(s.fileName ?? "").trim(),
          content: s.content == null ? "" : String(s.content),
        })),
        deletedFileNames: skillPendingDeletes.length > 0 ? skillPendingDeletes : undefined,
      });
      if (!skillResult.success) {
        setSaveError(skillResult.error ?? "Failed to save skills");
        return;
      }
      notifyBrainEditRecorded(taskSessionId);
      setPendingDeletes([]);
      setSkillPendingDeletes([]);
      setSaveOk(true);
      await load();
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog.Root open onOpenChange={(open) => { if (!open) onClose(); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-ink-900/20 backdrop-blur-sm animate-fade-in" />
        <Dialog.Content className="fixed inset-0 z-50 flex items-center justify-center px-4 py-8">
          <div className="w-full max-w-2xl rounded-2xl border border-ink-900/5 bg-surface shadow-elevated animate-scale-in flex flex-col max-h-[88vh]">
            <div className="flex items-center justify-between px-6 pt-6 pb-0 shrink-0 gap-3">
              <div>
                <Dialog.Title className="text-base font-semibold text-ink-800">Brain</Dialog.Title>
                <p className="text-xs text-muted-foreground mt-1 pr-4">
                  <strong className="text-ink-700">Memory</strong> files are merged into every model prompt.{" "}
                  <strong className="text-ink-700">Skills</strong> are top-level <code className="text-[11px] bg-ink-900/5 px-1 rounded">*.md</code> in
                  your skills folder (optional YAML frontmatter with <code className="text-[11px]">name</code> /{" "}
                  <code className="text-[11px]">description</code>) and are synced for the agent SDK.
                </p>
              </div>
              <Dialog.Close asChild>
                <button
                  type="button"
                  className="rounded-full p-1.5 text-muted-foreground hover:bg-surface-tertiary hover:text-ink-700 transition-colors shrink-0"
                  aria-label="Close"
                >
                  <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6L6 18M6 6l12 12" />
                  </svg>
                </button>
              </Dialog.Close>
            </div>

            <div className="flex-1 min-h-0 flex flex-col px-6 pb-6 pt-4 gap-4 overflow-y-auto">
              {loading ? (
                <div className="flex justify-center py-16">
                  <Spinner />
                </div>
              ) : loadError ? (
                <p className="text-sm text-error">{loadError}</p>
              ) : (
                <>
                  {memoriesDir ? (
                    <p className="text-[11px] text-muted-foreground font-mono truncate shrink-0" title={memoriesDir}>
                      Memory: {memoriesDir}
                    </p>
                  ) : null}

                  <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-600 shrink-0">Memory</h3>

                  <div className="flex flex-col gap-4">
                    {sections.map((sec) => (
                      <section
                        key={sec.fileName}
                        className="rounded-xl border border-ink-900/10 bg-surface-cream/80 p-4 flex flex-col gap-2"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <h3 className="text-sm font-semibold text-ink-800">
                              {sec.title || titleFromMemoryFileName(sec.fileName)}
                            </h3>
                            <p className="text-[11px] text-muted-foreground font-mono mt-0.5">{sec.fileName}</p>
                          </div>
                          <button
                            type="button"
                            className="shrink-0 text-xs text-muted-foreground hover:text-error transition-colors"
                            onClick={() => removeSection(sec.fileName)}
                          >
                            Remove
                          </button>
                        </div>
                        <BrainMdField
                          content={sec.content}
                          onChange={(value) => updateSectionContent(sec.fileName, value)}
                          ariaLabel={`Content of ${sec.fileName}`}
                        />
                      </section>
                    ))}
                  </div>

                  <div className="rounded-xl border border-dashed border-ink-900/15 p-4 flex flex-col gap-2 shrink-0">
                    <p className="text-xs font-medium text-ink-700">Add section</p>
                    <p className="text-[11px] text-muted-foreground">
                      Creates <code className="text-[11px] bg-ink-900/5 px-1 rounded">&lt;name&gt;.md</code>. Example:{" "}
                      <code className="text-[11px]">preferences</code> or <code className="text-[11px]">coding-style</code>.
                    </p>
                    <div className="flex flex-wrap items-center gap-2">
                      <input
                        type="text"
                        className="flex-1 min-w-[140px] rounded-lg border border-ink-900/10 bg-surface px-3 py-2 text-sm text-ink-800 focus:outline-none focus:ring-2 focus:ring-primary/25"
                        placeholder="e.g. preferences (becomes preferences.md)"
                        value={newSlug}
                        onChange={(e) => {
                          setNewSlug(e.target.value);
                          setAddError(null);
                        }}
                      />
                      <button
                        type="button"
                        className="rounded-lg border border-ink-900/10 bg-surface px-3 py-2 text-sm font-medium text-ink-700 hover:bg-ink-900/5 transition-colors"
                        onClick={handleAddSection}
                      >
                        Add file
                      </button>
                    </div>
                    {addError ? <p className="text-xs text-error">{addError}</p> : null}
                  </div>

                  <div className="border-t border-ink-900/10 pt-4 mt-2 flex flex-col gap-4">
                    {skillsDir ? (
                      <p className="text-[11px] text-muted-foreground font-mono truncate shrink-0" title={skillsDir}>
                        Skills: {skillsDir}
                      </p>
                    ) : null}
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-600 shrink-0">Skills</h3>
                    <div className="flex flex-col gap-4">
                      {skillSections.map((sec) => (
                        <section
                          key={`skill-${sec.fileName}`}
                          className="rounded-xl border border-ink-900/10 bg-primary-subtle/20 p-4 flex flex-col gap-2"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div>
                              <h3 className="text-sm font-semibold text-ink-800">
                                {sec.title || titleFromMemoryFileName(sec.fileName)}
                              </h3>
                              <p className="text-[11px] text-muted-foreground font-mono mt-0.5">{sec.fileName}</p>
                            </div>
                            <button
                              type="button"
                              className="shrink-0 text-xs text-muted-foreground hover:text-error transition-colors"
                              onClick={() => removeSkillSection(sec.fileName)}
                            >
                              Remove
                            </button>
                          </div>
                          <BrainMdField
                            content={sec.content}
                            onChange={(value) => updateSkillSectionContent(sec.fileName, value)}
                            ariaLabel={`Skill ${sec.fileName}`}
                          />
                        </section>
                      ))}
                    </div>

                    <div className="rounded-xl border border-dashed border-ink-900/15 p-4 flex flex-col gap-2 shrink-0">
                      <p className="text-xs font-medium text-ink-700">Add skill file</p>
                      <p className="text-[11px] text-muted-foreground">
                        Creates a new <code className="text-[11px] bg-ink-900/5 px-1 rounded">*.md</code> in the skills folder.
                        Use YAML frontmatter for <code className="text-[11px]">name</code> and <code className="text-[11px]">description</code> if you like.
                      </p>
                      <div className="flex flex-wrap items-center gap-2">
                        <input
                          type="text"
                          className="flex-1 min-w-[140px] rounded-lg border border-ink-900/10 bg-surface px-3 py-2 text-sm text-ink-800 focus:outline-none focus:ring-2 focus:ring-primary/25"
                          placeholder="e.g. run-tests (becomes run-tests.md)"
                          value={newSkillSlug}
                          onChange={(e) => {
                            setNewSkillSlug(e.target.value);
                            setSkillAddError(null);
                          }}
                        />
                        <button
                          type="button"
                          className="rounded-lg border border-ink-900/10 bg-surface px-3 py-2 text-sm font-medium text-ink-700 hover:bg-ink-900/5 transition-colors"
                          onClick={handleAddSkillSection}
                        >
                          Add file
                        </button>
                      </div>
                      {skillAddError ? <p className="text-xs text-error">{skillAddError}</p> : null}
                    </div>
                  </div>

                  <div className="flex items-center gap-3 flex-wrap shrink-0 pt-1">
                    <button
                      type="button"
                      className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-hover transition-colors disabled:opacity-50"
                      onClick={handleSave}
                      disabled={saving}
                    >
                      {saving ? "Saving…" : "Save all"}
                    </button>
                    {saveError ? <span className="text-sm text-error">{saveError}</span> : null}
                    {saveOk && !saveError ? <span className="text-sm text-primary">Saved.</span> : null}
                  </div>
                </>
              )}
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
