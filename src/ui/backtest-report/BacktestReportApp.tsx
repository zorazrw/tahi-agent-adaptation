import { useEffect, useMemo, useState } from "react";
import {
  AlertCircleIcon,
  ArrowDownIcon,
  ArrowRightIcon,
  ArrowUpIcon,
  CheckCircle2Icon,
  FileJsonIcon,
  FilterIcon,
  SearchIcon,
  UploadIcon,
  XCircleIcon,
} from "lucide-react";

type JudgeVerdict = "accurate" | "partially_accurate" | "inaccurate";

type Prediction = {
  actionType: string;
  draftText: string;
  confidence: number;
  rationale: string;
  rawResponse?: string;
};

type Judgment = {
  verdict: JudgeVerdict;
  score: number;
  rationale: string;
};

type PredictionRun = {
  prediction: Prediction | null;
  predictionError?: string;
  judge: Judgment;
};

type ReportRow = {
  caseId: string;
  sessionName: string;
  sessionUuid: string;
  stepIndex: number;
  transcript: string;
  workflowSummary: string;
  actualActionType: string;
  actualActionText: string;
  baseline: PredictionRun | null;
  personalized: PredictionRun;
};

type Metrics = {
  totalCases: number;
  accurate: number;
  partiallyAccurate: number;
  inaccurate: number;
  averageJudgeScore: number;
  nullPredictions: number;
};

type BacktestReport = {
  generatedAt: string;
  profilePath?: string;
  sessionsPath: string;
  reportName: string;
  caseSelection: string;
  includeBaseline: boolean;
  sessionCount: number;
  totalCases: number;
  metrics: {
    baseline: Metrics | null;
    personalized: Metrics;
    delta: {
      averageJudgeScore: number;
      accurate: number;
      partiallyAccurate: number;
      inaccurate: number;
      nullPredictions: number;
    } | null;
  };
  rows: ReportRow[];
};

type DeltaFilter = "all" | "improved" | "regressed" | "tied";
type VerdictFilter = "all" | JudgeVerdict | "null";

const verdictLabel: Record<JudgeVerdict, string> = {
  accurate: "Accurate",
  partially_accurate: "Partial",
  inaccurate: "Inaccurate",
};

function shortText(value: string | undefined, fallback = "Empty"): string {
  const trimmed = String(value ?? "").trim();
  return trimmed || fallback;
}

function score(run: PredictionRun | null): number {
  return run?.judge.score ?? 0;
}

function rowDelta(row: ReportRow): DeltaFilter {
  if (!row.baseline) return "tied";
  const diff = score(row.personalized) - score(row.baseline);
  if (diff > 0.001) return "improved";
  if (diff < -0.001) return "regressed";
  return "tied";
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
}

function verdictClasses(verdict: JudgeVerdict): string {
  switch (verdict) {
    case "accurate":
      return "border-success/25 bg-success-light text-success";
    case "partially_accurate":
      return "border-primary/25 bg-primary-subtle text-primary";
    case "inaccurate":
      return "border-error/25 bg-error-light text-error";
  }
}

function VerdictIcon({ verdict }: { verdict: JudgeVerdict }) {
  if (verdict === "accurate") return <CheckCircle2Icon className="size-3.5" aria-hidden />;
  if (verdict === "partially_accurate") return <AlertCircleIcon className="size-3.5" aria-hidden />;
  return <XCircleIcon className="size-3.5" aria-hidden />;
}

function MetricCard({ label, metrics }: { label: string; metrics: Metrics | null }) {
  return (
    <section className="border border-border bg-surface px-4 py-3">
      <div className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-500">{label}</div>
      {metrics ? (
        <>
          <div className="mt-2 flex items-end gap-2">
            <span className="text-3xl font-semibold tabular-nums text-ink-900">
              {metrics.averageJudgeScore.toFixed(3)}
            </span>
            <span className="mb-1 text-xs text-ink-500">avg score</span>
          </div>
          <div className="mt-3 grid grid-cols-4 gap-2 text-xs">
            <span className="text-success">{metrics.accurate} hit</span>
            <span className="text-primary">{metrics.partiallyAccurate} partial</span>
            <span className="text-error">{metrics.inaccurate} miss</span>
            <span className="text-ink-500">{metrics.nullPredictions} null</span>
          </div>
        </>
      ) : (
        <div className="mt-3 text-sm text-ink-500">Not run</div>
      )}
    </section>
  );
}

function PredictionCell({ title, run }: { title: string; run: PredictionRun | null }) {
  if (!run) {
    return (
      <div className="min-h-36 border border-border bg-bg-100 p-3 text-sm text-ink-500">
        <div className="font-semibold text-ink-700">{title}</div>
        <div className="mt-3">No prediction run.</div>
      </div>
    );
  }

  const prediction = run.prediction;
  return (
    <div className="min-h-36 border border-border bg-surface p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-500">{title}</div>
          <div className="mt-1 font-mono text-sm font-semibold text-ink-900">
            {prediction?.actionType ?? "null"}
          </div>
        </div>
        <span
          className={`inline-flex shrink-0 items-center gap-1 border px-2 py-1 text-xs font-medium ${verdictClasses(
            run.judge.verdict,
          )}`}
        >
          <VerdictIcon verdict={run.judge.verdict} />
          {verdictLabel[run.judge.verdict]} {run.judge.score.toFixed(2)}
        </span>
      </div>
      <div className="mt-3 max-h-24 overflow-auto whitespace-pre-wrap break-words border-l-2 border-bg-400 pl-3 text-sm leading-5 text-ink-800">
        {prediction ? shortText(prediction.draftText) : run.predictionError ?? "Invalid prediction"}
      </div>
      <div className="mt-3 grid grid-cols-[88px_minmax(0,1fr)] gap-x-2 gap-y-1 text-xs text-ink-500">
        <span>Confidence</span>
        <span className="font-mono text-ink-800">
          {prediction ? prediction.confidence.toFixed(2) : "n/a"}
        </span>
        <span>Rationale</span>
        <span className="break-words text-ink-700">{shortText(prediction?.rationale, "No rationale")}</span>
        <span>Judge</span>
        <span className="break-words text-ink-700">{shortText(run.judge.rationale, "No rationale")}</span>
      </div>
    </div>
  );
}

function GroundTruthCell({ row }: { row: ReportRow }) {
  return (
    <div className="min-h-36 border border-border bg-bg-100 p-3">
      <div className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-500">Ground truth</div>
      <div className="mt-1 font-mono text-sm font-semibold text-ink-900">{row.actualActionType}</div>
      <div className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap break-words border-l-2 border-ink-900 pl-3 text-sm leading-5 text-ink-800">
        {shortText(row.actualActionText)}
      </div>
    </div>
  );
}

function DeltaBadge({ row }: { row: ReportRow }) {
  const delta = rowDelta(row);
  const diff = row.baseline ? score(row.personalized) - score(row.baseline) : 0;
  if (delta === "improved") {
    return (
      <span className="inline-flex items-center gap-1 border border-success/25 bg-success-light px-2 py-1 text-xs font-medium text-success">
        <ArrowUpIcon className="size-3" aria-hidden />
        +{diff.toFixed(2)}
      </span>
    );
  }
  if (delta === "regressed") {
    return (
      <span className="inline-flex items-center gap-1 border border-error/25 bg-error-light px-2 py-1 text-xs font-medium text-error">
        <ArrowDownIcon className="size-3" aria-hidden />
        {diff.toFixed(2)}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 border border-border bg-bg-100 px-2 py-1 text-xs font-medium text-ink-500">
      <ArrowRightIcon className="size-3" aria-hidden />
      tied
    </span>
  );
}

function Select({
  value,
  onChange,
  children,
  label,
}: {
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
  label: string;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-ink-600">
      {label}
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-9 border border-border bg-surface px-2 text-sm text-ink-900 outline-none focus:border-primary"
      >
        {children}
      </select>
    </label>
  );
}

export function BacktestReportApp() {
  const [report, setReport] = useState<BacktestReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [sessionFilter, setSessionFilter] = useState("all");
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>("all");
  const [actionFilter, setActionFilter] = useState("all");
  const [deltaFilter, setDeltaFilter] = useState<DeltaFilter>("all");
  const [query, setQuery] = useState("");

  useEffect(() => {
    const reportParam = new URLSearchParams(window.location.search).get("report");
    if (!reportParam) return;
    fetch(reportParam)
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to fetch ${reportParam}: ${response.status}`);
        return response.json();
      })
      .then((json) => {
        setReport(json as BacktestReport);
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const sessions = useMemo(() => {
    const seen = new Map<string, string>();
    for (const row of report?.rows ?? []) {
      seen.set(row.sessionUuid, row.sessionName);
    }
    return [...seen.entries()];
  }, [report]);

  const actionTypes = useMemo(() => {
    return [...new Set((report?.rows ?? []).map((row) => row.actualActionType))].sort();
  }, [report]);

  const filteredRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (report?.rows ?? []).filter((row) => {
      if (sessionFilter !== "all" && row.sessionUuid !== sessionFilter) return false;
      if (actionFilter !== "all" && row.actualActionType !== actionFilter) return false;
      if (deltaFilter !== "all" && rowDelta(row) !== deltaFilter) return false;
      if (verdictFilter === "null" && row.personalized.prediction) return false;
      if (
        verdictFilter !== "all" &&
        verdictFilter !== "null" &&
        row.personalized.judge.verdict !== verdictFilter &&
        row.baseline?.judge.verdict !== verdictFilter
      ) {
        return false;
      }
      if (!q) return true;
      return [row.sessionName, row.actualActionText, row.baseline?.prediction?.draftText, row.personalized.prediction?.draftText]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(q));
    });
  }, [actionFilter, deltaFilter, query, report?.rows, sessionFilter, verdictFilter]);

  const selectedRow = useMemo(() => {
    return report?.rows.find((row) => row.caseId === selectedCaseId) ?? filteredRows[0] ?? null;
  }, [filteredRows, report?.rows, selectedCaseId]);

  async function loadFile(file: File): Promise<void> {
    try {
      const text = await file.text();
      setReport(JSON.parse(text) as BacktestReport);
      setError(null);
      setSelectedCaseId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <main className="min-h-full bg-bg-100 text-ink-900">
      <header className="border-b border-border bg-surface">
        <div className="mx-auto flex max-w-[1800px] flex-col gap-5 px-5 py-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-primary">
                <FileJsonIcon className="size-4" aria-hidden />
                User Simulator Backtest
              </div>
              <h1 className="mt-2 text-2xl font-semibold tracking-normal text-ink-900">
                Ground truth vs baseline vs personalized prediction
              </h1>
              {report ? (
                <p className="mt-2 text-sm text-ink-600">
                  {report.reportName} · {formatDate(report.generatedAt)} · {report.sessionCount} sessions ·{" "}
                  {report.totalCases} turns
                </p>
              ) : (
                <p className="mt-2 text-sm text-ink-600">Load a generated backtest JSON report to inspect turn-level comparisons.</p>
              )}
            </div>
            <label className="inline-flex h-10 cursor-pointer items-center gap-2 border border-ink-900 bg-ink-900 px-4 text-sm font-medium text-white hover:bg-ink-800">
              <UploadIcon className="size-4" aria-hidden />
              Load report JSON
              <input
                type="file"
                accept="application/json,.json"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void loadFile(file);
                }}
              />
            </label>
          </div>

          {error ? (
            <div className="border border-error/25 bg-error-light px-3 py-2 text-sm text-error">{error}</div>
          ) : null}

          {report ? (
            <div className="grid gap-3 md:grid-cols-3">
              <MetricCard label="Without personalization" metrics={report.metrics.baseline} />
              <MetricCard label="With personalization" metrics={report.metrics.personalized} />
              <section className="border border-border bg-surface px-4 py-3">
                <div className="text-xs font-semibold uppercase tracking-[0.08em] text-ink-500">Delta</div>
                <div className="mt-2 flex items-end gap-2">
                  <span className="text-3xl font-semibold tabular-nums text-ink-900">
                    {report.metrics.delta ? report.metrics.delta.averageJudgeScore.toFixed(3) : "n/a"}
                  </span>
                  <span className="mb-1 text-xs text-ink-500">personalized minus baseline</span>
                </div>
                <div className="mt-3 text-xs text-ink-500">
                  Profile: <span className="font-mono text-ink-700">{report.profilePath ?? "none"}</span>
                </div>
              </section>
            </div>
          ) : null}
        </div>
      </header>

      {report ? (
        <div className="mx-auto grid max-w-[1800px] grid-cols-1 gap-4 px-5 py-4 xl:grid-cols-[minmax(0,1fr)_430px]">
          <section className="min-w-0">
            <div className="sticky top-0 z-10 border border-border bg-surface p-3">
              <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.1em] text-ink-500">
                <FilterIcon className="size-4" aria-hidden />
                Filters
              </div>
              <div className="grid gap-3 md:grid-cols-[1.3fr_1fr_1fr_1fr_1fr]">
                <label className="flex flex-col gap-1 text-xs font-medium text-ink-600">
                  Search
                  <div className="relative">
                    <SearchIcon className="pointer-events-none absolute left-2 top-2.5 size-4 text-ink-400" aria-hidden />
                    <input
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      className="h-9 w-full border border-border bg-surface pl-8 pr-2 text-sm text-ink-900 outline-none focus:border-primary"
                      placeholder="Text, session, prediction"
                    />
                  </div>
                </label>
                <Select label="Session" value={sessionFilter} onChange={setSessionFilter}>
                  <option value="all">All sessions</option>
                  {sessions.map(([id, name]) => (
                    <option key={id} value={id}>
                      {name}
                    </option>
                  ))}
                </Select>
                <Select label="Verdict" value={verdictFilter} onChange={(value) => setVerdictFilter(value as VerdictFilter)}>
                  <option value="all">All verdicts</option>
                  <option value="accurate">Accurate</option>
                  <option value="partially_accurate">Partial</option>
                  <option value="inaccurate">Inaccurate</option>
                  <option value="null">Personalized null</option>
                </Select>
                <Select label="Action" value={actionFilter} onChange={setActionFilter}>
                  <option value="all">All actions</option>
                  {actionTypes.map((action) => (
                    <option key={action} value={action}>
                      {action}
                    </option>
                  ))}
                </Select>
                <Select label="Personalized" value={deltaFilter} onChange={(value) => setDeltaFilter(value as DeltaFilter)}>
                  <option value="all">All deltas</option>
                  <option value="improved">Improved</option>
                  <option value="regressed">Regressed</option>
                  <option value="tied">Tied</option>
                </Select>
              </div>
            </div>

            <div className="mt-4 space-y-4">
              {filteredRows.map((row) => (
                <article
                  key={row.caseId}
                  className={`border bg-surface ${
                    selectedRow?.caseId === row.caseId ? "border-primary shadow-card" : "border-border"
                  }`}
                >
                  <button
                    type="button"
                    className="flex w-full items-center justify-between gap-3 border-b border-border px-3 py-2 text-left hover:bg-bg-100"
                    onClick={() => setSelectedCaseId(row.caseId)}
                  >
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-ink-900">{row.sessionName}</div>
                      <div className="mt-0.5 font-mono text-xs text-ink-500">
                        step {row.stepIndex} · {row.actualActionType}
                      </div>
                    </div>
                    <DeltaBadge row={row} />
                  </button>
                  <div className="grid gap-3 p-3 lg:grid-cols-3">
                    <GroundTruthCell row={row} />
                    <PredictionCell title="No profile" run={row.baseline} />
                    <PredictionCell title="With profile" run={row.personalized} />
                  </div>
                </article>
              ))}
            </div>
          </section>

          <aside className="min-w-0 border border-border bg-surface xl:sticky xl:top-4 xl:max-h-[calc(100vh-2rem)] xl:overflow-auto">
            <div className="border-b border-border px-4 py-3">
              <div className="text-xs font-semibold uppercase tracking-[0.1em] text-ink-500">Turn Context</div>
              <div className="mt-1 truncate text-sm font-semibold text-ink-900">
                {selectedRow ? `${selectedRow.sessionName} · step ${selectedRow.stepIndex}` : "No row selected"}
              </div>
            </div>
            {selectedRow ? (
              <div className="space-y-4 p-4">
                <section>
                  <h2 className="text-sm font-semibold text-ink-900">Transcript Prefix</h2>
                  <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words border border-border bg-bg-100 p-3 text-xs leading-5 text-ink-800">
                    {shortText(selectedRow.transcript, "No transcript")}
                  </pre>
                </section>
                <section>
                  <h2 className="text-sm font-semibold text-ink-900">Workflow Summary</h2>
                  <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words border border-border bg-bg-100 p-3 text-xs leading-5 text-ink-800">
                    {shortText(selectedRow.workflowSummary, "No workflow summary")}
                  </pre>
                </section>
              </div>
            ) : (
              <div className="p-4 text-sm text-ink-500">Select a row to inspect hidden-turn context.</div>
            )}
          </aside>
        </div>
      ) : (
        <section className="mx-auto max-w-3xl px-5 py-16">
          <div className="border border-dashed border-ink-400 bg-surface p-8 text-center">
            <FileJsonIcon className="mx-auto size-10 text-primary" aria-hidden />
            <h2 className="mt-4 text-lg font-semibold text-ink-900">No report loaded</h2>
            <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-ink-600">
              Use the file picker above, or serve a report and open this page with{" "}
              <span className="font-mono text-ink-900">?report=/path/to/report.json</span>.
            </p>
          </div>
        </section>
      )}
    </main>
  );
}
