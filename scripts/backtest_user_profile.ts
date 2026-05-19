import { mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname, resolve } from "path";
import type {
  PredictedUserActionSuggestion,
  PredictionJudgeVerdict,
  UserPredictionJudgeResult,
} from "../src/electron/types.ts";
import {
  buildTranscriptFromExportedSteps,
  buildWorkflowSummaryFromExportedSteps,
  judgePredictedAction,
  loadUserProfileMarkdown,
  predictNextUserAction,
} from "../src/electron/libs/user-predict.ts";

type TrajectoryStep = {
  actor: "user" | "agent";
  action: string;
  environment?: unknown;
};

type SessionBlob = {
  uuid: string;
  name: string;
  trajectory: TrajectoryStep[];
};

type CaseSelection = "all" | "representative";

type CliOptions = {
  datasetPath?: string;
  caseSelection: CaseSelection;
  casesPerSession: number;
  includeBaseline: boolean;
  reportName: string;
  outDir: string;
};

type BacktestCase = {
  session: SessionBlob;
  stepIndex: number;
};

type PredictionRun = {
  prediction: PredictedUserActionSuggestion | null;
  predictionError?: string;
  judge: UserPredictionJudgeResult;
};

type BacktestRow = {
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

type MetricsBucket = {
  totalCases: number;
  accurate: number;
  partiallyAccurate: number;
  inaccurate: number;
  averageJudgeScore: number;
  nullPredictions: number;
  byActualActionType: Record<
    string,
    {
      totalCases: number;
      accurate: number;
      partiallyAccurate: number;
      inaccurate: number;
      averageJudgeScore: number;
      nullPredictions: number;
    }
  >;
};

function actionType(action: string): string {
  const m = String(action).match(/^([A-Za-z_]+)/);
  return m?.[1] ?? "unknown";
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    caseSelection: "all",
    casesPerSession: 2,
    includeBaseline: true,
    reportName: "user-simulator-backtest",
    outDir: "docs/backtests",
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      const value = argv[++i];
      if (!value) throw new Error(`Missing value for ${arg}`);
      return value;
    };

    switch (arg) {
      case "--case-selection": {
        const value = next();
        if (value !== "all" && value !== "representative") {
          throw new Error("--case-selection must be all or representative");
        }
        options.caseSelection = value;
        break;
      }
      case "--cases-per-session":
        options.casesPerSession = Number(next());
        if (!Number.isFinite(options.casesPerSession) || options.casesPerSession < 1) {
          throw new Error("--cases-per-session must be a positive number");
        }
        break;
      case "--include-baseline":
        options.includeBaseline = true;
        break;
      case "--no-baseline":
        options.includeBaseline = false;
        break;
      case "--report-name":
        options.reportName = next();
        break;
      case "--out-dir":
        options.outDir = next();
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
      default:
        if (arg.startsWith("-")) throw new Error(`Unknown option: ${arg}`);
        if (!options.datasetPath) {
          options.datasetPath = arg;
        } else {
          throw new Error(`Unexpected extra argument: ${arg}`);
        }
    }
  }

  return options;
}

function printHelp(): void {
  console.log(`Usage:
  bun scripts/backtest_user_profile.ts DATASET.json [options]

Options:
  --case-selection all|representative  Which user turns to evaluate (default: all)
  --cases-per-session N                Sampling count for representative mode (default: 2)
  --include-baseline                   Include empty-profile baseline (default)
  --no-baseline                        Skip baseline calls
  --report-name NAME                   Output basename (default: user-simulator-backtest)
  --out-dir DIR                        Output directory (default: docs/backtests)
`);
}

function loadSessions(path: string): SessionBlob[] {
  const raw = JSON.parse(readFileSync(path, "utf8")) as unknown;
  if (!Array.isArray(raw)) throw new Error("Expected an array of sessions.");
  return raw as SessionBlob[];
}

function ensureDirFor(filePath: string): void {
  mkdirSync(dirname(filePath), { recursive: true });
}

function pickAllUserCases(session: SessionBlob): BacktestCase[] {
  return session.trajectory
    .map((step, index) => ({ step, index }))
    .filter(({ step, index }) => step.actor === "user" && index > 0)
    .map(({ index }) => ({ session, stepIndex: index }));
}

function pickRepresentativeCases(session: SessionBlob, count: number): BacktestCase[] {
  const userIndices = pickAllUserCases(session).map((testCase) => testCase.stepIndex);

  if (userIndices.length <= count) {
    return userIndices.map((stepIndex) => ({ session, stepIndex }));
  }

  const picked: number[] = [];
  for (let i = 0; i < count; i++) {
    const position = Math.round((i * (userIndices.length - 1)) / Math.max(1, count - 1));
    const idx = userIndices[position];
    if (!picked.includes(idx)) picked.push(idx);
  }
  return picked.map((stepIndex) => ({ session, stepIndex }));
}

function pickCases(session: SessionBlob, options: CliOptions): BacktestCase[] {
  if (options.caseSelection === "representative") {
    return pickRepresentativeCases(session, options.casesPerSession);
  }
  return pickAllUserCases(session);
}

function failedPredictionRun(message: string): PredictionRun {
  return {
    prediction: null,
    predictionError: message,
    judge: {
      verdict: "inaccurate",
      score: 0,
      rationale: message,
    },
  };
}

async function runPredictionCase(args: {
  cwd: string;
  userProfileMarkdown: string;
  transcript: string;
  workflowSummary: string;
  sessionTitle: string;
  actualActionType: string;
  actualActionText: string;
}): Promise<PredictionRun> {
  try {
    const prediction = await predictNextUserAction({
      cwd: args.cwd,
      userProfileMarkdown: args.userProfileMarkdown,
      transcript: args.transcript,
      workflowSummary: args.workflowSummary,
      sessionTitle: args.sessionTitle,
    });

    if (!prediction) {
      return failedPredictionRun("Prediction model returned no valid executable action.");
    }

    const judge = await judgePredictedAction({
      cwd: args.cwd,
      transcript: args.transcript,
      workflowSummary: args.workflowSummary,
      prediction,
      actualActionType: args.actualActionType,
      actualActionText: args.actualActionText,
    });

    return { prediction, judge };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return failedPredictionRun(message);
  }
}

function verdictCount(rows: BacktestRow[], selector: (row: BacktestRow) => PredictionRun | null, verdict: PredictionJudgeVerdict): number {
  return rows.filter((row) => selector(row)?.judge.verdict === verdict).length;
}

function buildMetrics(rows: BacktestRow[], selector: (row: BacktestRow) => PredictionRun | null): MetricsBucket {
  const usableRows = rows.filter((row) => selector(row));
  const scoreSum = usableRows.reduce((sum, row) => sum + (selector(row)?.judge.score ?? 0), 0);
  const byActualActionType: MetricsBucket["byActualActionType"] = {};

  for (const row of usableRows) {
    const run = selector(row);
    if (!run) continue;
    const key = row.actualActionType;
    byActualActionType[key] ??= {
      totalCases: 0,
      accurate: 0,
      partiallyAccurate: 0,
      inaccurate: 0,
      averageJudgeScore: 0,
      nullPredictions: 0,
    };
    const bucket = byActualActionType[key];
    bucket.totalCases += 1;
    if (run.judge.verdict === "accurate") bucket.accurate += 1;
    if (run.judge.verdict === "partially_accurate") bucket.partiallyAccurate += 1;
    if (run.judge.verdict === "inaccurate") bucket.inaccurate += 1;
    bucket.averageJudgeScore += run.judge.score;
    if (!run.prediction) bucket.nullPredictions += 1;
  }

  for (const bucket of Object.values(byActualActionType)) {
    bucket.averageJudgeScore = bucket.totalCases > 0 ? bucket.averageJudgeScore / bucket.totalCases : 0;
  }

  return {
    totalCases: usableRows.length,
    accurate: verdictCount(rows, selector, "accurate"),
    partiallyAccurate: verdictCount(rows, selector, "partially_accurate"),
    inaccurate: verdictCount(rows, selector, "inaccurate"),
    averageJudgeScore: usableRows.length > 0 ? scoreSum / usableRows.length : 0,
    nullPredictions: usableRows.filter((row) => !selector(row)?.prediction).length,
    byActualActionType,
  };
}

function jsonForMarkdown(value: string, maxLength = 600): string {
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length > maxLength ? `${compact.slice(0, maxLength)}...` : compact;
}

function predictionSummary(run: PredictionRun | null): string {
  if (!run) return "_not run_";
  const prediction = run.prediction;
  const action = prediction ? `\`${prediction.actionType}\`` : "`null`";
  const draft = prediction?.draftText ? ` - ${jsonForMarkdown(prediction.draftText, 240)}` : "";
  return `${action}${draft} | ${run.judge.verdict} (${run.judge.score.toFixed(3)})`;
}

async function main(): Promise<void> {
  const cwd = process.cwd();
  const options = parseArgs(process.argv.slice(2));
  const datasetPath = options.datasetPath ?? process.env.USER_PREDICTION_DATASET;
  if (!datasetPath) {
    throw new Error("Pass a trajectory export path as argv[2] or set USER_PREDICTION_DATASET.");
  }

  const sessionsPath = resolve(cwd, datasetPath);
  const reportMdPath = resolve(cwd, options.outDir, `${options.reportName}.md`);
  const reportJsonPath = resolve(cwd, options.outDir, `${options.reportName}.json`);
  const sessions = loadSessions(sessionsPath);
  const { profileMarkdown, profilePath } = loadUserProfileMarkdown(cwd);
  if (!profileMarkdown.trim()) {
    throw new Error("USER_PROFILE.md was not found.");
  }

  const rows: BacktestRow[] = [];
  const cases = sessions.flatMap((session) => pickCases(session, options));

  for (let caseIndex = 0; caseIndex < cases.length; caseIndex++) {
    const testCase = cases[caseIndex];
    const session = testCase.session;
    const i = testCase.stepIndex;
    const current = session.trajectory[i];
    const prefix = session.trajectory.slice(0, i);
    const transcript = buildTranscriptFromExportedSteps(prefix, 14);
    const workflowSummary = buildWorkflowSummaryFromExportedSteps(prefix);
    const actualActionType = actionType(current.action);
    const actualActionText = current.action;

    console.error(
      `Backtest ${caseIndex + 1}/${cases.length}: ${session.name} step ${i} (${actualActionType})`
    );

    const common = {
      cwd,
      transcript,
      workflowSummary,
      sessionTitle: session.name,
      actualActionType,
      actualActionText,
    };

    const baseline = options.includeBaseline
      ? await runPredictionCase({ ...common, userProfileMarkdown: "" })
      : null;
    const personalized = await runPredictionCase({
      ...common,
      userProfileMarkdown: profileMarkdown,
    });

    rows.push({
      caseId: `${session.uuid}:${i}`,
      sessionName: session.name,
      sessionUuid: session.uuid,
      stepIndex: i,
      transcript,
      workflowSummary,
      actualActionType,
      actualActionText,
      baseline,
      personalized,
    });
  }

  const baselineMetrics = options.includeBaseline ? buildMetrics(rows, (row) => row.baseline) : null;
  const personalizedMetrics = buildMetrics(rows, (row) => row.personalized);
  const payload = {
    generatedAt: new Date().toISOString(),
    profilePath,
    sessionsPath,
    reportName: options.reportName,
    caseSelection: options.caseSelection,
    casesPerSession: options.caseSelection === "representative" ? options.casesPerSession : null,
    includeBaseline: options.includeBaseline,
    sessionCount: sessions.length,
    totalCases: rows.length,
    metrics: {
      baseline: baselineMetrics,
      personalized: personalizedMetrics,
      delta: baselineMetrics
        ? {
            averageJudgeScore: personalizedMetrics.averageJudgeScore - baselineMetrics.averageJudgeScore,
            accurate: personalizedMetrics.accurate - baselineMetrics.accurate,
            partiallyAccurate:
              personalizedMetrics.partiallyAccurate - baselineMetrics.partiallyAccurate,
            inaccurate: personalizedMetrics.inaccurate - baselineMetrics.inaccurate,
            nullPredictions: personalizedMetrics.nullPredictions - baselineMetrics.nullPredictions,
          }
        : null,
    },
    rows,
  };

  const markdown = [
    "# User Simulator Backtest",
    "",
    `Generated: ${payload.generatedAt}`,
    `Profile: ${profilePath ?? "USER_PROFILE.md"}`,
    `Data: ${sessionsPath}`,
    `Sessions: ${sessions.length}`,
    `Case selection: ${options.caseSelection}`,
    `Baseline included: ${options.includeBaseline ? "yes" : "no"}`,
    "",
    "## Summary",
    "",
    `- Total cases: ${rows.length}`,
    baselineMetrics
      ? `- Baseline average judge score: ${baselineMetrics.averageJudgeScore.toFixed(3)} (${baselineMetrics.accurate} accurate, ${baselineMetrics.partiallyAccurate} partial, ${baselineMetrics.inaccurate} inaccurate, ${baselineMetrics.nullPredictions} null)`
      : "- Baseline: not run",
    `- Personalized average judge score: ${personalizedMetrics.averageJudgeScore.toFixed(3)} (${personalizedMetrics.accurate} accurate, ${personalizedMetrics.partiallyAccurate} partial, ${personalizedMetrics.inaccurate} inaccurate, ${personalizedMetrics.nullPredictions} null)`,
    baselineMetrics
      ? `- Personalized score delta: ${(personalizedMetrics.averageJudgeScore - baselineMetrics.averageJudgeScore).toFixed(3)}`
      : "",
    "",
    "## Notes",
    "",
    "- Each case hides one real user action and predicts it from only the prior trajectory prefix.",
    "- Baseline uses the same model/context with an empty user profile.",
    "- Personalized uses the same model/context plus the local USER_PROFILE.md.",
    "- Judging is LLM-based and should be interpreted as directional.",
    "",
    "## Cases",
    "",
    ...rows.flatMap((row, index) => [
      `### ${index + 1}. ${row.sessionName} / step ${row.stepIndex}`,
      "",
      `- Ground truth: \`${row.actualActionType}\` - ${jsonForMarkdown(row.actualActionText)}`,
      `- Baseline: ${predictionSummary(row.baseline)}`,
      `- Personalized: ${predictionSummary(row.personalized)}`,
      "",
    ]),
  ]
    .filter(Boolean)
    .join("\n");

  ensureDirFor(reportMdPath);
  ensureDirFor(reportJsonPath);
  writeFileSync(reportMdPath, markdown, "utf8");
  writeFileSync(reportJsonPath, JSON.stringify(payload, null, 2), "utf8");

  console.log(JSON.stringify(payload, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
