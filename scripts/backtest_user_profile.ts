import { mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname, resolve } from "path";
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

type BacktestRow = {
  sessionName: string;
  sessionUuid: string;
  stepIndex: number;
  predictedActionType: string;
  predictedDraftText: string;
  predictedConfidence: number;
  actualActionType: string;
  actualActionText: string;
  judgeVerdict: "accurate" | "partially_accurate" | "inaccurate";
  judgeScore: number;
  judgeRationale: string;
};

type BacktestCase = {
  session: SessionBlob;
  stepIndex: number;
};

function actionType(action: string): string {
  const m = String(action).match(/^([A-Za-z_]+)/);
  return m?.[1] ?? "unknown";
}

function loadSessions(path: string): SessionBlob[] {
  const raw = JSON.parse(readFileSync(path, "utf8")) as unknown;
  if (!Array.isArray(raw)) throw new Error("Expected an array of sessions.");
  return raw as SessionBlob[];
}

function ensureDirFor(filePath: string): void {
  mkdirSync(dirname(filePath), { recursive: true });
}

function pickRepresentativeCases(session: SessionBlob, count: number): BacktestCase[] {
  const userIndices = session.trajectory
    .map((step, index) => ({ step, index }))
    .filter(({ step, index }) => step.actor === "user" && index > 0)
    .map(({ index }) => index);

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

async function main(): Promise<void> {
  const cwd = process.cwd();
  const sessionsPath = resolve(cwd, "assets/zora_chats.json");
  const reportMdPath = resolve(cwd, "docs/backtests/zora-user-profile-backtest.md");
  const reportJsonPath = resolve(cwd, "docs/backtests/zora-user-profile-backtest.json");
  const sessions = loadSessions(sessionsPath);
  const { profileMarkdown, profilePath } = loadUserProfileMarkdown(cwd);
  if (!profileMarkdown.trim()) {
    throw new Error("USER_PROFILE.md was not found.");
  }

  const rows: BacktestRow[] = [];
  const cases = sessions.flatMap((session) => pickRepresentativeCases(session, 2));

  for (const testCase of cases) {
    const session = testCase.session;
    const i = testCase.stepIndex;
    const current = session.trajectory[i];
    const prefix = session.trajectory.slice(0, i);
    const transcript = buildTranscriptFromExportedSteps(prefix, 14);
    const workflowSummary = buildWorkflowSummaryFromExportedSteps(prefix);
    const prediction = await predictNextUserAction({
      cwd,
      userProfileMarkdown: profileMarkdown,
      transcript,
      workflowSummary,
      sessionTitle: session.name,
    });
    const judgment = await judgePredictedAction({
      cwd,
      transcript,
      workflowSummary,
      prediction,
      actualActionType: actionType(current.action),
      actualActionText: current.action,
    });

    rows.push({
      sessionName: session.name,
      sessionUuid: session.uuid,
      stepIndex: i,
      predictedActionType: prediction.actionType,
      predictedDraftText: prediction.draftText,
      predictedConfidence: prediction.confidence,
      actualActionType: actionType(current.action),
      actualActionText: current.action,
      judgeVerdict: judgment.verdict,
      judgeScore: judgment.score,
      judgeRationale: judgment.rationale,
    });
  }

  const accurate = rows.filter((row) => row.judgeVerdict === "accurate");
  const partial = rows.filter((row) => row.judgeVerdict === "partially_accurate");
  const inaccurate = rows.filter((row) => row.judgeVerdict === "inaccurate");
  const averageJudgeScore =
    rows.length > 0 ? rows.reduce((sum, row) => sum + row.judgeScore, 0) / rows.length : 0;

  const payload = {
    generatedAt: new Date().toISOString(),
    profilePath,
    sessionsPath,
    totalCases: rows.length,
    accurate: accurate.length,
    partiallyAccurate: partial.length,
    inaccurate: inaccurate.length,
    averageJudgeScore,
    rows,
  };

  const markdown = [
    "# Zora User Profile Backtest",
    "",
    `Generated: ${payload.generatedAt}`,
    `Profile: ${profilePath ?? "USER_PROFILE.md"}`,
    `Data: ${sessionsPath}`,
    "",
    "## Summary",
    "",
    `- Total cases: ${rows.length}`,
    `- Accurate: ${accurate.length}`,
    `- Partially accurate: ${partial.length}`,
    `- Inaccurate: ${inaccurate.length}`,
    `- Average judge score: ${averageJudgeScore.toFixed(3)}`,
    "",
    "## Notes",
    "",
    "- This run samples 2 representative user-intervention points per historical chat session.",
    "- This is a prototype backtest using the same manually written user profile across all historical cases.",
    "- The evaluation is optimistic because the profile was written with knowledge of the same user history.",
    "- Judging is LLM-based and should be interpreted as directional, not definitive.",
    "",
    "## Cases",
    "",
    ...rows.flatMap((row, index) => [
      `### ${index + 1}. ${row.sessionName} / step ${row.stepIndex}`,
      "",
      `- Predicted action: \`${row.predictedActionType}\``,
      row.predictedDraftText ? `- Predicted draft: ${row.predictedDraftText}` : "- Predicted draft: (empty)",
      `- Actual action: \`${row.actualActionType}\``,
      `- Actual text: ${row.actualActionText}`,
      `- Judge verdict: \`${row.judgeVerdict}\``,
      `- Judge score: ${row.judgeScore.toFixed(3)}`,
      `- Judge rationale: ${row.judgeRationale}`,
      "",
    ]),
  ].join("\n");

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
