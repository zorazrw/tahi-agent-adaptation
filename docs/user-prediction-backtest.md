# User Prediction Backtest Runbook

This document explains how to export sessions, run the user-prediction backtest, and visualize the generated report.

## What the backtest measures

The backtest replays historical Agent Cowork sessions. For each selected user turn, it:

1. hides the real next user action
2. builds the same transcript and workflow summary used by the app
3. asks the predictor for the next user action
4. optionally runs a baseline with an empty user profile
5. when the baseline is enabled, asks an LLM judge to rank the two predictions after anonymizing and randomizing their order
6. when the baseline is skipped, falls back to the legacy single-prediction judge score

The script writes both a Markdown summary and a JSON report. The JSON report powers the interactive report viewer.

## Prerequisites

Run all commands from the repo root:

```bash
cd /path/to/agent-cowork
```

You need:

- a local `USER_PROFILE.md` file in the repo root
- an Agent Cowork `sessions.db` with sessions to export
- the same Pi/provider credentials and settings that the app uses for prediction calls
- `bun` for the TypeScript backtest script
- `python3` for the session export script

`USER_PROFILE.md`, exported datasets, and generated reports are local artifacts. `docs/backtests/` is ignored by git.

To create `USER_PROFILE.md`, open Agent Cowork settings, go to the **Profile** tab, choose how many recent chats to use, and click **Auto-generate**. The generated profile is saved to `USER_PROFILE.md` in the app/repo working directory. You can also edit and save the profile in that tab, or create `USER_PROFILE.md` manually in the repo root.

## 1. Export session data

The backtest expects the default export format from `scripts/export_task_sessions.py`.

Export the most recent sessions:

```bash
mkdir -p docs/backtests
python3 scripts/export_task_sessions.py \
  --limit 40 \
  --output docs/backtests/sessions-last-40.json
```

Export one session by id:

```bash
python3 scripts/export_task_sessions.py \
  --session-id SESSION_UUID \
  --output docs/backtests/session-SESSION_UUID.json
```

The backtest accepts both the multi-session array produced by `--limit` and the single-session object produced by `--session-id`.

Use a non-default database path:

```bash
AGENT_COWORK_DB="/path/to/sessions.db" \
python3 scripts/export_task_sessions.py \
  --limit 40 \
  --output docs/backtests/sessions-last-40.json
```

Equivalent explicit database flag:

```bash
python3 scripts/export_task_sessions.py \
  --db "/path/to/sessions.db" \
  --limit 40 \
  --output docs/backtests/sessions-last-40.json
```

Default database locations:

- macOS: `~/Library/Application Support/Agent Cowork/sessions.db`
- macOS alternate app id: `~/Library/Application Support/agent-cowork/sessions.db`
- Windows: `%APPDATA%\Agent Cowork\sessions.db`
- Windows alternate app id: `%APPDATA%\agent-cowork\sessions.db`
- Linux: `~/.config/Agent Cowork/sessions.db`
- Linux alternate app id: `~/.config/agent-cowork/sessions.db`

## 2. Run the backtest

Run a representative sample first. This keeps the number of Pi calls manageable while checking that the pipeline works.

```bash
bun scripts/backtest_user_profile.ts \
  docs/backtests/sessions-last-40.json \
  --case-selection representative \
  --cases-per-session 2
```

This writes:

- `docs/backtests/user-simulator-backtest-YY-MM-DD.md`
- `docs/backtests/user-simulator-backtest-YY-MM-DD.json`

Run every eligible user turn:

```bash
bun scripts/backtest_user_profile.ts \
  docs/backtests/sessions-last-40.json \
  --case-selection all \
  --report-name user-simulator-backtest-last-40-all
```

Skip the empty-profile baseline to halve the prediction calls:

```bash
bun scripts/backtest_user_profile.ts \
  docs/backtests/sessions-last-40.json \
  --case-selection representative \
  --cases-per-session 2 \
  --no-baseline \
  --report-name user-simulator-backtest-last-40-no-baseline
```

You can also pass the dataset path with an environment variable:

```bash
USER_PREDICTION_DATASET=docs/backtests/sessions-last-40.json \
bun scripts/backtest_user_profile.ts \
  --case-selection representative \
  --cases-per-session 2 \
  --report-name user-simulator-backtest-last-40-representative
```

## CLI options

```text
bun scripts/backtest_user_profile.ts DATASET.json [options]

Options:
  --case-selection all|representative  Which user turns to evaluate
  --cases-per-session N                Sampling count for representative mode
  --include-baseline                   Include empty-profile baseline
  --no-baseline                        Skip baseline calls
  --report-name NAME                   Output basename
  --out-dir DIR                        Output directory
```

Defaults:

- `--case-selection all`
- `--cases-per-session 2`
- `--include-baseline`
- `--report-name user-simulator-backtest-YY-MM-DD`
- `--out-dir docs/backtests`

When omitted, the report name uses the current local year, month, and day, for example `user-simulator-backtest-26-05-20`. If that default report already exists, the script appends the first available numeric suffix, for example `user-simulator-backtest-26-05-20-1`, then `user-simulator-backtest-26-05-20-2`. Passing `--report-name` uses the exact basename you provide.

## Interpreting outputs

The Markdown report is the fastest way to read the overall result. It includes:

- total cases
- pairwise ranking counts when baseline was enabled: personalized wins, baseline wins, and ties
- legacy single-prediction average judge score when baseline was skipped
- per-case ground truth, baseline prediction, personalized prediction, pairwise winner, and rationale

The JSON report contains the full payload:

- report metadata
- pairwise ranking metrics when baseline was enabled
- legacy score metrics when baseline was skipped
- per-case transcript and workflow summary
- ground truth action
- baseline and personalized prediction details
- pairwise judge winners and rationales

The pairwise judge is also an LLM call, so use the report to find patterns and examples rather than treating the ranking as a deterministic benchmark.

## 3. Visualize the JSON report

The report viewer is a Vite page at `backtest-report.html`.

Open the viewer:

```bash
bun run backtest:viewer
```

That command opens the Vite URL printed by the terminal, for example:

```text
http://127.0.0.1:5173/backtest-report.html
```

If another local app is already using Vite's default port, Vite will choose the next available port.

Use **Select report JSON** in the viewer, or drag a generated `.json` report onto the empty state. The selected file stays local in the browser and does not need to be committed or placed under `docs/backtests/`.

The viewer shows:

- pairwise baseline-vs-personalized ranking totals
- filters for session, action type, verdict, and improvement/regression
- searchable case rows
- side-by-side ground truth, baseline prediction, and personalized prediction
- per-case pairwise winner and rationale
- transcript and workflow summary for the selected case

## Troubleshooting

If the export fails with `sessions.db not found`, pass `--db PATH` or set `AGENT_COWORK_DB`.

If the backtest fails with `USER_PROFILE.md was not found`, use Settings > Profile > **Auto-generate** in Agent Cowork, or manually create/restore the local profile file in the repo root.

If the backtest fails before any cases run, check that the exported JSON is either an array of sessions or one session object, and that every session has a `trajectory`.

If the run is too slow, use `--case-selection representative`, reduce `--cases-per-session`, or add `--no-baseline`.

If the viewer opens but cannot load the report, make sure you selected the generated JSON report rather than the Markdown summary.
