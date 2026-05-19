# User Prediction Flow

This document describes the current user-prediction implementation in Agent Cowork:

- how the LLM is invoked
- which model/provider is used
- what context is sent into the prediction workflow
- how the UI suggestion is surfaced
- how backtesting works

## Overview

The current flow is:

1. A session reaches a stable point in the UI.
2. The renderer asks the Electron main process for a next-user prediction.
3. The main process loads the current session history and the local `USER_PROFILE.md`.
4. A small Pi-backed in-memory LLM session is created with **no tools**.
5. The model receives:
   - the markdown user profile
   - a compact transcript of recent session history
   - a compact workflow/verifier summary
   - the session title
6. The model returns JSON describing:
   - predicted next action
   - confidence
   - rationale
   - a validated executable payload for the predicted action
7. The UI shows the suggestion above the prompt bar.
8. If the suggestion is a `message`, pressing `Tab` sends it immediately.

Backtesting uses the same predictor, then runs a second LLM pass as a judge.

## Runtime Hookup

### Main entry point

The live prediction entry point is registered in [src/electron/main.ts](../src/electron/main.ts).

IPC handler:

- `predict-next-user-action`

That handler:

1. loads canonical session history from `sessions.getSessionHistory(sessionId)`
2. resolves the working directory
3. loads the local markdown user profile
4. builds transcript and workflow summary
5. calls `predictNextUserAction(...)`

### Preload bridge

The renderer calls prediction through [src/electron/preload.cts](../src/electron/preload.cts):

- `window.electron.predictNextUserAction(sessionId)`

### Renderer trigger

The renderer-side trigger lives in [src/ui/App.tsx](../src/ui/App.tsx).

The prediction request is made when:

- there is an active session
- the session is not running
- the session has messages
- the last message is not a `user_prompt`

So prediction is currently **post-turn**, not streaming and not per-keystroke.

## What LLM Is Used

The prediction flow uses the same Pi runtime stack as the rest of the app, but through a lightweight helper in [src/electron/libs/pi-prompt.ts](../src/electron/libs/pi-prompt.ts).

### Model/provider selection

The predictor does **not** hardcode a specific provider or model.

Instead it uses:

- `AuthStorage`
- `ModelRegistry`
- `SettingsManager`
- `createAgentSession(...)`

from `@mariozechner/pi-coding-agent`.

That means the active LLM is whatever the Pi settings currently resolve to as the default provider/model for the local user configuration.

In practice, prediction uses the same configured Pi provider stack as the app:

- Anthropic
- OpenAI
- OpenAI-compatible
- Tinker

depending on what the user has configured in Pi settings.

### Pi config resolution

The prompt helper resolves the Pi agent directory in this order:

1. `PI_AGENT_DIR` environment variable
2. `~/Library/Application Support/agent-cowork/pi-agent`
3. `~/.pi-agent`

From there it loads:

- `auth.json`
- `models.json`
- `tinker-provider.json`

So the prediction flow uses the same local auth and model registry data as the app runtime.

### Session shape

Prediction uses:

- `createAgentSession(...)`
- `SessionManager.inMemory(cwd)`
- `tools: []`

Important implications:

- prediction does not mutate the real task session
- prediction does not write prediction turns into the session history
- prediction has no tools, so it cannot browse/read/edit files during prediction
- it is a pure text-in / text-out model call

## Context Sent to the Predictor

The prediction input is assembled in [src/electron/libs/user-predict.ts](../src/electron/libs/user-predict.ts).

### 1. User profile markdown

The app loads a local `USER_PROFILE.md` from the Electron launch directory:

1. `<app launch cwd>/USER_PROFILE.md`

This is intentionally independent of the active session cwd, because session cwds can point at per-run workspace directories.

The content is passed to the model as raw markdown, not parsed into a rigid schema.

### 2. Recent transcript

The predictor builds a compact transcript from the last `14` session messages.

It includes:

- recent `user_prompt` messages
- recent assistant text blocks
- tool-use markers like `[tool:name]`
- `run_result`
- `node_completed`
- user structural edits such as:
  - `edit_workflow`
  - `edit_verifier`
  - `brain_edit`
  - `file_edit(path)`
- tool result markers

It does **not** include:

- full tool outputs
- full file contents
- full long-form session history

### 3. Workflow/verifier summary

If the session has a workflow tree, the predictor receives a compact line-based summary of:

- node descriptions
- node statuses
- verifier criteria
- verifier success/failure status

This is currently the only structured environment summary sent into prediction.

### 4. Session title

If available, the session title is included as lightweight task framing.

### 5. No additional hidden runtime state

The live predictor currently does **not** send:

- full memory files
- skill files
- output file contents
- screenshots
- preview state
- costs/token usage

So the current predictor is intentionally narrow.

## Prediction Prompt Contract

The predictor asks the LLM to return JSON with this shape:

```json
{
  "actionType": "message|edit_workflow|edit_verifier|file_edit|brain_edit|stop",
  "confidence": 0.0,
  "rationale": "string",
  "executable": "<ExecutableAction>"
}
```

Notes:

- `executable` is required and must validate against the shared executable action schema
- `actionType` is derived from `executable.type` after validation
- `message` sends a predicted prompt
- `edit_workflow` applies a patch to the current workflow tree
- `edit_verifier` replaces a node's verifier list
- `file_edit` writes full replacement contents to a path
- `brain_edit` records memory or skill edits
- `stop` means the user is likely done for now
- if the JSON cannot be parsed or the executable payload is invalid, no suggestion is surfaced

## UI Suggestion Flow

The current UI surface is implemented in [src/ui/components/PromptInput.tsx](../src/ui/components/PromptInput.tsx).

### What the user sees

Above the prompt bar, the app shows:

- predicted action type
- confidence
- optional draft text
- rationale

### Keyboard behavior

If there is a visible suggestion and:

- the suggested action is `message`
- the draft text is non-empty
- the prompt box is empty

then:

- `Tab` sends the suggested prompt immediately

Otherwise, the existing prompt behavior remains:

- `Tab` starts the next pending workflow step when applicable
- `Enter` sends the typed prompt
- `Esc` dismisses the suggestion

## Backtest Flow

The backtest entry point is [scripts/backtest_user_profile.ts](../scripts/backtest_user_profile.ts).

### Dataset

Backtesting expects a local trajectory export JSON. The data file is not committed to the repo.

Pass the export path as the first script argument or set `USER_PREDICTION_DATASET`.

### Replay method

For each sampled user intervention point:

1. take the trajectory prefix up to but not including the real next user step
2. build transcript + workflow summary from that prefix
3. run the same `predictNextUserAction(...)` call used by the app
4. when baseline is enabled, run the same predictor again with an empty user profile
5. run a second LLM pass that directly ranks the with-profile prediction against the empty-profile prediction
6. when baseline is skipped, use the legacy single-prediction judge score

### Judge LLM

The judge uses the **same Pi-backed text prompt path** as the predictor.

It is therefore typically the same configured provider/model unless the user changes Pi settings between calls.

For the normal baseline-vs-profile backtest, the judge returns:

```json
{
  "winner": "personalized|baseline|tie",
  "rationale": "string"
}
```

The judge is instructed to rank which prediction would have been more useful for anticipating the actual next user step, while caring more about underlying intent than exact UI action label.

When the baseline is skipped, the script falls back to the older single-prediction judge shape:

```json
{
  "verdict": "accurate|partially_accurate|inaccurate",
  "score": 0.0,
  "rationale": "string"
}
```

## Current Limitations

The current implementation is intentionally simple.

### The user profile is local-only

`USER_PROFILE.md` is expected to be local and ignored by git.

### The predictor is tool-free

It cannot inspect files or the rendered preview directly at prediction time.

### Context is compact

Only the last `14` messages plus a workflow summary are sent.

### The predictor and judge usually use the same LLM family

That is convenient operationally, but it may bias evaluation.

### No caching layer yet

Each prediction is a fresh Pi call after the session becomes idle.

## Relevant Files

- [src/electron/main.ts](../src/electron/main.ts)
- [src/electron/preload.cts](../src/electron/preload.cts)
- [src/electron/libs/pi-prompt.ts](../src/electron/libs/pi-prompt.ts)
- [src/electron/libs/user-predict.ts](../src/electron/libs/user-predict.ts)
- [src/ui/App.tsx](../src/ui/App.tsx)
- [src/ui/components/PromptInput.tsx](../src/ui/components/PromptInput.tsx)
- [scripts/backtest_user_profile.ts](../scripts/backtest_user_profile.ts)
- [docs/user-prediction-backtest.md](user-prediction-backtest.md)
