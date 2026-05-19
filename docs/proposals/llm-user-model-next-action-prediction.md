# Proposal: LLM-Based User Model and Next User Action Prediction

Date: 2026-04-20
Status: Draft

## Summary

I recommend starting with a **structured user-state model plus next user action predictor**, while also allowing an intentionally overfit **prototype next-message model** for fast iteration.

A small local trajectory export is a good seed dataset for problem framing. It is too small for a robust production user simulator, but it is still usable for a deliberately overfit prototype if the goal is to:

- validate the product surface
- identify useful state representations
- collect more trajectories over time

The practical path is:

1. Normalize exported session trajectories into a supervised dataset for **next user action classification**.
2. Use a strong base LLM to infer a compact **markdown user state** from the trajectory and environment snapshot.
3. Predict the next user action from a small closed label set.
4. Generate candidate next user messages only when the predicted next action is `message`.
5. Allow a prototype model to overfit on current sessions, but evaluate it by whether it reduces corrective user turns, verifier churn, and unnecessary agent work.

This proposal fits the current codebase because Agent Cowork already has:

- trajectory storage in `src/electron/libs/session-store.ts`
- environment snapshot export in `src/electron/libs/message-state-snapshot.ts`
- persistent memory primitives in `src/electron/libs/memory-store.ts`
- existing context-based and weight-based adaptation scripts under `scripts/`

## What a Seed Export Tells Us

A representative small seed export contains only a few sessions and a few hundred trajectory steps.

Observed user action types:

- `message`: 20
- `edit_workflow`: 19
- `edit_verifier`: 15
- `file_edit`: 7
- `brain_edit`: 1

Observed agent action types:

- `Edit`: 35
- `message`: 28
- `Bash`: 24
- `verify`: 23
- `update_verifiers`: 20
- `Read`: 17
- `Write`: 14
- `plan`: 3
- `result`: 1

Important pattern: user interventions often happen **after agent verification**.

Most common agent-to-user transitions:

- `verify -> message`: 11
- `verify -> file_edit`: 5
- `verify -> edit_workflow`: 3

This matters because it suggests the highest-value prediction point is not “at every turn,” but specifically **after the agent claims progress, verification, or completion**.

## Problem Framing

There are three related but different problems:

1. **User modeling**
   Infer stable preferences and temporary state from the trajectory.
2. **Next user action prediction**
   Predict whether the user will reply with a message, edit the workflow, edit verifiers, directly edit a file, or do nothing.
3. **Next user message generation**
   Generate the likely content of the next user message.

The first two are realistic now. The third is also reasonable as a **prototype objective** if we accept that it will overfit initially.

## Recommendation

Build a **two-stage user prediction system**, with an optional prototype generator from day one.

### Stage A: Markdown user state

Given:

- recent trajectory window
- current workflow tree
- verifier failures
- edited output files
- task title / initial request
- persistent memory snippets when available

infer a markdown note like:

```md
# User State

Current task goal: create a benchmark visualization

Stable preferences:
- prefers larger text
- prefers minimal charts
- prefers semantic color choices

Current focus:
- visual truncation
- readability

Likely intervention target:
- output rendering

Risk flags:
- agent may declare success too early
- chart export may be clipped

Likely next actions:
- message
- file_edit
```

This state should be treated as **ephemeral inference written in markdown**, not canonical truth.

### Stage B: Next user action prediction

Predict a label from a small ontology:

- `message`
- `edit_workflow`
- `edit_verifier`
- `file_edit`
- `brain_edit`
- optionally `no_intervention`

This should be the primary supervised objective because:

- the label space is small
- the signal is already present in exported trajectories
- action prediction is easier to evaluate than free-form message generation
- it can directly improve agent policy

### Optional Stage C: Candidate next message generation

Only when Stage B predicts `message`, ask the model for:

- 1 canonical next message
- 2-3 alternative candidate messages
- a short rationale explaining what likely triggered the reply

This should be used for:

- offline evaluation
- policy debugging
- ranking possible recovery actions
- early prototype user simulation

It should **not** directly impersonate the user in production without strong safeguards.

## Architecture Overview

The system should be split into four explicit layers:

1. **User model store**
   Persistent representation of what we think we know about the user.
2. **Prediction engine**
   Produces next-action probabilities and optional next-message drafts from conversation state plus user model.
3. **Backtest harness**
   Replays historical conversations and scores how well the model predicts the real next user action/message.
4. **UI integration**
   Surfaces proactive predicted replies in the app and lets the user accept them quickly.

This split matters because the project needs to support:

- a simple markdown-backed user model now
- stronger learned representations later
- offline evaluation without shipping product changes
- product experimentation without retraining the model each time

## Architecture: User Model Representation

For now, the user model should be stored as an **unstructured markdown file**, not as a schema-first JSON object and not only as hidden model weights.

That gives three benefits:

- easy inspection and manual editing
- easy bootstrapping from small data
- compatibility with the app’s existing memory-centric workflow

### Proposed representation

Use one markdown file per user or profile, for example:

- `memories/user-model.md`
- or `memories/user-models/<profile>.md`

Recommended structure:

```md
---
profile_id: zora-prototype
updated_at: 2026-04-20
source_sessions:
  - 6178b6e7-53b4-4d2f-a02e-cb0e3b21cbd3
  - fa3f152d-1924-4b98-b8c7-55249f3e2c0d
confidence: prototype
---

# Stable Preferences

- Prefers larger text in charts and figures.
- Prefers minimal visual design with semantic colors.
- Often corrects truncation, readability, and export issues.

# Interaction Patterns

- Frequently intervenes after verification or premature success claims.
- Uses direct file edits when layout fixes are faster than instruction loops.
- Often refines workflow and verifier definitions mid-task.

# Preferred Agent Behavior

- Be skeptical after verify/export steps.
- Check rendering and clipping before declaring success.
- Prefer concrete fixes over verbose explanations.

# Recent Situational State

- Current task family: visualization / formatting-heavy work
- Current likely sensitivity: truncation, label sizing, layout density

# Canonical Examples

## After verify

Agent tends to say output is complete.
Likely next user actions:
- `message`: "the png seems truncated? fix it"
- `file_edit`
```

### Runtime rule

Do not require a structured runtime schema for user state.

The source of truth should remain the markdown itself. The predictor should read raw markdown plus conversation context and reason over that text directly.

That keeps the system:

- easy to inspect
- easy to edit manually
- resilient to schema churn while the feature is still experimental
- aligned with the user’s preference for unstructured state

### Proposed app modules

- `src/electron/libs/user-model-store.ts`
  load/save markdown user models
- `src/electron/libs/user-model-induce.ts`
  infer or refresh markdown content from session history
- `src/electron/libs/user-predict.ts`
  run next-action / next-message prediction using raw markdown user state

This integrates naturally with the existing memory pipeline in [`memory-store.ts`](../../src/electron/libs/memory-store.ts).

## Architecture: Backtesting on Historical Conversations

Backtesting should be treated as a first-class subsystem, not an afterthought.

The goal is:

- given a historical session
- only expose the model to turns up to time `t`
- predict the real user’s next action or message at `t+1`
- compare prediction to the actual next turn

### Backtest modes

#### 1. Next-action stepwise replay

For every agent-to-user boundary in a session:

- build the user model from prior history only
- build the current context from the prefix only
- predict the next user action
- compare to the gold next user action

This should be the default and most stable benchmark.

#### 2. Next-message replay

When the gold next user action is `message`:

- generate 1-3 candidate next messages
- compare against the real next message
- score semantic utility, not only lexical overlap

#### 3. Rolling user-model update replay

This tests whether the markdown user model itself improves over time.

Loop:

- start with an empty or seed user model
- replay one session
- update the user model
- predict the next session

This is the best benchmark for “does the system learn the user over time?”

### Proposed backtest outputs

Each run should emit:

- per-turn prediction rows
- aggregate action metrics
- aggregate message metrics
- confusion matrix for action classes
- examples of strongest hits and misses
- user-model snapshots before and after each session

### Proposed scripts

- `scripts/export_user_prediction_data.py`
  emits normalized replay rows
- `scripts/backtest_user_model.py`
  runs stepwise replay and writes a report
- `scripts/render_user_backtest_report.py`
  optional HTML/Markdown summary for inspection

### Core evaluation principle

The backtest must simulate deployment honestly:

- no peeking at future turns
- no peeking at future user-model updates
- no mixing synthetic labels with gold human labels unless explicitly tagged

## Architecture: In-App Proactive Prediction Flow

The product goal is:

- when the agent reaches a point where user input is likely needed
- proactively draft the likely next user action or reply
- let the user accept it with one keypress

### Trigger conditions

Good initial trigger points:

- after `verify`
- after export/render steps
- after agent messages that imply completion
- when the model predicts high probability of `message`

Later, this can expand to:

- proactive `file_edit` or `edit_workflow` suggestions
- “likely user dissatisfaction” warnings

### UI flow

1. Agent finishes a turn.
2. Backend computes a next-user prediction.
3. If top prediction is `message`, create a draft reply suggestion.
4. The UI pre-fills or ghost-renders the draft in the input box.
5. User presses `Tab` to accept and send it.
6. User can keep typing to overwrite it, or dismiss it with `Esc`.

### Important interaction constraint

The app already uses `Tab` in [`PromptInput.tsx`](../../src/ui/components/PromptInput.tsx) to start the next pending workflow step when the prompt is empty.

So the safest precedence rule is:

- if a predictive reply suggestion is visible, `Tab` accepts and sends the suggestion
- otherwise preserve the existing `Tab` behavior

That avoids introducing a conflicting keyboard shortcut.

### Proposed UI state

Add app state like:

```ts
type PredictedUserReply = {
  sessionId: string;
  sourceTurnIndex: number;
  actionType: "message" | "edit_workflow" | "edit_verifier" | "file_edit";
  draftText: string;
  confidence: number;
  rationale?: string;
};
```

Suggested store additions in [`useAppStore.ts`](../../src/ui/store/useAppStore.ts):

- `predictedUserReply`
- `setPredictedUserReply`
- `clearPredictedUserReply`

### Proposed UI behavior in the input component

Relevant file:

- [`PromptInput.tsx`](../../src/ui/components/PromptInput.tsx)

Behavior:

- render the suggestion as ghost text or a visible inline chip above the input
- if the user presses `Tab`, copy suggestion into `prompt` and send immediately
- if the user starts typing, either:
  - replace the suggestion entirely, or
  - accept it into the input first and allow normal editing
- if the user presses `Esc`, dismiss the suggestion

For the first version, the simplest version is:

- do not auto-send invisible text
- show the predicted draft clearly
- `Tab` means “accept and send shown suggestion”

That keeps the feature legible and reversible.

### Proposed backend flow

Suggested new backend module:

- `src/electron/libs/user-prediction-service.ts`

Responsibilities:

- read session prefix and current environment snapshot
- load markdown user model
- call predictor
- return a structured prediction payload to the renderer

This service should be called only at specific moments, not every keystroke.

## What To Build First

The architecture should be implemented in this order:

1. Markdown-backed user model representation.
2. Honest historical backtest harness.
3. Prediction service returning structured next-action / next-message suggestions.
4. UI support for visible proactive suggestions and `Tab` acceptance.

That order is important because:

- without the markdown representation, the user model is hard to inspect
- without backtesting, the feature is hard to trust
- without structured UI behavior, the prediction feature will feel magical and brittle

## Overfitting Is Acceptable for the Prototype

With a small seed export alone, a trained next-message model will almost certainly overfit. For an initial prototype, that is acceptable.

That overfitting is tolerable if the immediate goal is:

- to model one user or a narrow user cluster
- to build the product integration before the larger dataset exists
- to learn which prediction targets are useful
- to bootstrap future data collection

Main reasons it will overfit:

- only 3 sessions
- only 20 user `message` actions
- most user signal is structured intervention, not free-form language
- tasks are narrow and visually oriented
- the same user appears to be represented repeatedly, so the data is not population-level

So the right framing is not “do not build it,” but:

- build it as a prototype
- expect memorization
- keep it advisory
- use new real sessions to continuously re-train and re-evaluate

## Data Plan

### 1. Build a dedicated export for user prediction

Add a new script, for example:

- `scripts/export_user_prediction_data.py`

Each training row should contain:

- session id
- turn index
- preceding trajectory window
- current environment snapshot
- previous user messages
- normalized user preference summary
- gold next user action
- gold next user message, if the gold action is `message`

### 2. Normalize actions

Do not train on raw strings like `message("...")`. Convert trajectory events into structured forms:

- `action_type`
- `action_payload`
- `target_file`
- `workflow_delta`
- `verifier_delta`

### 3. Expand beyond the initial seed export

Use `scripts/export_task_sessions.py` across the local session database and retain:

- real user corrections
- workflow edits
- verifier edits
- file edits
- memory snapshots when available

### 4. Keep synthetic data separate

Synthetic user continuations may help with coverage, but they should be tagged separately and never mixed invisibly with human gold data.

## Model Strategy

### Phase 1: Prompted LLM baseline

Implement a structured prompting baseline before any training.

Input:

- last `N` turns
- compact workflow summary
- failing verifier summary
- changed files summary

Output:

- user-state markdown
- next-action probabilities
- optional next-message candidate

This is likely the best first milestone because it needs no training and will show whether the feature is even useful.

### Phase 2: Overfit prototype model

Train a deliberately narrow model on currently available trajectories.

Recommended objectives:

- next user action classification
- next user message generation when the label is `message`
- optional rationale generation for likely dissatisfaction source

Recommended framing:

- optimize for one-user or few-user adaptation
- treat success as product usefulness, not generalization
- retrain frequently as more sessions arrive

### Phase 3: Lightweight action model

Once more sessions are exported, train a more stable model or adapter for next-action classification.

Practical options:

- classifier over LLM embeddings
- LoRA fine-tune for structured action prediction through the existing Tinker path
- reranker that scores possible next actions given trajectory state

### Phase 4: Message generator

Only after sufficient data exists, train or distill a generator for next user messages conditioned on:

- latent user state
- recent trajectory
- intervention target

This generator should support beam or candidate generation, not just one deterministic prediction.

## Product Integration

The predictor should be advisory and local to the agent loop.

Good trigger points:

- after `verify`
- after `message` summaries that claim completion
- after file export or rendering
- after repeated edit cycles

Possible uses:

- choose between “ask user”, “self-check”, and “continue”
- trigger a more skeptical verification pass when user dissatisfaction is likely
- prioritize readability / formatting fixes when the user historically corrects presentation issues
- decide whether to expose a checkpoint before continuing
- run a “predicted next complaint” check before declaring success

The model should not silently write to workflow or verifier state on the user’s behalf.

## Evaluation Plan

### Offline metrics

For next-action prediction, the primary metric should be an **LLM-judge score** rather than exact-label accuracy alone.

The reason is that many predicted actions can be directionally right without matching the gold label exactly. For example:

- predicting `message` may still be useful when the gold turn is `file_edit` but the predicted content correctly identifies the user’s complaint
- predicting the wrong intervention surface may still be high quality if it points to the same underlying problem

So the default evaluation should ask an LLM judge questions like:

- Does the predicted action identify the same underlying user intent as the real next turn?
- Would this predicted action have helped the agent avoid the failure that triggered the real user intervention?
- Is the predicted action a plausible next move for this user in this context?

Recommended judge outputs:

- `accurate`
- `partially_accurate`
- `inaccurate`
- short rationale

Secondary quantitative metrics can still be tracked:

- exact action-label match
- top-k action-label match
- exact match for action target metadata when applicable

For next-message generation:

- semantic similarity to the real next user message
- LLM-judge utility score: “Would this predicted message have caused the same agent correction?”
- human spot checks on a small evaluation set

### Online product metrics

- reduction in corrective user turns per task
- reduction in repeated `verify -> user complaint` loops
- fewer unnecessary tool calls before user correction
- time-to-accepted-output

The most important metric is whether the predictor helps the assistant avoid **predictable user corrections**.

For the overfit prototype, also track:

- leave-one-session-out performance, even if weak
- performance on the most recent session after training on older sessions
- whether the model predicts the right *kind* of complaint even when wording differs, as judged by the LLM evaluator

## Risks

### Simulator drift

Recent work warns that simulated users can diverge materially from real users. We should use simulation as a development aid, not as the only source of truth.

### Over-personalization

A model can become too certain about a user and suppress legitimate novelty. User state should be probabilistic and revisable.

### Privacy

User modeling turns conversation history into behavioral inference. Storage, retention, and export should be explicit and inspectable.

### Feedback loops

If the agent acts too aggressively on a predicted correction, it can create self-fulfilling behavior or become overly cautious.

## Proposed Milestones

1. Add `export_user_prediction_data.py` and normalize user action labels.
2. Build a prompted LLM baseline that returns structured user-state JSON plus next-action probabilities.
3. Train an intentionally overfit prototype for next-action and next-message prediction on current sessions.
4. Run offline evaluation on exported sessions with leave-one-session-out validation.
5. Integrate prediction after `verify` and completion-like messages only.
6. Measure whether the predictor reduces corrective user turns.
7. Continuously retrain as more sessions arrive, then harden the model once the dataset is large enough.

## Bottom Line

Agent Cowork should pursue **user-state inference + next user action prediction first**, but it is reasonable to also ship an intentionally overfit prototype next-message model now.

That is the highest-leverage path supported by the current data and codebase: use the current sessions to prototype aggressively, assume more data will arrive, and treat the first model as a bootstrapping mechanism rather than a final user simulator.

## References

- [`scripts/README.md`](../../scripts/README.md)
- [`docs/new-providers.md`](../new-providers.md)
- [User Behavior Simulation with Large Language Model based Agents (arXiv 2023)](https://arxiv.org/abs/2306.02552)
- [Simulating User Agents for Embodied Conversational-AI (arXiv 2024)](https://arxiv.org/abs/2410.23535)
- [PersonaX: A Recommendation Agent-Oriented User Modeling Framework for Long Behavior Sequence (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.300/)
- [SimulatorArena: Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants? (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1786/)
- [Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations (arXiv 2026)](https://arxiv.org/abs/2601.17087)
