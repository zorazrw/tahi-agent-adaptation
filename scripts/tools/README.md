# Tools

## Analyze shared rubrics across two verifier catalogs

`analyze_shared_rubrics.py` aligns tasks by paper title, then asks an LLM whether each **source** rubric is covered by any rubric on the matched **target** task. Prints a per-task list of binary `0`/`1` flags (and optionally writes a full JSON report).

```bash
python scripts/tools/analyze_shared_rubrics.py \
  scripts/log_writing/verifiers_ours/6-context.json \
  scripts/log_writing/verifiers_ours/7-context.json \
  -o scripts/log_writing/eval_log/shared_rubrics_6_vs_7.json

# One task only; inspect prompt without calling the LM
python scripts/tools/analyze_shared_rubrics.py \
  scripts/log_writing/verifiers_ours/6-context.json \
  scripts/log_writing/verifiers_ours/7-context.json \
  --task-index 0 --print-prompt
```

Requires Anthropic credentials (same as `induce.py`). Default model: `claude-sonnet-4-5` (`--model` to override).

## Summarize shared-rubric coverage across peers

`summarize_shared_rubrics.py` aggregates several `analyze_shared_rubrics.py` reports (same source catalog vs multiple peers). For each source rubric it reports whether it is **shared** (covered by all peers), **partial**, or **specific** (covered by none).

```bash
python scripts/tools/summarize_shared_rubrics.py \
  scripts/log_writing/verifiers_ours/shared_rubrics_6_vs_*.json \
  -o scripts/log_writing/verifiers_ours/shared_rubrics_6_summary.json \
  --md scripts/log_writing/verifiers_ours/shared_rubrics_6_summary.md
```

## Label shared/personal rubrics and score agent vs baseline

`calc_shared_improvement.py` consolidates the labeling + eval workflow:

1. Load `shared_rubrics_{id}_vs_*.json` peer coverage reports.
2. Label each source rubric `shared` if covered by ≥ `--min-shared` peers (default 3), else `personal`.
3. Write `{id}-context-labeled.json`.
4. Score agent/baseline PASS–FAIL rates within each label (and print Δ).

```bash
# Auto-discovers reports under verifiers_ours/ and verifiers_ours/shared-context/
python scripts/tools/calc_shared_improvement.py 6

# Use dataviz paths (scripts/log_dataviz/...)
python scripts/tools/calc_shared_improvement.py 16 --domain dataviz

# Weight-method users (eval_log/weight/, {id}-weight.json)
python scripts/tools/calc_shared_improvement.py 12 --method weight

python scripts/tools/calc_shared_improvement.py 8 \
  --shared-dir scripts/log_writing/verifiers_ours/shared-context \
  -o scripts/log_writing/verifiers_ours/8-shared-improvement.json

# Explicit report globs; label only
python scripts/tools/calc_shared_improvement.py 9 \
  --shared 'scripts/log_writing/verifiers_ours/shared_rubrics_9_vs_*.json' \
  --skip-eval
```

## Create summary verifiers from user history + human guidelines

`create_verifiers.py` asks an LLM for a **joint general** verifier list, then writes it in the same shape as the human guidelines file:

1. Summarize recurring criteria from high-quality user history (e.g. `verifiers-manual_*-*.json`).
2. Ensure the list **covers every** human-written guideline (first task entry only).
3. Output an array of `{uuid, instruction, verifiers}` (mirrored from `--human`) with the **same** summarized verifier set on every task.

```bash
python scripts/tools/create_verifiers.py \
  --manual scripts/log_dataviz/verifiers_evolve/verifiers-manual_17-context.json \
  --human scripts/log_dataviz/verifiers_human/verifiers-human_17-context.json \
  -o scripts/log_dataviz/verifiers_summary/verifiers-summary_17-context.json

# Inspect the prompt without calling the LM
python scripts/tools/create_verifiers.py \
  --manual scripts/log_dataviz/verifiers_evolve/verifiers-manual_17-context.json \
  --human scripts/log_dataviz/verifiers_human/verifiers-human_17-context.json \
  --print-prompt
```

Requires Anthropic credentials (same as `induce.py`). Default model: `claude-sonnet-4-5` (`--model` to override).

## Joint memory (intersection of induction runs)

`joint_memory.py` calls Anthropic to keep only preferences/facts shared across **all** provided memory files, and writes one consolidated `Fact:` / `Preference:` file.

```bash
python scripts/tools/joint_memory.py \
  scripts/brain/dataviz_25-offline/memories/data-viz-html.md \
  scripts/brain/dataviz_28-offline/memories/data-viz-html.md \
  -o scripts/brain/joint/memories/data-viz-html.md

# Inspect prompt / preview without writing
python scripts/tools/joint_memory.py a.md b.md --print-prompt
python scripts/tools/joint_memory.py a.md b.md --dry-run
```

Requires Anthropic credentials (same as `induce.py`). Default model: `claude-sonnet-4-5` (`--model` to override).

## Generate LLM verifiers from task instructions

`create_llm_verifiers.py` calls Claude (`claude-sonnet-4-5` by default) to create concrete, falsifiable evaluation criteria using only each task's `instruction`. It does not provide human outputs, existing verifiers, or session history to the model. Credentials are resolved the same way as `induce.py`.

```bash
# One abstract-writing task
python scripts/tools/create_llm_verifiers.py \
  expertise-examples/abstract-writing/tasks.json --task-id 1

# Full catalog
python scripts/tools/create_llm_verifiers.py \
  expertise-examples/abstract-writing/tasks.json \
  -o scripts/log_writing/verifiers_llm/all-context.json

# Direct instruction
python scripts/tools/create_llm_verifiers.py \
  --instruction "Write an abstract of the paper given the title and introduction..."

# Inspect the prompt without calling the LM
python scripts/tools/create_llm_verifiers.py \
  expertise-examples/abstract-writing/tasks.json --task-id 1 --print-prompt
```

File input produces a JSON array of `{uuid, id, type, instruction, verifiers}` entries compatible with the grading tools. Direct `--instruction` input produces one `{uuid, instruction, verifiers}` object.

## Extract initial verifiers

`extract_initial_verifiers.py` reads session exports such as `out_writing_13-weight.json`. For each session, it finds the **first** `task_units` entry whose `actor` is `"agent"`, then takes only the **last** top-level step in that unit's `environment.workflow` and extracts that step's verifiers.

```bash
python scripts/tools/extract_initial_verifiers.py \
  scripts/log_writing/out_writing_13-weight.json \
  -o scripts/initial_verifiers.json
```

The output is a JSON array of `{uuid, instruction, verifiers}`.

## Extract final verifiers

`extract_verifiers.py` reads exported session JSON (from `tasks/export_task_sessions.py`) and writes a JSON array of `{uuid, instruction, verifiers}`.

**Usage** (from repo root):

```bash
# Directory of per-session exports (e.g. scripts/sessions/*.json) → stdout
python scripts/tools/extract_verifiers.py scripts/sessions

# Write to a file
python scripts/tools/extract_verifiers.py scripts/sessions -o scripts/verifiers.json
```

**Output shape:**

```json
[
  {
    "uuid": "<session id>",
    "instruction": "<initial user instruction>",
    "verifiers": ["criterion 1", "criterion 2", "..."]
  }
]
```

## Grade redo session against rubrics

`grade_redo.py` scores a redo session against criteria from `verifiers.json` (matched by instruction overlap). Only the file(s) required by the **last** workflow step are sent to the grader (e.g. the final `.png`, not earlier `.py` scripts).

**Usage** (from repo root):

```bash
python scripts/tools/grade_redo.py \
  -j scripts/runs/headless_dpo_eval/task_001/session.json \
  --verifiers scripts/verifiers.json

# Inspect match without calling the LM
python scripts/tools/grade_redo.py -j path/to/session.json --verifiers scripts/verifiers.json --dry-run

# Save terminal grading output to a text file (still prints to the terminal)
python scripts/tools/grade_redo.py -j path/to/session.json --verifiers scripts/verifiers.json \
  --log-file grade_report.txt

# Use OpenAI as the final judge
python scripts/tools/grade_redo.py -j path/to/session.json --verifiers scripts/verifiers.json \
  --backend openai --model gpt-4.1-mini --json-out ratings.json
```

Requires Anthropic credentials (same as `induce.py`) for `--backend anthropic`, or `OPENAI_API_KEY` for `--backend openai`. Anthropic grading uses `claude-haiku-4-5` by default; OpenAI grading uses `gpt-4.1-mini` by default. Prints per-criterion PASS/FAIL and `average_success_rate` (passes / total criteria).

## Headless eval

Use the headless runner when you want to generate outputs and grade them in the same pass:

```bash
bun run headless:tasks -- \
  --tasks tasks.json \
  --limit 18 \
  --workplace-template trash/workplace-set/test-0527/dpo \
  --model-path "tinker://..." \
  --base-model Qwen/Qwen3.5-35B-A3B \
  --renderer-name qwen3_5 \
  --out runs/headless_dpo_eval \
  --resume \
  --eval \
  --eval-backend openai \
  --eval-model gpt-4.1-mini \
  --verifiers-json scripts/verifiers.json
```

Use `eval_headless_runs.py` when outputs already exist and you only want to re-grade them. It always overwrites each `task_*/ratings.json` it finds, then rebuilds `summary.json` and `scores.csv`.

```bash
# Re-grade every run under runs/
.venv/bin/python scripts/tools/eval_headless_runs.py runs \
  --verifiers scripts/verifiers.json \
  --backend openai \
  --model gpt-4.1-mini

# Re-grade one run directory only
.venv/bin/python scripts/tools/eval_headless_runs.py runs/headless_dpo_eval \
  --verifiers scripts/verifiers.json \
  --backend openai \
  --model gpt-4.1-mini

# Re-grade one task directory only
.venv/bin/python scripts/tools/eval_headless_runs.py runs/headless_dpo_eval/task_001 \
  --verifiers scripts/verifiers.json \
  --backend openai \
  --model gpt-4.1-mini

# Retry only previously failed tasks, up to 3 attempts each (default)
.venv/bin/python scripts/tools/eval_headless_runs.py runs \
  --failed-only \
  --verifiers scripts/verifiers.json \
  --backend openai \
  --model gpt-4.1-mini
```

## Rate file versions against workflow rubrics

`rate_file_versions.py` LM-scores **each unique file snapshot** per workflow step's output files. All versions are graded against the **last** workflow step's verifiers. Uses `claude-haiku-4-5` by default (`--model` to override). Use `--endpoints-only` for first+last snapshot only. Use `--exported-status` for human/exported pass-fail marks (no LM).

```bash
# Full version history + scatter plot
python scripts/tools/rate_file_versions.py -j out.json -s <uuid> -o ratings.json --plot

# First and last snapshot per step only
python scripts/tools/rate_file_versions.py -j out.json -s <uuid> --endpoints-only

# Exported verifier statuses (no LM)
python scripts/tools/rate_file_versions.py -j out.json -s <uuid> --exported-status
```
