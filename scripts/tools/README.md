# Tools

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
