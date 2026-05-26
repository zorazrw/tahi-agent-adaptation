# Data Export and Agent Update

## Prediction-tab telemetry

`prediction_stats.py` aggregates the `prediction_events` table (rows written
by the renderer for each user-prediction surface: `shown`, `accepted`,
`dismissed`, `ignored`).

```bash
python scripts/prediction_stats.py                    # overall accept / dismiss / ignore rates
python scripts/prediction_stats.py --since-days 7
python scripts/prediction_stats.py --session-id <uuid>
python scripts/prediction_stats.py --json
```

## Data Export

`export_task_sessions.py` is a standalone Python 3 script (stdlib only) to export task sessions from the Agent Cowork SQLite database to JSON.

**Usage:**
```bash
python scripts/tasks/export_task_sessions.py -o out.json

# Single session
python scripts/tasks/export_task_sessions.py --session-id <uuid> -o session.json

# Weight-based export (for training; see Formats table above)
python scripts/tasks/export_task_sessions.py --format weight -o out_weight.json
```

**Exported Data Structure:**

```json
{
  "uuid": "<session id>",
  "name": "<title>",
  "trajectory": [
    {
      "actor": "user | agent",
      "action": "<action string>",
      "message": "<optional decoded message text>",
      "tool_result": "<optional merged tool result text>",
      "environment": {
        "workflow": "<nested workflow tree with verifier statuses>",
        "file": "<output file snapshots>",
        "memory": "<memory file map>",
        "skill": "<skill file map>"
      }
    }
  ]
}
```


## Session file snapshots → GIF

`session_file_versions_to_gif.py` walks one exported session’s trajectory, collects **unique** non-null contents for a given workspace file key (e.g. `webarena_trend.html` from `environment.file`), writes them as `0.html`, `1.html`, …, then renders frames with Playwright and builds a palette-optimized GIF with **ffmpeg**.

**Dependencies:** `pip install playwright`, `playwright install chromium`, and `ffmpeg` on your PATH.

**Usage:**

```bash
# From repo root (default JSON: scripts/out.json)
python scripts/session_file_versions_to_gif.py \
  --session <uuid> \
  --file webarena_trend.html

# Custom export path and output directory
python scripts/session_file_versions_to_gif.py \
  -j path/to/sessions.json \
  -s <uuid> \
  -f chart.html \
  -o ./my-output

# Only extract numbered HTML files (no Playwright / ffmpeg)
python scripts/session_file_versions_to_gif.py -s <uuid> -f chart.html --html-only
```

See `python scripts/session_file_versions_to_gif.py --help` for viewport, FPS, step label, and scaling options.

## Context-Based Update

`induce.py` is a standalone Python 3 script (stdlib only) to extract memories and skills from session JSON (e.g. ``memories/{task-name}.md`` and ``skills/{task-name}.md``).

**Usage:**

```bash
python scripts/induce.py --data_path out.json --output_dir "."
```

## Weight-Based Update

1. DPO: `export_dpo_data.py` and `tinker_dpo.py`
2. OPD: `export_opd_data.py` and `tinker_opd.py`
3. REINFORCE: `export_reinforce_data.py` and `tinker_reinforce.py`

to run tinker training, use the following commands:

```bash
cd scripts
python tasks/export_task_sessions.py --format weight -o out.json

# DPO
python export_dpo_data.py out.json -o out_dpo.json
bash tinker_dpo.sh

# OPD
python export_opd_data.py out.json -o out_opd.json
bash tinker_opd.sh

# REINFORCE
python export_reinforce_data.py out.json -o out_reinforce.json
bash tinker_reinforce.sh

# Score a redo session's final outputs against final verifiers from baseline export
python score_redo_against_verifiers.py \
  --verifiers-json out.json \
  --outputs-json out_redo.json \
  --task task2

# Optional: print full request blocks (text + image metadata) before model call
python score_redo_against_verifiers.py \
  --verifiers-json out.json \
  --outputs-json out_redo.json \
  --task task2 \
  --debug-prompts
```
