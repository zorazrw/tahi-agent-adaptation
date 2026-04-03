# Scripts

## export_task_sessions.py

Standalone Python 3 script (stdlib only) to export task sessions from the Agent Cowork SQLite database to JSON.

**What’s in the database (and what this exports):**

| Data | In DB | Exported as |
|------|--------|-------------|
| Initial task instruction | `sessions.last_prompt` | `initial_task_instruction` |
| Workflow step descriptions | `sessions.steps` (JSON array) | `workflow_steps[].step_description` |
| Expected output files per step | `sessions.output_files` (JSON) | `workflow_steps[].expected_output_files` |
| Verifiers per step | `sessions.verification_criteria` + `verifier_marks` | `workflow_steps[].verifiers` (criterion + status: success/failure/unchecked) |
| Agent/user message turns | `messages.data` (JSON per row) | `action_trajectory` |

**Formats:**

| `--format` | Output | Use case |
|------------|--------|----------|
| `default` (default) | Human-readable action trajectory (`actor`, `action`, `environment`) | Context export pipeline |
| `weight-based` | Raw SDK messages split into `task_units` with `agent_trajectory`, `human_trajectory`, `verifiers`, `prompt_first_turn` | DPO / OPSD / RL training (`scripts/weight/`) |

**Usage:**

Use the `code` conda env if you have it: `conda activate code`

```bash
# Default DB path (e.g. macOS: ~/Library/Application Support/Agent Cowork/sessions.db)
python scripts/export_task_sessions.py --pretty

# Custom DB path
python scripts/export_task_sessions.py --db /path/to/sessions.db -o out.json

# Single session
python scripts/export_task_sessions.py --session-id <uuid> -o session.json

# Override via env
AGENT_COWORK_DB=/path/to/sessions.db python scripts/export_task_sessions.py -o out.json

# Weight-based export (for training; see Formats table above)
python scripts/export_task_sessions.py --format weight-based -o out_weight.json

# Weight-based, single session + custom DB
python scripts/export_task_sessions.py --db /path/to/sessions.db --session-id <uuid> --format weight-based -o out.json
```

Output: one JSON object per session (or `{"sessions": [...]}` when exporting all), with the structure described in the script docstring.
