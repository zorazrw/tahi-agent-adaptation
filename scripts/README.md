# Data Export and Agent Update

## Data Export

`export_task_sessions.py` is a standalone Python 3 script (stdlib only) to export task sessions from the Agent Cowork SQLite database to JSON.

**Usage:**
```bash
python scripts/export_task_sessions.py -o out.json

# Single session
python scripts/export_task_sessions.py --session-id <uuid> -o session.json

# Weight-based export (for training; see Formats table above)
python scripts/export_task_sessions.py --format weight -o out_weight.json
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
python export_task_sessions.py -o out.json

# DPO
python export_dpo_data.py out.json -o out_dpo.json
bash tinker_dpo.sh

# OPD
python export_opd_data.py out.json -o out_opd.json
bash tinker_opd.sh

# REINFORCE
python export_reinforce_data.py out.json -o out_reinforce.json
bash tinker_reinforce.sh
```