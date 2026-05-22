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

DPO, OPD, and REINFORCE now share the weight-format training stack under
`scripts/weight/`. Export sessions once with `--format weight`, then either run
the online server or invoke a specific trainer module.

```bash
python3 scripts/export_task_sessions.py --format weight -o scripts/out_weight.json

python3 -m scripts.weight.train.run_dpo \
  --train-path scripts/out_weight.json \
  --model-name Qwen/Qwen3-8B \
  --renderer-name qwen3 \
  --log-path logs/dpo

python3 -m scripts.weight.train.run_opd \
  --train-path scripts/out_weight.json \
  --model-name Qwen/Qwen3-8B \
  --renderer-name qwen3 \
  --log-path logs/opd

python3 -m scripts.weight.train.run_reinforce \
  --train-path scripts/out_weight.json \
  --model-name Qwen/Qwen3-8B \
  --renderer-name qwen3 \
  --log-path logs/reinforce
```

## Online RL Server

`server.py` runs a small **FastAPI** proxy in front of the Tinker OpenAI-compatible API. It accepts chat completions (streaming or not), enqueues exported **sessions** for training, and periodically runs **DPO**, **OPD**, or **REINFORCE** updates depending on `mode` in the YAML.

**Prerequisites**

- Set `TINKER_API_KEY` in the environment (or `.env` under the root directory).
- Install the dependencies: `pip install -r scripts/requirements.txt`

**Configuration**

Edit `config.yaml` (or any YAML with the same keys). Important fields:

- `mode`: `dpo` | `opd` | `reinforce` — which trainer runs when enough sessions are queued.
- `update_every_n_sessions`: enqueue this many sessions before triggering a training job.
- `proxy_host` / `proxy_port`: where the HTTP server listens (default `localhost:8000`).

**Run**

From the root directory:

```bash
python3 scripts/server.py --config scripts/config.yaml
```

`GET /healthz` returns `{"ok": true}` when the process is up.

**Frontend**

Point your provider at this proxy (base URL and port from `proxy_host` / `proxy_port`). Example layout:

![](assets/serverconfig.png)

**Training**

Training is triggered from the frontend with the **Train on this session** button in the lower-left corner. The frontend posts the current weight-format session to the server, and the server builds the dataset required by the configured training mode (`dpo`, `opd`, or `reinforce`) through `scripts/weight`.

Alternatively, you can also upload a session directly to the `/session` endpoint using the same weight format as `out_weight.json`. This triggers the same training flow as the frontend button, which can be useful for quick experiments or scripted runs.

The session is queued for training. Once `update_every_n_sessions` sessions have been processed, the server runs one Tinker update unless `dry_run` is enabled. During the checkpoint swap, inference waits until the new checkpoint is ready.

After training completes, the server broadcasts a `model-update` SSE event to the frontend. The frontend automatically points at the new checkpoint, so you normally do not need to switch models manually. If the new checkpoint is not visible in the model picker, click **Load models** to refresh the list.

The server persists the model registry, latest model path, and latest model-update event to `state_path` (default: `scripts/state.json`). On restart, this state is loaded so trained checkpoint slugs and the active model pointer are restored.
