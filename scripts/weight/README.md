# Weight Training Pipeline

End-to-end guide: install → train (DPO / OPD / REINFORCE) → serve LoRA adapters.

This pipeline supports **two backends**:

- **SkyRL** — local FSDP+vLLM server, see sections 1–4 below.
- **Tinker cloud API (default)** — managed service, no local server. See [section 5](#5-tinker-cloud-api-mode) for the alternative workflow. Switch to SkyRL by passing `--use-skyrl` to `run_dpo` / `run_opd`. REINFORCE works identically on both backends (no flag needed).

---

## 1. Setup

```bash
cd $SKYRL_DIR   # repo root that contains SkyRL/

# Hybrid-Mamba kernels (needed even for pure-Transformer models to satisfy imports)
uv pip install causal-conv1d==1.6.1 --no-build-isolation
uv pip install mamba-ssm --no-build-isolation
```

> **Compatible models**: pure-Transformer Qwen3 / Qwen2.5 / Qwen3-Instruct.  
> Qwen3.5 (hybrid Mamba-Transformer) causes CUDA errors in the FSDP backend — avoid it.

---

## 2. Start SkyRL Training Server

Run in a dedicated tmux pane. The server occupies one GPU for both training (FSDP) and inference (vLLM), which share memory.

```bash
cd $SKYRL_DIR

CUDA_VISIBLE_DEVICES=$GPU \
uv run --active --extra tinker --extra fsdp \
  -m skyrl.tinker.api \
  --base-model $MODEL \
  --backend fsdp \
  --port $PORT \
  --backend-config '{
    "generator.inference_engine.gpu_memory_utilization": 0.3,
    "generator.inference_engine.engine_init_kwargs": {"max_model_len": 32768}
  }'
```

Key parameters:

| Flag | Default | Notes |
|---|---|---|
| `--base-model` | — | HF model ID |
| `--port` | `8002` | Client must match |
| `gpu_memory_utilization` | `0.8` | **Lower to 0.3** to leave room for FSDP; raise only if OOM during sampling |
| `max_model_len` | model default | Reduce for long-context models to save KV-cache memory |
| `--backend-config '{"trainer.policy.language_model_only": true}'` | — | Add if server detects model as VLM and fails; forces text-only backbone |

**Single-model constraint**: SkyRL's FSDP backend supports only one model at a time. Wait ~5 min for session timeout, or fully restart the server between runs with different models.

**Cleanup** (stale processes / "Model already exists" error):
```bash
pkill -9 -f 'skyrl.tinker.api'
pkill -9 -f 'weight.train.run_'
uv run ray stop --force
rm -f tinker.db
```

---

## 3. Run Training

From `scripts/` directory. The server must already be running.

```bash
cd $REPO_ROOT/scripts

TINKER_API_KEY=tml-dummy \
TINKER_BASE_URL=http://localhost:$PORT \
python -m weight.train.run_dpo \       # or run_opd / run_reinforce
  --train-path $TRAIN_DATA \
  --model-name $MODEL \
  --renderer-name qwen3 \              # qwen3 | qwen3_5 | llama3 | …
  --log-path $LOG_DIR \
  --batch-size 2 \
  --num-epochs 5 \
  --lora-rank 16
```

**Renderer note**: use `qwen3` for both Qwen3 thinking and Qwen3-Instruct (non-thinking). The formatter converts the top-level `thinking` field to list-of-parts when present, so both data formats work automatically.

**LoRA adapter location**: after training, `checkpoints.jsonl` in `$LOG_DIR` records the `model_<id>`. The LoRA-only weights are at:
```
$CHECKPOINTS_BASE/model_<id>/sampler_weights/final.tar.gz   (~30 MB)
```
Extract and place in `$ADAPTER_DIR/<adapter_name>/`.

---

## 4. Serve with SGLang

Load the base model plus named LoRA adapters in one server:

```bash
CUDA_VISIBLE_DEVICES=$GPU python -m sglang.launch_server \
  --model-path $MODEL \
  --enable-lora \
  --lora-paths \
    adapter_a=$ADAPTER_DIR/adapter_a \
    adapter_b=$ADAPTER_DIR/adapter_b \
  --max-loras-per-batch 2 \
  --tp-size 1 \
  --mem-fraction-static 0.8 \
  --port 8200
```

> **Qwen3 (thinking) vs Qwen3-Instruct**: add `--reasoning-parser qwen3` for thinking models; omit for Instruct variants.

### Calling a specific adapter

Pass `"model": "<adapter_name>"` in your OpenAI-compatible request:

```python
client.chat.completions.create(model="adapter_a", messages=[...])
```

Or run a minimal port-proxy so each local port always routes to one adapter name:

```python
#!/usr/bin/env python3
import json
from aiohttp import web, ClientSession

UPSTREAM = "http://localhost:8200"
PORT_MODEL = {8212: "adapter_a", 8213: "adapter_b"}

async def proxy(request: web.Request):
    body = await request.read()
    if body:
        data = json.loads(body)
        data["model"] = PORT_MODEL[request.url.port]
        body = json.dumps(data).encode()
    async with ClientSession() as s:
        async with s.request(request.method, UPSTREAM + request.path_qs, data=body, headers={k: v for k, v in request.headers.items() if k.lower() != "host"}) as r:
            return web.Response(status=r.status, body=await r.read(), headers=r.headers)

app = web.Application()
app.router.add_route("*", "/{path:.*}", proxy)
web.run_app(app, host="0.0.0.0", port=8212)
```

### Hot-reload (no server restart)

**Option A — replace in place** (recommended, port mapping unchanged):
```bash
# Unload old
curl -X POST http://localhost:8200/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "adapter_a"}'

# Load new weights under the same name
curl -X POST http://localhost:8200/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "adapter_a", "lora_path": "$ADAPTER_DIR/adapter_a_v2", "pinned": true}'
```
Clients and proxy need no changes.

**Option B — new name**:
```bash
curl -X POST http://localhost:8200/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "adapter_a_v2", "lora_path": "$ADAPTER_DIR/adapter_a_v2", "pinned": true}'
```
Then add the new name to the port-proxy mapping and restart the proxy. Requests to the old port still use the old adapter until you update the mapping.

---

## 5. Tinker Cloud API Mode

When you have a Tinker API key, you can skip sections 1–2 entirely — no local GPU server, no `mamba-ssm`, no `tinker.db`. Tinker hosts the model and exposes the same training/sampling clients used by SkyRL, plus full `compute_logprobs_async` (which SkyRL's vLLM backend currently lacks).

### 5.1 Configure the API key

```bash
cd $REPO_ROOT/scripts/weight
cp .env.example .env
# Edit .env and set:
#   TINKER_API_KEY=tml-your-real-key
# Leave TINKER_BASE_URL commented out to use the production cloud.
```

`.env` is loaded automatically by `run_dpo.py`, `run_opd.py`, and `run_reinforce.py`. Existing process env vars override `.env`, so you can still do an inline override:

```bash
TINKER_API_KEY=tml-other-key python -m weight.train.run_dpo ...
```

### 5.2 Run training (default = Tinker)

Same module entry points; no extra flag needed for Tinker (and drop the `TINKER_BASE_URL=...` prefix):

```bash
cd $REPO_ROOT/scripts

# DPO on Tinker
python -m weight.train.run_dpo \
  --train-path $TRAIN_DATA \
  --model-name Qwen/Qwen3-4B \
  --renderer-name qwen3 \
  --log-path ./logs/dpo_run \
  --batch-size 2 \
  --num-epochs 5 \
  --lora-rank 16

# OPD on Tinker
python -m weight.train.run_opd \
  --train-path $TRAIN_DATA \
  --model-name Qwen/Qwen3-4B \
  --renderer-name qwen3 \
  --log-path ./logs/opd_run \
  --batch-size 2 \
  --num-epochs 5 \
  --lora-rank 16

# REINFORCE on Tinker (no flag — algorithm has no ref/teacher dependency)
python -m weight.train.run_reinforce \
  --train-path $TRAIN_DATA \
  --model-name Qwen/Qwen3-4B \
  --renderer-name qwen3 \
  --log-path ./logs/reinforce_run \
  --batch-size 2
```

#### What `--use-skyrl` changes

| Algorithm | Default (Tinker) | `--use-skyrl` |
|---|---|---|
| DPO | Snapshot once: `reference_client = training_client.save_weights_and_get_sampling_client()`. Per batch: `reference_client.compute_logprobs_async`. Canonical DPO recipe. | Pre-compute every datum's reference logprobs once via `training_client.forward()`, cache by content fingerprint. Workaround for SkyRL's missing `prompt_logprobs`. |
| OPD | Frozen base-model `teacher_client = service_client.create_sampling_client(base_model=...)`. Per batch: `teacher_client.compute_logprobs_async`. | Pre-compute teacher logprobs for every (student, teacher) pair via `training_client.forward()`. |
| REINFORCE | No ref/teacher dependency — same code path on both backends. | Same. |

The two paths produce mathematically equivalent training: both fix the reference / teacher distribution at the initial weights and use it to score student outputs. The Tinker path just does it batch-by-batch instead of caching everything up front.

### 5.3 Pick a model

Pass any Hugging Face model ID supported by Tinker as `--model-name`. Tinker handles GPU allocation; you don't choose hardware. Compatible families: Qwen3 / Qwen2.5 / Llama-3 / Qwen3-Instruct, etc. Qwen3.5 hybrid models (which fail under SkyRL's FSDP backend) may also work on Tinker — try if you need them.

### 5.4 View training results

Three options, in order of convenience:

1. **Tinker dashboard** — log in at [tinkerlabs.ai](https://tinkerlabs.ai). Each `create_lora_training_client` call registers a run with loss curves, custom metrics, and downloadable checkpoints.
2. **Weights & Biases** — pass `--wandb-project my-project` (and optionally `--wandb-name run-name`) to any training script. Streams live alongside the Tinker dashboard.
3. **Local files** — every run writes `metrics.jsonl`, `checkpoints.jsonl`, and `timing_spans.jsonl` under `--log-path`. Quick plot:

   ```python
   import json, pandas as pd, matplotlib.pyplot as plt
   rows = [json.loads(l) for l in open("logs/dpo_run/metrics.jsonl")]
   df = pd.DataFrame(rows)
   df.plot(x="epoch", y="dpo_loss")
   plt.show()
   ```

### 5.5 Get the LoRA adapter

After training, `checkpoints.jsonl` records the final `model_<id>` and a `state_path`. Download via the Tinker dashboard, or with the SDK:

```python
import tinker
sc = tinker.ServiceClient()
sc.download_sampler_weights(state_path, output_dir="./adapters/my_adapter")
```

The resulting directory is a standard LoRA adapter — load it the same way as section 4 (SGLang + `--lora-paths my_adapter=./adapters/my_adapter`). Hot-reload via the same `/load_lora_adapter` endpoint.
