# Weight Training Pipeline

End-to-end guide: install → train (DPO / OPD / REINFORCE) → serve LoRA adapters.

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
