# Training Agent Backbone LM

We use Tinker API to train the agent backbone LM by default. If you want to host the models yourself, we also offer the SkyRL backend.

## 1. Training with Tinker API

Set your Tinker API key in the environment variable `TINKER_API_KEY`.
```bash
TINKER_API_KEY={your-tinker-api-key}
```

Then run the training command, such as:
```bash
python -m weight.train.run_dpo \
  --train-path $TRAIN_DATA \
  --model-name Qwen/Qwen3.5-35B-A3B \
  --renderer-name qwen3_5 \
  --log-path ./logs/dpo_agentic_run \
  --agentic-rollout \
  --agentic-num-rollouts 4 \
  --agentic-include-first-last \
  --batch-size 2 \
  --num-epochs 4 \
  --learning-rate 5e-5 \
  --lora-rank 32
```
After training, `checkpoints.jsonl` records the trained model's `state_path` (for trainig) and `sampler_path` (for inference).

To supervise the training process, you can use:
1. **Tinker dashboard**. Log in at [tinkerlabs.ai](https://tinkerlabs.ai). Each `create_lora_training_client` call registers a run with loss curves, custom metrics, and downloadable checkpoints.
2. **Weights & Biases**. Pass `--wandb-project my-project` (and optionally `--wandb-name run-name`) to any training script. Streams live alongside the Tinker dashboard.
3. **Local files**. Every run writes `metrics.jsonl`, `checkpoints.jsonl`, and `timing_spans.jsonl` under `--log-path`. Quick plot:

   ```python
   import json, pandas as pd, matplotlib.pyplot as plt
   rows = [json.loads(l) for l in open("logs/dpo_run/metrics.jsonl")]
   df = pd.DataFrame(rows)
   df.plot(x="epoch", y="dpo_loss")
   plt.show()
   ```


## 2. Training with SkyRL

Install and run the SkyRL training server.

```bash
cd $SKYRL_DIR   # repo root that contains SkyRL/

# Hybrid-Mamba kernels (needed even for pure-Transformer models to satisfy imports)
uv pip install causal-conv1d==1.6.1 --no-build-isolation
uv pip install mamba-ssm --no-build-isolation
```

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

### Run Training

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


### Serve with SGLang

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
