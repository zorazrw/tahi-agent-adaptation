import json
import os
import tempfile
import yaml
import logging
import asyncio
import argparse
import time
import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from contextlib import asynccontextmanager

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer
from weight.train.formatter import WeightDPODataBuilder, WeightReinforceDataBuilder
from weight.train.run_dpo import (
    Config as DPOConfig,
    OnlineDPOAcceptedDataset,
    OnlineDPOAgenticDataset,
)
from weight.train.run_reinforce import Config as ReinforceConfig, OnlineReinforceRolloutDataset
from weight.train.run_opd import Config as ArtifactOPDConfig, OnlineOPDRolloutDataset

from model_manager import ModelManager, ModelUpdate
from trainer import DPOTrainer, OPDTrainer, REINFORCETrainer, Trainer, TrainingCheckpoint

from dotenv import load_dotenv
load_dotenv()


@dataclass
class Config:
    # -- Mode --
    mode: Literal["dpo", "opd", "reinforce"] = "reinforce"

    # -- Inference --
    temperature: float = 0.0
    logprobs: bool = False

    # -- Update --
    update_every_n_sessions: int = 1 # update the model after n new sessions are processed

    # -- Training --
    lora_rank: int = 32
    dry_run: bool = False # skip training if enabled
    learning_rate: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 1
    lr_schedule: str = "linear"
    max_steps: int | None = None
    max_length: int | None = None
    save_every: int = 20

    # -- DPO-specific --
    dpo_beta: float = 0.1
    dpo_pair_mode: Literal["first_last", "adjacent"] = "first_last"
    dpo_rpo_alpha: float = 0.0
    dpo_use_ipo: bool = False
    dpo_online_rollout: bool = False
    dpo_rollout_max_tokens: int = 4096
    dpo_rollout_temperature: float = 1.0
    dpo_rollout_attempts: int = 1
    dpo_log_rollout_samples: bool = True
    dpo_rollout_sample_log_chars: int = 4000
    dpo_artifact_only_rollout_instruction: bool = False

    # -- DPO agentic mode (dpo_agentic_rollout=True) --
    # The rejected side is produced by a full multi-turn tool-using rollout in a
    # live sandbox; the student's final artifact (matched to the chosen file by
    # basename) becomes the rejected write. Only the artifact write is trained on.
    dpo_agentic_rollout: bool = False
    # "on_policy": refresh the sampler each step so negatives track the current
    # policy. "off_policy": keep the frozen initial snapshot (== reference model)
    # for the whole run, using rollouts as a one-shot negative-augmentation source.
    dpo_agentic_policy_mode: Literal["on_policy", "off_policy"] = "on_policy"
    dpo_agentic_num_rollouts: int = 1        # parallel rejected rollouts per session
    # batch_size is the target #preference-pairs per batch; each batch draws
    # batch_size // dpo_agentic_pairs_per_session sessions, each contributing
    # exactly that many pairs (subsampled if it yields more, oversampled if fewer).
    dpo_agentic_pairs_per_session: int = 1
    # extra sessions fetched per batch to replace sessions that yield zero pairs
    dpo_agentic_session_reserve: int = 0
    # cap on rollouts running concurrently (0 = unlimited); bounds peak memory
    dpo_agentic_max_concurrent_rollouts: int = 0
    # additionally inject each session's first-written artifact version as a
    # synthetic rejected snapshot (the offline first_last pair), alongside the
    # on-policy model rollouts.
    dpo_agentic_include_first_last: bool = False
    # min content-similarity (0..1) for matching a student file to a chosen
    # artifact by content/type rather than filename (lower => more, looser pairs)
    dpo_agentic_match_min_similarity: float = 0.05
    dpo_agentic_max_turns: int = 48          # overall safety ceiling per episode
    dpo_agentic_max_turns_per_step: int = 8  # inner agent-loop cap within a step
    dpo_agentic_max_steps: int = 6           # max planned steps replayed
    dpo_agentic_enable_bash: bool = True
    dpo_agentic_tool_timeout_s: int = 20
    dpo_agentic_max_trajectory_tokens: int | None = None
    dpo_min_artifact_versions: int = 1

    # -- OPD-specific --
    opd_max_tokens: int = 2048
    opd_temperature: float = 1.0
    opd_topk: int = 20
    opd_max_context_length: int = 32768
    opd_pair_mode: Literal["first_last", "adjacent"] = "first_last"
    opd_use_gt: bool = True
    opd_use_student: bool = True
    opd_rollout_attempts: int = 1
    opd_teacher_temperature: float = 1.0
    opd_log_rollout_samples: bool = True
    opd_rollout_sample_log_chars: int = 4000
    opd_log_teacher_prompts: bool = True
    opd_artifact_only_rollout_instruction: bool = False
    # "v1"/"v2" use a historical completion per task unit; "agentic" runs one
    # continuous on-policy multi-turn tool-using rollout per session in a live
    # sandbox and distils the whole trajectory against a follow-up-augmented
    # teacher (see opd_agentic_* below).
    opd_extract_version: Literal["v1", "v2", "agentic"] = "v2"

    # -- OPD agentic mode (opd_extract_version="agentic") --
    # Each session is rolled out as a planning turn + one "Proceed with: <step>"
    # user turn per planned leaf step (steps derived from the model's own plan).
    opd_agentic_max_turns: int = 48          # overall safety ceiling per episode
    opd_agentic_max_turns_per_step: int = 8  # inner agent-loop cap within a step
    opd_agentic_max_steps: int = 6           # max planned steps replayed
    opd_agentic_enable_bash: bool = True
    opd_agentic_tool_timeout_s: int = 20
    opd_agentic_max_trajectory_tokens: int | None = None
    # Reasoning-renderer toggle (Qwen3/Kimi K2/DeepSeek thinking). When False,
    # the golden answer's ``<think>...</think>`` block survives in the
    # teacher prompt so the teacher can attend to the golden chain-of-thought.
    # When True, the renderer's default behavior strips CoT from non-last
    # assistant turns — which silently hides the golden CoT from the teacher
    # because SDFT teacher prompts always end with the ``redo`` user message.
    # Default False matches legacy ``tinker_opd.Config``.
    opd_strip_thinking_from_history: bool = False

    # OPD rollout pipeline.  "current" (default) uses the in-house
    # ``sampling_client.sample_async`` + parse-filter + canonicalization path
    # that the v2 extractor was originally paired with.  "legacy" routes
    # through ``tinker_cookbook.rl.rollouts.do_group_rollout_and_filter_constant_reward``
    # + ``assemble_training_data`` — the exact dispatch path the legacy
    # ``tinker_opd`` recipe used.  Pick "legacy" when you want the trained
    # tokens to be the raw sampled tokens (no canonicalization, no
    # parse-filter, no retry) — i.e. bit-identical training dynamics to the
    # legacy codebase.
    opd_rollout_pipeline: Literal["current", "legacy"] = "current"

    # -- REINFORCE-specific --
    reinforce_version: Literal["offline", "online"] = "offline"
    reward_alpha: float = 0.05
    initial_baseline: float = 0.0

    # REINFORCE online (reinforce_version="online")
    reinforce_rollout_max_tokens: int = 4096
    reinforce_rollout_temperature: float = 1.0
    reinforce_agentic_max_turns: int = 48
    reinforce_agentic_max_turns_per_step: int = 8
    reinforce_agentic_max_steps: int = 6
    reinforce_agentic_enable_bash: bool = True
    reinforce_agentic_tool_timeout_s: int = 20
    reinforce_agentic_max_trajectory_tokens: int | None = None
    reinforce_log_rollout_samples: bool = True
    reinforce_rollout_sample_log_chars: int = 4000

    # -- Logging --
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_root: str = "logs/online_training"
    experiment_name: str | None = None

    # -- Proxy Server --
    proxy_host: str = "localhost"
    proxy_port: int = 8000
    state_path: str | None = None
    # When true, /v1/chat/completions requests are rejected with HTTP 503
    # while a training round (or checkpoint swap) is in progress, rather
    # than being forwarded to Tinker. When false (default), inference
    # requests proceed concurrently with training and only briefly wait
    # for the checkpoint swap.
    block_during_training: bool = False

    # -- Model lifecycle --
    # Slug (from the registry) or base model id (e.g. "Qwen/Qwen3-8B") to
    # pre-warm a Tinker training client for at server startup. When None,
    # the training client is provisioned lazily on first use.
    preload_model: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        valid_keys = {fld.name for fld in fields(cls)}
        unknown = set(raw) - valid_keys
        if unknown:
            log.warning("Ignoring unknown config keys in %s: %s", path, ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in raw.items() if k in valid_keys})


@asynccontextmanager
async def lifespan(app: FastAPI):
    owner: "Server" = app.state.owner
    await owner.model_manager.start()
    t1 = asyncio.create_task(owner._process_sessions())
    t2 = asyncio.create_task(owner._training_consumer())
    try:
        yield
    finally:
        t1.cancel()
        t2.cancel()
        for t in (t1, t2):
            try:
                await t
            except asyncio.CancelledError:
                pass
        await owner.model_manager.aclose()
        

class Server:
    _TITLE = "Agent Cowork Tinker Proxy"
    _TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    _API_KEY = os.getenv("TINKER_API_KEY")

    def __init__(
        self,
        config: Config,
    ):
        if not self._API_KEY:
            raise RuntimeError("TINKER_API_KEY environment variable is required")

        self.config = config
        self.experiment_name = self._resolve_experiment_name(config.experiment_name, config.mode)
        self.experiment_dir = Path(config.log_root).expanduser() / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        resolved_state_path = self._resolve_state_path(config.state_path)
        self.config.state_path = str(resolved_state_path)

        log.info(
            "Experiment initialized: name=%s dir=%s state_path=%s",
            self.experiment_name,
            self.experiment_dir,
            self.config.state_path,
        )

        self.model_manager = ModelManager(
            tinker_api_key=self._API_KEY,
            tinker_base_url=self._TINKER_BASE_URL,
            state_path=self.config.state_path,
            mode=config.mode,
            lora_rank=config.lora_rank,
            preload_model=config.preload_model,
        )

        self.sessions_queue: asyncio.Queue = asyncio.Queue()
        self.training_queue: asyncio.Queue = asyncio.Queue()
        self.training_event = asyncio.Event()
        self.training_lock = asyncio.Lock()

    def _resolve_experiment_name(self, configured: str | None, mode: str) -> str:
        name = (configured or "").strip()
        if not name:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{mode}_{stamp}"
        sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
        return sanitized or f"{mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _resolve_state_path(self, configured: str | None) -> Path:
        raw = (configured or "").strip()
        if not raw:
            return self.experiment_dir / "state.json"
        return Path(raw).expanduser()

    def _new_round_log_path(self, mode: str) -> str:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.experiment_dir / mode / stamp
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
        
    def _build_app(self) -> FastAPI:
        app = FastAPI(title=self._TITLE, lifespan=lifespan)
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/session")
        async def _handle_session(request: Request):
            body = (await request.json())[0]
            
            session_id = body.get("uuid")
            name = body.get("name", "")
            
            await self.sessions_queue.put(body)
            
            log.info("Session enqueued: id=%s name=%r queue_depth=%d",
                     session_id, name, self.sessions_queue.qsize())
            return {"ok": True, "session_id": session_id}

        @app.get("/v1/models")
        async def _get_models(request: Request):
            data = [
                {
                    "id": slug,
                    "object": "model",
                    "created": None,
                    "owned_by": "agent-cowork",
                }
                for slug in self.model_manager.list_slugs()
            ]
            return JSONResponse({"object": "list", "data": data})

        @app.get("/v1/tinker/current")
        async def _get_tinker_current():
            """Return the most recent model update (or 204 if none yet).

            Used by the Electron main process to reconcile on startup/reconnect.
            """
            latest = self.model_manager.latest_update
            if latest is None:
                return Response(status_code=204)
            return JSONResponse(latest)

        @app.get("/v1/tinker/events")
        async def _stream_tinker_events(request: Request):
            """Server-Sent Events stream of model-update notifications.

            On connect we immediately emit the current latest_update (if any)
            so late subscribers converge without needing a separate GET.
            Heartbeat comments keep proxies/load balancers from closing the
            connection while training is idle.
            """
            queue = self.model_manager.subscribe()

            async def event_stream():
                try:
                    latest = self.model_manager.latest_update
                    if latest is not None:
                        yield f"event: model-update\ndata: {json.dumps(latest)}\n\n"
                    while True:
                        if await request.is_disconnected():
                            break
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        except asyncio.TimeoutError:
                            yield ": heartbeat\n\n"
                            continue
                        yield f"event: model-update\ndata: {json.dumps(event)}\n\n"
                finally:
                    self.model_manager.unsubscribe(queue)

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        @app.post("/v1/chat/completions")
        async def _handle_chat_completions(request: Request):
            # When configured, reject inference requests while a training
            # round (or checkpoint swap) is in progress instead of queueing.
            if self.config.block_during_training and (
                self.training_lock.locked() or not self.model_manager.model_ready.is_set()
            ):
                log.info("Rejecting chat completion: training in progress")
                return JSONResponse(
                    status_code=503,
                    headers={"Retry-After": "30"},
                    content={
                        "error": {
                            "message": (
                                "Training run in progress; inference is temporarily "
                                "unavailable. Please retry after training completes."
                            ),
                            "type": "service_unavailable",
                            "code": "training_in_progress",
                        }
                    },
                )

            # Block until any in-flight checkpoint swap finishes.
            await self.model_manager.model_ready.wait()

            body = await request.json()

            messages = body.pop("messages", [])
            stream = body.pop("stream", True)
            model_name = body.pop("model", None)
            if model_name is None:
                raise HTTPException(status_code=400, detail="model is required")
            if not self.model_manager.has_slug(model_name):
                raise HTTPException(status_code=400, detail=f"model {model_name} not found")

            model_path = self.model_manager.resolve_slug(model_name)
            
            # switch model if it is not the current model
            if model_path != self.model_manager.model_path or not self.model_manager.model_ready.is_set():
                self.model_manager.set_active(model_path)
                log.info("Switching model to: %s", model_path)
            else:
                log.info("Model is already active: %s", model_path)

            # Resolve renderer eagerly so we can (a) select a streaming
            # rewriter and (b) kick off lazy training-client warmup in the
            # background for whatever base model backs this slug.
            try:
                base_model, renderer_name = self.model_manager.resolve_renderer(model_path)
            except Exception:
                log.exception("Failed to resolve renderer for %s; streaming without rewrite", model_path)
                base_model, renderer_name = None, None
                
            print("=" * 80, flush=True)
            print("CHAT COMPLETIONS REQUEST:", flush=True)
            print(json.dumps(body, indent=2, default=str), flush=True)
            print("=" * 80, flush=True)


            if base_model is not None:
                # Fire-and-forget. If a warmup is already in flight, the
                # per-base-model lock inside ModelManager coalesces.
                asyncio.create_task(
                    self.model_manager.get_training_client(base_model)
                )

            log.info("Chat completion request: model=%s stream=%s messages=%d",
                     model_path, stream, len(messages))

            # Override inference parameters
            body["temperature"] = self.config.temperature
            body["logprobs"] = self.config.logprobs

            client = self.model_manager.inference_client
            
            if stream:

                async def generate():
                    chunks = []
                    state = {}

                    log.info("Forwarding chat completion request to Tinker...")
                    response = await client.chat.completions.create(
                        model=model_path,
                        messages=messages,
                        stream=True,
                        **body,
                    )
                    async for chunk in response:
                        log.info("Received chat completion chunk: %s", json.dumps(chunk.model_dump(), indent=4))
                        chunk_dict = chunk.model_dump()
                        yield f"data: {json.dumps(chunk_dict)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(generate(), media_type="text/event-stream")

            response = await client.chat.completions.create(
                model=model_path,
                messages=messages,
                **body,
            )
            log.info("Chat completion response: %s", response.model_dump_json(indent=4))
            return JSONResponse(response.model_dump())

        return app

    async def _process_sessions(self):
        while True:
            session_data = await self.sessions_queue.get()
            try:
                if self.config.mode == "dpo":
                    data = session_data
                elif self.config.mode == "opd":
                    data = session_data
                elif self.config.mode == "reinforce":
                    data = session_data
                else:
                    raise ValueError(f"Unknown training mode: {self.config.mode} (expected one of 'dpo', 'opd', 'reinforce')")

                if data:
                    await self.training_queue.put(data)
                    unit_count = len(data.get("task_units", [])) if isinstance(data, dict) else 0
                    if unit_count == 0 and isinstance(data, dict):
                        unit_count = len(data.get("learning_units", []))
                    log.info("Session processed: id=%s mode=%s units=%d training_queue=%d/%d",
                            session_data.get('uuid'), self.config.mode,
                            unit_count,
                            self.training_queue.qsize(), self.config.update_every_n_sessions)

                # trigger training if the queue has reached the update threshold
                if self.training_queue.qsize() >= self.config.update_every_n_sessions:
                    self.training_event.set()

            except Exception:
                log.exception("Session processing failed: id=%s mode=%s",
                              session_data.get('uuid'), self.config.mode)

            finally:
                self.sessions_queue.task_done()

    def _drain_queue(self) -> list:
        items = []
        cap = self.config.update_every_n_sessions
        while not self.training_queue.empty() and len(items) < cap:
            try:
                items.append(self.training_queue.get_nowait())
                self.training_queue.task_done()
            except asyncio.QueueEmpty:
                break
        return items

    async def _training_consumer(self):
        while True:
            await self.training_event.wait()
            self.training_event.clear()

            async with self.training_lock:
                items = self._drain_queue()

                # Skip training if dry run is enabled, for the sake of testing other parts of the server
                if self.config.dry_run:
                    log.info("Dry run: skipping training for %d sessions", len(items))
                    continue

                if not items:
                    log.info("No sessions to train, skipping")
                    continue

                log.info("Training started: mode=%s sessions=%d current_model=%s",
                         self.config.mode, len(items), self.model_manager.model_path)

                try:
                    checkpoint = await self._train_round(items)

                    prev_model = self.model_manager.model_path

                    # Gate inference while we swap in the new checkpoint.
                    self.model_manager.model_ready.clear()
                    try:
                        log.info("Updating model path to %s", checkpoint.sampler_path)
                        await self.model_manager.update_model_path(checkpoint.sampler_path, timeout=600.0)

                        new_model_name = self.model_manager.next_slug(self.config.mode)
                        self.model_manager.register_checkpoint(
                            new_model_name,
                            checkpoint.sampler_path,
                            state_path=checkpoint.state_path,
                        )
                        log.info(
                            "Model updated: name=%s sampler_path=%s state_path=%s",
                            new_model_name, checkpoint.sampler_path, checkpoint.state_path,
                        )

                        # Broadcast to SSE subscribers so the Electron app can
                        # atomically rotate the Tinker provider config onto the
                        # new slug without polling.
                        try:
                            base_model_name, renderer_name = self.model_manager.resolve_renderer(
                                checkpoint.sampler_path
                            )
                        except Exception:
                            log.exception(
                                "Failed to resolve renderer for %s; broadcasting without renderer",
                                checkpoint.sampler_path,
                            )
                            base_model_name, renderer_name = None, None

                        self.model_manager.broadcast(ModelUpdate(
                            slug=new_model_name,
                            model_path=checkpoint.sampler_path,
                            base_model=base_model_name,
                            renderer_name=renderer_name,
                            mode=self.config.mode,
                            state_path=checkpoint.state_path,
                        ))
                        # Persist the updated latest_update alongside the
                        # registry entry written by register_checkpoint.
                        self.model_manager.save_state()

                    except TimeoutError:
                        log.error("Checkpoint %s never became ready, keeping previous model %s",
                                checkpoint.sampler_path, prev_model)
                    finally:
                        self.model_manager.model_ready.set()

                    log.info("Training complete: mode=%s sessions=%d model=%s (was %s)",
                             self.config.mode, len(items), checkpoint.sampler_path, prev_model)
                except Exception:
                    log.exception("Training failed: mode=%s sessions=%d",
                                  self.config.mode, len(items))

            if self.training_queue.qsize() >= self.config.update_every_n_sessions:
                self.training_event.set()

    def _write_sessions_file(self, items: list) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"sessions": items}, f)
        f.close()
        return f.name

    def _load_checkpoint_path_for_active_model(self) -> str | None:
        """Return the resumable state path for the current active checkpoint."""
        model_path = self.model_manager.model_path
        if not model_path or not model_path.startswith("tinker://"):
            return None

        state_path = self.model_manager.training_state_path_for(model_path)
        if state_path is None:
            raise RuntimeError(
                f"Active checkpoint {model_path!r} has no persisted training state path. "
                "Select a base model or a checkpoint created after training_state_paths support was added."
            )
        return state_path

    async def _train_round(self, items: list) -> TrainingCheckpoint:
        """Run one online training round using the persistent mode-specific Trainer.

        The Trainer is obtained via :meth:`ModelManager.get_trainer`, which
        caches a single long-lived instance per base model. On the first call
        the mode-specific ``build`` closure constructs the Trainer (and, for
        DPO, snapshots the initial weights into a reference sampling client);
        subsequent rounds reuse the same Trainer so ``step_idx``, optimizer
        state, running baselines, etc. carry over.

        A fresh dataset is assembled from ``items`` every round and handed to
        ``trainer.do_update``, which returns sampler and state paths for the
        checkpoint produced this round.
        """
        train_path = self._write_sessions_file(items)
        try:
            model_path = self.model_manager.model_path
            if model_path is None:
                if self.config.preload_model:
                    model_path = (
                        self.model_manager.resolve_slug(self.config.preload_model)
                        if self.model_manager.has_slug(self.config.preload_model)
                        else self.config.preload_model
                    )
                    self.model_manager.set_active(model_path)
                    log.info("No active model; using preload_model=%s for training", model_path)
                else:
                    raise RuntimeError(
                        "No active model; send one chat completion or set preload_model before training"
                    )
            model_name, renderer_name = self.model_manager.resolve_renderer(model_path)
            load_checkpoint_path = self._load_checkpoint_path_for_active_model()

            if self.config.mode == "dpo":
                dataset, build = self._prepare_dpo(train_path, model_name, renderer_name)
            elif self.config.mode == "reinforce":
                dataset, build = self._prepare_reinforce(train_path, model_name, renderer_name)
            elif self.config.mode == "opd":
                dataset, build = self._prepare_opd(train_path, model_name, renderer_name)
            else:
                raise ValueError(
                    f"Unknown training mode: {self.config.mode} "
                    f"(expected one of 'dpo', 'reinforce', 'opd')"
                )

            trainer = await self.model_manager.get_trainer(
                base_model=model_name,
                build=build,
                load_checkpoint_path=load_checkpoint_path,
            )
            if self.config.mode == "opd":
                checkpoint = await trainer.do_update(dataset, num_epochs=self.config.num_epochs)
            else:
                checkpoint = await trainer.do_update(dataset)
            return checkpoint
        finally:
            os.unlink(train_path)

    def _prepare_dpo(self, train_path: str, model_name: str, renderer_name: str):
        if self.config.dpo_agentic_rollout:
            tokenizer = get_tokenizer(model_name)
            renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
            dataset_builder = None
            dataset = OnlineDPOAgenticDataset.from_weight_json(
                path=train_path,
                renderer=renderer,
                max_length=self.config.max_length,
                batch_size=self.config.batch_size,
                pairs_per_session=self.config.dpo_agentic_pairs_per_session,
                session_reserve=self.config.dpo_agentic_session_reserve,
                min_versions=self.config.dpo_min_artifact_versions,
                artifact_only_instruction=self.config.dpo_artifact_only_rollout_instruction,
            )
            if not dataset._rows:
                raise ValueError(
                    f"No DPO final-artifact session rows in {train_path} "
                    "(need accepted output_files + an initial task in the weight JSON)."
                )
            log.info(
                "DPO agentic: %d session rows, batch_size=%d",
                len(dataset._rows), self.config.batch_size,
            )
        elif self.config.dpo_online_rollout:
            tokenizer = get_tokenizer(model_name)
            renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
            dataset_builder = None
            dataset = OnlineDPOAcceptedDataset.from_weight_json(
                path=train_path,
                renderer=renderer,
                max_length=self.config.max_length,
                batch_size=self.config.batch_size,
                artifact_only_instruction=self.config.dpo_artifact_only_rollout_instruction,
            )
            if not dataset._rows:
                raise ValueError(
                    f"No DPO accepted artifact rows in {train_path} "
                    "(need accepted output_files in the weight JSON)."
                )
            log.info(
                "DPO online: %d accepted artifact rows, batch_size=%d",
                len(dataset._rows), self.config.batch_size,
            )
        else:
            dataset_builder = WeightDPODataBuilder(
                train_path=train_path,
                pair_mode=self.config.dpo_pair_mode,
                common_config=ChatDatasetBuilderCommonConfig(
                    model_name_for_tokenizer=model_name,
                    renderer_name=renderer_name,
                    max_length=self.config.max_length,
                    batch_size=self.config.batch_size,
                ),
            )
            dataset, _ = dataset_builder()

        log_path = self._new_round_log_path("dpo")
        trainer_config = DPOConfig(
            log_path=log_path,
            model_name=model_name,
            dataset_builder=dataset_builder,
            renderer_name=renderer_name,
            learning_rate=self.config.learning_rate,
            lr_schedule=self.config.lr_schedule,
            dpo_beta=self.config.dpo_beta,
            rpo_alpha=self.config.dpo_rpo_alpha,
            use_ipo=self.config.dpo_use_ipo,
            online_rollout=self.config.dpo_online_rollout,
            rollout_max_tokens=self.config.dpo_rollout_max_tokens,
            rollout_temperature=self.config.dpo_rollout_temperature,
            rollout_attempts=self.config.dpo_rollout_attempts,
            log_rollout_samples=self.config.dpo_log_rollout_samples,
            rollout_sample_log_chars=self.config.dpo_rollout_sample_log_chars,
            artifact_only_rollout_instruction=self.config.dpo_artifact_only_rollout_instruction,
            agentic_rollout=self.config.dpo_agentic_rollout,
            agentic_policy_mode=self.config.dpo_agentic_policy_mode,
            agentic_num_rollouts=self.config.dpo_agentic_num_rollouts,
            agentic_pairs_per_session=self.config.dpo_agentic_pairs_per_session,
            agentic_session_reserve=self.config.dpo_agentic_session_reserve,
            agentic_max_concurrent_rollouts=self.config.dpo_agentic_max_concurrent_rollouts,
            agentic_match_min_similarity=self.config.dpo_agentic_match_min_similarity,
            agentic_include_first_last=self.config.dpo_agentic_include_first_last,
            agentic_max_turns=self.config.dpo_agentic_max_turns,
            agentic_max_turns_per_step=self.config.dpo_agentic_max_turns_per_step,
            agentic_max_steps=self.config.dpo_agentic_max_steps,
            agentic_enable_bash=self.config.dpo_agentic_enable_bash,
            agentic_tool_timeout_s=self.config.dpo_agentic_tool_timeout_s,
            agentic_max_trajectory_tokens=self.config.dpo_agentic_max_trajectory_tokens,
            min_artifact_versions=self.config.dpo_min_artifact_versions,
            num_epochs=self.config.num_epochs,
            lora_rank=self.config.lora_rank,
            save_every=self.config.save_every,
            max_steps=self.config.max_steps,
            wandb_project=self.config.wandb_project,
            wandb_name=self.config.wandb_name,
        )

        def build(
            training_client: tinker.TrainingClient,
            service_client: tinker.ServiceClient,
        ) -> Trainer:
            reference_client = training_client.save_weights_and_get_sampling_client()
            return DPOTrainer(
                logger=log,
                config=trainer_config,
                training_client=training_client,
                service_client=service_client,
                reference_client=reference_client,
            )

        return dataset, build

    def _prepare_reinforce(self, train_path: str, model_name: str, renderer_name: str):
        log_path = self._new_round_log_path("reinforce")
        is_online = self.config.reinforce_version == "online"

        if is_online:
            dataset = OnlineReinforceRolloutDataset.from_weight_json(
                train_path, self.config.batch_size,
            )
            if not dataset._rows:
                raise ValueError(
                    f"No REINFORCE online rollout seeds in {train_path} "
                    "(need sessions with initial_task_instruction)."
                )
            dataset_builder = None
            log.info(
                "REINFORCE online: %d rollout seeds, batch_size=%d",
                len(dataset._rows), self.config.batch_size,
            )
        else:
            dataset_builder = WeightReinforceDataBuilder(
                train_path=train_path,
                common_config=ChatDatasetBuilderCommonConfig(
                    model_name_for_tokenizer=model_name,
                    renderer_name=renderer_name,
                    max_length=self.config.max_length,
                    batch_size=self.config.batch_size,
                ),
            )
            dataset, _ = dataset_builder()

        trainer_config = ReinforceConfig(
            log_path=log_path,
            model_name=model_name,
            dataset_builder=dataset_builder,
            renderer_name=renderer_name,
            reinforce_version=self.config.reinforce_version,
            max_length=self.config.max_length,
            learning_rate=self.config.learning_rate,
            lr_schedule=self.config.lr_schedule,
            num_epochs=self.config.num_epochs,
            lora_rank=self.config.lora_rank,
            reward_alpha=self.config.reward_alpha,
            initial_baseline=self.config.initial_baseline,
            save_every=self.config.save_every,
            max_steps=self.config.max_steps,
            wandb_project=self.config.wandb_project,
            wandb_name=self.config.wandb_name,
            rollout_max_tokens=self.config.reinforce_rollout_max_tokens,
            rollout_temperature=self.config.reinforce_rollout_temperature,
            agentic_max_turns=self.config.reinforce_agentic_max_turns,
            agentic_max_turns_per_step=self.config.reinforce_agentic_max_turns_per_step,
            agentic_max_steps=self.config.reinforce_agentic_max_steps,
            agentic_enable_bash=self.config.reinforce_agentic_enable_bash,
            agentic_tool_timeout_s=self.config.reinforce_agentic_tool_timeout_s,
            agentic_max_trajectory_tokens=self.config.reinforce_agentic_max_trajectory_tokens,
            log_rollout_samples=self.config.reinforce_log_rollout_samples,
            rollout_sample_log_chars=self.config.reinforce_rollout_sample_log_chars,
        )

        def build(
            training_client: tinker.TrainingClient,
            service_client: tinker.ServiceClient,
        ) -> Trainer:
            return REINFORCETrainer(
                logger=log,
                config=trainer_config,
                training_client=training_client,
                service_client=service_client,
            )

        return dataset, build

    def _prepare_opd(self, train_path: str, model_name: str, renderer_name: str):
        """Build the weight-format online artifact OPD dataset and trainer."""
        tokenizer = get_tokenizer(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
        if hasattr(renderer, "strip_thinking_from_history"):
            renderer.strip_thinking_from_history = self.config.opd_strip_thinking_from_history
            log.info(
                "Dataset renderer %s: strip_thinking_from_history=%s",
                type(renderer).__name__,
                self.config.opd_strip_thinking_from_history,
            )
        artifact_only = self.config.opd_artifact_only_rollout_instruction
        if self.config.opd_extract_version in ("v2", "agentic") and artifact_only:
            log.warning(
                "opd_artifact_only_rollout_instruction=True is v1-only and is "
                "incompatible with opd_extract_version=%r. Forcing it to False "
                "for this OPD training run.",
                self.config.opd_extract_version,
            )
            artifact_only = False
        dataset = OnlineOPDRolloutDataset.from_weight_json(
            path=train_path,
            renderer=renderer,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
            pair_mode=self.config.opd_pair_mode,
            use_gt=self.config.opd_use_gt,
            use_student=self.config.opd_use_student,
            artifact_only_instruction=artifact_only,
            extract_version=self.config.opd_extract_version,
        )

        log_path = self._new_round_log_path("opd")
        trainer_config = ArtifactOPDConfig(
            model_name=model_name,
            renderer_name=renderer_name,
            log_path=log_path,
            lora_rank=self.config.lora_rank,
            learning_rate=self.config.learning_rate,
            lr_schedule=self.config.lr_schedule,
            num_epochs=self.config.num_epochs,
            save_every=self.config.save_every,
            max_steps=self.config.max_steps,
            wandb_project=self.config.wandb_project,
            wandb_name=self.config.wandb_name,
            pair_mode=self.config.opd_pair_mode,
            use_gt=self.config.opd_use_gt,
            use_student=self.config.opd_use_student,
            topk=self.config.opd_topk,
            teacher_temperature=self.config.opd_teacher_temperature,
            online_rollout=True,
            rollout_max_tokens=self.config.opd_max_tokens,
            rollout_temperature=self.config.opd_temperature,
            rollout_attempts=self.config.opd_rollout_attempts,
            log_rollout_samples=self.config.opd_log_rollout_samples,
            rollout_sample_log_chars=self.config.opd_rollout_sample_log_chars,
            log_teacher_prompts=self.config.opd_log_teacher_prompts,
            artifact_only_rollout_instruction=artifact_only,
            extract_version=self.config.opd_extract_version,
            strip_thinking_from_history=self.config.opd_strip_thinking_from_history,
            rollout_pipeline=self.config.opd_rollout_pipeline,
            agentic_max_turns=self.config.opd_agentic_max_turns,
            agentic_max_turns_per_step=self.config.opd_agentic_max_turns_per_step,
            agentic_max_steps=self.config.opd_agentic_max_steps,
            agentic_enable_bash=self.config.opd_agentic_enable_bash,
            agentic_tool_timeout_s=self.config.opd_agentic_tool_timeout_s,
            agentic_max_trajectory_tokens=self.config.opd_agentic_max_trajectory_tokens,
        )

        def build(
            training_client: tinker.TrainingClient,
            service_client: tinker.ServiceClient,
        ) -> Trainer:
            return OPDTrainer(
                logger=log,
                config=trainer_config,
                training_client=training_client,
                service_client=service_client,
                max_context_length=self.config.opd_max_context_length,
            )

        return dataset, build


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    assert args.config.endswith(".yaml"), "Config file must be a YAML file"

    config = Config.from_yaml(args.config)
    server = Server(config)
    app = server._build_app()
    uvicorn.run(app, host=config.proxy_host, port=config.proxy_port)
