import json
import os
import tempfile
import yaml
import logging
import asyncio
import argparse
import time

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
from weight.train.run_dpo import Config as DPOConfig
from weight.train.run_reinforce import Config as ReinforceConfig
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
    opd_artifact_only_rollout_instruction: bool = False

    # -- REINFORCE-specific --
    reward_alpha: float = 0.05
    initial_baseline: float = 0.0

    # -- Logging --
    wandb_project: str | None = None
    wandb_name: str | None = None

    # -- Proxy Server --
    proxy_host: str = "localhost"
    proxy_port: int = 8000
    state_path: str = "state.json"
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

        self.model_manager = ModelManager(
            tinker_api_key=self._API_KEY,
            tinker_base_url=self._TINKER_BASE_URL,
            state_path=config.state_path,
            mode=config.mode,
            lora_rank=config.lora_rank,
            preload_model=config.preload_model,
        )

        self.sessions_queue: asyncio.Queue = asyncio.Queue()
        self.training_queue: asyncio.Queue = asyncio.Queue()
        self.training_event = asyncio.Event()
        self.training_lock = asyncio.Lock()
        
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

        log_path = f"logs/weight_dpo/{int(time.time())}"
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

        log_path = f"logs/weight_reinforce/{int(time.time())}"
        trainer_config = ReinforceConfig(
            log_path=log_path,
            model_name=model_name,
            dataset_builder=dataset_builder,
            renderer_name=renderer_name,
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
        dataset = OnlineOPDRolloutDataset.from_weight_json(
            path=train_path,
            renderer=renderer,
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
            pair_mode=self.config.opd_pair_mode,
            use_gt=self.config.opd_use_gt,
            use_student=self.config.opd_use_student,
            artifact_only_instruction=self.config.opd_artifact_only_rollout_instruction,
        )

        log_path = f"logs/weight_opd/{int(time.time())}"
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
            artifact_only_rollout_instruction=self.config.opd_artifact_only_rollout_instruction,
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
